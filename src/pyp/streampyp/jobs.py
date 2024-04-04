import json
import math
import os
import subprocess
from pathlib import Path

from genericpath import exists

from pyp.streampyp.web import Web
from pyp.system import project_params, slurm
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_pyp_configuration, run_pyp, run_slurm, run_ssh
from pyp.utils import get_relative_path, symlink_force

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def _absolutize_path(path):
    if path[0] == "/":
        return path
    else:
        return os.path.join(os.getcwd(), path)

def get_gres_option(use_gpu,gres):
    options = []
    if use_gpu and "gpu:" not in gres:
        options.append("gpu:1")
    if len(gres) > 0:
        options.append(gres)
    gpu_gres = ",".join(options)
    if not Web.exists and len(gpu_gres) > 0:
        gpu_gres = "--gres=" + gpu_gres
    return gpu_gres

def submit_commands(
    submit_dir,
    command_file,
    jobtype,
    jobname,
    queue,
    threads,
    memory,
    gres,
    account,
    walltime,
    dependencies,
    tasks_per_arr,
    csp_no_stacks,
    use_gpu,
):

    # example inputs:
    # submit_dir: swarm
    # command_file: pre_process.swarm
    # FILE:
    # cd /var/out/swarm; /opt/pyp/python/checknode.py; export sprswarm=sprswarm; /opt/pyp/pyp/pyp_main.py--file raw/May08_03.05.02.bin > ../log/20201015_170810_May08_03.05.02.bin.log
    # jobname: sprswarm
    # queue:
    # threads: 7
    # memory: 700
    # walltime: 2:00:00
    # dependencies:

    if threads == 0:
        message = "Number of threads must be positive"
        logger.error(message)
        raise Exception(message)

    if "sess_" in jobtype:
        task_file = command_file
    else:
        task_file = os.path.join(os.getcwd(), submit_dir, command_file)

    with open(os.path.join(submit_dir, command_file)) as f:
        commands = [line.replace("\n", "") for line in f if len(line) > 0]

    # count the number of commands, assign to processes
    processes = len(commands)

    # only perform the following when tasks_per_arr > 1
    if "sess_" in jobtype:
        cmdlist = [
            # filter the commands list: pick the line of the script that matches the task id
            "eval `cat '%s' | awk -v line=$SLURM_ARRAY_TASK_ID '{if (NR == line) print $0}'`"
            % command_file,
        ]
    else:

        cmdlist = [
            "export OPENBLAS_NUM_THREADS=1\n",
            "TASKS_PER_ARR={}\n".format(tasks_per_arr),
            """
if [[ ! -z $SLURM_ARRAY_JOB_ID ]]
then
#        echo In slurm array mode
        OUTPUT_BASENAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
#        echo In normal slurm mode
        OUTPUT_BASENAME=${SLURM_JOB_ID}_1
fi

#echo Output basename is $OUTPUT_BASENAME

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $TASKS_PER_ARR + 1 ))
MAX_TASK=$(wc -l < %s)
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $TASKS_PER_ARR ))
END_NUM=$(( $MAX_TASK < $END_NUM ? $MAX_TASK : $END_NUM ))

# Print the task and run range
# echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM
            """
            % command_file,
            """
for (( run=$START_NUM; run<=END_NUM; run++ )); do
# echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
#Do your stuff here
eval `cat '%s' | awk -v line=$run '{if (NR == line) print $0}'`
done
            """
            % command_file,
        ]

        if csp_no_stacks and "classmerge" not in jobtype:
            cmdlist.append("unset %s \n" % jobtype)
            cmdlist.append(
                "export csp_local_merge=csp_local_merge; {0} --stacks_files stacks.txt --par_files pars.txt --ordering_file ordering.txt --project_path_file project_dir.txt --output_basename $OUTPUT_BASENAME --path '{1}/$OUTPUT_BASENAME'\n".format(
                    run_pyp(command="pyp", script=True, cpus=threads),
                    Path(os.environ["PYP_SCRATCH"]),
                ),
            )

    # limit bundle size to number of nodes
    if os.path.exists(".pyp_config.toml"):
        par_dir = "."
    elif os.path.exists("../.pyp_config.toml"):
        par_dir = ".."
    elif os.path.exists("../../.pyp_config.toml"):
        par_dir = "../.."
    else:
        raise Exception("can't find .pyp_config.toml")

    # enforce max number of threads
    all_cpu_nodes = int(project_params.load_parameters(par_dir)["slurm_max_cpus"])
    bundle_size = all_cpu_nodes / threads

    # enforce max amount of memory
    all_memory_nodes = int(project_params.load_parameters(par_dir)["slurm_max_memory"])
    bundle_by_memory = all_memory_nodes / memory

    # keep the most limiting of the two
    bundle_size = min( bundle_size, bundle_by_memory )

    net_processes = int(math.ceil(float(processes) / tasks_per_arr))
    if bundle_size > 0 and net_processes < bundle_size:
        if Web.exists:
            bundle = None
        else:
            bundle = ""
    else:
        if Web.exists:
            bundle = int(bundle_size)
        else:
            bundle = "%" + str(int(bundle_size))

    if Web.exists:

        # convert dependencies into a list
        if dependencies == "":
            dependencies = []
        else:
            dependencies = dependencies.split(",")

        if "sess_" in jobtype:
            csp_local_merge_command = ""
            tasks_per_arr = 1
        else:
            if processes > tasks_per_arr:
                # echo In slurm array mode
                output_basename = "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
            else:
                # echo In normal slurm mode
                output_basename = "${SLURM_JOB_ID}_1"

            scratch = Path(os.environ["PYP_SCRATCH"]).parent
            csp_local_merge_command = f"export OPENBLAS_NUM_THREADS=1; unset {jobtype}; export csp_local_merge=csp_local_merge; {run_pyp(command='pyp', script=True, cpus=threads)} --stacks_files stacks.txt --par_files pars.txt --ordering_file ordering.txt --project_path_file project_dir.txt --output_basename {output_basename} --path '{scratch}/{output_basename}'"

        cmdgrid = [[]]
        job_counter = array_counter = array_job_counter = 0
        while job_counter < processes:
            if array_job_counter < tasks_per_arr:
                array_job_counter += 1
            else:
                array_job_counter = 1
                array_counter += 1
                cmdgrid.append([])
            cmdgrid[array_counter].append(
                '/bin/bash -c "' + commands[job_counter] + '"'
            )
            job_counter += 1
        if csp_no_stacks and "classmerge" not in jobtype and len(csp_local_merge_command) > 0:
            for batch in cmdgrid:
                batch.append('/bin/bash -c "' + csp_local_merge_command + '"')

        # add MPI settings if needed
        mpi = None
        if threads > 1:
            mpi = {"oversubscribe": True, "cpus": threads}

        # launch job via streampyp
        if len(csp_local_merge_command) == 0:
            return Web().slurm_sbatch(
                web_name=jobname,
                cluster_name="pyp_"+jobtype,
                commands=Web.CommandsScript(cmdlist, processes, bundle),
                dir=_absolutize_path(submit_dir),
                args=get_slurm_args( queue=queue, threads=threads, walltime=walltime, memory=memory, jobname=jobname, gres=get_gres_option(use_gpu,gres), account=account),
                deps=dependencies,
                mpi=mpi,
            )
        else:
            return Web().slurm_sbatch(
                web_name=jobname,
                cluster_name="pyp_"+jobtype,
                commands=Web.CommandsGrid(cmdgrid, bundle),
                dir=_absolutize_path(submit_dir),
                args=get_slurm_args( queue=queue, threads=threads, walltime=walltime, memory=memory, jobname=jobname, gres=get_gres_option(use_gpu,gres), account=account),
                deps=dependencies,
                mpi=mpi,
            )
    else:

        if "sess_" in jobtype:
            multirun_file = "{0}/commands.swarm".format(submit_dir)
            multirun_file = os.path.join(
                submit_dir, command_file.replace(".swarm", ".array")
            )
        elif "-train" in jobtype:
            multirun_file = os.path.join(os.getcwd(), submit_dir, "train_commands.swarm")
        elif "classmerge" in jobtype:
            multirun_file = os.path.join(os.getcwd(), submit_dir, "classmerge_commands.swarm")
        else:
            multirun_file = os.path.join(os.getcwd(), submit_dir, "commands.swarm")

        with open(multirun_file, "w") as f:
            f.write("#!/bin/bash --login\n")
            f.write("#SBATCH --open-mode=append\n")
            f.write(f"#SBATCH --output='{submit_dir}/slurm-%A.out'\n")
            f.write("cd '%s'\n" % (submit_dir))

            if False and "gres=gpu" in queue:
                f.write("""
available_devs=""
for devidx in $(seq 0 15);
do
    string=$(nvidia-smi -i $devidx --query-compute-apps=pid --format=csv,noheader)
    if ! [[ $string = *"No devices were found"* ]]; then
	 if [[ -z "$available_devs" ]] ; then
            available_devs=$devidx
        else
            available_devs=$available_devs,$devidx
        fi
    fi
done
export CUDA_VISIBLE_DEVICES=$available_devs

"""
            )

            for line in cmdlist:
                f.write(line)

            run_shell_command("chmod u+x '{0}'".format(multirun_file), verbose=False)

        # format dependencies based on the environment/batch system
        if len(dependencies) == 0:
            depend = ""
        else:
            depend = " --dependency=afterany:{0}".format(dependencies)

        command = """{0} --array=1-{7}{8} --time={9} --mem={5}G --cpus-per-task={6} {1} {2} --job-name={4} {10} {3}""".format(
            run_slurm(command="sbatch", path=os.getcwd()),
            "--partition=%s" % queue if queue != '' else '',
            depend,
            multirun_file,
            "pyp_"+jobtype,
            memory,
            threads,
            net_processes,
            bundle,
            walltime,
            get_gres_option(use_gpu,gres),
        )
        command = run_ssh(command)
        [output, error] = run_shell_command(command, verbose=False)
        if "error" in error or "failed" in error:
            logger.warning(command)
            if not "sleeping and retrying" in error:
                raise Exception(error)
            else:
                logger.warning(error)
                id = output.split()[-1]
        else:
            id = output.split()[-1]
        return id

def get_slurm_args( queue, threads, walltime, memory, jobname, gres = None, account = None):
    args = [
        ("--partition=%s" % queue) if queue != '' else '',
        "--cpus-per-task=%d" % threads,
        "--time=%s" % walltime,
        "--mem=%sG" % memory,
        "--job-name='%s'" % jobname,
    ]
    if gres != "" and gres != None:
        args.append("--gres=%s" % json.dumps(gres))
    if account != "" and account != None:
        args.append("--account=%s" % json.dumps(account))
    return args

def submit_script(
    submit_dir,
    command_file,
    jobtype,
    jobname,
    queue,
    threads,
    memory,
    gres,
    account,
    walltime,
    dependencies,
    is_script,
    use_gpu=False,
):

    # example inputs:
    # submit_dir: swarm
    # command_file: /opt/pyp/python/pyp_main.py
    # FILE:
    # (oh yes... it's really the whole pyp_main.py script)
    # jobname: sprmerge
    # queue:
    # threads: 4
    # memory: 64
    # walltime: 48:00:00
    # dependencies: 5f89a4a4564747d02e5323b4

    if threads == 0:
        message = "Number of threads has to be positive"
        logger.error(message)
        raise Exception(message)

    # make sure the script path starts with a / or ./
    cmd = command_file
    if not cmd.startswith("/"):
        cmd = "./%s" % cmd
    if os.path.exists(cmd):
        cmd = "'%s'" % cmd

    if Web.exists:

        # add MPI settings if needed
        mpi = None
        if threads > 1:
            mpi = {"oversubscribe": True, "cpus": threads}

        # convert dependencies into a list
        if dependencies == "":
            dependencies = []
        else:
            dependencies = dependencies.split(",")

        # launch job via streampyp
        return Web().slurm_sbatch(
            web_name=jobname,
            cluster_name="pyp_"+jobtype,
            commands=Web.CommandsScript([cmd]),
            dir=_absolutize_path(submit_dir),
            env=[(jobtype, jobtype)],
            args=get_slurm_args( queue, threads, walltime, memory, jobname, get_gres_option(use_gpu,gres), account),
            deps=dependencies,
            mpi=mpi,
        )

    else:

        # format dependencies
        if dependencies == "":
            depend = ""
        else:
            depend = " --dependency=afterany:{0}".format(dependencies)

        if not is_script:
            command_name = os.path.split(command_file)[-1]
            command_file = os.path.join(os.getcwd(), submit_dir, jobtype + ".swarm")
            slurm.create_script_file(
                command_file,
                run_pyp(command=command_name, script=True, cpus=threads),
                path=os.path.join(os.getcwd(), submit_dir),
            )
        partition = f"--partition={queue}" if queue != '' else ''
        command = "{0} {2} {3} --mem={6}G --job-name={1} {4} {7} {5}".format(
            run_slurm(
                command="sbatch",
                path=os.path.join(os.getcwd(), submit_dir),
                env=jobtype,
            ),
            "pyp_"+jobtype,
            partition,
            "--cpus-per-task=%d" % threads,
            depend,
            command_file,
            memory,
            get_gres_option(use_gpu,gres),
        )
        command = run_ssh(command)
        [id, error] = run_shell_command(command, verbose=False)
        if "error" in error or "failed" in error:
            logger.error(error)
            raise Exception(error)
        return id
