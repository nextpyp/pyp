import math
import os
import multiprocessing
from pathlib import Path

from pyp.streampyp.web import Web
from pyp.system import project_params, slurm
from pyp.system.local_run import run_shell_command, stream_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_pyp_configuration, standalone_mode, run_pyp, run_slurm, run_ssh
from pyp.utils import get_relative_path
from pyp.system.mpi import submit_jobs_to_workers

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

def calculate_effective_bundle_size(parameters,processes):
    """Calculate effective bundle size taking into account any specified resource limits

    Args:
        parameters (dict): pyp parameters
        processes (int): total number of processes

    Returns:
        int: size of bundle
    """    
    
    net_processes = math.ceil( float(processes) / parameters["slurm_bundle_size"])
    
    all_cpu_nodes = int(parameters["slurm_max_cpus"])
    threads = int(parameters["slurm_tasks"])
    if all_cpu_nodes > 0:
        simultaneous_tasks_by_cpus = math.floor(all_cpu_nodes / threads )
    else:
        simultaneous_tasks_by_cpus = net_processes

    # enforce max amount of memory
    all_memory_nodes = int(parameters["slurm_max_memory"])
    memory = parameters["slurm_tasks"]*parameters["slurm_memory_per_task"]
    if all_memory_nodes > 0:
        simultaneous_tasks_by_memory = math.floor(all_memory_nodes / memory)
    else:
        simultaneous_tasks_by_memory = net_processes

    # keep the most limiting of the two
    slurm_bundle_size = min( simultaneous_tasks_by_cpus, simultaneous_tasks_by_memory )
    
    return slurm_bundle_size, net_processes    

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

    with open(os.path.join(submit_dir, command_file)) as f:
        commands = [line.replace("\n", "") for line in f if len(line) > 0]

    # count the number of commands, assign to processes
    processes = len(commands)

    # limit bundle size to number of nodes
    if os.path.exists(".pyp_config.toml"):
        par_dir = "."
    elif os.path.exists("../.pyp_config.toml"):
        par_dir = ".."
    elif os.path.exists("../../.pyp_config.toml"):
        par_dir = "../.."
    else:
        raise Exception("can't find .pyp_config.toml")

    # calculate effective bundle size
    parameters = project_params.load_parameters(par_dir)
    slurm_bundle_size, net_processes = calculate_effective_bundle_size(parameters,processes)

    # only perform the following when tasks_per_arr > 1
    if "sess_" in jobtype:
        cmdlist = [
            # filter the commands list: pick the line of the script that matches the task id
            "eval `cat '%s' | awk -v line=$SLURM_ARRAY_TASK_ID '{if (NR == line) print $0}'`"
            % command_file,
        ]
    else:

        config = get_pyp_configuration()
        scratch_config = config["pyp"]["scratch"]
        if "$" in scratch_config:
            os.environ["PYP_SCRATCH"] = os.path.expandvars(config["pyp"]["scratch"])
        else:
            os.environ["PYP_SCRATCH"] = scratch_config
        os.environ["PYP_SCRATCH"] = str(Path(os.environ["PYP_SCRATCH"]) / os.environ["USER"])

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
                "export csp_local_merge=csp_local_merge; {0} --stacks_files stacks.txt --par_files pars.txt --ordering_file ordering.txt --project_path_file project_dir.txt --output_basename $OUTPUT_BASENAME --path {1}/$OUTPUT_BASENAME\n".format(
                    run_pyp(command="pyp", script=True, cpus=threads),
                    Path(os.environ["PYP_SCRATCH"]),
                ),
            )

    if slurm_bundle_size > 1 and net_processes > slurm_bundle_size:
        if Web.exists:
            bundle = int(slurm_bundle_size)
        else:
            bundle = "%" + str(int(slurm_bundle_size))
    else:
        if Web.exists:
            bundle = None
        else:
            bundle = ""

    if Web.exists or standalone_mode():

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
                
            # handle the standalone case separately since we don't have SLURM env variables in this case
            if standalone_mode():
                output_basename = f"{os.environ['SLURM_JOB_ID']}"

            scratch = Path(os.environ["PYP_SCRATCH"])
            csp_local_merge_command = f"export OPENBLAS_NUM_THREADS=1; unset {jobtype}; export csp_local_merge=csp_local_merge; {run_pyp(command='pyp', script=True, cpus=threads)} --stacks_files stacks.txt --par_files pars.txt --ordering_file ordering.txt --project_path_file project_dir.txt --output_basename {output_basename} --path '{scratch}/{output_basename}'"

        cmdgrid = [[]]
        job_counter = array_counter = array_job_counter = 0
        
        # manually manage resources when in standalone mode so we don't overload the server
        if Web.exists:
            cpus = ""
        elif standalone_mode():
            tasks_per_arr = max( parameters["slurm_bundle_size"], math.floor(processes*parameters["slurm_tasks"]/multiprocessing.cpu_count()) )
            cpus = f"export SLURM_CPUS_PER_TASK={parameters['slurm_tasks']}; SLURM_NTASKS={parameters['slurm_tasks']}; export OMP_NUM_THREADS={parameters['slurm_tasks']}; export MKL_NUM_THREADS={parameters['slurm_tasks']}; "

        while job_counter < processes:
            if array_job_counter < tasks_per_arr:
                array_job_counter += 1
            else:
                array_job_counter = 1
                array_counter += 1
                cmdgrid.append([])
            if Web.exists:
                cmdgrid[array_counter].append(
                    '/bin/bash -c "' + cpus + commands[job_counter] + '"'
                )
            elif standalone_mode():
                cmdgrid[array_counter].append(
                    '/bin/bash -c "' + cpus + commands[job_counter].replace('/opt/pyp/bin/run/pyp','python -u /opt/pyp/src/pyp_main.py') + '"'
                )
            job_counter += 1

        # do not run csp_local_merge as last command when in standalone mode since we only need to run one instance
        if csp_no_stacks and "classmerge" not in jobtype and len(csp_local_merge_command) > 0 and Web.exists:
            for batch in cmdgrid:
                batch.append('/bin/bash -c "' + csp_local_merge_command + '"')

        # add MPI settings if needed
        mpi = None
        if threads > 1:
            mpi = {"oversubscribe": True, "cpus": threads}

        if Web.exists:
            
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
            # standalone mode
            commands = []
            for batch in cmdgrid:
                commands.append( "; ".join(batch) )
            submit_jobs_to_workers(commands, working_path=submit_dir)
            
            # run only one instance of csp_local_merge once all cspswarm processes finish
            if csp_no_stacks and "classmerge" not in jobtype and len(csp_local_merge_command) > 0:
                stream_shell_command('/bin/bash -c "' + csp_local_merge_command.replace('/opt/pyp/bin/run/pyp','python -u /opt/pyp/src/pyp_main.py') + '"')
            return "standalone"
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
        "--cpus-per-task=%d" % threads,
        "--time=%s" % walltime,
        "--mem=%sG" % memory,
    ]
    if gres != "" and gres != None:
        args.append("--gres=%s" % gres)
    if account != "" and account != None:
        args.append("--account=%s" % account)
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
    if not cmd.startswith("/") and not cmd.startswith('mkdir'):
        cmd = "./%s" % cmd
    if os.path.exists(cmd):
        cmd = "'%s'" % cmd

    if Web.exists or standalone_mode():

        # add MPI settings if needed
        mpi = None
        if threads > 1:
            mpi = {"oversubscribe": True, "cpus": threads}

        if Web.exists:
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
            
        elif standalone_mode():
            cpus = f"export SLURM_CPUS_PER_TASK={threads}; SLURM_NTASKS={threads}; export OMP_NUM_THREADS={threads}; export MKL_NUM_THREADS={threads}; "
            new_cmd = cmd.replace("'/opt/pyp/bin/run/pyp'","python -u /opt/pyp/src/pyp_main.py")
            command = '/bin/bash -c "' + cpus + f"export {jobtype}={jobtype}; cd {submit_dir}; {new_cmd}" + '"'
            stream_shell_command(command,verbose=True)
            return "standalone"

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
        command = "{0} {2} {3} --mem={6}G --time {8} --job-name={1} {4} {7} {5}".format(
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
            walltime,
        )
        command = run_ssh(command)
        [id, error] = run_shell_command(command, verbose=False)
        if "error" in error or "failed" in error:
            logger.error(error)
            raise Exception(error)
        return id
