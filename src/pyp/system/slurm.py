import math
import multiprocessing
import os
import shutil
import sys
from time import time
from pathlib import Path

import numpy as np

from pyp.inout.metadata.frealign_parfile import Parameters
from pyp.streampyp import jobs
from pyp.streampyp.web import Web
from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_pyp_configuration, run_pyp, run_slurm, run_ssh
from pyp.system.utils import get_shell_multirun_path, is_atrf, is_biowulf2, is_dcc, qos, get_slurm_path
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def check_sbatch_job_finish(jobname):
    import time

    # import getpass
    # user = getpass.getuser()
    sbatch_job = 1
    while sbatch_job > 0:
        # command = 'squeue -u %s -o ' % user
        command = "squeue --me -o %j"
        command = run_ssh(command)
        [info, error] = run_shell_command(command, verbose=False)
        if jobname in info:
            time.sleep(5)
            logger.info("waiting for %s job finish", jobname)
        else:
            sbatch_job = 0
    return


def get_array_job(parameters, series):
    # get job array number from series
    """
    START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $TASKS_PER_ARR + 1 ))
    MAX_TASK=$(wc -l < %s)
    END_NUM=$(( $SLURM_ARRAY_TASK_ID * $TASKS_PER_ARR ))
    """
    tasks_per_arr = int(parameters["slurm_bundle_size"])
    return (series - 1) // tasks_per_arr + 1


def calculate_rec_swarm_required_resources(mparameters, fparameters, particles):
    nodes = 32
    threads = 32

    # TODO: should this be configurable by the user?
    threads = 17
    if "dcc" in fparameters["slurm_queue"]:
        nodes = len(
            fparameters["slurm_queue"].split("nodelist=")[1].split()[0].split(",")
        )
    elif "--nodes" in fparameters["slurm_queue"]:
        nodes = int(fparameters["slurm_queue"].split("--nodes=")[1].split(" ")[0])
    else:
        nodes = 10
    cores_per_node = multiprocessing.cpu_count()

    # split in 20 or minimum of 1000 particles
    increment = max(int(particles / nodes + 1), 1000)

    increment = max(int(particles / (nodes * cores_per_node / threads) + 1), 1000,)

    if int(mparameters["extract_box"]) > 450:
        increment = 2500

    return increment, threads


def create_pyp_swarm_file(parameters, files, timestamp, swarm_file="pre_process.swarm"):
    with open(os.path.join("swarm", swarm_file), "w") as f:
        if "extract_fmt" in parameters.keys() and "frealign_local" in parameters["extract_fmt"]:
            # remove existing files
            try:
                [
                    os.remove("../log/per_particle_refinement_{0}.log".format(logfile))
                    for logfile in files
                ]
            except:
                pass
            f.write(
                "\n".join(
                    [
                        "cd {3}/swarm; export {2}swarm={2}swarm; {0} --keep --file raw/{1} --path {3} 2>&1 | tee ../log/{1}_per_particle_refinement.log".format(
                            run_pyp(command="pyp", script=True),
                            s,
                            parameters["data_mode"],
                            os.getcwd(),
                            timestamp,
                        )
                        for s in files
                    ]
                )
            )
        else:
            f.write(
                "\n".join(
                    [
                        "cd {4}/swarm; export {3}swarm={3}swarm; {0} --file raw/{2} --path {4} 2>&1 | tee ../log/{2}.log".format(
                            run_pyp(
                                command="pyp",
                                script=True,
                                cpus=parameters["slurm_tasks"],
                            ),
                            timestamp,
                            s,
                            parameters["data_mode"],
                            os.getcwd(),
                        )
                        for s in files
                    ]
                )
            )
        f.write("\n")

    return swarm_file


def create_train_swarm_file(parameters, timestamp, swarm_file="train.swarm"):
    with open(os.path.join("swarm", swarm_file), "w") as f:
        f.write(
            "cd {2}; export {1}train={1}train; {0} 2>&1 | tee log/{3}_{1}train.log".format(
                run_pyp(command="pyp", script=True),
                parameters["data_mode"],
                os.getcwd(),
                timestamp,
            )
        )
        f.write("\n")

    return swarm_file


def create_csp_swarm_file(files, parameters, iteration, swarm_file="cspswarm.swarm"):
    f = open(swarm_file, "w")
    f.write(
        "\n".join(
            [
                "cd {0}; export cspswarm=cspswarm; {1} --file {2} --iter {3} --no-skip --no-debug 2>&1 | tee ../log/{2}_csp.log".format(
                    os.getcwd(),
                    run_pyp(command="pyp", script=True, cpus=parameters["slurm_tasks"]),
                    s,
                    iteration,
                )
                for s in files
            ]
        )
    )
    f.write("\n")
    f.close()

    return swarm_file


def create_csp_classmerge_file(iteration, parameters, swarm_file="csp_class_merge.swarm"):
    f = open(swarm_file, "w")
    # this function would not be called if iteration == 2, but just in case
    class_num = parameters["class_num"] if parameters["refine_iter"] > 2 else 1
    f.write(
        "\n".join(
            [
                "cd {0}; export classmerge=classmerge; {1} --iter {3} --classId {2} --no-skip --no-debug 2>&1 | tee ../log/r{2:02d}_csp_classmerge.log".format(
                    os.getcwd(),
                    run_pyp(command="pyp", script=True, cpus=parameters["slurm_tasks"]),
                    class_id+1, 
                    iteration,
                )
                for class_id in range(class_num)
            ]
        )
    )
    f.write("\n")
    f.close()

    return swarm_file


def create_script_file(swarm_file, command, path):
    with open(swarm_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("cd %s\n" % path)
        f.write(command)
        f.write("\n")

    return swarm_file


def create_def_swarm_file(
    fp, parfile, tolerance, def_swarm_file="swarm/frealign_dsearch.swarm"
):
    with open(def_swarm_file, "w") as fsave:

        # input = np.array( [line.split() for line in open( args.parfile ) if not line.startswith('C') ], dtype=float )
        input = Parameters.from_file(parfile).data
        films = np.unique(input[:, 7])

        for f in films:

            if "frealignx" in fp["metric"]:
                tilts = np.unique(input[input[:, 7] == f][:, 20])
            else:
                tilts = np.unique(input[input[:, 7] == f][:, 19])

            for t in tilts:

                fsave.write(
                    "cd {0}/swarm; umask=33; export frealign_def=frealign_def; {1} --parfile {2} --film {3} --scanor {4} --tolerance {5}".format(
                        os.getcwd(),
                        run_pyp(command="fyp", script=True),
                        parfile,
                        int(f),
                        int(t),
                        tolerance,
                    )
                )
                fsave.write("\n")

    return def_swarm_file


def create_def_merge_file(
    parfile, tolerance, def_merge_file="swarm/frealign_mdefocus.swarm"
):
    with open(def_merge_file, "w") as f:
        f.write(
            "#!/bin/bash\ncd {0}/swarm; export frealign_def_merge=frealign_def_merge; {1} --parfile {2} --tolerance {3}".format(
                os.getcwd(), run_pyp(command="fyp", script=True), parfile, tolerance
            )
        )

    return def_merge_file


def create_rec_swarm_file(iteration, alignment_option):
    rec_swarm_file = "swarm/frealign_mrecons_%02d.swarm" % (iteration)
    f = open(rec_swarm_file, "w")
    f.write(
        "#!/bin/bash\ncd {3}/swarm; export frealign_rec=frealign_rec; {0} --iteration {1} --alignment_option {2}".format(
            run_pyp(command="fyp", script=True),
            iteration,
            alignment_option,
            os.getcwd(),
        )
    )
    f.close()
    run_shell_command("chmod u+x %s" % rec_swarm_file)

    return rec_swarm_file


def create_rec_split_swarm_file(iteration, particles, classes, increment):
    """Within frealign_rec, create rec split swarm file"""

    rec_split_swarm_file = "swarm/frealign_mrecons_split_%02d.swarm" % (iteration)
    f = open(rec_split_swarm_file, "w")
    for ref in range(classes):
        first = count = 0
        last = min(first + increment - 1, particles - 1)
        for first in range(1, particles + 1, increment):
            last = min(first + increment - 1, particles)
            if particles - last < increment / 2:
                last = particles
            f.write(
                "cd {0}/swarm; export frealign_rec_split=frealign_rec_split; {1} --iteration {2} --ref {3} --first {4} --last {5} --count {6}\n".format(
                    os.getcwd(),
                    run_pyp(command="fyp", script=True),
                    iteration,
                    ref + 1,
                    first,
                    last,
                    count,
                )
            )
            count += 1
            if last == particles:
                break
    f.close()

    return rec_split_swarm_file


def create_rec_merge_swarm_file(iteration):
    """Within frealign_rec, create rec merge swarm file"""

    rec_swarm_file = "swarm/frealign_mrecons_%02d.swarm" % (iteration)
    f = open(rec_swarm_file, "w")
    f.write(
        "#!/bin/bash\ncd {0}/swarm; export frealign_rec_merge=frealign_rec_merge; {1} --iteration {2}".format(
            os.getcwd(), run_pyp(command="fyp", script=True), iteration
        )
    )
    f.close()
    run_shell_command("chmod u+x %s" % rec_swarm_file)

    return rec_swarm_file


def create_ref_swarm_file(fp, iteration, classes, particles, metric, increment):
    ref_swarm_file = "swarm/frealign_msearch_%02d.swarm" % (iteration)
    f = open(ref_swarm_file, "w")

    if is_biowulf2():
        if "-g" in fp["queue"]:
            mem = int(fp["queue"].split("-g")[1])
            threads = 125 / mem
            if "ibfdr" in fp["queue"]:
                threads = 58 / mem
        else:
            threads = 32

        threads = 1
    else:
        threads = 1

    first = count = 0
    last = min(first + increment - 1, particles - 1)
    thread_count = 0
    for first in range(1, particles + 1, increment):
        last = min(first + increment - 1, particles)
        for ref in range(classes):
            f.write(
                "cd {6}/swarm; umask=33; export frealign_ref=frealign_ref; {0} --iteration {1} --ref {2} --first {3} --last {4} -metric {5}".format(
                    run_pyp(command="fyp", script=True),
                    iteration,
                    ref + 1,
                    first,
                    last,
                    metric.replace("-metric", ""),
                    os.getcwd(),
                )
            )
            thread_count += 1
            if thread_count == threads:
                f.write("\n")
                thread_count = 0
            else:
                f.write(" & ")
        count += 1
    f.close()

    return ref_swarm_file


def submit_jobs(
    submit_dir,
    command_file,
    jobtype,
    jobname,
    queue,
    scratch=0,
    threads=2,
    memory=2,
    walltime="72:00:00",
    dependencies="",
    tasks_per_arr=1,
    csp_no_stacks=False,
):
    """Submit jobs to batch system"""

    id = ""

    if queue == None:
        queue = ""

    # determine if this is a command file or an executable
    if os.path.isfile(command_file):
        myfile = command_file
    elif os.path.isfile(os.path.join(submit_dir, command_file)):
        myfile = os.path.join(submit_dir, command_file)
    else:
        myfile = ""

    # There are three classes of commands
    # 1) File with one command per line ()
    # 2) Bash script with single command (bash in first line)
    # 3) Binary or executable command (# and not bash)

    is_list = False
    is_script = False

    if os.path.isfile(myfile):
        with open(myfile) as f:
            firstline = f.readline()
            is_list = "#" not in firstline
            is_script = "#" in firstline and "bash" in firstline

    if is_list:
        with open(myfile) as file:
            nonempty_lines = [line for line in file if line != "\n"]
            procs = len(nonempty_lines)
    else:
        procs = 1

    # format dependencies based on the environment/batch system
    if len(dependencies) == 0:
        depend = ""
    else:
        depend = " --dependency=afterany:{0}".format(dependencies)

    # call the corresponding submission function
    if is_list:
        id = jobs.submit_commands(
            submit_dir,
            command_file,
            jobtype,
            jobname,
            queue,
            threads,
            memory,
            walltime,
            dependencies,
            tasks_per_arr,
            csp_no_stacks,
        )
    else:
        id = jobs.submit_script(
            submit_dir,
            command_file,
            jobtype,
            jobname,
            queue,
            threads,
            memory,
            walltime,
            dependencies,
            is_script,
        )

    logger.info("Submitting {0} job(s) ({1})".format(procs, id.strip()))

    return id if jobtype != "cspswarm" and jobtype != "classmerge" else (id, procs)

def transfer_stack_to_scratch(dataset):
    stack = "%s_stack.mrc" % dataset

    if not os.path.exists("/scratch/%s_stack.mrc" % dataset):
        logger.info("starting copying the stack file over to /scratch")
        start = time.time()
        shutil.copy2(stack, "/scratch")
        end = time.time()
        time_elapsed = end - start
        logger.info(("{} seconds taken to transfer stack file\n".format(time_elapsed)))
    else:
        logger.info("stack file already in /scratch")


def get_frealign_queue(mp, fp, iteration):
    if "cc" not in project_params.param(fp["metric"], iteration):
        # make sure we use a high memory node since we need enough scratch space
        if (
            "hrem" in fp["queue"]
            and int(fp["queue"].split(":")[0]) > 1
            and int(mp["extract_box"]) > 512
        ):
            queue = fp["queue"] + ",mem=384g"
        else:
            queue = fp["queue"]
    else:
        queue = fp["queue"]

    return queue


def get_total_seconds(slurm_walltime: str) -> int:
    
    time = slurm_walltime.split("-")
    total_seconds = 0

    MIN = 60 
    HOUR = 60 * 60
    DAY = 60 * 60 * 24

    if len(time) > 1:
        day = int(time[0])
        time = time[1].split(":")
        total_seconds += day * DAY
    else:
        time = time[0].split(":")

    seconds = int(time[-1])
    minutes = int(time[-2])
    hours = int(time[-3])
    total_seconds += hours * HOUR + minutes * MIN + seconds 

    return total_seconds


def launch_csp(micrograph_list: list, parameters: dict, swarm_folder: Path):
    """launch_csp Launch csp

    Parameters
    ----------
    micrograph_list : list
        List of tilt-series/micrographs to submit 
    parameters : dict
        PYP parameters
    swarm_folder : Path
        Path to the swarm folder
    """

    current_directory = Path().cwd()
    os.chdir(swarm_folder)

    iteration = parameters["refine_iter"]

    if len(micrograph_list) > 0:
        swarm_file = create_csp_swarm_file(
            micrograph_list, parameters, iteration, "cspswarm.swarm"
        )
    else:
        parameters["slurm_merge_only"] = True

    jobtype = "cspswarm"
    jobname = "Iteration %d (split)" % parameters["refine_iter"] if Web.exists else "cspswarm"

    # submit jobs to batch system
    if parameters["slurm_merge_only"]:
        id = ""
    else:
        if parameters["csp_parx_only"]:
            id = submit_jobs(
                ".",
                swarm_file,
                jobtype,
                jobname,
                queue=parameters["slurm_queue"] if "slurm_queue" in parameters else "",
                scratch=0,
                threads=2,
                memory=20,
            ).strip()
        else:
            (id, procs) = submit_jobs(
                ".",
                swarm_file,
                jobtype,
                jobname,
                queue=parameters["slurm_queue"] if "slurm_queue" in parameters else "",
                threads=parameters["slurm_tasks"],
                memory=parameters["slurm_memory"],
                walltime=parameters["slurm_walltime"],
                tasks_per_arr=parameters["slurm_bundle_size"],
                csp_no_stacks=parameters["csp_no_stacks"],
            )

            # just use the first array job as prerequisite
            id = id.strip() + "_1"

    if parameters["class_num"] > 1 and iteration > 2:

        swarm_classmerge_file = create_csp_classmerge_file(
            iteration, parameters, "csp_class_merge.swarm"
        )
            
        jobtype = "classmerge"
        jobname = "Iteration %d (classmerge)" % parameters["refine_iter"] if Web.exists else "classmerge"
        
        (id, procs) = submit_jobs(
            ".",
            swarm_classmerge_file,
            jobtype,
            jobname,
            queue=parameters["slurm_queue"] if "slurm_queue" in parameters else "",
            threads=parameters["slurm_tasks"],
            memory=parameters["slurm_memory"],
            walltime=parameters["slurm_walltime"],
            tasks_per_arr=1, # one class per array job
            csp_no_stacks=parameters["csp_no_stacks"],
            dependencies=id,
        )

    jobtype = "cspmerge"
    jobname = "Iteration %d (merge)" % parameters["refine_iter"] if Web.exists else "cspmerge"

    submit_jobs(
        ".",
        run_pyp(command="pyp"),
        jobtype,
        jobname,
        queue=parameters["slurm_queue"] if "slurm_queue" in parameters else "",
        scratch=0,
        threads=parameters["slurm_merge_tasks"],
        memory=parameters["slurm_merge_memory"],
        walltime=parameters["slurm_merge_walltime"],
        dependencies=id,
    )

    os.chdir(current_directory)