import math
import multiprocessing
import os
import socket
import subprocess
from pathlib import Path

from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_mpirun_command, run_pyp
from pyp.utils import get_relative_path, timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def create_pyp_multirun_file(
    parameters, files, timestamp, nodes, mpirunfile="swarm/pre_process.swarm"
):
    f = open(mpirunfile, "w")
    f.write("#\!/bin/bash\n")
    f.write("cd '{0}/swarm'\n".format(os.getcwd()))
    f.write("export {0}swarm={0}swarm\n".format(parameters["data_mode"]))
    f.write("case $MP_CHILD in\n")
    group = 0
    f.write("{0})\n".format(group))
    # clean scratch
    process_per_node = len(files) / nodes
    additional_jobs = len(files) - nodes * process_per_node
    counter = 0
    for i in files:
        if counter == (process_per_node + 1):
            counter = 0
            group += 1
            additional_jobs -= 1
            f.write(";;\n")
            f.write("{0})\n".format(group))
            # f.write('clearscratch\n')
        f.write(
            """find %s -user $USER -exec rm -fr {} \;\n""" % os.environ["PYP_SCRATCH"]
        )
        f.write("mkdir -p %s\n" % Path(os.environ["PYP_SCRATCH"]) / str(i))
        f.write(
            "{0}/pyp_main.py --file raw/{2} > ../log/{1}_{2}.log".format(
                os.environ["PYTHONDIR"], timestamp, i
            )
        )
        # run multiple jobs per node
        # if counter == 0 or counter % 4:
        #    f.write('&')
        f.write("\n")
        counter += 1
        if additional_jobs == 0:
            process_per_node -= 1
            additional_jobs -= 1

    f.write(";;\n")
    f.write("esac\n")
    f.close()

    return mpirunfile


def run_shell_command(command, verbose=False):
    if verbose:
        logger.info(command)
    [output, error] = subprocess.Popen(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
    ).communicate()
    if verbose and len(output) > 0:
        logger.info("\n".join([s for s in output.split("\n") if s]))
    if len(error) > 0 and "BZIP2" not in error and "no version information available" not in error and "Format: lossy" not in error and "p_observed" not in error and "it/s" not in error and "sleeping and retrying" not in error and "TIFFReadDirectory: Warning" not in error and "Found device" not in error:
        logger.error(error)
    return output, error


def create_initial_multirun_file(
    fparameters, dataset, multirun_file="swarm/commands.swarm"
):
    with open(multirun_file, "w") as f:
        f.write("#\!/bin/bash\n")
        f.write("case $MP_CHILD in\n")
        lscratch = os.environ["PYP_SCRATCH"]
        stack = "%s_stack.mrc" % dataset
        group = 0
        # TODO: scontrol won't work inside the container
        [output, error] = run_shell_command(
            "%s/scontrol show hostname $SLURM_JOB_NODELIST" % slurm_prefix(),
        )
        nodes = output.splitlines()
        for i in nodes:
            f.write("{0})\n".format(group))
            f.write("""find %s -user $USER -exec rm -fr {} \;\n""" % lscratch)
            if "t" in fparameters["scratch_copy_stack"].lower():
                f.write(
                    "cp {0} {1}\n".format(
                        os.environ["SLURM_SUBMIT_DIR"] + "/" + stack, lscratch
                    )
                )
                if os.path.exists("%s_recstack.mrc" % dataset):
                    f.write("cp %s_recstack.mrc %s" % (dataset, lscratch))
                f.write(";;\n")
            group += 1
        f.write("esac\n")
    run_shell_command("chmod u+x '{0}'".format(multirun_file),)

    return nodes, multirun_file


def create_rec_split_multirun_file(
    iteration, particles, classes, increment, legacy=True
):
    """Within frealign_rec, create rec split file"""

    mpirunfile = "swarm/frealign_rec_split.multirun"
    f = open(mpirunfile, "w")

    if legacy:
        f.write("#\!/bin/bash\n")
        f.write("cd '{0}/swarm'\n".format(os.getcwd()))
        f.write("unset frealign_rec\n")
        f.write("export frealign_rec_split=frealign_rec_split\n")
        f.write("case $MP_CHILD in\n")

    # first = count = 0
    first = 0
    count = 1
    last = min(first + increment - 1, particles - 1)
    for first in range(1, particles + 1, increment):
        last = min(first + increment - 1, particles)
        if particles - last < increment / 2:
            last = particles
        if legacy:
            f.write("{0})\n".format(count))
        for ref in range(classes):
            f.write(
                "cd '{0}/swarm'; ".format(os.getcwd())
                + "{0} --iteration {1} --ref {2} --first {3} --last {4} --count {5}\n".format(
                    run_pyp(command="fyp", script=False),
                    iteration,
                    ref + 1,
                    first,
                    last,
                    count,
                )
            )
        if legacy:
            f.write(";;\n")
        count += 1
        if last == particles:
            break
    if legacy:
        f.write("esac\n")
    f.close()

    run_shell_command("chmod u+x '{0}/{1}'".format(os.getcwd(), mpirunfile),)

    # manage enviroment variables
    my_env = os.environ.copy()
    my_env["frealign_rec_split"] = "frealign_rec_split"
    if "frealign_rec" in my_env:
        del my_env["frealign_rec"]

    return mpirunfile, count


def num_particles_per_core(particles: int, tilts: int, frames: int, boxsize: int, images: str, cores: int, memory: float) -> int:
    """
    Return the maximal number of particles each core can process
    """
    KB_PER_PIXEL = 0.004
    TOLERANCE = 1 * cores # GB, basic system operating memory 

    mem_per_particle = KB_PER_PIXEL * tilts * frames * boxsize * boxsize # KB
    max_mem_per_core = ((memory-TOLERANCE) * 1.0 / cores) * 1024 * 1024 # KB
    
    from pyp.inout.image.core import get_image_dimensions
    if images.endswith(".mrc") or images.endswith(".tif"):
        # w/o frames
        # all the cores read entire tilt-series at the same time
        dims = get_image_dimensions(images)
        x, y, z = dims[0], dims[1], dims[2]
    
    elif images.endswith(".txt"):
        # w/ frames
        # all the cores read one tilted movie at the same time
        with open(images, 'r') as f:
            dims = get_image_dimensions(f.readlines()[0].strip())
            x, y, z = dims[0], dims[1], dims[2]
          
    mem_per_tilt_image = x * y * z * KB_PER_PIXEL
    assert (mem_per_tilt_image <= max_mem_per_core), f"Do not have enough memory for reading tilted images per core in parallel. Please increase your memory or decrease number of tasks."
    assert (mem_per_particle <= max_mem_per_core), f"Do not have enough memory for processing even one particle per core in parallel. Please increase your memory or decrease number of tasks."
    
    particles_per_core = min(math.floor((max_mem_per_core-mem_per_tilt_image)/mem_per_particle), \
                            math.ceil(particles*1.0/cores)
                            )
    assert (particles_per_core > 0), f"Do not have enough memory for processing even one particle per core in parallel. Please increase your memory or decrease number of tasks."
    return particles_per_core


def create_csp_split_commands(
    csp_command, parameter_file, mode, cores, name, merged_stack, ptlind_list, scanord_list, frame_list, parameters, use_frames=False, cutoff=0
):

    """Within frealign_rec, create rec split file"""

    commands = []
    movie_list = []

    first = 0
    count = 1

    name = name.split("_r")[0]
    frame_tag = ""
    if use_frames:
        images = "frames_csp.txt"
        frame_tag = "_local"
    else:
        images = os.path.join("frealign", "%s.mrc" % (name))


    # Patch-based csp refinement - parfile is splitted into pieces 
    if isinstance(parameter_file, list):
        
        refine_frames = '1' if not use_frames or (use_frames and parameters["csp_frame_refinement"]) else '0'

        if mode == 3 and not parameters["csp_frame_refinement"]:
            mode = 6
        if mode == 2:
            mode = 5
        
        for core, region in enumerate(parameter_file[::-1]):
            
            stack = merged_stack
            split_parameter_file = region[0]
            extended_parameter_file = split_parameter_file.replace(".cistem", "_extended.cistem")

            # Micrograph patch-based refinement 
            if mode == 3 or mode == 6 or mode == 4:
                
                # only separate csp processes based on micrograph index if not doing frame refinement 
                micrographs_list = range(1) if parameters["csp_frame_refinement"] else region[2]
                
                # iterate over micrograph index (scanning order) present in this patch
                for micrograph_first_index in micrographs_list:

                    # if the number of particles in this patch doesn't meet the threshold, do not refine it
                    patch_micrograph_mode = mode
                    
                    micrograph_last_index = -1 if parameters["csp_frame_refinement"] else micrograph_first_index

                    logfile = "%s_csp_region%s_%06d_%06d.log" % (
                        name,
                        region[0].split("region")[-1].split("_")[0],
                        micrograph_first_index,
                        micrograph_last_index,
                    ) if core == 0 else "/dev/null"
                    commands.append(
                        "{0} {1} {2} {3} {4} {5} {6} {7} {8} > {9}".format(
                            csp_command,
                            split_parameter_file,
                            extended_parameter_file,
                            patch_micrograph_mode,
                            micrograph_first_index,
                            micrograph_last_index,
                            refine_frames,
                            images,
                            stack,
                            logfile,
                        )
                    )

            # Particle patch-based refinement
            else:
                # iterate over particle index present in this patch 
                for particle_first_index in region[1]:

                    particle_last_index = particle_first_index

                    logfile = "%s_csp_region%s_%06d_%06d.log" % (
                        name,
                        region[0].split("region")[-1].split("_")[0],
                        particle_first_index,
                        particle_last_index,
                    ) if core == 0 else "/dev/null"
                    commands.append(
                        "{0} {1} {2} {3} {4} {5} {6} {7} {8} > {9}".format(
                            csp_command,
                            split_parameter_file,
                            extended_parameter_file,
                            mode,
                            particle_first_index,
                            particle_last_index,
                            refine_frames,
                            images,
                            stack,
                            logfile,
                        )
                    )

    # Global refinement 
    else:
        extract_frame = 1
        extended_parameter_file = parameter_file.replace(".cistem", "_extended.cistem")

        if mode == 2 or mode == -2:
            # refine particle parameters & particle extraction 
            mode = 5 if mode == 2 else mode
            max_index = len(ptlind_list)-1
            increment = num_particles_per_core(particles=len(ptlind_list), 
                                                tilts=len(scanord_list), 
                                                frames=len(frame_list), 
                                                boxsize=parameters["extract_box"], 
                                                images=images,
                                                cores=cores, 
                                                memory=parameters["slurm_memory"] 
                                                )
            increment = 1 if (use_frames and mode == 5) else increment
            list_to_iterate = ptlind_list

        elif mode == 3 and not use_frames:
            # refine micrograph parameters (w/o frames)
            mode = 6 
            max_index = len(scanord_list)-1
            increment = 1
            list_to_iterate = scanord_list

        
        elif mode == 3 and use_frames:
            # refine micrograph parameters (w/ frames)
            extract_frame = 0
            max_index = len(ptlind_list)-1
            increment = 1
            list_to_iterate = ptlind_list
        
        for first_index in range(0, max_index+1, increment):
            last_index = min(first_index+increment-1, max_index)
            
            first = int(list_to_iterate[first_index])
            last = int(list_to_iterate[last_index])

            logfile = "%s_csp_%06d_%06d.log" % (name, first, last) if first == 0 else "/dev/null"
            
            stack = merged_stack if mode != -2 else "frealign/%s_stack_%04d_%04d.mrc" % (name, first, last)
            commands.append(
                "{0} {1} {2} {3} {4} {5} {6} {7} {8} > {9}".format(
                    csp_command,
                    parameter_file,
                    extended_parameter_file,
                    mode,
                    first,
                    last,
                    extract_frame,
                    images,
                    stack,
                    logfile,
                )
            )
            movie_list.append("frealign/%s_stack_%04d_%04d.mrc" % (name, first, last))
            count += 1
            
    return commands, count, movie_list

@timer.Timer(
    "creat_split_commands", text="Creating split commands took: {}", logger=logger.info
)
def create_split_commands(
    mp, name, frames, cores, scratch, step="", num_frames=1, ref=1, current_path=".", iteration=2,
):
    """Function to write the script for directly calling refine3d/reconstruct3d in parallel using multirun

    Parameters
    ----------
    mp : dict
        parameters for main pyp
    name : str
        name of the movie
    frames : int
        the total number of frames contained in the parfile
    metric : str
        alignment metric for refine3d
    cores : int
        the number of cpus/cores for parallelism
    scratch : str
        name of the scratch folder 
    current_path : str
        project dir to find the global parfile
    ref : int
        reference's class number
    Returns
    -------
    int
        it's trivial, just use it to stop the function if mode == 0
    """

    fp = mp
    from pyp.refine.frealign import frealign

    if "reconstruct3d" in step:
        boff, thresh = frealign.mreconstruct_pre(mp, fp, iteration, ref)

    increment = math.ceil(frames / cores)

    count = 0

    commands = []

    for first in range(1, frames + 1, increment + 1):
        last = min(first + increment, frames)

        ranger = "%07d_%07d" % (first, last)


        from pyp.system import project_params
        if "refine3d" in step:

            if fp["refine_debug"] or first == 1:
                logfile = "%s_msearch_n.log_%s" % (name, ranger)
            else:
                logfile = "/dev/null"

            command = frealign.mrefine_version(
                mp,
                first,
                last,
                iteration,
                ref,
                current_path,
                name,
                ranger,
                logfile,
                scratch,
                refine_beam_tilt=False,
            )
        elif "refine_ctf" in step:

            if fp["refine_debug"] or first == 1:
                logfile = "%s_msearch_ctf_n.log_%s" % (name, ranger)
            else:
                logfile = "/dev/null"

            command = frealign.mrefine_version(
                mp,
                first,
                last,
                iteration,
                ref,
                current_path,
                name,
                ranger,
                logfile,
                scratch,
                refine_beam_tilt=True,
            )

        elif "reconstruct3d" in step:
            command = frealign.split_reconstruction(
                mp,
                first,
                last,
                iteration,
                ref,
                count + 1,
                boff,
                thresh,
                dump_intermediate="yes",
                num_frames=num_frames,
                run=False,
            )

        commands.append(command)
        count += 1

    return commands, count


def create_stack_multirun_file(csp_command, mode, particles, cmin, cmax, cores):

    """Within frealign_rec, create rec split file"""

    mpirunfile = "csp_split.multirun"
    with open(mpirunfile, "w") as f:

        f.write("#\!/bin/bash\n")
        f.write("case $MP_CHILD in\n")

        increment = math.ceil(particles / cores)

        first = 0
        count = 1

        for first in range(0, particles, increment):
            last = min(first + increment, particles - 1)
            if last < first + increment:
                last = particles - 1
            f.write("{0})\n".format(count))
            f.write(
                "{0} parameters.config 0 {1} {2} {3} {4} {5}\n".format(
                    csp_command, mode, first, last, cmin, cmax,
                )
            )
            f.write(";;\n")
            count += 1
        f.write("esac\n")

    run_shell_command("chmod u+x '{0}/{1}'".format(os.getcwd(), mpirunfile),)

    return mpirunfile, count


def create_ref_multirun_file(
    iteration,
    classes,
    particles,
    metric,
    increment,
    cores,
    mpirunfile="swarm/frealign_ref.multirun",
):

    f = open(mpirunfile, "w")

    f.write("#\!/bin/bash\n")
    f.write("cd '{0}/swarm'\n".format(os.getcwd()))
    f.write("export frealign_ref=frealign_ref\n")

    if cores < 50:

        first = count = 0
        last = min(first + increment - 1, particles - 1)
        for first in range(1, particles + 1, increment):
            last = min(first + increment - 1, particles)
            for ref in range(classes):
                f.write(
                    "{0} --iteration {1} --ref {2} --first {3} --last {4} --metric {5} &\n".format(
                        run_pyp(command="fyp", script=False),
                        iteration,
                        ref + 1,
                        first,
                        last,
                        metric.replace("-metric", ""),
                    )
                )
            count += 1

    else:

        f.write("case $MP_CHILD in\n")

        first = count = 0
        last = min(first + increment - 1, particles - 1)
        for first in range(1, particles + 1, increment):
            last = min(first + increment - 1, particles)
            f.write("{0})\n".format(count))
            for ref in range(classes):
                f.write(
                    "{0} --iteration {1} --ref {2} --first {3} --last {4} --metric {5}\n".format(
                        run_pyp(command="fyp", script=True),
                        iteration,
                        ref + 1,
                        first,
                        last,
                        metric.replace("-metric", ""),
                    )
                )
            f.write(";;\n")
            count += 1

        f.write("esac\n")
    f.close()

    run_shell_command("chmod u+x '{0}/{1}'".format(os.getcwd(), mpirunfile),)

    return mpirunfile


def create_ref_multirun_file_from_missing(
    machinefile, missing_ali_swarm_file, mpirunfile="../swarm/frealign_ref.multirun"
):
    with open(missing_ali_swarm_file) as f:
        lines = f.read().splitlines()

    nodes = len(open(machinefile, "r").read().split())
    if "MYCORES" in os.environ and int(os.environ["MYCORES"]) > 0:
        cores = int(os.environ["MYCORES"])
    else:
        cores = multiprocessing.cpu_count() * nodes
    lines_per_node = int(math.ceil(len(lines) / float(nodes)))

    f = open(mpirunfile, "w")
    f.write("#\!/bin/bash\n")
    f.write("cd '{0}/../swarm'\n".format(os.getcwd()))
    f.write("export frealign_ref=frealign_ref\n")
    f.write("case $MP_CHILD in\n")

    count = change = 0
    for line in lines:
        if change == 0:
            f.write("{0})\n".format(count))

        f.write(line + "\n")
        change += 1

        if change == lines_per_node:
            f.write(";;\n")
            change = 0
            count += 1

    f.write("esac\n")
    f.close()

    run_shell_command("chmod u+x '{0}/{1}'".format(os.getcwd(), mpirunfile),)

    return count, cores, mpirunfile


def multiprocessing_multirun_command(command, my_env):

    print(
        subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True, text=True, env=my_env
        )
    )


def run_multirun(command_list, cpus=0, logfile="/dev/null"):
    """Run a list of tasks in parallel using multirun

    Parameters
    ----------
    command_list : list
        List of commands to run
    cpus : int, optional
        Number of parallel processes to use, by default 0 (use all tasks available through SLURM)
    """
    if cpus == 0:
        cpus = int(os.environ["SLURM_NTASKS"])

    lines = len(command_list)

    machinefile = "machinefile"
    with open(machinefile, "w") as f:
        f.write(socket.gethostname())

    # compose multirun file
    multirunfile = "multitun.multirun"
    with open(multirunfile, "w") as f:

        f.write("#\!/bin/bash\n")
        f.write("case $MP_CHILD in\n")

        increment = math.ceil(lines / cpus)

        count = 0
        for core in range(cpus):
            f.write("{0})\n".format(core))
            if count < lines:
                for line in range(increment):
                    f.write(command_list[count] + "\n")
                    count += 1
                    if count == lines:
                        break
            f.write(";;\n")
        f.write("esac\n")

    run_shell_command("chmod u+x '{0}/{1}'".format(os.getcwd(), multirunfile),)

    command = "{0} -machinefile {1} -np {2} {3}/external/multirun/multirun -m {4} > {5}".format(
        get_mpirun_command(),
        machinefile,
        cpus,
        os.environ["PYP_DIR"],
        os.path.join(os.getcwd(), multirunfile),
        logfile,
    )

    # execute multirun
    [output, error] = run_shell_command(command)
