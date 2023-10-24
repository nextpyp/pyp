#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import os
import subprocess
import sys
import time

import numpy

from pyp import analysis
from pyp.inout.image import mrc, writepng
from pyp.system import local_run, project_params, user_comm
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import eman_load_command, qos
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def run_relion(rparameters, sparameters, run=True):

    mem_per_cpu = int(
        float(rparameters["mem_per_task"]) / int(rparameters["cpus_per_task"])
    )

    # convert particle_rad to RELION's diameter
    actual_pixel = (
        float(sparameters["scope_pixel"])
        * float(sparameters["data_bin"])
        * float(sparameters["extract_bin"])
    )
    radius_in_A = numpy.array(sparameters["particle_rad"].split(","), dtype=float).max()
    diameter = 2.5 * radius_in_A

    initial_model = rparameters["dataset"] + "_initial_model.mrc"

    working_dir = os.environ["SLURM_SUBMIT_DIR"]
    lscratch = "/lscratch/" + os.environ["SLURM_JOBID"]
    lscratch = "/scratch"

    logger.info("Running on %s", working_dir)

    os.environ["TMPDIR"] = "/lscratch/%s/TMPDIR" % os.environ["SLURM_JOBID"]
    os.environ["TMPDIR"] = "/scratch/TMPDIR"
    try:
        os.mkdir(os.environ["TMPDIR"])
    except:
        pass

    os.chdir(working_dir)

    data = rparameters["dataset"]

    # 2D Classification
    if "2d" in rparameters["mode"].lower():

        if not os.path.exists(working_dir + "/Class2D"):
            os.makedirs(working_dir + "/Class2D")

        command = "`which relion_refine_mpi` --o Class2D/{0} --i {0}.star --particle_diameter {1} --angpix {2} --ctf --iter {4} --tau2_fudge {7} --K {3} --flatten_solvent --zero_mask --oversampling 1 --psi_step 10 --offset_range 5 --offset_step 2 --norm --scale --j {5} --memory_per_thread {6} --dont_check_norm --ctf_intact_first_peak".format(
            data,
            diameter,
            actual_pixel,
            rparameters["classes"],
            rparameters["iter"],
            rparameters["cpus_per_task"],
            mem_per_cpu,
            rparameters["tau"],
        )
        # command='`which relion_refine_mpi` --o Class2D/{0} --i {0}.star --particle_diameter {1} --angpix {2} --ctf --iter {4} --tau2_fudge {7} --K {3} --flatten_solvent --zero_mask --oversampling 1 --psi_step 10 --offset_range 5 --offset_step 2 --norm --scale --j {5} --memory_per_thread {6} --dont_check_norm'.format( data, diameter, actual_pixel, rparameters['classes'], rparameters['iter'], rparameters['cpus_per_task'], mem_per_cpu, rparameters['tau'] )
        command = "`which relion_refine_mpi` --o Class2D/{0} --i {0}.star --particle_diameter {1} --ctf --iter {4} --tau2_fudge {7} --pad 2 --pool 3 --K {3} --flatten_solvent --zero_mask --oversampling 1 --psi_step 12 --offset_range 5 --offset_step 2 --norm --scale --j {5} --dont_check_norm".format(
            data,
            diameter,
            actual_pixel,
            rparameters["classes"],
            rparameters["iter"],
            rparameters["cpus_per_task"],
            mem_per_cpu,
            rparameters["tau"],
        )

        # command += ' --ctf_intact_first_peak'
        command += " --fast_subsets"
        command += " --scratch_dir /scratch"

    # Initial Model
    elif "init" in rparameters["mode"].lower():

        if not os.path.exists(working_dir + "/Init3D"):
            os.makedirs(working_dir + "/Init3D")

        command = "`which relion_refine_mpi` --o Init3D/{0} --i {0}.star --particle_diameter {1} --sgd_ini_iter 50 --sgd_inbetween_iter 200 --sgd_fin_iter 50 --sgd_write_iter 10 --sgd_ini_resol 35 --sgd_fin_resol 15 --sgd_ini_subset 100 --sgd_fin_subset 500 --sgd  --denovo_3dref --ctf --K {2} --flatten_solvent --zero_mask --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 4 --sym {3} --norm --scale --pad 2  --j {4} --dont_ckeck_norm --pool 3".format(
            data,
            diameter,
            rparameters["classes"],
            sparameters["particle_sym"],
            rparameters["cpus_per_task"],
        )

    # 3D Classification
    elif "class" in rparameters["mode"].lower():

        if not os.path.exists(working_dir + "/Class3D"):
            os.makedirs(working_dir + "/Class3D")

        command = "`which relion_refine_mpi` --o Class3D/{0} --i {0}.star --particle_diameter {1} --ref {2} --firstiter_cc --ini_high {10} --ctf -ctf_corrected_ref --iter {3} --tau2_fudge {9} --K {4} --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 --sym {5} --norm --scale  --j {6} --memory_per_thread {7} --angpix {8} --dont_ckeck_norm".format(
            data,
            diameter,
            initial_model,
            rparameters["iter"],
            rparameters["classes"],
            sparameters["particle_sym"],
            rparameters["cpus_per_task"],
            mem_per_cpu,
            actual_pixel,
            rparameters["tau"],
            rparameters["res"],
        )

    # Movie Mode 3D Refinement
    elif "movie" in rparameters["mode"].lower():

        # use most recent optimizer file
        # optimiser = sorted( glob.glob( 'Refine3D/{0}_it???_optimiser.star'.format( rparameters['continue'] ) ) )[0]

        # use optimiser file from latest 3D refinement
        command = "`which relion_refine_mpi` --o Refine3D/{0}_frames --continue {1} --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --realign_movie_frames {0}_frames.star --movie_frames_running_avg {2} --sigma_off 1 --skip_rotate --skip_maximize  --j {3}".format(
            data,
            rparameters["continue"],
            rparameters["runavg"],
            rparameters["cpus_per_task"],
            mem_per_cpu,
        )

    # Particle Polishing
    elif "polish" in rparameters["mode"].lower():

        bg_radius = int(float(sparameters["particle_rad"]) / actual_pixel)

        # use result of 3D movie refinement as input
        command = "`which relion_particle_polish_mpi` --i Refine3D/{0}_frames_data.star --o {0}_shiny --angpix {1} --movie_frames_running_avg {2} --dont_read_old_files --sigma_nb {3} --mask {4} --sym {5} --bg_radius {6} --white_dust -1 --black_dust -1".format(
            data,
            actual_pixel,
            rparameters["runavg"],
            rparameters["sigma_nb"],
            rparameters["mask"],
            sparameters["particle_sym"],
            bg_radius,
        )

        if "f" in rparameters["skip_bfactor_weighting"].lower():
            command += " --perframe_highres {0} --autob_lowres {1}".format(
                rparameters["perframe_highres"], rparameters["autob_lowres"]
            )
        else:
            command += " --skip_bfactor_weighting"

    # 3D Refinement
    else:

        if not os.path.exists(working_dir + "/Refine3D"):
            os.makedirs(working_dir + "/Refine3D")

        # figure out if we are using orientation parameters from previous runn
        if len(rparameters["continue"]) > 0 and os.path.exists(rparameters["continue"]):

            # Continue old run from last _optimiser.star file (error recovery)
            if "optimiser" in rparameters["continue"]:

                command = "`which relion_refine_mpi` --o Refine3D/{0} --continue {1} --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --j {2} --memory_per_thread {3} --dont_check_norm".format(
                    data,
                    rparameters["continue"],
                    rparameters["cpus_per_task"],
                    mem_per_cpu,
                )

            # Local refinement from existing _data.star file accounting for different binning factors and box sizes
            else:

                # read input star file
                input = numpy.array(
                    [line.split() for line in open(data + ".star") if ".mrcs" in line]
                )

                # read star file containing alignment parameters
                refine = numpy.array(
                    [
                        line.split()
                        for line in open(rparameters["continue"])
                        if ".mrcs" in line
                    ]
                )

                rlnMagnification = [
                    int(line.split()[1].replace("#", ""))
                    for line in open(rparameters["continue"])
                    if "_rlnMagnification" in line
                ][0] - 1
                Magnification = refine[0, rlnMagnification]
                data_bin = float(sparameters["data_bin"]) * float(
                    sparameters["extract_bin"]
                )
                pixel = float(sparameters["scope_pixel"]) * data_bin
                magnification = float(sparameters["scope_mag"]) / data_bin
                dstep = pixel * magnification / 10000.0
                factor = magnification / float(Magnification)

                # relevant alignment parameters
                rlnAngleRot = [
                    int(line.split()[1].replace("#", ""))
                    for line in open(rparameters["continue"])
                    if "_rlnAngleRot" in line
                ][0] - 1
                rlnAngleTilt = [
                    int(line.split()[1].replace("#", ""))
                    for line in open(rparameters["continue"])
                    if "_rlnAngleTilt" in line
                ][0] - 1
                rlnAnglePsi = [
                    int(line.split()[1].replace("#", ""))
                    for line in open(rparameters["continue"])
                    if "_rlnAnglePsi" in line
                ][0] - 1
                rlnOriginX = [
                    int(line.split()[1].replace("#", ""))
                    for line in open(rparameters["continue"])
                    if "_rlnOriginX" in line
                ][0] - 1
                rlnOriginY = [
                    int(line.split()[1].replace("#", ""))
                    for line in open(rparameters["continue"])
                    if "_rlnOriginY" in line
                ][0] - 1

                # compose output parameter file
                output = numpy.empty(
                    [input.shape[0], input.shape[1] + 5], dtype=input.dtype
                )
                output[:, : input.shape[1]] = input
                output[:, -5] = refine[:, rlnAngleRot]
                output[:, -4] = refine[:, rlnAngleTilt]
                output[:, -3] = refine[:, rlnAnglePsi]
                output[:, -2] = (factor * refine[:, rlnOriginX].astype("f")).astype(
                    "string"
                )
                output[:, -1] = (factor * refine[:, rlnOriginY].astype("f")).astype(
                    "string"
                )

                # write new star file with the correct alingment parameters

                header = numpy.array(
                    [
                        line.split()
                        for line in open(data + ".star")
                        if not ".mrcs" in line
                    ]
                )
                header = header[:-1]
                f = open(data + "_continue.star", "wb")
                [f.write(line) for line in open(data + ".star") if not ".mrcs" in line]
                cols = input.shape[1]
                f.write(
                    """_rlnAngleRot #%d\n_rlnAngleTilt #%d\n_rlnAnglePsi #%d\n_rlnOriginX #%d\n_rlnOriginY #%d\n"""
                    % (cols + 1, cols + 2, cols + 3, cols + 4, cols + 5)
                )

                for line in range(output.shape[0]):
                    f.write("\t".join(output[line, :]) + "\n")
                f.close()

                """ rescale initial model if needed
                if factor > 1 and os.path.exists( rparameters['model'] ):
                    oldsize = mrc.readHeaderFromFile( rparameters['model'] )['nz']
                    newsize = mrc.readHeaderFromFile( dstack[0] )['nx']
                    com = 'module load EMAN1; proc3d {0} {1} scale={2} clip={3},{3},{3}'.format( rparameters['model'], rparameters['model'].replace('.mrc','_continue.mrc'), factor, newsize )
                    print dstack[0]
                    print newsize
                    print com
                    print commands.getoutput(com)
                    rparameters['model'] = rparameters['model'].replace('.mrc','_continue.mrc')
                """

                # figure out current resolution

                com = """grep _rlnCurrentResolution %s | awk '{ print $2 }'""" % rparameters[
                    "continue"
                ].replace(
                    "_data", "_half1_model"
                )
                current_resolution = subprocess.getoutput(com)

                # run auto-refine using fine initial angular sampling (1.8 degrees) and Stddev on all three Euler angles for local angular searches (5) and initial reference filtered to 6A
                command = "`which relion_refine_mpi` --o Refine3D/{0} --auto_refine --split_random_halves --i {0}_continue.star --particle_diameter {1} --ref {2} --firstiter_cc --ini_high {3} --ctf_corrected_ref --ctf --flatten_solvent --zero_mask --oversampling 1 --healpix_order 4 -sigma_ang 1.5 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym {4} --low_resol_join_halves {5} --norm --scale  --j {6} --memory_per_thread {7} --angpix {8} --dont_check_norm".format(
                    data,
                    diameter,
                    initial_model,
                    current_resolution,
                    sparameters["particle_sym"],
                    rparameters["res"],
                    rparameters["cpus_per_task"],
                    mem_per_cpu,
                    actual_pixel,
                )

        # Normal 3D Refinement
        else:

            command = "`which relion_refine_mpi` --o Refine3D/{0} --auto_refine --split_random_halves --i {0}.star --particle_diameter {1} --ref {2} --firstiter_cc --ini_high {8} --ctf_corrected_ref --ctf --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym {3} --low_resol_join_halves {4} --norm --scale  --j {5} --memory_per_thread {6} --angpix {7} --dont_check_norm".format(
                data,
                diameter,
                initial_model,
                sparameters["particle_sym"],
                rparameters["res"],
                rparameters["cpus_per_task"],
                mem_per_cpu,
                actual_pixel,
                rparameters["res"],
            )

    # print com

    com = "module load RELION/1.4-beta-2; srun --mpi=pmi2 " + command
    com = "module load RELION/1.4; srun --mpi=pmi2 " + command
    com = "module load RELION/1.4; mpirun " + command
    com = "export PATH=$PATH:/dscrhome/ab690/code/relion/build/bin && mpirun " + command
    com = (
        "module load Anaconda2/2.7.13; export PATH=$PATH:/dscrhome/ab690/code/relion-3.0_beta/build/bin && mpirun "
        + command
    )

    com += " --dont_combine_weights_via_disc"

    cmd_name = sorted(glob.glob("*" + rparameters["dataset"] + ".err"))[-1].replace(
        ".err", ".cmd"
    )

    with open(cmd_name, "w") as f:
        f.write(com)
    logger.info(com)
    if run:
        logger.info(subprocess.getoutput(com))

    # post processing
    post_processing(sparameters, rparameters)

    # merge shiny particle stack into single file
    if "polish" in rparameters["mode"].lower():

        star_file = rparameters["dataset"] + "_shiny.star"

        rlnImageName = [
            int(line.split()[1].replace("#", ""))
            for line in open(star_file)
            if "_rlnImageName" in line
        ][0] - 1

        # extract stack list from file
        stacklist = numpy.array(
            [
                line.split()[rlnImageName].split("@")[-1]
                for line in open(star_file)
                if "000001@" in line
            ]
        )

        relion_stack = os.getcwd() + "/{}_shiny_stack.mrcs".format(
            rparameters["dataset"]
        )

        mrc.merge(stacklist, relion_stack)

        new_star_file = rparameters["dataset"] + "_shiny.star"

        # save _particles.star file
        com = "mv {0} {1}".format(
            new_star_file, new_star_file.replace(".star", "_particles.star")
        )
        subprocess.getoutput(com)

        # write star file
        output = open(new_star_file, "w")
        with open(star_file.replace(".star", "_particles.star"), "r") as f:
            counter = 1
            for line in f:
                if not "mrcs" in line:
                    output.write(line)
                else:
                    stack = line.split()[rlnImageName]
                    output.write(
                        line.replace(
                            stack, "%.6d@%s/%s" % (counter, os.getcwd(), relion_stack)
                        )
                    )
                    counter += 1
        output.close()


def launch_relion(rparameters, mparameters):

    if "2d" in rparameters["mode"].lower():
        jobname = "r2C_"
    elif "class" in rparameters["mode"].lower():
        jobname = "r3C_"
    else:
        jobname = "r3R_"

    if "time" in rparameters["slurm_queue"]:
        wtime = ""
    else:
        wtime = "--time 5-00:00:00"

    # ignore walltime if in quick queue
    if "quick" in rparameters["slurm_queue"]:
        wtime = ""

    timestamp = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + "_"
        + rparameters["mode"].lower()
    )

    # manage resources automatically

    relion_stack = rparameters["dataset"] + "_stack.mrcs"
    boxsize = int(mrc.readHeaderFromFile(relion_stack)["nx"])

    if len(rparameters["nodes"]) == 0:
        nodes = 8

        # set memory requiremnts according to image size
        factor = boxsize * boxsize / 256.0 / 256.0

        # memory per MPI process ( 256x256 = 8GB )
        rparameters["mem_per_task"] = str(int(numpy.ceil(factor * 9)))

        # assume total node memory is 128GB and core count is 32
        rparameters["cpus_per_task"] = str(
            int(32 * float(rparameters["mem_per_task"]) / 128.0)
        )
        rparameters["cpus_per_task"] = str(
            max(1, int(70 * float(rparameters["mem_per_task"]) / 768.0))
        )

        # use 10-core jobs
        # rparameters['nodes'] = 10        # str( int( 8 * 128 / int( rparameters['mem_per_task'] ) ) )

        project_params.save_relion_parameters(rparameters)

    else:
        nodes = rparameters["nodes"]

        # each ntask has cpus_per_task threads and each thread needs memory_per_cpu GB of memory

    # print rparameters['cpus_per_task']
    mem_per_cpu = int(
        float(rparameters["mem_per_task"]) / int(rparameters["cpus_per_task"])
    )

    command = "( export pyprunrln=pyprunrln && /opt/slurm/bin/sbatch --exclusive --gres=lscratch:100 --cpus-per-task {0} --nodes {1} --ntasks-per-core=2 --mem-per-cpu={2}g --job-name={3} {4} {5} --output {9}_{6}.out --error {9}_{6}.err {7} {8}/refine/relion/relion.py )".format(
        rparameters["cpus_per_task"],
        nodes,
        mem_per_cpu,
        jobname + rparameters["dataset"][-6:],
        rparameters["slurm_queue"],
        qos(rparameters["slurm_queue"]),
        rparameters["dataset"],
        wtime,
        os.environ["PYTHONDIR"],
        timestamp,
    )
    command = "( export pyprunrln=pyprunrln && /opt/slurm/bin/sbatch --exclusive --cpus-per-task {0} --nodes {1} --ntasks-per-core=2 --mem-per-cpu={2}g --job-name={3} {4} {5} --output {9}_{6}.out --error {9}_{6}.err {7} {8}/refine/relion/relion.py )".format(
        rparameters["cpus_per_task"],
        nodes,
        mem_per_cpu,
        jobname + rparameters["dataset"][-6:],
        rparameters["slurm_queue"],
        qos(rparameters["slurm_queue"]),
        rparameters["dataset"],
        wtime,
        os.environ["PYTHONDIR"],
        timestamp,
    )
    # command='( export pyprunrln=pyprunrln && /opt/slurm/bin/sbatch --gres=lscratch:400 --cpus-per-task 4 --ntasks 64 --ntasks-per-core=2 --mem=62g {0}/refine/relion/relion.py )'.format( os.environ['PYTHONDIR'] )

    logger.info(command)
    # return commands.getoutput(command).split()[-1]
    return subprocess.getoutput(command)
    # [ output, error ] = subprocess.Popen( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE ).communicate()
    # return output


def launch_relion_refinement(parameters, new_name, actual_pixel):
    if "2d" in parameters["extract_fmt"].lower() and int(parameters["class_num"]) > 0:

        command = "-mode Class2D -classes {0}".format(parameters["class_num"])

    elif os.path.exists(parameters["class_ref"]):

        # create initial model with correct pixel size and dimensions
        relion_initial_model = (
            os.getcwd() + "/relion/" + new_name + "_initial_model.mrc"
        )
        if os.path.exists(relion_initial_model):
            os.remove(relion_initial_model)

        original_box_size = int(mrc.readHeaderFromFile(parameters["class_ref"])["nx"])
        original_pixel_size = float(
            mrc.readHeaderFromFile(parameters["class_ref"])["xlen"]
        ) / float(original_box_size)

        if (
            original_pixel_size != actual_pixel
            or int(parameters["extract_box"]) != original_box_size
        ):
            load_eman_cmd = eman_load_command()
            command = "{0}; e2proc3d.py {1} {2} --scale={3} --clip={4}".format(
                load_eman_cmd,
                parameters["class_ref"],
                relion_initial_model,
                "%.2f" % (original_pixel_size / actual_pixel),
                parameters["extract_box"],
            )
            local_run.run_shell_command(command)
        else:
            os.symlink(parameters["class_ref"], relion_initial_model)

        if int(parameters["class_num"]) > 0:

            # 3D Classification
            command = "-mode Class3D -classes {0} -model {1}".format(
                parameters["class_num"], relion_initial_model
            )

        else:

            # 3D Auto-refine
            command = "-mode Refine3D -model {0}".format(relion_initial_model)

    else:

        command = ""

    if len(command) > 0:

        command = 'cd relion; {0}/refine/relion/relion.py -dataset {1} -queue "{2}" {3}'.format(
            os.environ["PYTHONDIR"], new_name, parameters["slurm_queue"], command
        )
        logger.info(command)
        # print commands.getoutput(command)
        # [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE ).communicate()
        # print output
        # print error


def parse_arguments():

    # parse arguments
    parser = argparse.ArgumentParser(description="RELION refinement")
    parser.add_argument("-dataset", "--dataset", help="Name of dataset")
    # General RELION
    parser.add_argument(
        "-mode",
        "--mode",
        help="Mode of execution is one of Class2D, Init3D, Class3D, or Refine3D",
        type=str,
    )
    parser.add_argument(
        "-model",
        "--model",
        help="Initial model to use as reference for refinement",
        type=str,
    )
    parser.add_argument("-iter", "--iter", help="Number of iterations (25)", type=int)
    parser.add_argument(
        "-tau", "--tau", help="RELION's Regularization parameter T (4)", type=str
    )
    parser.add_argument(
        "-res", "--res", help="Resolution to filter inital model in A (40)", type=str
    )
    parser.add_argument(
        "-mask",
        "--mask",
        help="Mask determined automatically during post-processing before movie-processing.",
        type=str,
    )
    parser.add_argument(
        "-classes",
        "--classes",
        help="B-factor to apply to particle image projections before orientation determination or refinement (0)",
        type=str,
    )
    parser.add_argument(
        "-continue", "--continue", help="Use pre-existing alignments.", type=str
    )
    # Polishing
    parser.add_argument(
        "-runavg",
        "--runavg",
        help="Number of frames for running averages during movie refinement (5)",
        type=int,
    )
    parser.add_argument(
        "-sigma_nb",
        "--sigma_nb",
        help="Standard deviation for a Gaussian weight on the particle distance (300)",
        type=str,
    )
    parser.add_argument(
        "-skip_bfactor_weighting",
        "--skip_bfactor_weighting",
        help="Skip bfactor weighting during particle polishing (False)",
        type=str,
    )
    parser.add_argument(
        "-perframe_highres",
        "--per_frame_highres",
        help="Highres-limit per-frame maps in A (6)",
        type=float,
    )
    parser.add_argument(
        "-autob_lowres",
        "--autob_lowres",
        help="Lowres-limit B-factor estimation in A (20)",
        type=float,
    )
    # Running
    parser.add_argument(
        "-queue", "--queue", help="Use specific queue to run jobs (" ")", type=str
    )
    parser.add_argument("-nodes", "--nodes", help="Number of nodes ().", type=str)
    parser.add_argument(
        "-mem_per_task", "--mem_per_task", help="Memory per cpu (4).", type=str
    )
    parser.add_argument(
        "-cpus_per_task", "--cpus_per_task", help="CPUs per task (8).", type=str
    )
    parser.add_argument("-debug", "--debug", help="Do a dry run (False).", type=str)

    args = parser.parse_args()

    # create empty parameter file
    empty_parameters = collections.OrderedDict(
        [
            (
                "dataset",
                os.path.split(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))[
                    -1
                ],
            )
        ]
    )
    # General RELION
    empty_parameters.update(
        [
            ("mode", ""),
            ("model", ""),
            ("iter", 25),
            ("tau", "4"),
            ("res", "40"),
            ("mask", ""),
            ("classes", 1),
            ("continue", ""),
        ]
    )
    # Polishing
    empty_parameters.update(
        [
            ("runavg", "5"),
            ("sigma_nb", "300"),
            ("skip_bfactor_weighting", "F"),
            ("perframe_highres", 6.0),
            ("autob_lowres", 20.0),
        ]
    )
    # Running
    empty_parameters.update(
        [
            ("queue", ""),
            ("nodes", ""),
            ("mem_per_task", "16"),
            ("cpus_per_task", "4"),
            ("debug", "F"),
        ]
    )

    # load existing parameters
    parameters = project_params.load_relion_parameters()

    if parameters == 0:
        parameters = empty_parameters
    else:
        if len(parameters) is not len(empty_parameters):
            logger.warning("Parameter file format has changed. Adding new entries:")
            for key in list(empty_parameters.keys()):
                if key not in parameters:
                    print("\t", key, empty_parameters[key])
                    parameters[key] = empty_parameters[key]

    for k, v in vars(args).items():
        if v != None and parameters[k] is not v:
            logger.info("Updating {0} from {1} to {2}".format(k, parameters[k], v))
            parameters[k] = v

    # check required parameters
    if parameters["dataset"] == 0:
        logger.error("-dataset is required.")
        sys.exit(1)

    project_params.save_relion_parameters(parameters)

    return parameters


def launch_pyp(rparameters, sparameters, dependency):

    swarm_file = "runpyp.swarm"

    dirs = ["Class2D", "Init3D", "Class3D", "Refine3D"]
    extract_cls = int(sparameters["extract_cls"])

    if extract_cls < 3:

        if "2d" in dirs[extract_cls + 1].lower():
            class_num = 20
        else:
            class_num = 5

        with open(swarm_file, "w") as x:
            x.write(
                "posfile=`ls -tr {1}_it???_data.star | tail -1`; cd ..; {0}/byp -parfile relion/$posfile -classes auto -extract_cls {2}; {0}/pyp_main.py -extract_fmt relion_{3} -extract_cls {2} -class_ref {4} -class_num {5}".format(
                    os.environ["PYTHONDIR"],
                    dirs[extract_cls] + "/" + rparameters["dataset"],
                    extract_cls + 1,
                    dirs[extract_cls + 1],
                    sparameters["class_ref"],
                    class_num,
                )
            )

        com = (
            'swarm -f %s --dependency=afterany:%s %s --sbatch "%s" --job-name pyprunrln'
            % (
                swarm_file,
                dependency,
                sparameters["slurm_queue"],
                qos(sparameters["slurm_queue"]),
            )
        )
        logger.info(com)
        logger.info(subprocess.getoutput(com))


def post_processing(sp, rp):

    dataset = rp["dataset"]

    # 2D Classification, Movies and Polishing
    if (
        "2d" in rp["mode"].lower()
        or "movie" in rp["mode"].lower()
        or "polish" in rp["mode"].lower()
    ):

        # email result
        if "class2d" not in rp["mode"].lower():
            user_comm.notify(
                "Relion job %s (%s) finished." % (rp["dataset"], rp["mode"])
            )
        else:
            png_plot = "Class2D/%s_classes2D.png" % rp["dataset"]
            com = "source activate eman-env; sxmontage.py Class2D/{0}_it{1}_classes.mrcs {2} --N=15 --bg=3 --scale".format(
                rp["dataset"], "%03d" % (int(rp["iter"]) - 1), png_plot
            )
            logger.info(com)
            logger.info(subprocess.getoutput(com))
            attach = os.getcwd() + "/" + png_plot
            logger.info(attach)
            user_comm.notify(rp["dataset"] + " (RLN 2D)", attach)

            # clean up mess
            if "projects_mmc" in os.getcwd():
                [
                    os.remove(f)
                    for f in glob.glob("Class2D/" + rp["dataset"] + "*.*")
                    if not "png" in f
                ]
                [
                    os.remove(f)
                    for f in glob.glob("*" + rp["dataset"] + "*.*")
                    if not "_stack.mrcs" in f
                ]
            else:
                [
                    os.remove(f)
                    for f in glob.glob("Class2D/" + rp["dataset"] + "*.*")
                    if not "png" in f
                    and not "_it%03d" % int(rp["iter"]) in f
                    and not "unmasked" in f
                ]
                [
                    os.remove(f)
                    for f in glob.glob("*" + rp["dataset"] + "*.*")
                    if not "_stack.mrcs" in f and not ".star" in f
                ]
        return

    # 3D Initial Model
    elif "init" in rp["mode"].lower():
        path = "Init3D"
        last_iteration = sorted(glob.glob(path + "/" + dataset + "_?t???_data.star"))[
            -1
        ][-13:-10]
        classes = glob.glob(
            path + "/" + dataset + "_?t%s_class???.mrc" % last_iteration
        )
        starfile = glob.glob(path + "/" + dataset + "_?t%s_data.star" % last_iteration)[
            -1
        ]

        # plot current resolution curve
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        com = (
            """grep _rlnCurrentResolution %s/%s_it0??_model.star | awk '{ print $2}' """
            % (path, dataset)
        )
        resolution = numpy.array(subprocess.getoutput(com).split()[1:], dtype="f")
        plt.clf()
        plt.plot(resolution, label=dataset)
        plt.title("Current Resolution", fontsize=12)
        plt.xlabel("Iteration")
        plt.ylabel("Resolution (A)")
        plt.legend(prop={"size": 10})
        plt.savefig("%s/%s_evolution.png" % (path, dataset))

    # 3D Classification
    elif "class" in rp["mode"].lower():
        path = "Class3D"
        last_iteration = sorted(glob.glob(path + "/" + dataset + "_?t???_data.star"))[
            -1
        ][-13:-10]
        classes = glob.glob(
            path + "/" + dataset + "_?t%s_class???.mrc" % last_iteration
        )
        starfile = glob.glob(path + "/" + dataset + "_?t%s_data.star" % last_iteration)[
            -1
        ]

        # plot current resolution curve
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        com = (
            """grep _rlnCurrentResolution %s/%s_it0??_model.star | awk '{ print $2}' """
            % (path, dataset)
        )
        resolution = numpy.array(subprocess.getoutput(com).split()[1:], dtype="f")
        plt.clf()
        plt.plot(resolution, label=dataset)
        plt.title("Current Resolution", fontsize=12)
        plt.xlabel("Iteration")
        plt.ylabel("Resolution (A)")
        plt.legend(prop={"size": 10})
        plt.savefig("%s/%s_evolution.png" % (path, dataset))

    # 3D Auto-Refine
    else:
        path = "Refine3D"
        classes = glob.glob(path + "/" + dataset + "_class???.mrc")
        starfile = path + "/" + dataset + "_data.star"
        half1 = classes[0].replace("class001", "half1_class001_unfil")
        half2 = classes[0].replace("class001", "half2_class001_unfil")

        # compute FSC between half maps
        model_box_size = int(mrc.readHeaderFromFile(classes[0])["nx"])
        apix = float(mrc.readHeaderFromFile(classes[0])["xlen"]) / float(model_box_size)

        # calculate correlation
        com = "module load EMAN2; e2proc3d.py {0} {4}/{2}_fsc.txt --apix={3} --calcfsc={1}".format(
            half1, half2, dataset, apix, path
        )
        logger.info(com)
        subprocess.getoutput(com)

        # plot curves
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        FSCs = numpy.loadtxt(path + "/{0}_fsc.txt".format(dataset))
        ax.plot(FSCs[:, 0], FSCs[:, 1], label=classes[0])
        legend = ax.legend(loc="upper right", shadow=True)
        ax.set_ylim((-0.1, 1.05))
        # ax.set_xlim((FSCs[0,0],FSCs[-1,0]))
        plt.title("FSC for %s (unmasked)" % dataset, fontsize=12)
        plt.xlabel("Frequency (1/A)")
        plt.ylabel("FSC")
        plt.legend(prop={"size": 10})
        plt.savefig("%s/%s_fsc.png" % (path, dataset))
        os.remove(path + "/{0}_fsc.txt".format(dataset))

        # plot current resolution curve
        com = (
            """grep _rlnCurrentResolution %s/%s_it0??_half1_model.star | awk '{ print $2}' """
            % (path, dataset)
        )
        resolution = numpy.array(subprocess.getoutput(com).split()[1:], dtype="f")
        plt.clf()
        plt.plot(resolution, label=dataset)
        plt.title("Current Resolution", fontsize=12)
        plt.xlabel("Iteration")
        plt.ylabel("Resolution (A)")
        plt.legend(prop={"size": 10})
        plt.savefig("%s/%s_evolution.png" % (path, dataset))

    # plot orientations and defocuses
    analysis.plot.generate_plots_relion(starfile)

    # reconstruction montage
    for map in classes:

        # map slices
        rec = mrc.read(map)
        z = rec.shape[0]

        # what if radius is larger than box
        model_box_size = int(mrc.readHeaderFromFile(map)["nx"])
        apix = float(mrc.readHeaderFromFile(map)["xlen"]) / float(model_box_size)
        radius = float(sp["particle_rad"]) / apix
        if radius > z / 2:
            logger.warning("Particle radius falls outside box %f > %f", radius, z / 2)
            radius = z / 2 - 1

        lim = int(z / 2 - radius)
        nz = z - 2 * lim
        montage = numpy.zeros([nz * 2, nz * 3])

        # 2D central slices
        i = 0
        j = 0
        I = rec[z / 2, lim:-lim, lim:-lim]
        montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (I - I.mean()) / I.std()
        j = 1
        I = rec[lim:-lim, z / 2, lim:-lim]
        montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (I - I.mean()) / I.std()
        j = 2
        I = rec[lim:-lim, lim:-lim, z / 2]
        montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (I - I.mean()) / I.std()

        # 2D projections
        i = 1
        for j in range(3):
            I = numpy.average(rec[lim:-lim, lim:-lim, lim:-lim], j)
            montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (
                I - I.mean()
            ) / I.std()

        name = map[:-4]

        writepng(montage, "%s_map.png" % name)

        # plot angular accuracy curves
        if "class" in rp["mode"].lower():
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            com = """grep %s %s/%s_it0??_model.star | awk '{ print $3}' """ % (
                map[-13:],
                path,
                dataset,
            )
            accuracy = numpy.array(subprocess.getoutput(com).split()[1:], dtype="f")
            plt.plot(accuracy, label=dataset)
            legend = ax.legend(loc="upper right", shadow=True)
            plt.title("Accuracy of Rotations", fontsize=12)
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.legend(prop={"size": 10})
            plt.savefig("%s/%s_fsc.png" % (path, dataset))

        # create composite montage
        command = "montage {0}_map.png {1}/{2}_fsc.png {3}_prs.png {1}/{2}_evolution.png -geometry 690x460 {0}.png".format(
            name, path, dataset, starfile.replace(".star", "")
        )
        subprocess.getoutput(command)

        # delete temporary files
        for i in "{0}/{1}_fsc.png {2}_map.png".format(path, dataset, name).split():
            os.remove(i)

        # email result
        if False:
            png_plot = "%s.png" % name
            attach = os.getcwd() + "/" + png_plot
            user_comm.notify(os.path.split(name)[-1] + " (3D)", attach)

    # delete temporary files
    for i in "{0}/{1}_evolution.png {2}_prs.png".format(
        path, dataset, starfile.replace(".star", "")
    ).split():
        os.remove(i)


if __name__ == "__main__":

    os.environ["PYTHONDIR"] = "{0}/python".format(os.environ["PYP_DIR"])

    # check if we are in frealign directory
    if "relion" not in os.path.split(os.getcwd())[-1]:
        logger.error("You are not in the relion directory.")
        sys.exit(1)

    # parse relion parameters
    rparameters = parse_arguments()

    # load PYP parameters
    sparameters = project_params.load_pyp_parameters("..")

    if "pyprunrln" in os.environ:

        """
        com = 'module load RELION/1.4-beta-2'
        print com
        print commands.getoutput(com)
        """
        # for i in os.environ:
        #    # if 'slurm' in i.lower() or 'mpi' in i.lower():
        #    print i, os.environ[i]

        logger.info("Running RELION")
        run_relion(rparameters, sparameters, "f" in rparameters["debug"].lower())

    else:

        # create initial model with correct sampling and size
        if os.path.exists(rparameters["model"]):
            actual_pixel = (
                float(sparameters["scope_pixel"])
                * float(sparameters["data_bin"])
                * float(sparameters["extract_bin"])
            )
            model_box_size = int(mrc.readHeaderFromFile(rparameters["model"])["nx"])
            model_pixel_size = float(
                mrc.readHeaderFromFile(rparameters["model"])["xlen"]
            ) / float(model_box_size)
            relion_initial_model = rparameters["dataset"] + "_initial_model.mrc"
            if (
                model_pixel_size != actual_pixel
                or int(sparameters["extract_box"]) != model_box_size
            ):
                command = "module load EMAN1; proc3d {0} {1} scale={2} clip={3},{3},{3}".format(
                    rparameters["model"],
                    relion_initial_model,
                    "%.2f" % (model_pixel_size / actual_pixel),
                    sparameters["extract_box"],
                )
                logger.info("Making initial model compatible with particle stack")
                logger.info(command)
                logger.info(subprocess.getoutput(command))
            else:
                os.symlink(rparameters["model"], relion_initial_model)

        # check parameter consistency
        if "movie" in rparameters["mode"].lower():
            if not os.path.split(rparameters["continue"])[-1][
                -15:
            ] == "_optimiser.star" or not os.path.exists(rparameters["continue"]):
                logger.error(
                    "Please specify corresponding _optimiser.star file while in movie mode."
                )
                sys.exit()

            frames_file = rparameters["dataset"] + "_frames.star"
            if not os.path.exists(frames_file):
                logger.error("%s does not exist.", frames_file)
                sys.exit()

        if "polish" in rparameters["mode"].lower():

            if not os.path.exists(rparameters["mask"]):
                logger.error(
                    "%s does not exist. Needed for particle polishing",
                    rparameters["mask"],
                )
                sys.exit()

            frames_file = "Refine3D/%s_frames_data.star" % rparameters["dataset"]
            if not os.path.exists(frames_file):
                logger.error("%s not found.", frames_file)
                sys.exit()

        # manage resources automatically
        """
        relion_initial_model = rparameters['dataset'] + '_initial_model.mrc'
        boxsize = int(mrc.readHeaderFromFile( relion_initial_model )['nx'])
        if len( rparameters['nodes'] ) == 0:
           
            # each ntask has cpus_per_task threads and each thread needs memory_per_cpu GB of memory 

            # set memory requiremnts according to image size
            factor = boxsize * boxsize / 256.0 / 256.0
            
            # memory per MPI process ( 256x256 = 8GB )
            rparameters['mem_per_task'] = str( int( numpy.ceil( factor * 8 ) ) )
            
            # assume total node memory is 128GB and core count is 32
            rparameters['cpus_per_task'] = str( int( 32 * float(rparameters['mem_per_task']) / 128.0 ) )
            
            # use 10-core jobs
            rparameters['nodes'] = 10        # str( int( 8 * 128 / int( rparameters['mem_per_task'] ) ) )

            project_params.save_relion_parameters(rparameters)
            
            if boxsize < 384:
                # Movie Refinement
                if 'movie' in rparameters['mode'].lower():
                    rparameters['cpus_per_task'] = '2' 
                    rparameters['mem_per_cpu'] = '15'
                    rparameters['ntasks'] = '32'
                # Particle Polishing
                elif 'polish' in rparameters['mode'].lower():
                    rparameters['cpus_per_task'] = '2'
                    rparameters['mem_per_cpu'] = '32'
                    rparameters['ntasks'] = '12'
                # Regular Classification and Refinement
                else:
                    rparameters['cpus_per_task'] = '8'
                    rparameters['mem_per_cpu'] = '4'
                    rparameters['ntasks'] = '64'
            
            elif boxsize < 512:
                # Movie Refinement
                if 'movie' in rparameters['mode'].lower():
                    rparameters['cpus_per_task'] = '2' 
                    rparameters['mem_per_cpu'] = '15'
                    rparameters['ntasks'] = '32'
                # Particle Polishing
                elif 'polish' in rparameters['mode'].lower():
                    rparameters['cpus_per_task'] = '2'
                    rparameters['mem_per_cpu'] = '32'
                    rparameters['ntasks'] = '12'
                # Regular Classification and Refinement
                else:
                    rparameters['cpus_per_task'] = '4'
                    rparameters['mem_per_cpu'] = '15'
                    rparameters['ntasks'] = '16'

            else:
                rparameters['cpus_per_task'] = '4'
                rparameters['mem_per_cpu'] = '15'
                rparameters['ntasks'] = '16'
        """

        # keep track of issued commands
        with open(".rly_history", "a") as f:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
                "%Y/%m/%d %H:%M:%S "
            )
            f.write(timestamp + " ".join(sys.argv) + "\n")

        logger.info("Launching RELION")

        # launch next relion command
        dependency = launch_relion(rparameters, sparameters)

        logger.info("Job ID %s", dependency)

        if False and "t" in sparameters["data_auto"].lower():
            launch_pyp(rparameters, sparameters, dependency)

    project_params.save_relion_parameters(rparameters)
