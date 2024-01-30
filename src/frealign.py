#!/usr/bin/env python

# SBATCH --job-name="fyp"
# SBATCH --output="fyp_%j.out"
# SBATCH --error="fyp_%j.err"

##PBS -k oe
##PBS -N fyp


import datetime
import os
import sys
import socket
import subprocess
import shutil
import time
import toml
from pathlib import Path, PosixPath

from pyp.inout.image import mrc
from pyp.inout.metadata import get_particles_from_par
from pyp.refine.frealign.frealign import (
    parse_def_split_arguments,
    frealign_def_split,
    parse_def_merge_arguments,
    frealign_def_merge,
    parse_def_arguments,
    frealign_def,
    parse_ref_arguments,
    split_refinement,
    parse_rec_arguments,
    frealign_rec,
    parse_rec_split_arguments,
    split_reconstruction,
    parse_rec_merge_arguments,
    frealign_rec_merge,
    parse_arguments,
    frealign_iterate
)
from pyp.stream import fyp_daemon
from pyp.streampyp.web import Web
from pyp.system import local_run, project_params, mpi
from pyp.system.utils import (
    get_imod_path,
    get_multirun_path,

)
from pyp.system.singularity import run_slurm, get_pyp_configuration
from pyp.system.set_up import prepare_frealign_dir
from pyp.utils import symlink_relative
from pyp.system.logging import initialize_pyp_logger

logger = initialize_pyp_logger(log_name=__name__)

if __name__ == "__main__":

    mpi_tasks = mpi.initialize_worker_pool()

    # retrieve version number
    version = toml.load(os.path.join(os.environ['PYP_DIR'],"nextpyp.toml"))['version']
    memory = f"and {int(os.environ['SLURM_MEM_PER_NODE'])/1024:.0f} GB of RAM" if "SLURM_MEM_PER_NODE" in os.environ else ""

    logger.info(
        "Job (v{}) launching on {} using {} task(s) {}".format(
        version, socket.gethostname(), mpi_tasks, memory
        )
    )

    os.environ["IMAGICDIR"] = "/usr/bin"
    os.environ["SHELLDIR"] = "{0}/shell".format(os.environ["PYP_DIR"])
    os.environ["PYTHONDIR"] = "{0}/pyp".format(os.environ["PYP_DIR"])

    config = get_pyp_configuration()
    os.environ["PYP_SCRATCH"] = config["pyp"]["scratch"]

    if "SLURM_ARRAY_JOB_ID" in os.environ:
        subdir = f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        os.environ["PYP_SCRATCH"] = str(Path(os.environ["PYP_SCRATCH"]) / subdir)
    elif "SLURM_JOB_ID" in os.environ:
        os.environ["PYP_SCRATCH"] = str(Path(os.environ["PYP_SCRATCH"]) / os.environ["SLURM_JOB_ID"])

    if not os.path.exists(os.environ["PYP_SCRATCH"]):
        os.mkdir(os.environ["PYP_SCRATCH"])
    if not os.environ.get("PBS_O_WORKDIR"):
        os.environ["PBS_O_WORKDIR"] = os.getcwd()
    os.environ["MPI_BIN"] = "/opt/apps/rhel7/anaconda2/bin"

    # if fyp was launched by the webserver, do some additional initialization
    if Web.exists:
        Web.init_env()

    if True:
        if "sess_ref" in os.environ:

            del os.environ["sess_ref"]
            local_scratch = os.environ["PYP_SCRATCH"]
            project_directory = Path().cwd().parent
            frealign_directory = Path().cwd() / "class2d"
            flag = None
            while True:
                if flag is None:
                    flag = fyp_daemon.fyp_daemon(existing_unique_name=None, existing_boxes_lists=dict())
                else:
                    if flag["type"] == "clear":
                        shutil.rmtree(local_scratch)
                        os.mkdir(local_scratch)
                        os.chdir(frealign_directory)
                        logger.warning("Clean local scratch and re-start 2D classification")
                        flag = fyp_daemon.fyp_daemon(existing_unique_name=None, existing_boxes_lists=dict())
                    elif flag["type"] == "restart":
                        os.chdir(frealign_directory)
                        logger.warning("Re-start 2D classification")
                        flag = fyp_daemon.fyp_daemon(existing_unique_name=flag["existing_name"], existing_boxes_lists=flag["existing_boxes"])

                    elif flag["type"] == "stop":
                        logger.warning("Terminate 2D classification")
                        sys.exit()


        if "refine2d" in os.environ:

            del os.environ["refine2d"]

            mparameters = project_params.load_pyp_parameters("..")
            fparameters = project_params.load_fyp_parameters()

        elif "frealign_def_split" in os.environ:

            del os.environ["frealign_def_split"]

            fp = project_params.load_fyp_parameters()

            args = parse_def_split_arguments()

            frealign_def_split(fp, args.parfile, args.tolerance)

        elif "frealign_def_merge" in os.environ:

            del os.environ["frealign_def_merge"]

            # go to frealign directory
            os.chdir("..")

            fparameters = project_params.load_fyp_parameters(".")

            args = parse_def_merge_arguments()

            frealign_def_merge(fparameters, args.parfile, args.tolerance)

        elif "frealign_def" in os.environ:

            del os.environ["frealign_def"]

            # go to frealign directory
            os.chdir("..")

            # microscope parameters
            mparameters = project_params.load_pyp_parameters("..")

            # frealign parameters
            fparameters = project_params.load_fyp_parameters(".")

            args = parse_def_arguments()

            frealign_def(
                mparameters,
                fparameters,
                args.parfile,
                args.film,
                args.scanor,
                args.tolerance,
            )

        elif "frealign_ref" in os.environ:

            del os.environ["frealign_ref"]

            args = parse_ref_arguments()

            # go to frealign directory
            os.chdir("..")

            fparameters = project_params.load_fyp_parameters()

            mparameters = project_params.load_pyp_parameters("..")

            split_refinement(
                mparameters,
                fparameters,
                args.ref,
                args.first,
                args.last,
                args.iteration,
                args.metric,
            )

        elif "frealign_rec" in os.environ:

            del os.environ["frealign_rec"]

            args = parse_rec_arguments()

            # go to frealign directory
            os.chdir("..")

            fparameters = project_params.load_fyp_parameters()

            mparameters = project_params.load_pyp_parameters("..")

            os.chdir("scratch")

            frealign_rec(mparameters, fparameters, args.iteration, args.alignment_option)

        elif "frealign_rec_split" in os.environ:

            del os.environ["frealign_rec_split"]

            args = parse_rec_split_arguments()

            # go to frealign directory
            os.chdir("..")

            fparameters = project_params.load_fyp_parameters()

            mparameters = project_params.load_pyp_parameters("..")

            os.chdir("scratch")

            split_reconstruction(
                mparameters,
                fparameters,
                args.first,
                args.last,
                args.iteration,
                args.ref,
                args.count,
            )

        elif "frealign_rec_merge" in os.environ:

            del os.environ["frealign_rec_merge"]

            args = parse_rec_merge_arguments()

            # go to frealign directory
            os.chdir("..")

            fparameters = project_params.load_fyp_parameters()

            mparameters = project_params.load_pyp_parameters("..")

            os.chdir("scratch")

            frealign_rec_merge(mparameters, fparameters, args.iteration)

        else:

            # initialize

            # cd to working directory if running with qsub
            if os.environ.get("SLURM_SUBMIT_DIR"):
                if not "swarm" in os.environ.get(
                    "SLURM_SUBMIT_DIR"
                ) and "frealign" in os.environ.get("SLURM_SUBMIT_DIR"):
                    os.chdir(os.environ["SLURM_SUBMIT_DIR"])

            logger.info(f"Running on directory {os.getcwd()}")

            # check if we are in frealign directory
            if "frealign" not in os.path.split(os.getcwd())[-1]:
                logger.error("You are not in the frealign directory.")
                sys.exit(1)

            # Create directories if needed
            prepare_frealign_dir()

            # load frealign parameters
            fparameters = parse_arguments()

            # go onother level up to project directory
            mparameters = project_params.load_pyp_parameters("..")

            # check existence of input files
            dataset = fparameters["dataset"]
            if fparameters["model"]:
                initial_model = fparameters["model"]
            else:
                initial_model = frealign_initial_model = dataset + "_01.mrc"
            stack_file = dataset + "_stack.mrc"

            # print initial_model
            par_file = dataset + "_01.par"

            for f in {initial_model, stack_file, par_file}:
                if not os.path.isfile(f):
                    logger.error("{0} not found.".format(f))
                    sys.exit(1)

            if int(fparameters["iter"]) > 2:
                frealign_initial_model = (
                    "maps/" + dataset + "_r01_%02d.mrc" % (int(fparameters["iter"]) - 1)
                )
            else:
                frealign_initial_model = dataset + "_01.mrc"

            # create proper initial model
            # TODO: can we generalize this? there are several parts in fyp where we conditionate initial model
            if fparameters["model"] and int(fparameters["iter"]) == 2:
                actual_pixel = (
                    float(mparameters["scope_pixel"])
                    * float(mparameters["data_bin"])
                    * float(mparameters["extract_bin"])
                )
                model_box_size = int(mrc.readHeaderFromFile(initial_model)["nx"])
                model_pixel_size = float(
                    mrc.readHeaderFromFile(initial_model)["xlen"]
                ) / float(model_box_size)

                if (
                    model_pixel_size != actual_pixel
                    or int(mparameters["extract_box"]) != model_box_size
                ):

                    # scale and crop initial model
                    scaling = model_pixel_size / actual_pixel
                    logger.info(f"Rescaling initial model {initial_model} to {scaling} A per pixel")
                    new_size = int(mparameters["extract_box"])
                    command = f"{get_imod_path()}/bin/matchvol -size {new_size},{new_size},{new_size} -3dxform {scaling},0,0,0,0,{scaling},0,0,0,0,{scaling},0 {initial_model} {frealign_initial_model}; rm {frealign_initial_model}~"
                    local_run.run_shell_command(command=command,verbose=mparameters["slurm_verbose"])

                else:
                    symlink_relative(initial_model, frealign_initial_model)

            # check size consistency between model and stack
            model = mrc.readHeaderFromFile(frealign_initial_model)
            stack = mrc.readHeaderFromFile(stack_file)
            if not stack["nx"] == model["nx"] or not stack["ny"] == model["ny"]:
                logger.error(
                    "Initial model dimensions do not match particle stack: {0} != {1}.".format(
                        model["nx"], stack["nx"]
                    )
                )
                sys.exit(1)

            # get non-empty files from .par file
            parlines = get_particles_from_par(par_file)
            if not stack["nz"] == parlines:
                logger.error(
                    "number of particles in parameter file and stack file do not match ({0} != {1})".format(
                        parlines, stack["nz"]
                    )
                )
                sys.exit(1)

            # initialize multirun
            if (
                os.environ.get("SLURM_NODELIST")
                or os.environ.get("MYNODES")
                or os.environ.get("PBS_NODEFILE")
            ):

                if os.environ.get("MYNODES"):
                    shutil.copy(os.environ["MYNODES"], "mpirun.mynodes")

                    # signal dummy processes to stop
                    open(os.environ["MYNODES"].replace("mynodes", "stopit"), "w").close()
                elif os.environ.get("SLURM_NODELIST"):

                    nodes = subprocess.check_output(
                        "{0}/scontrol show hostname $SLURM_JOB_NODELIST".format( run_slurm( command='scontrol' ) ),
                        stderr=subprocess.STDOUT,
                        shell=True,
                        text=True,
                    ).splitlines()
                    with open("mpirun.mynodes", "w") as f:
                        for n in nodes:
                            f.write(n + "\n")

                    # signal dummy processes to stop
                    open("mpirun.stopit", "w").close()
                else:
                    shutil.copy(os.environ["PBS_NODEFILE"], "mpirun.mynodes")

                # initialize nodes and copy particle stack
                nodes, multirun_file = local_run.create_initial_multirun_file(
                    fparameters, dataset
                )

                command = "{3}/mpirun -np {0} -map-by node --mca btl_tcp_if_include eth0 {1}/multirun -m {2}\n".format(
                    len(nodes),
                    get_multirun_path(),
                    os.getcwd() + "/" + multirun_file,
                    os.environ["MPI_BIN"],
                    )
                logger.info(command)
                logger.info(subprocess.getoutput("eval `{0}`".format(command)))

            # cleanup
            else:
                if os.path.isfile("mpirun.mynodes"):
                    os.remove("mpirun.mynodes")
                if os.path.isfile("mpirun.myrecnodes"):
                    os.remove("mpirun.myrecnodes")

            # launch iterative refinement
            first_iteration = int(fparameters["iter"])
            if first_iteration > 0:
                frealign_iterate(mparameters, fparameters, first_iteration)

            # signal dummy processes to continue
            if os.environ.get("MYNODES"):
                os.remove(os.environ["MYNODES"].replace("mynodes", "stopit"))

            # keep track of issued commands
            if "scratch" not in os.getcwd():
                with open(".fry_history", "a") as f:
                    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
                        "%Y/%m/%d %H:%M:%S "
                    )
                    f.write(timestamp + " ".join(sys.argv) + "\n")
    else:
        mpi.initialize_worker_pool()
