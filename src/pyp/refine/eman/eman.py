import os
import subprocess

from pyp.inout.image import mrc
from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import eman_load_command, is_atrf, is_biowulf2, qos
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def eman_2d_classify(parameters, new_name, imagic_stack, radius):
    # this is the radius before the binning used for classification
    classes = int(
        mrc.readHeaderFromFile(imagic_stack)["nz"] / int(parameters["class_num"])
    )

    # use user provided script if available, otherwise use the standard script in the shell directory
    if os.path.exists("eman/eman2_classify_mpi.sh"):
        emanpath = "eman"
    else:
        emanpath = os.environ["SHELLDIR"]

    if is_biowulf2():
        command = "cd eman; sbatch --time 5-0:00:00 --gres=lscratch:400 --mem=58g --ntasks=32 --export data={0},shrink={1},radius={2},classes={3} {4} {5}/eman2_classify_mpi.sh".format(
            new_name + "_stack",
            parameters["class_bin"],
            radius * 1.25,
            classes,
            parameters["slurm_queue"] + " " + qos(parameters["slurm_queue"]),
            emanpath,
        )  # umask=33
    elif is_atrf():
        command = "cd eman; qsub -v data={0},shrink={1},radius={2},classes={3} -W umask=33 -l nodes=1:gpfs {4} {5}/eman2_classify_mpi.sh".format(
            new_name + "_stack",
            parameters["class_bin"],
            radius * 1.25,
            classes,
            parameters["slurm_queue"],
            emanpath,
        )
    else:
        command = "cd eman; sbatch --export=data={0},shrink={1},radius={2},classes={3} --nodes=1 {4} {5}/eman2_classify_mpi.sh".format(
            new_name + "_stack",
            parameters["class_bin"],
            radius * 1.25,
            classes,
            parameters["slurm_queue"],
            emanpath,
        )

    logger.info(command)
    logger.info(
        subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True, text=True
        )
    )


def eman_3davg(parameters):
    # HF Liu: do eman subtomogram averaging
    # clean up the last vol lst
    if os.path.exists("eman/subvols.lst"):
        os.remove("eman/subvols.lst")

    # make new vol lst
    load_eman_cmd = eman_load_command()

    # make sub-list for each tilt series
    if os.path.exists(parameters["data_set"] + ".films"):
        film = open(parameters["data_set"] + ".films")
        film_list = film.readlines()

        for tilt in film_list:
            tilt_name = tilt.strip("\n")

            command = "{0}; cd eman; e2proclst.py ../sva/{1}_*.rec --create {1}_subvols.lst".format(
                load_eman_cmd, tilt_name,
            )
            logger.info(command)
            logger.info(
                subprocess.check_output(
                    command, stderr=subprocess.STDOUT, shell=True, text=True
                )
            )
    # merge sub-list as one subvols.lst
    com = "{0}; cd eman; e2proclst.py *_subvols.lst --merge subvols_all.lst".format(
        load_eman_cmd
    )
    logger.info(com)
    logger.info(
        subprocess.check_output(com, stderr=subprocess.STDOUT, shell=True, text=True)
    )

    # Submit the subtomogram avg to queue and rewrite parameter file for CSPT
    command_e2avg = "e2spt_refine.py subvols_all.lst --reference={0} --niter=5 --sym={1} --mass={2} --goldstandard=30 --pkeep=0.8 --maxtilt=90.0 --parallel=mpi:280:/scratch".format(
        project_params.resolve_path(parameters["refine_model"]),
        parameters["particle_sym"],
        parameters["particle_mw"],
    )

    command = "{0}; cd eman; sbatch --ntasks=280 --cpus-per-task=1 --mem=600G --job-name=e2avg_{2} --output=%x.out --error=%x.err --wrap='{4} && python {5}/pyp/eman2cspt.py'".format(
        load_eman_cmd,
        parameters["data_set"][-6:],
        parameters["data_set"],
        parameters["slurm_queue"] + " " + qos(parameters["slurm_queue"]),
        command_e2avg,
        os.environ["PYP_DIR"],
    )
    logger.info(
        subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True, text=True
        )
    )
