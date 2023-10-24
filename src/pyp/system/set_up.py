import glob
import os
import shutil
import subprocess

from pyp.inout.metadata import frealign_parfile
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_frealign_paths, get_parameter_files_path
from pyp.utils import get_relative_path, makedirs_list, symlink_force

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def prepare_spr_dir():
    folders = [
        os.getcwd() + "/" + s
        for s in (
            "raw",
            "mrc",
            "webp",
            "log",
            "pkl",
            "swarm",
            "train",
            "frealign",
            "csp",
        )
    ]
    makedirs_list(folders)


def prepare_spr_daemon_dir():
    folders = [
        os.getcwd() + "/" + s
        for s in (
            "raw",
            "mrc",
            "webp",
            "log",
            "pkl",
            "swarm",
            "class2d",
        )
    ]
    makedirs_list(folders)


def prepare_tomo_dir():
    folders = [
        os.getcwd() + "/" + s
        for s in (
            "raw",
            "next",
            "mrc",
            "webp",
            "train",
            "sva",
            "log",
            "pkl",
            "swarm",
            "frealign",
            "csp",
        )
    ]
    makedirs_list(folders)


def prepare_frealign_dir():
    folders = ["scratch", "maps", "swarm", "log", "data"]
    makedirs_list(folders)


def prepare_3davg_dir():
    folders = ["protocol", "swarm"]
    makedirs_list(folders)


def prepare_3davg_xml(dataset):
    """Prepare xml files needed for paramterizing 3DAVG sub-tomogram averaging

    Args:
        dataset (String): Name of dataset
    """
    # only prepare xml if they don't exist
    for xmlfile in glob.glob("{0}/*.xml".format(get_parameter_files_path())):
        basename = os.path.basename(xmlfile)
        if not os.path.exists(f"protocol/{basename}"):
            with open(xmlfile, "r") as f:
                contents = f.read().replace("DEFAULT_PATTERN", dataset)
            with open(f"protocol/{basename}", "w") as f:
                f.write(contents)


def initialize_classification(
    mp, iteration, dataset, classes, references_only=False, parameters_only=False
):
    frealign_paths = get_frealign_paths()

    if not references_only:
        if not mp["reconstruct_weights"]:
            # Initialize parameter files for each class using RSAMPLE
            actual_pixel = (
                float(mp["scope_pixel"])
                * float(mp["data_bin"])
                * float(mp["extract_bin"])
            )

            # save extended version of par file
            prefix = "maps/%s_r01_%02d" % (dataset, iteration - 1)
            shutil.copy2(prefix + ".par", prefix + ".parx")

            # WARNING: rsample only exists for cc3m (need to find out how this is handled in the newer versions of frealign)
            command = """
%s/bin/rsample.exe << eot
maps/%s_r01_%02d.par
%s
%s
maps/%s_%02d_r.par
eot
""" % (
                frealign_paths["cc3m"],
                dataset,
                iteration - 1,
                actual_pixel,
                classes,
                dataset,
                iteration - 1,
            )
            logger.info(command)
            subprocess.Popen(command, shell=True, text=True).wait()
            for ref in range(classes):
                # rename parameter files to follow general convention dataset_r01_01.par
                source = "maps/%s_%02d_r%d.par" % (dataset, iteration - 1, ref + 1)
                target = "maps/%s_r%02d_%02d.par" % (dataset, ref + 1, iteration - 1)
                shutil.copy2(prefix + ".parx", target)
                from pyp import align

                align.concatenate_par_files(target, source, mp)
                os.remove(source)
        else:
            input_par_file = "maps/%s_r01_%02d.par" % (dataset, iteration - 1)
            for res in range(1, classes):
                mask = ["0", "0", "0", "0", "0"]
                mask[(res - 1) % 5] = "1"
                frealign_parfile.Parameters.generate_par_file(
                    input_par_file,
                    "maps/%s_r%02d_%02d.par" % (dataset, res + 1, iteration - 1),
                    ",".join(mask),
                )

    if not parameters_only:
        for ref in range(1, classes):
            # create initial models and fsc.txt files for each reference
            try:
                source = "maps/%s_r01_%02d.mrc" % (dataset, iteration - 1)
                target = "maps/%s_r%02d_%02d.mrc" % (dataset, ref + 1, iteration - 1)
                symlink_force(os.path.join(os.getcwd(), source), target)
                shutil.copy2(
                    "maps/%s_r01_fsc.txt" % dataset,
                    "maps/%s_r%02d_fsc.txt" % (dataset, ref + 1),
                )
            except:
                pass
