#!/usr/bin/env python

import os
import sys

sys.dont_write_bytecode = True

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"

# prevent matplotlib warning
os.environ["MPLCONFIGDIR"] = "/tmp"
import matplotlib

matplotlib.use("Agg")

import argparse
import cProfile
import datetime
import filecmp
import glob
import math
import multiprocessing
import random
import shutil
import socket
import sys
import time
import json
import pickle
import re
import toml
from pathlib import Path
from uuid import uuid4
import numpy as np

from pyp import align
from pyp import ctf as ctf_mod
from pyp import detect, extract, merge, postprocess, preprocess
from pyp.analysis import plot
from pyp.analysis.image import (
    compute_running_avg,
    contrast_stretch,
    normalize_frames,
)
from pyp.analysis import statistics
from pyp.analysis.occupancies import occupancy_extended, classification_initialization, get_statistics_from_par
from pyp.analysis.scores import clean_particles_tomo, score_particles_fromparx, particle_cleaning
from pyp.ctf import utils as ctf_utils
from pyp.detect import joint, topaz, tomo_subvolume_extract_is_required
from pyp.detect import tomo as detect_tomo
from pyp.inout.image import mergeImagicFiles, mergeRelionFiles, mrc, img2webp, decompress
from pyp.inout.image.core import get_gain_reference, get_image_dimensions, generate_aligned_tiltseries, get_tilt_axis_angle, cistem_mask_create
from pyp.inout.utils import pyp_edit_box_files as imod
from pyp.inout.metadata import (
    compileDatabase,
    csp_extract_coordinates,
    frealign_parfile,
    pyp_metadata,
    generateGlobalFrameWeights,
    generateRelionParFileNew,
    get_new_input_list,
    tomo_load_frame_xf,
    use_existing_alignments,
    get_image_particle_index,
    get_particles_tilt_index,
    compute_global_weights,
)
from pyp.postprocess.pyp_fsc import fsc_cutoff
from pyp.refine.csp import particle_cspt
from pyp.refine.eman import eman
from pyp.refine.frealign import frealign
from pyp.refine.relion import relion
from pyp.stream import pyp_daemon
from pyp.streampyp.web import Web
from pyp.system import local_run, mpi, project_params, set_up, slurm, user_comm
from pyp.system.db_comm import (
    load_config_files,
    load_csp_results,
    load_spr_results,
    load_tomo_results,
    save_csp_results,
    save_micrograph_to_website,
    save_spr_results,
    save_spr_results_lean,
    save_tiltseries_to_website,
    save_refinement_to_website,
    save_reconstruction_to_website,
    save_tomo_results,
    save_tomo_results_lean,
)
from pyp.system.logging import initialize_pyp_logger
from pyp.system.set_up import prepare_frealign_dir
from pyp.system.singularity import (
    get_mpirun_command,
    get_pyp_configuration,
    run_pyp,
    run_slurm,
    run_ssh,
)
from pyp.system.utils import get_imod_path, get_multirun_path, get_parameter_files_path, get_gpu_queue
from pyp.system.wrapper_functions import (
    avgstack,
    replace_sections,
    write_current_particle,
)
from pyp.utils import timer, movie2regex, symlink_relative

__author__ = "Alberto Bartesaghi"
__maintainer__ = "Alberto Bartesaghi"
__email__ = "alberto.bartesaghi@duke.edu"

from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

# set seed
random.seed(11)

def spr_swarm_check_error(parameters, name):
    if "extract_box" in parameters.keys() and parameters["extract_box"] > 0 and os.path.exists("{}.boxx".format(name)):

        p_in_stack = 0
        if "frealign" in parameters["extract_fmt"].lower() and os.path.exists(
            name + "_stack.mrc"
        ):
            p_in_stack = int(mrc.readHeaderFromFile(name + "_stack.mrc")["nz"])
        elif "relion" in parameters["extract_fmt"].lower() and os.path.exists(
            name + "_particles.mrcs"
        ):
            p_in_stack = int(mrc.readHeaderFromFile(name + "_particles.mrcs")["nz"])
        elif "eman" in parameters["extract_fmt"].lower() and os.path.exists(
            name + "_phase_flipped_stack.mrc"
        ):
            p_in_stack = int(
                mrc.readHeaderFromFile(name + "_phase_flipped_stack.mrc")["nz"]
            )

        boxx = np.loadtxt("{}.boxx".format(name), ndmin=2)
        box = boxx[
            np.logical_and(
                boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"])
            )
        ]
        # box = boxx[ boxx[:,5] >= int(parameters['extract_cls']) ]
        p_in_box = box.shape[0]

        if p_in_stack != p_in_box:
            logger.error(
                "Number of particles does not match %s, stack = %d, box = %d",
                name,
                p_in_stack,
                p_in_box,
            )
            return True
    return False


def parse_arguments(block):

    # load existing parameters or from data_parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data_parent", "--data_parent")
    parser.add_argument("-data_mode", "--data_mode")
    parser.add_argument("-data_import", "--data_import",action="store_true",default=False)
    parser.add_argument("-csp_no_stacks", "--csp_no_stacks",action="store_true",default=False)
    args, unknown = parser.parse_known_args()
    parent_parameters = vars(args)

    # check if parent project is valid
    valid_parent = "data_parent" in parent_parameters.keys() and parent_parameters["data_parent"] is not None and os.path.exists(project_params.resolve_path(parent_parameters["data_parent"]))
    need_new_project = len(glob.glob("raw/*")) == 0 or project_params.load_parameters() == 0 or parent_parameters["data_import"] or "postproc" in os.environ
    if valid_parent and need_new_project:
        # copy parameter file from parent
        parameters_existing = project_params.load_parameters(
            project_params.resolve_path(parent_parameters["data_parent"])
        )
        # reset these parameters
        if parameters_existing:

            # read specification file
            import toml
            specifications = toml.load("/opt/pyp/config/pyp_config.toml")

            # figure out which parameters need to be reset to their default values
            for t in specifications["tabs"].keys():
                if not t.startswith("_"):
                    for p in specifications["tabs"][t].keys():
                        if "copyToNewBlock" in specifications["tabs"][t][p] and not specifications["tabs"][t][p]["copyToNewBlock"]:
                            if "default" in specifications["tabs"][t][p]:
                                parameters_existing[f"{t}_{p}"] = specifications["tabs"][t][p]["default"]
                            elif f"{t}_{p}" in parameters_existing:
                                del parameters_existing[f"{t}_{p}"]

    else:
        parameters_existing = project_params.load_parameters()

    # parse parameters
    try:
        if parameters_existing != 0:
            data_mode = parameters_existing["data_mode"]
        elif parent_parameters != 0:
            data_mode = parent_parameters["data_mode"]
        if data_mode is None:
            data_mode = "spr"
        parameters = project_params.parse_parameters(parameters_existing, block, data_mode )
    except:
        type, value, traceback = sys.exc_info()
        if type == SystemExit:
            # argparse throws this error when user input is incorrect
            # it's safe to ignore these errors
            pass
        else:
            # but show other kinds of exceptions, those are usually bugs
            sys.__excepthook__(type, value, traceback)
        return 0

    # default to current directory if not provided
    parent_dataset = parameters["data_set"]

    if parameters["data_set"] == None or parent_parameters["data_parent"] is not None:
        parameters["data_set"] = parameters["refine_dataset"] = os.path.split(os.getcwd())[1]

    # set slurm queues to default values if not provided
    if "slurm_queue" not in parameters.keys():
        parameters["slurm_queue"] = None

    if "slurm_queue_gpu" not in parameters.keys():
        parameters["slurm_queue_gpu"] = None

    if "slurm_merge_queue" not in parameters.keys():
        parameters["slurm_merge_queue"] = None

    # deal with the case of bnd defaulting to box
    if (
        "extract_bnd" in parameters.keys() and isinstance(parameters["extract_bnd"], dict)
        and "ref" in parameters["extract_bnd"].keys()
    ):
        parameters["extract_bnd"] = parameters["extract_box"]

    # in tomo mode, check if virion radius != 0 when doing virion detection 
    if ( parameters["data_mode"] == "tomo" and "tomo_vir_method" in parameters and parameters["tomo_vir_method"] != "none" and ( "tomo_vir_rad" not in parameters or "tomo_vir_rad" in parameters and parameters["tomo_vir_rad"] == 0)):
        raise Exception("Must specify non-zero virion radius (-tomo_vir_rad)")

    # initialize if no images are present
    if not len(glob.glob("raw/*")) and block != "export_session":
        if parameters["data_parent"] != None and not parameters["data_parent"] == ".":

            # link all necessary metadata directories
            if not parameters["data_import"]:
                folders = ["raw", "sva", "mrc", "webp"]
                for f in folders:
                    source = os.path.join(parameters["data_parent"], f)
                    if os.path.exists(source):
                        symlink_relative(source, f)

                # create empty log folder
                os.makedirs("log")

                # link individual files
                for folder in ["pkl", "csp"]:
                    os.makedirs(folder)
                    files = glob.glob(os.path.join(parameters["data_parent"], folder, "*"))
                    if folder == "csp":
                        exclude_file = ["micrograph_particle.index", "particle_tilt.index"]
                        files = [f for f in files if f not in exclude_file]
                    for source in files:             
                        destination = os.path.join(os.getcwd(), os.path.relpath(source,parameters["data_parent"]))
                        symlink_relative(source, destination)
            else:
                # create new folders and links to individual files
                folders = ["raw", "mrc", "webp", "sva", "pkl"]
                for f in folders:
                    os.makedirs(f)
                    files = glob.glob(
                        os.path.join(parameters["data_parent"], f, "*")
                    )
                    for source in files:
                        destination = os.path.relpath(source, parameters["data_parent"])
                        symlink_relative(source, destination)

            # link micrographs and films
            dataset = os.path.split(os.getcwd())[-1]
            extensions = [".micrographs", ".films"]

            # check if using new .micrographs file
            reinitialize = False
            micrographs = dataset + extensions[0]
            if os.path.exists(micrographs):
                shutil.copy2(micrographs, dataset + extensions[1])
                reinitialize = True
            else:
                for e in extensions:
                    source = os.path.join(parameters["data_parent"], parent_dataset + e)
                    target = dataset + e
                    if not os.path.exists(target) and os.path.exists(source):
                        symlink_relative(source, target)

            # copy configuration files
            files = [".pyp_config.toml"]
            for f in files:
                if os.path.exists(f):
                    shutil.copy2(os.path.join(parameters["data_parent"], f), f)

            # use default par file from block upstream if none specified
            if os.path.exists(micrographs) and (
                not "refine_parfile" in parameters.keys()
                or parameters["refine_parfile"] is None
                or project_params.resolve_path(parameters["refine_parfile"]) == "auto"
                or not Path(
                    project_params.resolve_path(parameters["refine_parfile"])
                ).exists
                or not "refine_parfile_tomo" in parameters.keys()
                or parameters["refine_parfile_tomo"] is None
                or project_params.resolve_path(parameters["refine_parfile_tomo"]) == "auto"
                or not Path(
                    project_params.resolve_path(parameters["refine_parfile_tomo"])
                ).exists
                or reinitialize
            ):
                # if not using all micrographs, we need to generate new .par file
                if not os.path.islink(parameters["data_set"] + ".micrographs"):
                    os.mkdir("frealign")
                    spr_merge(parameters, check_for_missing_files=False)
                    reference_par_file = os.path.join(
                        os.getcwd(), "frealign", "maps", parameters["data_set"] + "_r01_02.par.bz2"
                    )
                    logger.warning("Generated new parameter file " + reference_par_file)
                    reference_par_file = 'none'
                    reference_model_file = 'none'
                    mask_file = 'none'
                # else, we can use existing .par file from parent
                else:
                    parent_project_name = Path(parameters["data_parent"]).name
                    reference_par_file = os.path.join(
                        parent_parameters["data_parent"],
                        "frealign", "maps",
                        parent_project_name + "_r01_02.par.bz2",
                    )
                    # find out what is the most recent parameter file
                    reference_par_file = sorted(glob.glob(os.path.join(parent_parameters["data_parent"],"frealign","maps","*_r01_??.par*") ))
                    if len(reference_par_file) > 0:
                        reference_par_file = reference_par_file[-1]
                        reference_model_file = reference_par_file.replace(".bz2","").replace(".par",".mrc")
                    elif data_mode == "tomo":
                        reference_par_file = sorted(glob.glob( os.path.join(parent_parameters["data_parent"],"frealign","tomo-preprocessing-*.txt") ))
                        if len(reference_par_file) > 0:
                            reference_par_file = reference_par_file[-1]
                        reference_model_file = ""
                    else:
                        reference_par_file = ""
                        reference_model_file = ""

                    mask_file = 'none'
                    if "refine_maskth" in parameters and parameters["refine_maskth"] is not None:
                        mask_path: str = project_params.resolve_path(parameters["refine_maskth"])
                        mask_file = project_params.get_mask_from_projects() if mask_path == 'auto' else mask_path

                if os.path.exists(reference_par_file):
                    if data_mode == "tomo":
                        parameters["refine_parfile_tomo"] = reference_par_file
                    else:
                        parameters["refine_parfile"] = reference_par_file
                    if block != "spr_tomo_post_process":
                        logger.info("Using parameter file " + reference_par_file)
                if os.path.exists(reference_model_file):
                    parameters["refine_model"] = reference_model_file
                    if block != "spr_tomo_post_process":
                        logger.info("Using initial model " + reference_model_file)
                if os.path.exists(mask_file):
                    parameters["refine_maskth"] = mask_file
                    if block != "spr_tomo_post_process":
                        logger.info("Using shape mask " + mask_file)

        elif parameters["data_path"] is not None:

            # try to get parameters from autoprocessing daemon
            logger.info(
                "Attempting to load existing parameters and metadata from %s",
                os.path.split(str(project_params.resolve_path(parameters["data_path"])).split(",")[0])[0].replace(
                    "/raw", ""
                ),
            )

            if "data_retrieve" in parameters.keys() and parameters["data_retrieve"]:

                logger.info(
                    "Attempting to load parameter file from %s",
                    os.path.split(str(parameters["data_path"]).split(",")[0])[
                        0
                    ].replace("/raw", ""),
                )

                existing_parameters = project_params.load_parameters(
                    os.path.split(str(parameters["data_path"]).split(",")[0])[
                        0
                    ].replace("/raw", "")
                )

                # use existing parameters
                if existing_parameters != 0:
                    for key in existing_parameters.keys():
                        if (
                            not "data_path" in key
                            and not "data_set" in key
                            and not "queue" in key
                        ):
                            parameters[key] = existing_parameters[key]

            # create and populate raw directory if data_path provided
            if not os.path.exists("raw"):
                os.mkdir("raw")

            files = []
            for current_data_path in str(project_params.resolve_path(parameters["data_path"])).split(","):
                files += glob.glob(current_data_path)
            # remove gain reference from list
            [files.remove(file) for file in files if "Gain" in file]

            logger.info(
                "{0} found {1} files to link into raw folder".format(
                    project_params.resolve_path(parameters["data_path"]), len(files)
                )
            )
            for file in files:
                # check for duplicates
                if os.path.exists("raw/" + os.path.basename(file)):
                    time_stamp = time.strftime(
                        "%Y%m%d_%H%M", time.gmtime(os.path.getmtime(file))
                    )
                    target = (
                        os.path.splitext(os.path.basename(file))[0]
                        + "_"
                        + time_stamp
                        + os.path.splitext(os.path.basename(file))[1]
                    )
                else:
                    target = os.path.basename(file)
                symlink_relative(file, os.path.join("raw", target))

                # copy all necessary metadata as well
                name = Path(target).stem
                mfiles = dict()
                mfiles["raw"] = ".xml .rawtlt .order"
                if "data_retrieve" in parameters.keys() and parameters["data_retrieve"]:
                    if "movie_ali" in parameters and not parameters["movie_ali"]:
                        mfiles["ali"] = ".xf .tlt .fid.txt _tiltalignScript.txt"
                        mfiles["ctf"] = ".def .param .ctf _CTFprof.txt _avgrot.txt"
                    mfiles["box"] = ".box .boxx"
                    mfiles["log"] = "_RAPTOR.log"
                    mfiles["csp"] = "_boxes3d.txt"
                    mfiles["mod"] = ".spk _xray.mod _gold3d.mod _boxes.mod"
                # mfiles["tomo"] = ".rec _bin.ali _bin.mrc"
                for d in mfiles.keys():
                    if not os.path.exists(d):
                        os.mkdir(d)
                    for f in mfiles[d].split():
                        if d == "raw":
                            source = os.path.join(
                                Path(file).parents[0],
                                name + f,
                            )
                        else:
                            source = os.path.join(
                                Path(file).parents[0],
                                d,
                                name + f,
                            )
                        destination = d + "/" + name + f
                        if os.path.isfile(source) and not os.path.exists(destination):
                            if "slurm_verbose" in parameters and parameters["slurm_verbose"]:
                                logger.info("Retrieving " + source)
                            shutil.copy2(source, destination)

            ctffile = "ctf/" + os.path.splitext(os.path.basename(files[0]))[0] + ".ctf"
            if os.path.isfile(ctffile):
                ctf = np.loadtxt(ctffile)
                if (
                    parameters["scope_pixel"] == "0"
                    or "auto" in str(parameters["scope_pixel"]).lower()
                ):
                    parameters["scope_pixel"] = str(ctf[9])
                if (
                    parameters["scope_voltage"] == "0"
                    or "auto" in str(parameters["scope_voltage"]).lower()
                ):
                    parameters["scope_voltage"] = str(ctf[10])
                if (
                    parameters["scope_mag"] == "0"
                    or "auto" in str(parameters["scope_mag"]).lower()
                ):
                    parameters["scope_mag"] = str(ctf[11])
    
    if "class_num" in parameters.keys() and parameters["class_num"] > 1 or "tomo" in parameters["data_mode"]:
        parameters["refine_metric"] = "new"

    # enable _force depending on parameter changes
    if parameters_existing != 0:
        parameters = project_params.parameter_force_check(parameters_existing, parameters)

    if "extract_use_clean" in parameters.keys() and parameters["extract_use_clean"]:
        parameters["extract_cls"] = 1

    return parameters

@timer.Timer(
    "spr_merge", text="Total time elapsed (spr_merge): {}", logger=logger.info
)
def spr_merge(parameters, check_for_missing_files=True):
    """Compiles multiple out files from SPR swarm function.

    Parameters
    ----------
    parameters : dict
        Main configurations taken from .pyp_config
    """

    try:
        data_set = parameters["data_set"]
    except KeyError:
        data_set = None
    micrographs = "{}.micrographs".format(data_set)
    with open(micrographs) as f:
        input_all_list = [line.strip() for line in f]

    if os.path.exists(micrographs + "_missing"):
        micrographs += "_missing"
        with open(micrographs) as f:
            inputlist = [line.strip() for line in f]
    else:
        inputlist = input_all_list

    # check if all processes ended successfully
    if check_for_missing_files:
        missing_files = project_params.get_missing_files(parameters, inputlist)

        if len(missing_files) > 0:
            if micrographs.endswith("_missing"):
                # missing files remaining after retrying
                try:
                    os.remove(micrographs)
                    logger.warning("Detected errors even after retrying. Stopping.")
                except:
                    pass

            else:
                logger.warning(
                    "{0} jobs failed, attempting to re-submit".format(
                        len(missing_files)
                    )
                )
                micrographs_file = micrographs + "_missing"
                with open(micrographs_file, "w") as f:
                    f.write("\n".join([m for m in missing_files]))
                # re-submit only jobs that failed
                split(parameters)
                return

    logger.info(f"Total number of micrographs = {len(input_all_list)}")

    inputlist = get_new_input_list(parameters, input_all_list)

    logger.info(f"Number of micrographs used = {len(inputlist)}")

    # save final micrograph list (excluding micrographs with no boxes)
    if len(inputlist) > 0:
        films = "{}.films".format(data_set)
        if Path(films).is_symlink():
            os.remove(films)
        with open(films, "w") as f:
            f.write("\n".join(inputlist))
    else:
        logger.error("Either all micrographs failed or no particles were found, stopping")
        inputlist = input_all_list
        raise

    # use given naming convention when extracting frames for relion
    if False:
        if (
            "extract_fmt" in parameters.keys()
            and "relion_frames" in parameters["extract_fmt"].lower()
            and len(parameters["extract_fmt"].split(":")) > 1
        ):
            timestamp = parameters["extract_fmt"].split(":")[-1].split(",")[0]
        else:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
                "%Y%m%d_%H%M%S"
            )

        # figure out number of last frame
        last = project_params.find_last_frame(parameters, inputlist)

        new_name = "{0}_frames_{1}_{2}".format(
            data_set, "%02d" % int(parameters["movie_first"]), "%02d" % last,
        )

    # override with dataset name
    new_name = data_set

    frealign_stack = "frealign/{}_stack.mrc".format(new_name)
    relion_stack = "relion/{}_stack.mrcs".format(new_name)
    imagic_stack = "eman/{}_stack.mrc".format(new_name)

    if not Web.exists:
        # compile "database"
        compileDatabase(inputlist, "{}_dbase.txt".format(data_set))

        # select clean set of micrographs
        clean_list = project_params.select_clean_micrographs(
            parameters, micrographs, inputlist
        )

        # plot dataset results
        plot.plot_dataset(parameters)
    else:
        clean_list = inputlist

    pool = multiprocessing.Pool()
    if "detect_rad" in parameters and parameters["detect_rad"] > 0:

        # update/compute global frame weights
        # pool.apply_async( generateGlobalFrameWeights, args=( parameters['data_set'] ) )

        # pool.apply_async( frealign_parfile.Parameters.generateParameterFiles, args=( inputlist, parameters['data_set'], parameters['ctf_use_ast'] ) )
        if "extract_fmt" in parameters.keys():
            if "frealign" in parameters["extract_fmt"].lower():
                pool.apply_async(
                    frealign_parfile.Parameters.generateFrealignParFile,
                    args=(
                        clean_list,
                        data_set,
                        new_name,
                        parameters["ctf_use_ast"],
                        parameters["ctf_use_lcl"],
                        parameters["extract_cls"],
                    ),
                )
            if "relion" in parameters["extract_fmt"].lower():
                pool.apply_async(
                    generateRelionParFileNew,
                    args=(
                        clean_list,
                        new_name,
                        float(parameters["data_bin"]) * float(parameters["extract_bin"]),
                        parameters["scope_cs"],
                        parameters["scope_wgh"],
                        parameters,
                        parameters["ctf_use_ast"],
                    ),
                )
                # if not 'frames' in parameters['extract_fmt']:
                pool.apply_async(mergeRelionFiles, args=(clean_list, relion_stack))
            if "eman" in parameters["extract_fmt"].lower():
                pool.apply_async(mergeImagicFiles, args=(clean_list, imagic_stack))
    # pool.apply_async( compileDatabase, args=( inputlist, '{}_dbase.txt'.format( parameters['data_set'] ) ) )
    pool.close()

    # Wait for all processes to complete
    pool.join()

    generateGlobalFrameWeights(data_set)

    # cleanup
    stacklist = ["frealign/" + line + "_stack.mrc" for line in inputlist]
    null = [os.remove(i) for i in stacklist if os.path.isfile(i)]

    stacklist = ["relion/" + line + "_stack.mrcs" for line in inputlist]
    null = [os.remove(i) for i in stacklist if os.path.isfile(i)]

    stacklist = ["eman/" + line + "_phase_flipped_stack.mrc" for line in inputlist]
    null = [os.remove(i) for i in stacklist if os.path.isfile(i)]

    # launch processing jobs
    if False and int(parameters["extract_box"]) > 0:

        actual_pixel = (
            float(parameters["scope_pixel"])
            * float(parameters["data_bin"])
            * float(parameters["extract_bin"])
        )
        radius_in_A = np.array(parameters["particle_rad"].split(","), dtype=float).max()
        diameter = 2.5 * radius_in_A
        radius = int(radius_in_A / actual_pixel)

        # EMAN's 2D classification
        if (
            "eman" in parameters["extract_fmt"].lower()
            and int(parameters["class_num"]) > 0
        ):
            eman.eman_2d_classify(parameters, new_name, imagic_stack, radius)

        # RELION
        if (
            "relion" in parameters["extract_fmt"].lower()
            and not "relion_frames" in parameters["extract_fmt"].lower()
        ):
            relion.launch_relion_refinement(parameters, new_name, actual_pixel)

        # FREALIGN
        if "frealign" in parameters["extract_fmt"].lower() and os.path.exists(
            parameters["class_ref"]
        ):

            # use existing alignments if available
            if os.path.exists(parameters["class_par"]):
                use_existing_alignments(parameters, new_name)

            fp = project_params.load_fyp_parameters("frealign")

            # launch FREALIGN refinement
            frealign.launch_frealign_refinement(parameters, new_name, fp)

            # csp_final_merge
            tasks_per_arr = int(parameters["slurm_bundle_size"])
            if tasks_per_arr > 1 and "t" in parameters["csp_no_stacks"].lower():
                path = os.path.join(os.getcwd(), "frealign", "scratch")
                particle_cspt.run_merge(path)


def tomo_merge(parameters, check_for_missing_files=True):
    """Compiles multiple out files from TOMO swarm function.

    Parameters
    ----------
    parameters : dict
        Main configurations taken from .pyp_config
    """

    try:
        data_set = parameters["data_set"]
    except KeyError:
        data_set = None
    micrographs = "{}.micrographs".format(data_set)
    with open(micrographs) as f:
        input_all_list = [line.strip() for line in f]

    if os.path.exists(micrographs + "_missing"):
        micrographs += "_missing"
        with open(micrographs) as f:
            inputlist = [line.strip() for line in f]
    else:
        inputlist = input_all_list

    # check if all processes ended successfully
    if check_for_missing_files:
        missing_files = project_params.get_missing_files(parameters, inputlist)

        if len(missing_files) > 0:
            if micrographs.endswith("_missing"):
                # missing files remaining after retrying
                try:
                    os.remove(micrographs)
                    logger.error("Second attempt failed, stopping. Please check for errors in the logs.")
                    raise
                except:
                    pass
            else:
                logger.warning(
                    "{0} jobs failed, attempting to re-submit".format(
                        len(missing_files)
                    )
                )
                micrographs_file = micrographs + "_missing"
                with open(micrographs_file, "w") as f:
                    f.write("\n".join([m for m in missing_files]))
                # re-submit only jobs that failed
                split(parameters)
                return
        else:
            if micrographs.endswith("_missing"):
                # remove the missing file list 
                try:
                    os.remove(micrographs)
                    logger.warning("Image(s) that failed last time succeeded in new job(s), deleting missing file list ")
                except:
                    pass

    # save final micrograph list (excluding micrographs with no boxes)
    inputlist = get_new_input_list(parameters, input_all_list)

    if len(inputlist) > 0: 
        films = parameters["data_set"] + ".films"
        if Path(films).is_symlink():
            os.remove(films)
        with open(films, "w") as f:
            f.write("\n".join(inputlist))
            f.close()
    else:
        logger.error("Either all micrographs failed or no particles were found, stopping")
        inputlist = input_all_list
        raise

    if detect.tomo_spk_is_required(parameters) > 0:
        # produce .txt file for 3DAVG
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y%m%d_%H%M%S"
        )
        new_volumes_file = "frealign/{0}_{1}_original_volumes.txt".format(
            timestamp, parameters["data_set"]
        )
        f = open(new_volumes_file, "w")
        f.write(
            """number  lwedge  uwedge  posX    posY    posZ    geomX   geomY   geomZ   normalX normalY normalZ matrix[0]       matrix[1]       matrix[2]        matrix[3]       matrix[4]       matrix[5]       matrix[6]       matrix[7]       matrix[8]       matrix[9]       matrix[10]       matrix[11]      matrix[12]      matrix[13]      matrix[14]      matrix[15]      magnification[0]       magnification[1]      magnification[2]        cutOffset       filename\n"""
        )

        count = 1
        # combine projections into one stack if they exist
        proj_stack = []

        for file in sorted(glob.glob("sva/*_vir????.txt")):
            for volume in [
                line
                for line in open(file).read().split("\n")
                if "number" not in line and line != ""
            ]:
                vector = volume.split("\t")
                # print vector
                vector[0] = str(count)
                # randomize phi angle in +/- 180
                if parameters["tomo_vir_detect_rand"] and parameters["tomo_spk_rand"]:
                    vector[10] = "%.4f" % (360 * (random.random() - 0.5))
                vector[-1] = os.getcwd() + "/sva/" + vector[-1]
                f.write("\t".join([v for v in vector]) + "\n")

                if os.path.exists(vector[-1].split(".")[0] + ".proj"):
                    proj_stack.append(vector[-1].split(".")[0] + ".proj")

                count += 1

            os.remove(file)
        f.close()

        # combine all projections (from sub-volumes) into one stack if they exist
        if len(proj_stack) != 0:
            mrc.merge(proj_stack, "sva/%s_proj.mrc" % (parameters["data_set"]))
            for proj in proj_stack:
                try:
                    os.remove(proj)
                except OSError:
                    logger.exception("Error while deleting files.")

        # create link(s) for compatibility with 3DAVG
        volumes_file = "frealign/{0}_original_volumes.txt".format(parameters["data_set"])
        if os.path.lexists(volumes_file):
            os.remove(volumes_file)
        symlink_relative(os.path.join(os.getcwd(), new_volumes_file), volumes_file)


    if not Web.exists:
        # compile "database"
        compileDatabase(inputlist, "{}_dbase.txt".format(parameters["data_set"]))

        # plot dataset results
        # plot.plot_dataset(parameters)

    if "tomo_ext_size" in parameters and parameters["tomo_ext_size"] > 0:

        if "eman" in parameters["tomo_ext_fmt"].lower():
            eman.eman_3davg(parameters)

        else:
            pass
            # sub_tomo_avg.run_3davg(parameters)


def split(parameters):
    """Split function for both SPR and TOMO.

    Splits a large cryo-EM project with multiple movies into smaller chunks that are run
    using the swarm functions. Option to split jobs into array run via slurm or
    run locally using the multirun binary.

    Parameters
    ----------
    parameters : dict
        Main configurations taken from .pyp_config
    """

    # prepare directory structure

    if parameters["data_mode"] == "spr":
        set_up.prepare_spr_dir()
    else:
        set_up.prepare_tomo_dir()

    # create list of micrographs

    micrographs, files = project_params.create_micrographs_list(parameters)

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    # check parameter consistency before submitting jobs to swarm
    project_params.check_parameter_consistency(parameters)

    # launch pre-processing
    if not os.path.isfile("frealign/mpirun.mynodes"):

        swarm_file, gpu = slurm.create_pyp_swarm_file(parameters, files, timestamp)

        config = get_pyp_configuration()

        if parameters["slurm_queue"] == None and "slurm" in config:

            try:
                parameters["slurm_queue"] = config["slurm"]["queues"][0]
            except:
                logger.warning("No CPU partitions configured for this instance?")
                parameters["slurm_queue"] = ""
                pass

        tomo_train = parameters["data_mode"] == "tomo" and ( parameters["tomo_vir_method"] == "pyp-train" or parameters["tomo_spk_method"] == "pyp-train" )
        spr_train = parameters["data_mode"] == "spr" and "train" in parameters["detect_method"]

        if gpu or tomo_train or spr_train:
            # try to get the gpu partition
            partition_name = get_gpu_queue(parameters)
            job_name = "Split (gpu)"
        else:
            partition_name = parameters["slurm_queue"]
            job_name = "Split (cpu)"

        if ( tomo_train or spr_train ):
            if os.path.exists(os.path.join("train","current_list.txt")):
                train_swarm_file = slurm.create_train_swarm_file(parameters, timestamp)

                # submit swarm jobs
                id_train = slurm.submit_jobs(
                    "swarm",
                    train_swarm_file,
                    jobtype="milotrain" if "tomo_spk_method" in parameters and parameters["tomo_spk_method"] == "milo-train" else parameters["data_mode"] + "train",
                    jobname="Train (gpu)",
                    queue=partition_name,
                    scratch=0,
                    threads=parameters["slurm_merge_tasks"],
                    memory=parameters["slurm_merge_memory"],
                    gres=parameters["slurm_merge_gres"],
                    account=parameters.get("slurm_merge_account"),
                    walltime=parameters["slurm_merge_walltime"],
                    tasks_per_arr=parameters["slurm_bundle_size"],
                    csp_no_stacks=parameters["csp_no_stacks"],
                    use_gpu=gpu,
                ).strip()
            else:
                raise Exception("Please select a list of coordinates for training")
        else:
            id_train = ""

            # submit swarm jobs
            id = slurm.submit_jobs(
                "swarm",
                swarm_file,
                jobtype=parameters["data_mode"] + "swarm",
                jobname=job_name,
                queue=partition_name,
                scratch=0,
                threads=parameters["slurm_tasks"],
                memory=parameters["slurm_memory"],
                gres=parameters["slurm_gres"],
                account=parameters.get("slurm_account"),
                walltime=parameters["slurm_walltime"],
                tasks_per_arr=parameters["slurm_bundle_size"],
                dependencies=id_train,
                csp_no_stacks=parameters["csp_no_stacks"],
                use_gpu=gpu,
            ).strip()

            # submit merge job dependent on swarm jobs
            slurm.submit_jobs(
                "swarm",
                run_pyp(command="pyp", script=True),
                jobtype=parameters["data_mode"] + "merge",
                jobname="Merge",
                queue=parameters["slurm_queue"],
                scratch=0,
                threads=parameters["slurm_merge_tasks"],
                gres=parameters["slurm_merge_gres"],
                account=parameters.get("slurm_merge_account"),
                memory=parameters["slurm_merge_memory"],
                walltime=parameters["slurm_merge_walltime"],
                dependencies=id,
                csp_no_stacks=parameters["csp_no_stacks"],
            )

    else:
        # initialize multirun

        machinefile = "frealign/mpirun.mynodes"

        nodes = len(open(machinefile, "r").read().split())

        mpirunfile = local_run.create_pyp_multirun_file(
            parameters, files, timestamp, nodes
        )

        local_run.run_shell_command("chmod u+x {0}/{1}".format(os.getcwd(), mpirunfile),verbose=parameters["slurm_verbose"])

        mpirun = get_mpirun_command()

        open("%s/mpirun.stopit" % os.environ["HOME"], "w").close()

        command = "{0} -machinefile {1} -np {2} {3}/multirun -m {4}/{5}".format(
            mpirun, machinefile, nodes, get_multirun_path(), os.getcwd(), mpirunfile,
        )
        local_run.run_shell_command(command)

        # Merge results
        command = "cd swarm; ( export {0}merge={0}merge && {1}/bin/pyp )".format(
            parameters["data_mode"], os.environ["PYP_DIR"]
        )
        local_run.run_shell_command(command)

        os.remove("%s/mpirun.stopit" % os.environ["HOME"])


@timer.Timer(
    "spr_swarm", text="Total time elapsed (spr_swarm): {}", logger=logger.info
)
def spr_swarm(project_path, filename, debug = False, keep = False, skip = False ):
    """Main workhorse function for SPR. Performs preprocessing, frame alignment, merging, 
    ctf estimation, particle detection and extraction.

    Pseudo code
    -----------
	dbase.pull()
    image.read_2d()
        image.import()
		image.gain_correct()
		image.remove_xrays()
	align.frame_alignment()
	merge.merge_2d() [implicit]
	ctf.ctffind_2d()
	particle.detect_2d()
	particle.extract_2d()
	dbase.push()

    Parameters
    ----------
    project_path : str, Path
        Path to main project directory
    filename : str, Path
        Movie filename
    debug : bool
        Whether to save results to storage/database
    keep : bool
        Whether to keep results in scratch
    skip : bool
        Whether to ignore existing results
    """

    # manage directories
    os.chdir(project_path)

    parameters = project_params.load_parameters()

    # get file name
    name = os.path.basename(filename)

    # set-up working area
    current_path = Path.cwd()

    working_path = Path(os.environ["PYP_SCRATCH"]) / name

    logger.info(f"Running on directory {working_path}")

    if not keep:
        shutil.rmtree(working_path, "True")

    working_path.mkdir(parents=True, exist_ok=True)
    os.chdir(working_path)

    # use this to save intermediate files generated by NN particle picking
    with open("project_folder.txt", "w") as f:
        f.write(project_params.resolve_path(current_path))

    # align, pick, extract              -> image_boxed.png                        \__ montage
    # periodogram averaging, ctffind3   -> powers.png + radial.png + ctffind3.png /

    # retrieve available results
    metadata = None

    if "data_set" in parameters:
        dataset = parameters["data_set"]
    elif "stream_session_name" in parameters:
        dataset = parameters["stream_session_name"]
    else:
        raise Exception("Unknown dataset or session name")
    load_config_files(dataset, current_path, working_path)
    if not skip:
        load_spr_results(name, parameters, current_path, working_path, verbose=parameters["slurm_verbose"])

        # unpack pkl file
        metadata = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=True)
        # clean entries according to _force in parameters
        metadata.refresh_entries(parameters=parameters, update_virion=False)

        metadata.meta2PYP(path=working_path)

    else:
        logger.info("Ignoring existing results")

    # ali step
    if not parameters["movie_force"] and project_params.spr_is_done(name):
        logger.info("Skipping frame alignment and using existing frame displacements")
        aligned_average = mrc.read(f"{name}.avg")
    else:
        with timer.Timer(
            "prepare_files", text="Copy raw and gain took: {}", logger=logger.info
        ):
            basename = str(current_path / filename)
            if os.path.exists(basename + ".tif"):
                extension = ".tif"
            elif os.path.exists(basename + ".tiff"):
                extension = ".tiff"
            elif os.path.exists(basename + ".mrc"):
                extension = ".mrc"
            elif os.path.exists(basename + ".mrcs"):
                extension = ".mrcs"
            elif os.path.exists(basename + ".eer"):
                extension = ".eer"
            elif os.path.exists(basename + ".ccp4"):
                extension = ".ccp4"
            elif os.path.exists(basename + ".dm3"):
                extension = ".dm3"
            elif os.path.exists(basename + ".dm4"):
                extension = ".dm4"
            elif os.path.exists(basename + ".dm"):
                extension = ".dm"
            elif os.path.exists(basename + ".tgz"):
                extension = ".tgz"
            elif os.path.exists(basename + ".bz2"):
                extension = ".bz2"
            elif os.path.exists(basename + ".tbz"):
                extension = ".tbz"
            elif os.path.exists(basename + ".tar.bz2"):
                extension = ".tar.bz2"
            else:
                raise Exception(f"Could not find a valid extension for {basename}")

            # copy raw images to working dir
            raw_name = str(current_path / filename) + extension
            if "z" in extension:
                raw_name = decompress(raw_name, threads=parameters["slurm_tasks"])
                extension = Path(raw_name).suffix
            else:
                shutil.copy2(raw_name, os.getcwd())

            # we can't really deal with dm4's properly, so we just convert to mrc and continue on (not the most efficient, but it works)
            if extension == ".dm4":
                stem = Path(filename).stem
                local_run.run_shell_command(f"{get_imod_path()}/bin/newstack {stem}.dm4 {stem}.mrc")
                os.remove(stem+".dm4")
                extension = ".mrc"
                raw_name = stem + extension

        with timer.Timer(
            "movie_alignment", text="Movie frame alignment took: {}", logger=logger.info
        ):
            # check raw image is movie or not
            frame_num = get_image_dimensions(raw_name)[-1]
            if frame_num == 1:
                logger.info("Linking raw image %s to %s" % (name + extension, os.getcwd() + "/" + name + ".avg"))
                os.symlink(name + extension, name + ".avg")
                # assuming gain corrected for single frame image
                aligned_average = mrc.read(raw_name) 
                # saving a xf file with 0 shifts for x, y
                xfshifts = np.zeros((1, 6))
                xfshifts[:, 0] = 1
                xfshifts[:, 3] = 1
                np.savetxt(name + ".xf", xfshifts, newline="", fmt="%13.7f")
            else:

                if "gain_reference" in parameters.keys():
                    gain_reference_file = project_params.resolve_path(parameters["gain_reference"])
                    if os.path.exists(gain_reference_file):
                        shutil.copy2(gain_reference_file, os.getcwd())
                logger.info("Aligning frames using: " + parameters['movie_ali'])
                aligned_average = align.align_movie_super(
                    parameters, name, extension
                )

    auto_binning = align.generate_thumbnail(aligned_average, name, parameters)

    # ctf estimation step
    mpiF = []
    mpiARG = []
    if not parameters["ctf_force"] and ctf_mod.is_done(metadata.data, parameters, name=name, project_dir=current_path):
        logger.info("Skipping CTF estimation and using existing CTF parameters")
        ctf = np.loadtxt(f"{name}.ctf")

        # retrieve parameters if they were not set
        parameters = ctf_utils.update_pyp_params_using_ctf(parameters, ctf, save=False)

        # retrieve movie frames if we are doing movie processing
        if (
            "extract_fmt" in parameters.keys() and ( "relion_frames" in parameters["extract_fmt"]
            or "local" in parameters["extract_fmt"] )
        ):
            logger.info("Retrieving frames for movie processing")
            # TODO: a very costly way to read dims, can get from .ctf
            # x, y, z
            dims = ctf[6:9]

    else:
        # ctf = ctf_mod.ctffind4_quad(
        #    name, aligned_average, parameters, save_ctf=True, movie=0
        # )
        ctf_args = [(name, aligned_average, parameters, True, 0)]
        mpiF.append(ctf_mod.ctffind4_quad)
        mpiARG.append(ctf_args)
        # pctf = multiprocessing.Process(group=None,target=ctf_mod.ctffind4_quad, args=ctf_args)
        # pctf.start()

    actual_pixel = float(parameters["scope_pixel"]) * float(parameters["data_bin"])

    # pick and extract particles
    # TODO: split pick and extract particles
    # then change to if detect.is_required(parameters) and not detect.is_done(name):
    if ( parameters["detect_force"] or detect.is_required(parameters,name) ) and parameters["detect_method"] != "pyp-train" and parameters["detect_method"] != "none":
        detect_args = [(name, aligned_average, parameters, actual_pixel)]
        mpiF.append(detect.pick_particles)
        mpiARG.append(detect_args)
    else:
        logger.info("Skipping particle picking and using existing positions, generating boxx from box files")
        if "extract_box" in parameters and parameters["extract_box"] > 0 and "extract_bin" in parameters:
            size = parameters["extract_box"] * parameters["extract_bin"]
        else:
            size = 0
        inside = 1
        selection = parameters["extract_cls"] = 0
        particles = name + ".box"
        if os.path.exists(particles):
            coord = np.loadtxt(name + ".box", ndmin=2)
            # box files top-left corner x, y, x-size, y-size
            if coord.shape[0] > 0:
                sel_col = np.zeros((coord.shape[0], 2))
                sel_col[:, 0] = inside
                sel_col[:, 1] = selection

                if coord.shape[1] > 2:
                    coord[:, 0] = coord[:, 0] + coord[:, 2]/2 - size/2 
                    coord[:, 1] = coord[:, 1] + coord[:, 2]/2 - size/2
                    coord[:, 2] = size
                    coord[:, 3] = size
                    boxx = np.hstack((coord, sel_col))
                else:
                    size_col = np.full((coord.shape[0], 2), size)
                    coord[:, 0] = coord[:, 0] - size/2 
                    coord[:, 1] = coord[:, 1] - size/2
                    boxx = np.hstack((coord, size_col, sel_col))

                np.savetxt(name + ".boxx", boxx, fmt='%.02f', delimiter='\t')
            else:
                logger.warning("No particles from box for image " + name)
        else:
            logger.warning("No particles from box for image " + name)

    if len(mpiF) > 0:
        mpi.submit_function_to_workers(mpiF, mpiARG, verbose=parameters["slurm_verbose"])

    # store binning factor in CTF metadata
    ctf = np.loadtxt(f"{name}.ctf")
    ctf[11] = auto_binning
    np.savetxt("{}.ctf".format(name), ctf)

    # save average as mrc file
    if not os.path.exists(name + ".mrc") or not os.path.samefile(name + ".avg", name + ".mrc"):
        shutil.copy2(name + ".avg", name + ".mrc")
    else:
        logger.info("The average and the output are the same file, this is probably because the raw data has no frames")

    # save results
    if not debug:

        # pack all metadata into one pickle file
        data = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=True)
        data.loadFiles()

        if Web.exists:
            save_spr_results_lean(name, current_path, verbose=parameters["slurm_verbose"])
        else:
            save_spr_results(name, parameters, current_path, verbose=parameters["slurm_verbose"])

        save_micrograph_to_website(name,'slurm_verbose' in parameters and parameters['slurm_verbose'])

    # clean-up
    if not keep:
        shutil.rmtree(working_path)

@timer.Timer(
    "tomo_swarm", text="Total time in tomo_swarm {} seconds", logger=logger.info
)
def tomo_swarm(project_path, filename, debug = False, keep = False, skip = False ):
    """Main workhorse function for TOMO. Performs preprocessing (includes frame alignment),
    tilt-series alignment, ctf estimation & correction, TOMO reconstruction,
    particle detection and extraction.

    Pseudo code
    -----------
	dbase.pull()
	image.read_3d()
    for tilt in tilt_series:
        image.read_2d()
        image.import()
        image.gain_correct()
        image.remove_xrays()
        align.frame_alignment()
        merge.merge_2d()
    align.tilt_alignment()
    ctf.ctffind_tomo_estimate()
    ctf.ctffind_tomo_correct()
    merge.reconstruct_tomo()
    particle.detect_3d()
    particle.extract_3d()
    dbase.push()

    Parameters
    ----------
    project_path : str, Path
        Path to main project directory
    filename : str, Path
        Movie filename
    debug : bool
        Whether to save results to storage/database
    keep : bool
        Whether to keep results in scratch
    skip : bool
        Whether to ignore existing results
    """

    project_path = project_params.resolve_path(project_path)
    filename = project_params.resolve_path(filename)

    # manage directories
    os.chdir(project_path)

    parameters = project_params.load_pyp_parameters()

    # get file name
    name = os.path.basename(filename)

    # set-up working area
    current_path = Path.cwd()

    working_path = Path(os.environ["PYP_SCRATCH"]) / name

    logger.info(f"Working path: {working_path}")

    if not keep:
        shutil.rmtree(working_path, "True")

    working_path.mkdir(parents=True, exist_ok=True)
    os.chdir(working_path)

    metadata = None

    # use this to save intermediate files generated by NN particle picking
    with open("project_folder.txt", "w") as f:
        f.write(project_params.resolve_path(current_path))

    t = timer.Timer(text="Loading results took: {}", logger=logger.info)
    t.start()
    # retrieve available results
    if "data_set" in parameters:
        dataset = parameters["data_set"]
    elif "stream_session_name" in parameters:
        dataset = parameters["stream_session_name"]
    else:
        raise Exception("Unknown dataset or session name")
    load_config_files(dataset, current_path, working_path)
    if not skip:
        load_tomo_results(name, parameters, current_path, working_path, verbose=parameters["slurm_verbose"])

        if parameters["tomo_vir_method"] != "none" and os.path.exists("virion_thresholds.next") and os.stat("virion_thresholds.next").st_size > 0:
            # virion exlusion input from website
            seg_thresh = np.loadtxt("virion_thresholds.next", dtype=str, ndmin=2)
            TS_seg = seg_thresh[seg_thresh[:, 0] == name]
            if np.all(TS_seg[:, -1].astype(int) == 1):
                # defaut segmentation threshold
                updating_virion = False
            else:
                logger.info("Virion segmentation thresholds changed, will re-pick particles")
                updating_virion = True
                logger.info("Removing virion spike txt files")
                [ os.remove(f) for f in glob.glob( os.path.join(current_path, "sva", "*_cut.txt") ) ]
                [ os.remove(f) for f in glob.glob( os.path.join(working_path, "*_cut.txt") ) ]

        else:
            updating_virion = False

        # unpack pkl file
        if os.path.exists(f"{name}.pkl"):

            metadata_object = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=False)

            # clean entries based on _force in parameters
            metadata_object.refresh_entries(parameters, update_virion=updating_virion)

            metadata = metadata_object.data

            if "tomo_rec_force" in parameters and parameters["tomo_rec_force"]:
                logger.info(
                    f"Tomogram will be recomputed"
                )

            # convert metadata to files
            metadata_object.meta2PYP(path=working_path,data_path=os.path.join(current_path,"raw/"))

        # convert nextpyp coordinates to imod model
        if os.path.exists(f"{name}_exclude_views.next"):
            # convert next file to imod model
            com = f"{get_imod_path()}/bin/point2model {name}_exclude_views.next {name}_exclude_views.mod -scat"
            local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])
    else:
        logger.info("Ignoring existing results")

    t.stop()
    # Read tilt-series (and imaging parameters)

    # if we already have the combined tilt-series, use that instead of the raw data
    if not os.path.exists(name+".mrc"):
        [
            x,
            y,
            z,
            scope_pixel,
            scope_voltage,
            scope_mag,
            tilt_axis,
            tilt_metadata,
        ] = preprocess.read_tilt_series(
            str(current_path / filename),
            parameters,
            metadata,
            current_path=current_path,
            working_path=working_path,
            project_path=project_path,
        )
    else:
        parameters['movie_no_frames'] = True
        shutil.copy( name + ".tlt", name + ".rawtlt")
        [
            x,
            y,
            z,
            scope_pixel,
            scope_voltage,
            scope_mag,
            tilt_axis,
            tilt_metadata,
        ] = preprocess.read_tilt_series(
            name,
            parameters,
            metadata,
            current_path=current_path,
            working_path=working_path,
            project_path=project_path,
        )

    # remove x-rays
    if 'movie_no_frames' in parameters and parameters['movie_no_frames'] and "gain_remove_hot_pixels" in parameters and parameters["gain_remove_hot_pixels"]:
        t = timer.Timer(text="Removing hot pixels took: {}", logger=logger.info)
        t.start()
        preprocess.remove_xrays_from_file(name)
        t.stop()
    else:
        os.symlink(name + ".mrc", name + ".st")

    parameters = ctf_utils.update_pyp_scope_params(
        parameters, scope_pixel, scope_voltage, scope_mag, save=False
    )

    # actual stack sizes
    originalx = x
    originaly = y
    originalz = z
    headers = mrc.readHeaderFromFile(name + ".mrc")
    x = int(headers["nx"])
    y = int(headers["ny"])
    z = int(headers["nz"])

    # binned reconstruction
    binning = parameters["tomo_rec_binning"]

    # TODO: write description
    if os.path.exists("IMOD/%s.zfac" % name):
        zfact = "-ZFACTORFILE IMOD/%s.zfac" % name
    else:
        zfact = ""

    mpi_funcs, mpi_args = [ ], [ ]

    # tilt-series alignment
    if project_params.tiltseries_align_is_done(metadata):
        logger.info("Using existing tilt-series alignments")
    else:
        mpi_funcs.append(align.align_tilt_series)
        mpi_args.append( [(name, parameters, tilt_axis)] )

        t = timer.Timer(text="Tilt-series alignment + convert to tif took: {}", logger=logger.info)
        t.start()
        if len(mpi_funcs) > 0:
            mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=parameters["slurm_verbose"])
        t.stop()

    # generate full-size aligned tiltseries only if we do not yet have binned tomogram OR 
    # we need .ali for either sub-volume or virion extraction
    tilt_metadata["tilt_axis_angle"] = get_tilt_axis_angle(name, parameters)
    if not merge.tomo_is_done(name, os.path.join(project_path, "mrc")) or \
        ( parameters["tomo_vir_method"] != "none" and parameters["detect_force"] ) or \
        parameters["tomo_vir_force"] or \
        parameters["tomo_rec_force"] or \
        tomo_subvolume_extract_is_required(parameters) or \
        detect.tomo_vir_is_required(parameters) or \
        not ctf_mod.is_done(metadata, parameters, name=name, project_dir=current_path):

        t = timer.Timer(text="Apply alignment to tiltseries took: {}", logger=logger.info)
        t.start()
        generate_aligned_tiltseries(name, parameters, tilt_metadata)
        t.stop()

    # Refined tilt angles
    tltfile = f"{name}.tlt"
    tilt_angles = np.loadtxt(tltfile) if os.path.exists(tltfile) else metadata["tlt"].to_numpy()

    exclude_virions = merge.do_exclude_virions(name)
    if len(exclude_virions) > 0:
        # redo spike picking
        [ os.remove(f) for f in glob.glob( os.path.join(current_path, name + "_vir????_cut.txt") ) ]
        logger.info("Excluded virions list from mod file:")
        logger.info(exclude_virions)
    exclude_views = merge.do_exclude_views(name, tilt_angles)

    # Reconstruction options # -RADIAL 0.125,0.15, -RADIAL 0.25,0.15 (autoem2), 0.35,0.05 (less stringent)
    tilt_options = "-MODE 2 -OFFSET 0.00 -PERPENDICULAR -RADIAL {0},{1} -SCALE 0.0,0.002 -SUBSETSTART 0,0 -XAXISTILT 0.0 -FlatFilterFraction 0.0 {2}".format(
        parameters["tomo_rec_lpradial_cutoff"], parameters["tomo_rec_lpradial_falloff"], exclude_views
    )

    # Run the following steps in parallel
    # 1. per-tilt CTF estimation
    # 2. tomogram reconstruction
    mpi_funcs, mpi_args = [ ], [ ]

    # produce binned tomograms
    need_recalculation = parameters["tomo_rec_force"]
    if not merge.tomo_is_done(name, os.path.join(project_path, "mrc")) or need_recalculation:
        mpi_funcs.append(merge.reconstruct_tomo)
        mpi_args.append( [(parameters, name, x, y, binning, zfact, tilt_options)] )

    ctffind_tilt = False
    if ctf_mod.is_required_3d(parameters):
        if ctf_mod.is_done(metadata,parameters, name=name, project_dir=current_path):
            logger.info("Using existing CTF estimation")
            ctf = metadata["global_ctf"].to_numpy()
        else:
            # per-tilt CTF estimation
            ctf, ctffind_tilt_args = ctf_mod.ctffind_tomo_estimate(name, parameters)
            if len(ctffind_tilt_args) > 0:
                ctffind_tilt = True
                mpi_funcs.append(ctf_mod.ctffind_tilt_multiprocessing)
                mpi_args.append(ctffind_tilt_args)

            # write global ctf file
            ctf[6] = originalx
            ctf[7] = originaly
            ctf[8] = parameters['tomo_rec_thickness'] + parameters['tomo_rec_thickness'] % 2
            ctf[11] = parameters['tomo_rec_binning']
            np.savetxt("{}.ctf".format(name), ctf)

    if len(mpi_funcs) > 0:
        t = timer.Timer(text="Tomogram reconstruction + ctffind tilt took: {}", logger=logger.info)
        t.start()
        mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=parameters["slurm_verbose"], silent=True)
        t.stop()

    if ctffind_tilt:
        ctf_mod.detect_handedness_tilt_range(name=name,
                                             tilt_angles=tilt_angles, 
                                             lower_tilt=parameters["ctf_handedness_mintilt"], 
                                             upper_tilt=parameters["ctf_handedness_maxtilt"],)


    # package CTF metadata into dictionary
    ctf_profiles = {}
    ctf_values = {}
    for index in range(len(tilt_angles)):
        profile_file = "%s_%04d_ctffind4_avrot.txt" % ( name, index )
        if os.path.exists(profile_file):
            ctf_profiles[index] = np.loadtxt( profile_file, comments="#")
        values_file = "%s_%04d.txt" % ( name, index )
        if os.path.exists(values_file):
            ctf_values[index] = np.loadtxt( values_file, comments="#")
    tilt_metadata["ctf_values"] = ctf_values
    tilt_metadata["ctf_profiles"] = ctf_profiles

    # erase fiducials if needed
    if parameters["tomo_ali_method"] == "imod_gold" and parameters["tomo_rec_erase_fiducials"] and ( not os.path.exists(name+"_rec.webp") or parameters["tomo_rec_force"] ):

        # create binned aligned stack
        if not os.path.exists(f'{name}_bin.ali'):
            command = "{0}/bin/newstack -input {1}.ali -output {1}_bin.ali -mode 2 -origin -linear -bin {2}".format(
                get_imod_path(), name, binning
            )
            local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])

        detect.detect_gold_beads(parameters, name, x, y, binning, zfact, tilt_options)

        # save projected gold coordinates as txt file
        com = f"{get_imod_path()}/bin/model2point {name}_gold.mod {name}_gold_ccderaser.txt"
        local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])

        # calculate unbinned tilt-series coordinates
        gold_coordinates = np.loadtxt(name + "_gold_ccderaser.txt",ndmin=2)
        gold_coordinates[:,:2] *= binning
        np.savetxt(name + "_gold_ccderaser.txt",gold_coordinates)

        # convert back to imod model using one point per contour
        com = f"{get_imod_path()}/bin/point2model {name}_gold_ccderaser.txt {name}_gold_ccderaser.mod -scat -number 1"
        local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])

        # erase gold on (unbinned) aligned tilt-series
        erase_factor = parameters["tomo_rec_erase_factor"]
        com = f"{get_imod_path()}/bin/ccderaser -input {name}.ali -output {name}.ali -model {name}_gold_ccderaser.mod -expand 5 -order 0 -merge -exclude -circle 1 -better {parameters['tomo_ali_fiducial'] * erase_factor / parameters['scope_pixel']} -verbose"
        local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])

        try:
            os.remove(name + "_gold_ccderaser.txt")
            os.remove(name + "_gold_ccderaser.mod")
        except:
            pass

        # re-calculate reconstruction using gold-erased tilt-series
        merge.reconstruct_tomo(parameters, name, x, y, binning, zfact, tilt_options, force=True)

    # link binned tomogram to local scratch in case we need it for particle picking
    if not os.path.exists(f"{name}.rec"):
        symlink_relative(os.path.join(project_path, "mrc", f"{name}.rec"), f"{name}.rec")

    t = timer.Timer(text="Virion and spike detection took: {}", logger=logger.info)
    t.start()
    # remove environment LD_LIBRARY_PATH conflicts
    
    # particle detection and extraction
    virion_coordinates, spike_coordinates = detect_tomo.detect_and_extract_particles( 
        name,
        parameters,
        current_path,
        binning,
        x, y,
        zfact,
        tilt_angles,
        tilt_options,
        exclude_virions )
    t.stop()

    if (
        "tomo_vir_force" in parameters and parameters["tomo_vir_force"]
        or metadata is None or metadata is not None and "vir" not in metadata or parameters["data_import"]
    ):
        tilt_metadata["virion_coordinates"] = virion_coordinates

    tilt_metadata["spike_coordinates"] = spike_coordinates

    mpi_funcs, mpi_args = [ ], [ ]
    if ctffind_tilt:
        mpi_funcs.append(ctf_mod.plot_ctffind_tilt)
        mpi_args.append( [(name,parameters,ctf)] )

    if not os.path.exists(f"{name}.webp"):
        mpi_funcs.append(plot.plot_tomo_ctf)
        mpi_args.append( [(name,parameters["slurm_verbose"])] )

    if not os.path.exists(f"{name}_rec.webp") or parameters["tomo_rec_force"]:
        mpi_funcs.append(plot.tomo_slicer_gif)
        mpi_args.append( [(f"{name}.rec", f"{name}_rec.webp", True, 2, parameters["slurm_verbose"])] )

    if os.path.exists(f"{name}_bin.mrc") and not os.path.exists(name + "_raw.webp"):
        mpi_funcs.append(plot.tomo_montage)
        mpi_args.append( [(name + '_bin.mrc', name + "_raw.webp")] )

    if os.path.exists(f"{name}_bin.ali"):
        mpi_funcs.append(plot.tomo_montage)
        mpi_args.append( [(name + '_bin.ali', name + "_ali.webp")] )

    if len(mpi_funcs):
        t = timer.Timer(text="Ploting ctf and tomo webp's took: {}", logger=logger.info)
        t.start()
        mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=parameters["slurm_verbose"], silent=True)
        t.stop()

    # convert to jpg to fool nextPYP
    if os.path.exists(name + ".webp"):
        local_run.run_shell_command(f"convert {name}.webp {name}.jpg",verbose=False)

    t = timer.Timer(text="Save result and clean scratch took: {}", logger=logger.info)
    t.start()

    # save results
    if not debug:
        # replace squared tiltseries with raw one (after frame alignment)
        if os.path.exists(f"{name}.raw.mrc"):
            os.remove(f"{name}.mrc")
            os.rename(f"{name}.raw.mrc", f"{name}.mrc") 

        with open( f"{name}.pickle", 'wb') as f:
            pickle.dump(tilt_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # pack all metadata and save into one pickle file
        data = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=False)
        data.loadFiles()

        if Web.exists:
            save_tomo_results_lean(name, parameters, current_path, verbose=parameters["slurm_verbose"])
        else:
            save_tomo_results(name, parameters, current_path, verbose=parameters["slurm_verbose"])

        save_tiltseries_to_website(name, tilt_metadata, 'slurm_verbose' in parameters and parameters['slurm_verbose'])

    # clean-up
    if not keep:
        shutil.rmtree(working_path)
    t.stop()

# this is a legacy method that may be needed for the moving average used by frame refinement
def write_current_particle_multiprocessing(
    parameters,
    stackfile,
    film,
    particle,
    sections,
    indexes,
    num_particles,
    actual_pixel,
):

    # write stack for current particle
    particle_stack = write_current_particle(
        parameters, stackfile, film, particle, sections
    )

    # compute running frame averages
    window_averaging = float(parameters["extract_wgh"].lower())
    if window_averaging > 0:
        frames = len(indexes)
        all_weights = compute_running_avg(
            particle, num_particles, frames, window_averaging
        )

        particle_frames = mrc.read(particle_stack)
        mrc.write(
            merge.weight_stack_array(particle_frames, all_weights), particle_stack,
        )
        replace_sections(particle_stack, stackfile, sections)

    # frame normalization
    if len(indexes) > 0:
        particle_frames = np.array(mrc.read(particle_stack), ndmin=3)
        normalize_frames(parameters, actual_pixel, indexes, particle_frames)
    # print 'after = ', particles[indexes,:,:].mean(0).mean(), particles[indexes,:,:].mean(0).std()

    # write out MRC file
    mrc.write(particle_frames, particle_stack)
    replace_sections(particle_stack, stackfile, sections)


def csp_split(parameters, iteration):
    """Split function for both SP SPR and SP TOMO.

    Splits a large cryo-EM project with multiple movies into smaller chunks that are run
    using the swarm functions.

    Parameters
    ----------
    parameters : dict
        Main configurations taken from .pyp_config
    """

    # read list of micrographs
    with open("{}.films".format(parameters["data_set"])) as f:
        files = [
            line.strip() for line in f
        ]
    workdir = os.getcwd()

    # cleanup any previous runs
    if not parameters["slurm_merge_only"]:
        try:
            shutil.rmtree(os.path.join("frealign", "scratch"))
        except:
            pass

    # clean-up csp logs and webp files
    path_to_logs = Path(os.getcwd(), "log")
    [os.remove(path_to_logs / f) for f in os.listdir(path_to_logs) if f.endswith("_csp.log")]
    path_to_webps = Path(os.getcwd(), "frealign")
    [os.remove(path_to_webps / f) for f in os.listdir(path_to_webps) if f.endswith("_weights_local.webp")]

    classes = int(project_params.param(parameters["class_num"], iteration))
    dataset = parameters["data_set"]
    use_frames = "local" in parameters["extract_fmt"].lower()
    # collect FREALIGN statistics
    for ref in range(classes):
        name = "%s_r%02d" % (dataset, ref + 1)
        compressed_par = Path(os.getcwd(), "frealign", "maps", f"{name}_{iteration-1:02d}.par.bz2")

        parfile = None
        if compressed_par.exists():
            decompressed_par = frealign_parfile.Parameters.decompress_parameter_file(str(compressed_par), parameters["slurm_tasks"])
            parfile = decompressed_par
            if classes > 1:
                shutil.copy2(parfile, Path(compressed_par.parent, os.path.basename(parfile)))

        elif "refine_parfile" in parameters and parameters["refine_parfile"] != None and os.path.exists(project_params.resolve_path(parameters["refine_parfile"])) and ".par" in project_params.resolve_path(parameters["refine_parfile"]):
            refine_parfile = frealign_parfile.Parameters.decompress_parameter_file(project_params.resolve_path(parameters["refine_parfile"]), parameters["slurm_tasks"])
            parfile = refine_parfile

        if use_frames and ref == 0 or iteration == 2:
            try:
                os.unlink("./csp/micrograph_particle.index")
            except:
                pass
        if not os.path.isfile("./csp/micrograph_particle.index") and ( (iteration > 2 and ref == 0) or (iteration == 2 and parfile != None)):
            get_image_particle_index(parameters, parfile, path="./csp")


        if parameters["dose_weighting_enable"] and ref == 0:

            # create weights folder for storing weights.txt
            weights_folder = Path.cwd() / "frealign" / "weights"
            if not weights_folder.exists():
                os.mkdir(weights_folder)

            # compute global weights using enitre parfile
            if parameters["dose_weighting_global"] and parfile != None:

                global_weight_file = str(weights_folder / "global_weight.txt")
                compute_global_weights(parfile=parfile, weights_file=global_weight_file)

                parameters["dose_weighting_weights"] = global_weight_file
                project_params.save_pyp_parameters(parameters=parameters, path=".")

            elif "dose_weighting_weights" in parameters and parameters["dose_weighting_weights"] is not None and project_params.resolve_path(parameters["dose_weighting_weights"]) == "auto":
                weight_file = project_params.get_weight_from_projects(weight_folder=weights_folder, parameters=parameters)
                if weight_file is not None:
                    parameters["dose_weighting_weights"] = weight_file
                    project_params.save_pyp_parameters(parameters=parameters, path=".")

        if (classes > 1
        and iteration > 2
        and not os.path.isfile("./csp/particle_tilt.index")
        and "tomo" in parameters["data_mode"]
        and ref == 0
        ):
            get_particles_tilt_index(parfile, path="./csp")

        previous = "frealign/maps/%s_%02d.par" % (name, iteration - 1)
        current = "%s_%02d" % (name, iteration)
        raw_stats_file = f"frealign/maps/{name}_{(iteration-1):02d}_statistics.txt_raw"
        smooth_stats_file = f"frealign/maps/{name}_{(iteration-1):02d}_statistics.txt"

        # smooth part FSC curves
        if project_params.param(parameters["refine_metric"], iteration) == "new" and parameters["refine_fssnr"]:

            plot_name = "frealign/maps/" + current + "_snr.png"

            if not os.path.exists(raw_stats_file) and os.path.exists(smooth_stats_file):
                postprocess.smooth_part_fsc(smooth_stats_file, plot_name)
            elif os.path.exists(raw_stats_file):
                postprocess.smooth_part_fsc(raw_stats_file, plot_name)

    if classes > 1 and iteration > 2:

        os.chdir("frealign")

        force_init = parameters["class_force_init"]

        # initialize classes if needed
        if not os.path.exists(
            os.path.join(
                "maps", "%s_r%02d_%02d.mrc" % (dataset, classes, iteration - 1),
            )
            or force_init
        ):
            use_frame = "local" in parameters["extract_fmt"]
            is_tomo = "tomo" in parameters["data_mode"]
            classification_initialization(
            parameters, dataset, classes, iteration, use_frame = use_frame, is_tomo = is_tomo, references_only=False
        )
            # skip occupancy calculation twice
            parameters["refine_skip"] = True
            parameters["class_force_init"] = False
            project_params.save_pyp_parameters(parameters=parameters, path="..")
        else:

            occupancy_extended(
                parameters, dataset, iteration - 1, classes, path="maps", is_frealignx=False, local=False
            )

        os.chdir("..")

    # get the statistics for refine3d and reconstruct3d
    if classes > 1 and iteration > 2:
        for ref in range(classes):
            name = "%s_r%02d" % (dataset, ref + 1)
            # previous = "frealign/maps/%s_%02d.par" % (name, iteration - 1)
            previous = os.path.join("frealign", "maps" , "%s_%02d.par" % (name, iteration - 1))
            par_stat = "frealign/maps/parfile_constrain_r%02d.txt" % (ref + 1)
            get_statistics_from_par(previous, par_stat)

    os.makedirs("swarm", exist_ok=True)
    os.chdir("swarm")

    slurm.launch_csp(micrograph_list=files,
                    parameters=parameters,
                    swarm_folder=Path().cwd(),
                    )

    os.chdir(workdir)


@timer.Timer(
    "csp_extract_frames", text="Particle extraction took: {}", logger=logger.info
)
def csp_extract_frames(
    allboxes,
    allparxs,
    parameters,
    filename,
    imagefile,
    parxfile,
    stackfile,
    working_path,
    current_path,
):

    totalboxes = len(allboxes)
    iteration = parameters["refine_iter"]
    metric = project_params.param(parameters["refine_metric"], iteration)
    if totalboxes > 0:
         # write .parx file for each class
        if type(allparxs[0]) == np.ndarray:
            par_col = allparxs[0].shape[1]
            if par_col > 15:
                if par_col < 45:
                    metricfmt = "new"
                    format = frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO
                else:
                    metricfmt = "frealignx"
                    format = frealign_parfile.EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE_WO_NO
            else:
                format = frealign_parfile.NEW_PAR_STRING_TEMPLATE_WO_NO
            with timer.Timer(
                "write_allparx", text = "Writing parx file from allparxs took: {}", logger=logger.info
            ):
                for current_class in range(len(allparxs)):
                    parfilename = parxfile.replace("_r01", "_r%02d" % (current_class + 1))
                    np.savetxt(parfilename.replace(".parx", ""), allparxs[current_class], fmt=format)
        else:
            for current_class in range(len(allparxs)):
                parfilename = parxfile.replace("_r01", "_r%02d" % (current_class + 1))
                with open(parfilename.replace(".parx", ""), "w") as f:
                    f.writelines("%s\n" % item for item in allparxs[current_class])

        if not parameters["csp_parx_only"]:

            if os.path.exists(parameters["data_set"] + ".films"):

                film = 0

                os.chdir(working_path)

                actual_pixel = float(parameters["scope_pixel"]) * float(
                    parameters["data_bin"]
                )

                use_frames = "local" in parameters["extract_fmt"].lower()
                is_tomo = "tomo" in parameters["data_mode"].lower()

                if os.path.exists(f"{filename}.mrc"):
                    os.rename(f"{filename}.mrc", f"{filename}.avg")

                if is_tomo and use_frames:

                    # copy over all the frames
                    arguments = []
                    # [shutil.copy2(current_path + "/raw/" + f, f) for f in imagefile]
                    for f in imagefile:
                        arguments.append((current_path + "/raw/" + f, f))
                    mpi.submit_function_to_workers(shutil.copy2, arguments, verbose=parameters["slurm_verbose"])
                    raw_image = imagefile

                else:
                    if os.path.exists(os.path.join(current_path, imagefile + ".mrc")):
                        raw_image = filename + ".mrc"
                    elif os.path.exists(os.path.join(current_path, imagefile + ".tif")):
                        raw_image = filename + ".tif"
                    elif os.path.exists(os.path.join(current_path, imagefile + ".tiff")):
                        raw_image = filename + ".tiff"
                    else:
                        logger.error("Cannot figure out image filename")

                    # copy over all the frames
                    shutil.copy2(
                        os.path.join(current_path, imagefile + Path(raw_image).suffix),
                        raw_image,
                    )

                    # pack it into list to adapt it into tomo extraction pipeline
                    if use_frames:
                        raw_image = [raw_image]

                # convert tif to mrc so csp can read the tilt-series
                if not use_frames and (Path(raw_image).suffix == ".tif" or Path(raw_image).suffix == ".tiff"):

                    t = timer.Timer(text="Convert tif to mrc took: {}", logger=logger.info)
                    t.start()

                    command = "{0}/bin/tif2mrc {1}.tif {1}.mrc; rm {1}.tif".format(
                        get_imod_path(), filename
                    )
                    local_run.run_shell_command(command)
                    raw_image = f"{filename}.mrc"

                    t.stop()
                """
                elif use_frames:
                    dims = read(current_path + "/" + imagefile, parameters)
                else:
                    if os.path.exists( os.path.join( current_path, "mrc", filename + ".tif" ) ):
                        command = "{0}/bin/tif2mrc {1}.tif {1}.mrc"
                    else:
                        shutil.copy2(
                            os.path.join(current_path, "mrc", raw_image), raw_image,
                        )
                """

                # remove hot pixels
                if False:
                    preprocess.remove_xrays_from_movie_file(filename, True)

                # extract particles
                particles = extract.extract_particles(
                    raw_image,
                    stackfile,
                    allboxes,
                    parameters["particle_rad"] * float(parameters["data_bin"]),
                    parameters["extract_box"],
                    parameters["extract_bin"],
                    actual_pixel,
                    parameters["slurm_tasks"],
                    parameters,
                    normalize=True,
                    fixemptyframes=True,
                    method=parameters["extract_method"],
                    is_tomo=is_tomo,
                    use_frames=use_frames,
                )

                # remove original image to save space
                """
                if use_frames:
                    [
                        os.remove(f)
                        for frame in raw_image
                        for f in glob.glob(frame + "*")
                    ]
                    [
                        os.remove(f)
                        for frame in raw_image
                        for f in glob.glob("frealign/" + frame + "*")
                    ]
                else:
                    [os.remove(f) for f in glob.glob(raw_image + "*")]
                    [os.remove(f) for f in glob.glob("frealign/" + raw_image + "*")]
                """
                if True:  # "tomo" in parameters["data_mode"]:
                    actual_number_of_particles = particles
                else:
                    actual_number_of_particles = mrc.readHeaderFromFile(stackfile)["nz"]

                # check if all particles extracted correctly
                if totalboxes != actual_number_of_particles:
                    logger.error(
                        "Only {0} particles extracted from requested {1}".format(
                            actual_number_of_particles, totalboxes
                        )
                    )
                else:
                    logger.info(
                        f"Total number of particle frames extracted: {totalboxes:,}"
                    )

                # fix empty particles
                # logger.info("Detecting empty particles")
                # temp_stack = filename + "_temp_stack.mrc"
                # fix_empty_particles(stackfile, actual_number_of_particles, temp_stack)

                imod_path = get_imod_path()

                # create individual per-particle, per-micrograph stacks
                if (
                    False
                    and "frealign" in parameters["extract_fmt"].lower()
                    and "tomo" in parameters["data_mode"].lower()
                ):

                    root_stack = os.path.join(
                        working_path,
                        "frealign",
                        "data",
                        parameters["data_set"] + "_frames_T%04d" % (film),
                    )

                    # convert metadata to numpy array
                    allparxs_array = np.genfromtxt(allparxs[0])

                    # get the particle indexes
                    local_particle = np.unique(allparxs_array[:, 15].astype("int"))

                    num_particles = len(local_particle)

                    logger.info("Found {} particles.".format(num_particles))

                    command_list = []

                    # template to use for naming the per-particle stacks
                    root_particle_stack = root_stack + "_P??????_stack.mrc"

                    for particle in local_particle:

                        # find all lines corresponding to current particle
                        indexes = np.argwhere(allparxs_array[:, 15] == particle)

                        # name of output stack
                        particle_stack = root_particle_stack.replace(
                            "??????", "%06d" % particle
                        )

                        # format slice numbers for extraction with newstack
                        if len(indexes) == 1:

                            sections = str(indexes.squeeze())

                            # newstack command
                            command = "{0}/bin/newstack {1} {2} -secs {3}".format(
                                get_imod_path(), stackfile, particle_stack, sections
                            )

                            command_list.append(command)

                        else:
                            # work around newstack's -secs limitation
                            chunk_size = 50
                            list = indexes.astype("str").squeeze().tolist()
                            chunks = [
                                list[i : i + chunk_size]
                                for i in range(0, len(list), chunk_size)
                            ]
                            split_sections = ""
                            for chunk in chunks:
                                split_sections += " -secs " + ",".join(chunk)

                            # newstack command
                            command = "{0}/bin/newstack {1} {2} {3}".format(
                                get_imod_path(),
                                stackfile,
                                split_sections,
                                particle_stack,
                            )

                            command_list.append(command)

                    local_frames = np.unique(allparxs_array[:, 18].astype("int"))

                    logger.info("Found {} micrographs.".format(len(local_frames)))

                    # template to use for naming the per-micrograph stacks
                    root_micrograph_stack = root_stack + "_M??????_stack.mrc"

                    for frame in local_frames:

                        # find all lines corresponding to current micrograph
                        indexes = np.argwhere(allparxs_array[:, 18] == frame)

                        # name of output stack
                        name = root_micrograph_stack.replace("??????", "%06d" % frame)

                        # work around newstack's -secs limitation
                        chunk_size = 50
                        list = indexes.astype("str").squeeze().tolist()
                        chunks = [
                            list[i : i + chunk_size]
                            for i in range(0, len(list), chunk_size)
                        ]
                        split_sections = ""
                        for chunk in chunks:
                            split_sections += " -secs " + ",".join(chunk)

                        # newstack command
                        command = "{0}/bin/newstack {1} {2} {3}".format(
                            get_imod_path(), stackfile, split_sections, name
                        )

                        command_list.append(command)

                    cpus = int(parameters["slurm_tasks"])

                    logger.info(
                        "Writing {} per-particle, per-micrograph stacks using {} cores".format(
                            len(command_list), cpus - 1
                        )
                    )

                    # run multirun on list of commands
                    mpi.submit_jobs_to_workers(command_list, os.getcwd())

                elif "relion" in parameters["extract_fmt"].lower():
                    # write one stack per micrograph
                    mrc.write(
                        -particles,
                        current_path + "/" + "relion/" + filename + "_stack.mrcs",
                    )

                os.chdir(current_path)

                if not parameters["csp_stacks"]:
                    # clear up space
                    # shutil.rmtree(working_path)

                    # only remove unnecesary files
                    [os.remove(f) for f in glob.glob(filename + ".*")]
                else:
                    [
                        shutil.copy2(f, os.path.join(current_path, "frealign", "data"))
                        for f in glob.glob(
                            os.path.join(
                                working_path, "frealign", "data", "*_P??????_stack.mrc"
                            )
                        )
                    ]
            else:
                logger.info("{}.films does not exist".format(parameters["data_set"]))

            return actual_number_of_particles


@timer.Timer(
    "csp_swarm", text="Total time elapsed (csp_swarm): {}", logger=logger.info
)
def csp_swarm(filename, parameters, iteration, skip, debug):
    """Workhorse function for CSP frames and metadata extraction.

    Parameters
    ----------
    filename : str, Path
        Movie filename
    parameters : dict
        Main configurations taken from .pyp_config
    """

    # store current project directory
    current_path = os.getcwd()

    use_frames = "local" in parameters["extract_fmt"].lower()
    is_tomo = "tomo" in parameters["data_mode"].lower()

    dataset = parameters["data_set"]

    # remove _local.allboxes automatically
    local_allboxes = Path(current_path, "csp", f"{filename}_local.allboxes")
    if iteration == 2 and use_frames and local_allboxes.exists() and not local_allboxes.is_symlink():
        os.remove(local_allboxes)

    # setup frealign enviroment in local scratch
    working_path = os.path.join(os.environ["PYP_SCRATCH"], filename)
    shutil.rmtree(working_path, ignore_errors=True)

    local_frealign_folder = os.path.join(working_path, "frealign")
    os.makedirs(local_frealign_folder)
    os.chdir(local_frealign_folder)
    prepare_frealign_dir()

    shutil.copy2(os.path.join(current_path, ".pyp_config.toml"), working_path)
    if iteration > 2 and parameters["refine_fssnr"]:
        for ref in range(parameters["class_num"]):
            try:
                shutil.copy2(
                    os.path.join(
                        current_path,
                        "frealign",
                        "maps",
                        "statistics_r%02d.txt" % (ref + 1),
                    ),
                    os.path.join(local_frealign_folder, "scratch"),
                )
            except:
                logger.warning(
                    "Could not find frealign statistics file in maps folder"
                )
                pass

    os.chdir(current_path)

    t = timer.Timer(text="Retrieve metadata took: {}", logger=logger.info)
    t.start()
    if not skip:
        load_csp_results(filename, parameters, Path(current_path), Path(working_path), verbose=parameters["slurm_verbose"])

    metafile = os.path.join(current_path, "pkl", filename + ".pkl")
    if os.path.exists(metafile):
        try:
            shutil.copy2(metafile, working_path)
            metafile = os.path.join(working_path, filename + ".pkl")

            # TODO: just a test, delete later
            if "spr" in parameters["data_mode"]:
                is_spr = True
            else:
                is_spr = False
            metadata = pyp_metadata.LocalMetadata(metafile, is_spr=is_spr)

            metadata.meta2PYP(path=working_path)
            """
            # only pickle file, need boxx file in ./box first time to run csp
            boxxfile = os.path.join(current_path, "box", filename + ".boxx")
            boxfrompkl = os.path.join(working_path, filename + ".boxx")
            if not os.path.exists(boxxfile) and os.path.isfile(boxfrompkl):
                shutil.copy2(boxfrompkl, boxxfile)

            # ctf file from pickle
            ctffile = os.path.join(current_path, "ctf", filename + ".ctf")
            ctffrompkl = os.path.join(working_path, filename + ".ctf")
            if not os.path.exists(ctffile) and os.path.isfile(ctffrompkl):
                shutil.copy2(ctffrompkl, ctffile)

            # xf file from pickle
            xffile = os.path.join(current_path, "ali", filename + ".xf")
            xffrompkl = os.path.join(working_path, filename + ".xf")
            if not os.path.exists(xffile) and os.path.isfile(xffrompkl):
                shutil.copy2(xffrompkl, xffile)
            """
        except:
            logger.warning("Unable to retrieve metadata")
            trackback()
            pass

    statistic_file = glob.glob(
            current_path + "/frealign/" + "maps/" + f"{dataset}_r01_{(iteration-1):02d}_statistics.txt"
    )
    for statics in statistic_file:
        try:
            shutil.copy2(
                statics, os.path.join(local_frealign_folder, "scratch"),
            )
            logger.info("Copying frealign statistics to local scratch")
        except:
            logger.warning(
                "Cannot find frealign statistics file in scratch folder, skipping"
            )
            pass

    # extract/retrieve particle coordinates
    [allboxes, allparxs] = csp_extract_coordinates(
        filename,
        parameters,
        working_path,
        current_path,
        skip,
        only_inside=False,
        use_frames=use_frames,
        use_existing_frame_alignments=True,
    )

    if is_tomo:
        if use_frames:

            # TODO: add movie filenames into pkl
            # this solution won't work if not using movie_pattern during preprocessing (i.e. using mdoc)

            os.chdir(working_path)
            # compile movie_pattern used during preprocessing into RegEx
            pattern = parameters["movie_pattern"]
            regex = movie2regex(pattern, filename)
            r = re.compile(regex)

            # look for the position of tilt angle in the filename
            labels = ["TILTSERIES", "SCANORD", "ANGLE"]
            labels = [l for l in labels if pattern.find(l) >= 0]
            labels.sort(key=lambda x: int(pattern.find(x)))
            pos_tiltangle = labels.index("ANGLE") + 1

            # search files in project directory and sort the list based on tilt angle
            detected_movies = [
                    [r.match(f).group(0), float(r.match(f).group(pos_tiltangle))] 
                    for f in os.listdir(os.path.join(current_path, "raw")) 
                    if r.match(f)
                    ]
            sorted_tilts = sorted(detected_movies, key=lambda x: x[1])
            imagefile = [_[0] for _ in sorted_tilts]

        elif os.path.exists(os.path.join("mrc", filename + ".mrc")):
            imagefile = "mrc/" + filename
            parameters["gain_reference"] = None
        elif os.path.exists(os.path.join("raw", filename + ".mrc")):
            imagefile = "raw/" + filename
        else:
            raise Exception("Image not found")
    elif use_frames:
        imagefile = "raw/" + filename

    else:
        imagefile = "mrc/" + filename
        parameters["gain_reference"] = None


    parxfile = os.path.join(
        working_path, "frealign", "maps", filename + "_r01_%02d.parx" % (iteration - 1)
    )
    stackfile = os.path.join(working_path, "frealign", filename + "_stack.mrc")

    os.chdir(current_path)

    # save copy of all boxes
    allboxes_saved = allboxes.copy()

    # extract paticle frames and write parameter and stack files:
    # 1) parxfile's: working_path/frealign/maps/filename_r??_??.parx
    # 2) stackfile: working_path/filename_stack.mrc
    actual_number_of_frames = csp_extract_frames(
        allboxes,
        allparxs,
        parameters,
        filename,
        imagefile,
        parxfile,
        stackfile,
        working_path,
        current_path,
    )

    # refinement
    # outputs: working_path/frealign/maps/filename_r??_??.parx
    align.csp_refinement(
        parameters,
        filename,
        current_path,
        working_path,
        use_frames,
        parxfile,
        iteration,
    )

    # save results
    if not debug:
        save_csp_results(
            filename, parameters, current_path, verbose=parameters["slurm_verbose"]
        )
        save_refinement_to_website(filename, iteration, 'slurm_verbose' in parameters and parameters['slurm_verbose'])

    # clean-up unnecesary files
    try:
        [
            os.remove(f)
            for f in glob.glob(
                os.path.join(local_frealign_folder, "data", "*_stack.mrc*")
            )
        ]
        [
            os.remove(f)
            for f in glob.glob(
                os.path.join(working_path, filename, "{}.mrc*".format(filename))
            )
        ]
    except:
        logger.error("Failed to clean up local scratch")
        pass


def csp_merge(parameters):
    """Merge function that uses the new stack-less paradigm.

    Parameters
    ----------
    parameters : dict
        PYP parameters loaded from .pyp_config
    """
    # csp_final_merge
    if parameters["csp_no_stacks"]:
        path = os.path.join(os.getcwd(), "frealign", "scratch")
        particle_cspt.run_merge(path)


def box_edit(skip, startat):
    parameters = project_params.load_pyp_parameters()

    micrographs = "{}.micrographs".format(parameters["data_set"])

    if not os.path.isfile(micrographs):
        logger.warning("Cannot find %s", micrographs)
        files = [
            s.replace("mrc/", "").replace(".mrc", "") for s in glob.glob("mrc/*.mrc")
        ]
    else:
        files = open(micrographs, "r").read().split()

    for f in files:

        if startat != "":
            if f == startat:
                logger.info(f"Starting at micrograph {startat}")
                startat = ""
            else:
                continue

        ydim = mrc.readHeaderFromFile("mrc/{0}.mrc".format(f))["ny"]

        if ydim > 6096:
            factor = 8.0
        else:
            factor = 4.0

        if os.path.exists("box/{0}.box".format(f)):

            # extract coordinates from box file
            com = 'cat box/{0}.box | awk \'{{ print $1" "{1}-$2" "$3}}\' > {2}'.format(
                f, ydim, Path(os.environ["PYP_SCRATCH"]) / f"{f}.box"
            )
            local_run.run_shell_command(com)

            # convert box file to imod model
            com = "{0}/bin/point2model {3}.box {3}.mod -scat -circle 20 -pixel {2},{2},{2}".format(
                get_imod_path(), f, 1 / factor, Path(os.environ["PYP_SCRATCH"]) / str(f)
            )
            local_run.run_shell_command(com)

            # save copy for detecting change
            shutil.copy2(
                Path(os.environ["PYP_SCRATCH"]) / f"{f}.mod",
                Path(os.environ["PYP_SCRATCH"]) / "dummy.mod",
            )
            os.remove(Path(os.environ["PYP_SCRATCH"]) / f"{f}.box")

        if os.path.exists("box/{0}.jpg".format(f)):
            # load raw binned image and imod model
            local_run.run_shell_command(
                "{0}/bin/3dmod box/{1}.jpg {2}.mod".format(
                    get_imod_path(), f, Path(os.environ["PYP_SCRATCH"]) / str(f)
                )
            )

            # only of model updated
            if not filecmp.cmp(
                Path(os.environ["PYP_SCRATCH"]) / f"{f}.mod",
                Path(os.environ["PYP_SCRATCH"]) / "dummy.mod",
            ):

                # convert box file to imod model
                com = "{0}/bin/model2point {2}.mod {2}.txt -float".format(
                    get_imod_path(), f, Path(os.environ["PYP_SCRATCH"]) / str(f)
                )
                local_run.run_shell_command(com)

                coords = np.loadtxt(Path(os.environ["PYP_SCRATCH"]) / f"{f}.txt")
                coords[:, 0] = factor * coords[:, 0]
                coords[:, 1] = factor * (ydim / factor - coords[:, 1])
                coords = np.hstack((coords, np.zeros((coords.shape[0], 1))))
                logger.info("%d boxes saved to boxer/%s.box", coords.shape[0], f)
                np.savetxt("box/{0}.box".format(f), coords, delimiter="\t", fmt="%.0f")

                os.remove(Path(os.environ["PYP_SCRATCH"]) / f"{f}.txt")
            else:
                logger.info("No changes made to box/{}.box".format(f))
            os.remove(Path(os.environ["PYP_SCRATCH"]) / f"{f}.mod")
            os.remove(Path(os.environ["PYP_SCRATCH"]) / "dummy.mod")
        else:
            logger.warning(f"box/{f}.jpg no such file.")


def tomo_edit(startat, raw, ali, rec, reg, seg, vir, spk, skip, clean):
    parameters = project_params.load_pyp_parameters()

    micrographs = "{}.micrographs".format(parameters["data_set"])

    if not os.path.isfile(micrographs):
        logger.warning("Cannot find %s", micrographs)
        files = [
            s.replace("raw/", "").replace(".tbz", "") for s in glob.glob("raw/*.tbz")
        ]
    else:
        files = open(micrographs, "r").read().split()

    keep_files = files[:]
    remove_files = []

    done = False

    # clean particles based on the parfile
    # clean[0, 1, 2] - clean, parx, dist
    if clean[0]:

        try:
            with open("{}.films".format(parameters["data_set"]), "r") as f:
                films = [film.strip() for film in f.readlines()]
        except:
            raise Exception(
                "{} does not exists".format("{}.films".format(parameters["data_set"]))
            )

        # if we're not discarding particles with 0 occ
        if not clean[3]:

            # check if re-computing pre-particle scores based on the parfile
            if clean[1] == "":
                logger.warning("Use existing scores in boxes3d for particle sorting")
            elif not os.path.exists(clean[1]):
                raise Exception(f"{clean[1]} does not exists")
            else:
                # parse metadata into numpy array
                par_data = frealign_parfile.Parameters.from_file(clean[1]).data
                # compute per-particle scores and store them into dictionary
                scores = score_particles_fromparx(par_data)

                # parse new scores into 3d boxes
                for film, tiltseries in enumerate(films):
                    box3dfile = "csp/{}_boxes3d.txt".format(tiltseries)
                    newbox3dfile = "csp/{}_boxes3d_clean.txt".format(tiltseries)
                    if not os.path.exists(box3dfile):
                        logger.warning("{} does not exist".format(box3dfile))
                    else:
                        newf = open(newbox3dfile, "w")
                        with open(box3dfile, "r") as f:
                            for line in f.readlines():
                                if line.strip().startswith("PTLIDX"):
                                    newf.write(line)
                                else:
                                    data = line.split()
                                    # PTLIDX, X, Y, Z, Score, Keep
                                    (
                                        data[0],
                                        data[1],
                                        data[2],
                                        data[3],
                                        data[4],
                                        data[5],
                                    ) = (
                                        int(data[0]),
                                        float(data[1]),
                                        float(data[2]),
                                        float(data[3]),
                                        float(data[4]),
                                        str(data[5]),
                                    )

                                    try:
                                        data[4] = scores[film][data[0]]
                                    except:
                                        data[4] = 0
                                        # logger.error(f"Parfile does not match {tiltseries} boxes3d.")

                                    newf.write(
                                        "%8d\t%8.1f\t%8.1f\t%8.1f\t%8.2f\t%8s\n"
                                        % tuple(data)
                                    )
                        newf.close()
                        shutil.copy2(newbox3dfile, box3dfile)
                        os.remove(newbox3dfile)

        # clean particles based on distance, scores ...etc, and update in boxes3d
        if clean[2] > 0:

            films_used_particles = [[film, 0] for film in films]
            film_count = 0
            particle_used_count = 0
            particle_all_count = 0

            for film, tiltseries in enumerate(films):
                box3dfile = "csp/{}_boxes3d.txt".format(tiltseries)
                newbox3dfile = "csp/{}_boxes3d_clean.txt".format(tiltseries)
                particle_used_film = 0

                if not os.path.exists(box3dfile):
                    logger.warning("{} does not exist".format(box3dfile))
                else:
                    newf = open(newbox3dfile, "w")
                    newf.write(
                        "%8s\t%8s\t%8s\t%8s\t%8s\t%8s\n"
                        % ("PTLIDX", "X", "Y", "Z", "Score", "Keep_CSP")
                    )
                    with open(box3dfile, "r") as f:
                        box3d = [line.split() for line in f.readlines()]
                        newbox3d = clean_particles_tomo(box3d, clean[2])

                        for line in newbox3d:
                            newf.write(
                                "%8d\t%8.1f\t%8.1f\t%8.1f\t%8.2f\t%8s\n" % tuple(line)
                            )
                            particle_all_count += 1
                            if "Yes" in line[5]:
                                particle_used_count += 1
                                particle_used_film += 1

                    newf.close()
                    shutil.copy2(newbox3dfile, box3dfile)
                    os.remove(newbox3dfile)

                films_used_particles[film][1] = particle_used_film
                film_count += 1

            logger.warning(
                "{} particles ({:.1f}%) from {} tilt-series will be left after filtering.".format(
                    particle_used_count,
                    (particle_used_count / particle_all_count * 100),
                    film_count,
                )
            )
            """
            # DO NOT update film file because it will mess up retrieving metadata from previous parfile
            # write a new .film file sorted by particle number (more to less) to make cspswarm more efficient
            films_used_particles = sorted(
                films_used_particles, key=lambda x: x[1], reverse=True
            )
            with open("{}.films".format(parameters["data_set"]), "w") as f:
                f.write("\n".join([film[0] for film in films_used_particles]))
            """

        # remove particles that have 0 occ from parx and corresponding lines in allboxes
        if os.path.exists(clean[1]) and clean[3]:

            FILM_COL = 8 - 1
            OCC_COL = 12 - 1
            ORI_TAG = "_original"
            parx_name, parx_format = os.path.splitext(clean[1])
            par = frealign_parfile.Parameters.from_file(clean[1])
            par_data = par.data

            par_data_clean = par_data[par_data[:, OCC_COL] != 0.0]
            allboxes_line_count = 0

            for idx, film in enumerate(films):
                # check which allboxes we should use: frame format (*local.allboxes) is the first priority
                if os.path.exists(os.path.join("csp", film + "_local.allboxes")):
                    allboxes_file = os.path.join("csp", film + "_local.allboxes")
                    allboxes = np.loadtxt(
                        os.path.join("csp", film + "_local.allboxes"), ndmin=2
                    )
                elif os.path.exists(os.path.join("csp", film + ".allboxes")):
                    allboxes_file = os.path.join("csp", film + ".allboxes")
                    allboxes = np.loadtxt(
                        os.path.join("csp", film + ".allboxes"), ndmin=2
                    )
                else:
                    raise Exception(f"{film} allboxes not found")

                allboxes_name, allboxes_format = os.path.splitext(allboxes_file)

                par_data_film = par_data[par_data[:, FILM_COL] == idx]

                if par_data_film.shape[0] != allboxes.shape[0]:
                    raise Exception(
                        f"The number of lines in {clean[1]} and {allboxes} does not match - {par_data_film.shape[0]} v.s. {allboxes.shape[0]}"
                    )

                allboxes = np.delete(
                    allboxes, np.argwhere(par_data_film[:, OCC_COL] == 0), axis=0
                )

                allboxes_line_count += allboxes.shape[0]

                os.rename(allboxes_file, allboxes_name + ORI_TAG + allboxes_format)
                np.savetxt(allboxes_file, allboxes.astype(int), fmt="%i")

            if par_data_clean.shape[0] != allboxes_line_count:
                [
                    os.rename(f, f.replace(ORI_TAG + allboxes_format, allboxes_format))
                    for f in [
                        os.path.join("csp", file)
                        for file in os.listdir("csp")
                        if file.endswith(ORI_TAG + allboxes_format)
                    ]
                ]
                raise Exception(
                    f"After cleaning the total number of lines in {clean[1]} and allboxes does not match - {par_data_clean.shape[0]} v.s. {allboxes_line_count}"
                )

            par_data_clean[:, 0] = np.array(
                [(_i + 1) % 10000000 for _i in range(par_data_clean.shape[0])]
            )
            par.data = par_data_clean
            os.rename(clean[1], clean[1].replace(parx_format, ORI_TAG + parx_format))
            par.write_file(clean[1])

            logger.info("Successfully remove particles from the parfile and allboxes!")

    for f in files:

        if startat != "":
            if f == startat:
                logger.info(f"Starting at tilt-series {startat}")
                startat = ""
            else:
                continue

        # load raw tilt series
        if raw:
            local_run.run_shell_command(
                "{0}/bin/3dmod mrc/{1}_bin.mrc".format(get_imod_path(), f)
            )

        # load aligned tilt series to select bad views
        if ali:
            try:
                if not os.path.isfile("mod/{0}_exclude_views.mod".format(f)):
                    shutil.copy2(
                        "{0}/xclude_model.mod".format(get_parameter_files_path()),
                        "mod/%s_exclude_views.mod" % f,
                    )
                local_run.run_shell_command(
                    "{0}/bin/3dmod mrc/{1}_bin.ali mod/{1}_exclude_views.mod".format(
                        get_imod_path(), f
                    )
                )
            except:
                local_run.run_shell_command(
                    "{0}/bin/3dmod mrc/{1}_bin.ali".format(get_imod_path(), f)
                )

        # load virion model to get rid of false positives
        if rec:
            local_run.run_shell_command(
                "{0}/bin/3dmod -Y -xyz mrc/{1}.rec mod/{1}.vir".format(
                    get_imod_path(), f
                )
            )

        # load tomograms and model with selected areas of interest
        if reg:
            if not os.path.isfile("mod/{0}_regions.mod".format(f)):
                shutil.copy2(
                    "{0}/xclude_regions.mod".format(get_parameter_files_path()),
                    "mod/%s_regions.mod" % f,
                )

            local_run.run_shell_command(
                "{0}/bin/3dmod -Y -xyz mrc/{1}.rec mod/{1}_regions.mod".format(
                    get_imod_path(), f
                )
            )
        # load segmented virions to select bad ones
        if seg:
            if not os.path.isfile("mod/{0}_segment.mod".format(f)):
                shutil.copy2(
                    "{0}/xclude_segment.mod".format(get_parameter_files_path()),
                    "mod/%s_segment.mod" % f,
                )
            local_run.run_shell_command(
                "{0}/bin/3dmod `ls mrc/{1}_vir????_binned_nad.mrc` mod/{1}_segment.mod".format(
                    get_imod_path(), f
                )
            )

        # load segmented virions to select bad ones
        # if os.path.exists( 'mod/{0}_exclude_virions.mod'.format(f) ):
        if vir:
            # HF: add -E 1,2 to enable model mode by deault
            if not os.path.isfile("next/{0}_exclude_virions.mod".format(f)):
                shutil.copy2(
                    "{0}/xclude_virions.mod".format(get_parameter_files_path()),
                    "next/%s_exclude_virions.mod" % f,
                )
            [ local_run.run_shell_command("convert %s %s" % (vir_img, vir_img.replace(".webp", ".png"))) 
                for vir_img in glob.glob(f"webp/{f}_vir????_binned_nad.webp") ]

            local_run.run_shell_command(
                "{0}/bin/3dmod -E 1,2 `ls webp/{1}_vir????_binned_nad.png` next/{1}_exclude_virions.mod".format(
                    get_imod_path(), f
                )
            )

        if spk:
            if not os.path.isfile("mod/{0}.spk".format(f)):
                shutil.copy2(
                    "{0}/xclude_regions.mod".format(get_parameter_files_path()),
                    "mod/%s.spk" % f,
                )
            local_run.run_shell_command(
                "{0}/bin/3dmod -Y -xyz mrc/{1}.rec mod/{1}.spk".format(
                    get_imod_path(), f
                )
            )

        if skip:
            happy = True
        else:
            happy = False

        while not happy:
            nb = input("Keep {0} (yes/no/done)? ".format(f))
            # print nb
            if nb == "n" or nb == "no":
                keep_files.remove(f)
                remove_files.append(f)
                logger.info(f"Removing {f}. {len(keep_files)} images left.")
                happy = True
            elif nb == "d" or nb == "done":
                done = True
                happy = True
            elif nb == "y" or nb == "yes":
                happy = True
            else:
                logger.info("Invalid entry. Please select from (yes,no,done,cancel).")

        if done:
            break

    if not skip:
        if len(remove_files) > 0:
            nb = input("Permanently save image selection (yes/no)?")
            if nb.lower() == "y" or nb.lower() == "yes":

                # backup previous micrograph list
                old_micrographs = (
                    micrographs
                    + "_"
                    + datetime.datetime.fromtimestamp(time.time()).strftime(
                        "%Y%m%d_%H%M%S"
                    )
                )
                shutil.move(micrographs, old_micrographs)

                # save new micrograph list
                f = open(micrographs, "w")
                for k in keep_files:
                    f.write("%s\n" % k)
                f.close()
                logger.info(
                    "%d files kept. Original micrograph list saved to %s",
                    len(keep_files),
                    old_micrographs,
                )

                # save bad micrograph list
                bad_micrographs = (
                    micrographs
                    + "_"
                    + datetime.datetime.fromtimestamp(time.time()).strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    + "_bad"
                )
                f = open(bad_micrographs, "w")
                for k in remove_files:
                    f.write("%s\n" % k)
                f.close()
                logger.info(
                    "%d files removed and saved to %s",
                    len(remove_files),
                    bad_micrographs,
                )
        else:
            logger.info("No images have been discarded.")


def cryolo_3d(
    boxsize, flipyz, recdir, a, inputsize, lpcutoff, thr, tsr, tmem, tmin, skip
):
    config = get_pyp_configuration()
    if not 'cryolo' in config["pyp"]:
        raise Exception('Configuration for crYOLO not found. Please add the location to the cryolo conda environment in the configuration file (section: [pyp], entry: cryolo).')

    if not os.path.isdir("cryolo"):
        os.mkdir("cryolo")

    # flipyz tomo .rec files
    parameters = project_params.load_pyp_parameters()
    micrographs = "{}.reclist".format(parameters["data_set"])

    if not os.path.isfile(micrographs):
        logger.warning("Cannot find %s", micrographs)
        files = [
            s.replace("mrc/", "").replace(".rec", "") for s in glob.glob("mrc/*.rec")
        ]
        with open(micrographs, "w") as reclist:
            for file in files:
                reclist.write("%s\n" % file)
    else:
        files = open(micrographs, "r").read().split()

    os.chdir("./cryolo")
    n = len(files)
    ## this part need to improve as parallel run. may be array is better.
    sysqueue = parameters["slurm_queue"]
    if not skip:

        header = f"""
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition={sysqueue}
#SBATCH --job-name=flipyz
#SBATCH --array=1-{n}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --ntasks-per-core=1

"""
        with open("job_submit.sh", 'w') as f:
            f.write("#!/bin/bash" + "\n")
            f.write(header)
            f.write("eval `cat jobs.swarm | awk -v line=$SLURM_ARRAY_TASK_ID '{if (NR == line) print}'`")

        for recfile in files:
            folder_name = recfile + "_rec"
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)
            if flipyz:
                if not os.path.exists("%s/%s.rec" % (folder_name, recfile)):
                    # need version IMOD_4.11.12?
                    command_clip = "{0}/bin/clip flipyz ../{2}/{1}.rec {3}/{1}.rec".format(
                        get_imod_path(), recfile, recdir, folder_name
                    )

                    with open("jobs.swarm", 'a') as f:
                        f.write(command_clip + "\n")
                else:
                    logger.info(".rec files already exist, pass the flipyz step.")
                    pass


            else:
                os.symlink(
                    "../../{0}/{1}.rec".format(recdir, recfile),
                    "{0}/{1}.rec".format(folder_name, recfile),
                )
        if flipyz and os.path.isfile("jobs.swarm"):
            slurm.submit_jobs(".", "job_submit.sh", "flipyz", "flipyz", sysqueue)
            # wait the flipyz jobs finish
            slurm.check_sbatch_job_finish("flipyz")
            if os.path.isfile("jobs.swarm"):
                os.remove("jobs.swarm")

        yolo_ini_dir = "picked_input"
        if not os.path.isdir(yolo_ini_dir):
            os.mkdir(yolo_ini_dir)

        picked_files = [
            s.replace("../next/", "").replace(".next", "")
            for s in glob.glob("../next/*.next")
        ]
        name = picked_files[0]
        pwd = os.getcwd()
        logger.info(pwd)
        for file in picked_files:
            if not os.path.isfile("picked_input/" + file + ".rec"):
                symlink_relative(
                    os.path.join(pwd, file + "_rec", file + ".rec"),
                    os.path.join("picked_input","%s.rec" % file)
                )
            else:
                pass

        x = int(mrc.readHeaderFromFile(yolo_ini_dir + "/" + name + ".rec")["nx"])
        z = int(mrc.readHeaderFromFile(yolo_ini_dir + "/" + name + ".rec")["nz"])
        if not int(inputsize) == x:
            logger.warning(
                "The actual image size in x dimension is %s. Using actual size now.", x
            )
        inputsize = str(x)
        z_height = str(z)
        # convert mod to cryolo CBOX
        for file in picked_files:
            command = "python {3}/src/pyp/analysis/geometry/pyp_convert_coord.py -mod2cryolo -input ../next/{0}.next -output picked_input/{0}.cbox -boxsize {1} -z {2} -s 1".format(
                file, boxsize, z_height, os.environ["PYP_DIR"]
            )
            local_run.run_shell_command(command)

        # cryolo config
        ## do we need conda environment to run the cryolo_gui.py
        cryolo_path = config["pyp"]["cryolo"]
        command = "source activate {cryolo_path}; {cryolo_path}/bin/cryolo_gui.py --ignore-gooey config "\
            + f"--train_image_folder {yolo_ini_dir} --train_annot_folder {yolo_ini_dir} --saved_weights_name cryolo_model.h5 -a {a} --input_size {inputsize} "\
            + f"-nm STANDARD --num_patches 1 --overlap_patches 200 --filtered_output filtered_tmp -f LOWPASS --low_pass_cutoff {lpcutoff} --janni_overlap 24 "\
            + "--janni_batches 3 --train_times 10 --batch_size 4 --learning_rate 0.0001 --nb_epoch 200 --object_scale 5.0 --no_object_scale 1.0 --coord_scale 1.0 "\
            + f"--class_scale 1.0 --debug --log_path logs/ -- config_cryolo.json {boxsize}"

        local_run.run_shell_command(command, verbose=True)

        # cryolo train submit jobs to GPU node
        submit_script = "cryolo_train_submit.sh"
        with open(submit_script, "w") as s:
            header = f"#!/usr/bin/env bash\n#SBATCH --job-name cryolo_train\n#SBATCH -n {parameters['slurm_ntasks']}\n#SBATCH --gres=gpu:1\n#SBATCH -p {parameters['slurm_queue_gpu']}\n#SBATCH --mem={parameters['slurm_memory']}g\n#SBATCH --output cryolo_train.out\n#SBATCH --error cryolo_train.err\n"
            s.write(header)
            s.write(f"source activate {cryolo_path}\n")
            command = "cryolo_gui.py --ignore-gooey train -c config_cryolo.json -w 5 -nc -1 --gpu_fraction 1.0 -e 10 -lft 2 --seed 10 "\
            + """&& echo ${SLURM_JOB_ID} && squeue -j ${SLURM_JOB_ID} \n"""
            s.write(command)

        command = run_slurm(command="sbatch", path=os.getcwd()) + " " + submit_script
        command = run_ssh(command)
        local_run.run_shell_command(command)

        slurm.check_sbatch_job_finish("cryolo_train")

    else:
        picked_files = [
            s.replace("../next/", "").replace(".next", "")
            for s in glob.glob("../next/*.next")
        ]
        name = picked_files[0]
        yolo_ini_dir = "picked_input"
        z = int(mrc.readHeaderFromFile(yolo_ini_dir + "/" + name + ".rec")["nz"])
        z_height = str(z)
        logger.info("Skip cryolo configuration and training. Run prediction now:")

    # crylo_predict submit jobs to GPU nodes
    submit_script = "cryolo_predict_submit.sh"
    with open(submit_script, "w") as s:
        header = f"#!/usr/bin/env bash\n#SBATCH --job-name cryolo_predict\n#SBATCH -n {parameters['slurm_ntasks']}\n#SBATCH --gres=gpu:1\n#SBATCH -p {parameters['slurm_queue_gpu']}\n#SBATCH --mem={parameters['slurm_memory']}g\n#SBATCH --output cryolo_predict.out\n#SBATCH --error cryolo_predict.err\n"
        s.write(header)
        s.write(f"source activate {cryolo_path}\n")
        command = "ls -d *_rec | while read lines; do cryolo_gui.py predict -c config_cryolo.json -w cryolo_model.h5 -i $lines "\
            + f"-o cryolo_output -t {thr} -pbs 3 --gpu_fraction 1.0 -nc -1 -mw 100 -sr 1.41 --tomogram -tsr {tsr} -tmem {tmem} -tmin {tmin} ; done "\
            + """&& echo ${SLURM_JOB_ID} && squeue -j ${SLURM_JOB_ID}\n"""
        s.write(command)
    command = run_slurm(command="sbatch", path=os.getcwd()) + " " + submit_script
    command = run_ssh(command)
    local_run.run_shell_command(command, verbose=True)

    slurm.check_sbatch_job_finish("cryolo_predict")

    # convert back to model
    for boxfile in files:
        if os.path.isfile("cryolo_output/COORDS/%s.coords" % boxfile):
            com_format = (
                """awk '{printf("%4.1f\\t%4.1f\\t%4.1f\\n", $1,$2,$3)}' """
                + "cryolo_output/COORDS/%s.coords > cryolo_output/COORDS/%s.box"
                % (boxfile, boxfile)
            )
            local_run.run_shell_command(com_format)

            command = "python {3}/src/pyp/analysis/geometry/pyp_convert_coord.py -cryolo2mod -input cryolo_output/COORDS/{0}.box -output ../mod/{0}.mod -boxsize {1} -z {2} -s 1".format(
                boxfile, boxsize, z_height, os.environ["PYP_DIR"]
            )
            local_run.run_shell_command(command, verbose=True)
        else:
            pass

    # clean temporary files
    shutil.rmtree("filtered_tmp")
    os.chdir("..")

def enable_profiler(parameters=None,path=None):
    if parameters == None:
        parameters = project_params.load_parameters(path)
    if "slurm_profile" in parameters.keys() and parameters["slurm_profile"]:
        pr = cProfile.Profile()
        pr.enable()
        return pr
    else:
        return None 

def disable_profiler(profiler,path=os.getcwd()):
    if profiler != None:
        profiler.disable()
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        filename = os.path.join( path, timestamp + ".prof" )
        profiler.dump_stats(filename)
        os.chdir(path)
        command = "/usr/local/envs/pyp/bin/gprof2dot -f pstats {0} | dot -Tpdf -o {1}".format(filename,filename.replace(".prof",".pdf"))
        local_run.run_shell_command(command)
        # os.remove(filename)


def trackback():
    type, value, traceback = sys.exc_info()
    sys.__excepthook__(type, value, traceback)

    # try to tell web website about the failure, if possible
    if Web.exists:
        Web().failed()

def get_free_space(scratch):
    # report space available on local scratch
    command = f"df -h {scratch} 2> /dev/null"
    [ output, error ] = local_run.run_shell_command(command, verbose=False)
    for line in output.split("\n"):
        if len(line) > 0:
            logger.info(line)

"""
def clear_scratch(scratch):
    # clear up local scratch directory from non-running jobs
    command = run_ssh(run_slurm(command="squeue --me -o %i --noheader"))
    [ running_jobs, error ] = local_run.run_shell_command(command, verbose=True)
    logger.info("running_jobs")
    logger.info(command)
    logger.info(running_jobs)
    for dir in [ name for name in os.listdir(scratch) if os.path.isdir(os.path.join(scratch, name)) ]:
        # check if directory is in the form {SLURM_JOB_ID}_{SLURM_ARRAY_TASK_ID}
        if bool(re.match('[\d/_]+$',dir)) and dir not in running_jobs:
            try:
                logger.warning(f"Slurm job {dir} is no longer running. Attempting to delete")
                os.rmdir(dir)
            except:
                logger.error(f"Failed to delete folder {dir}")
                pass
"""

# remove any leftover scratch directories that are older than 1 hour
def clear_scratch(scratch):
    # list all top level directories under scratch folder
    if os.path.exists(scratch):
        for dir in [ name for name in os.listdir(scratch) if os.path.isdir(os.path.join(scratch, name)) ]:
            # check if directory is in the form {SLURM_JOB_ID}_{SLURM_ARRAY_TASK_ID}
            if bool(re.match('[\d/_]+$',dir)):
                # get list of all files in this directory
                list_of_files = glob.glob(f'{os.path.join(scratch,dir)}/**/*.*',recursive=True)
                if len(list_of_files) > 0:
                    try:
                        # get timestamp of most recent file
                        latest_file = max(list_of_files, key=os.path.getctime)
                        age_in_minutes = ( time.time() - os.path.getctime(latest_file) ) / 60.
                        # if age of most recent file is more than 1 hour, assume this is a zombie folder
                        if age_in_minutes > 60:
                            try:
                                logger.warning(f"Detected zombie run at {dir}, clearing up files")
                                shutil.rmtree(os.path.join(scratch,dir), ignore_errors=True)
                            except:
                                logger.error(f"Failed to delete folder {dir}")
                                pass
                    except:
                        pass

if __name__ == "__main__":

    try:

        mpi_tasks = mpi.initialize_worker_pool()

        jobid = None
        if "SLURM_ARRAY_JOB_ID" in os.environ and "SLURM_ARRAY_TASK_ID" in os.environ:
            jobid = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
        elif "SLURM_JOB_ID" in os.environ:
            jobid = os.environ["SLURM_JOB_ID"]

        # initialize various parameters
        if not "PYP_DIR" in os.environ:
            raise Exception("You must define environment variable $PYP_DIR")

        # retrieve version number
        version = toml.load(os.path.join(os.environ['PYP_DIR'],"nextpyp.toml"))['version']
        memory = f"and {int(os.environ['SLURM_MEM_PER_NODE'])/1024:.0f} GB of RAM" if "SLURM_MEM_PER_NODE" in os.environ else ""

        if jobid is None:
            logger.info(
                "Job (v{}) launching on {} using {} task(s) {}".format(
                version, socket.gethostname(), mpi_tasks, memory
                )
            )
        else:
            logger.info(
                "Job {} (v{}) launching on {} using {} task(s) {}".format(
                jobid, version, socket.gethostname(), mpi_tasks, memory
                )
            )

        config = get_pyp_configuration()

        os.environ["OMP_NUM_THREADS"] = os.environ["IMOD_PROCESSORS"] = "1"

        os.environ["PYTHONDIR"] = "{0}/src".format(os.environ["PYP_DIR"])
        os.environ["SHELLDIR"] = "{0}/shell".format(os.environ["PYP_DIR"])

        os.environ["PYP_PYTHON"] = "Anaconda2/2.7.13"
        scratch_config = config["pyp"]["scratch"]
        if "$" in scratch_config:
            scratch_split = config["pyp"]["scratch"].split("$")
            os.environ["PYP_SCRATCH"] = os.path.join(
                scratch_split[0], os.environ[scratch_split[1]]
            )
        else:
            os.environ["PYP_SCRATCH"] = scratch_config
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["PBS_O_WORKDIR"] = os.getcwd()

        if "SLURM_ARRAY_JOB_ID" in os.environ:
            subdir = f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        elif "SLURM_JOB_ID" in os.environ:
            subdir = os.environ["SLURM_JOB_ID"]
        else:
            subdir = ""
        os.environ["PYP_SCRATCH"] = str(
            Path(os.environ["PYP_SCRATCH"]) / os.environ["USER"] / subdir
        )
        if not os.path.exists(os.environ["PYP_SCRATCH"]):
            try:
                os.makedirs(os.environ["PYP_SCRATCH"], exist_ok=True)
            except:
                pass

        # TODO: switch to pyp.system.utils.get_imod_path()
        os.environ["IMAGICDIR"] = "/usr/bin"
        os.environ["IMOD_DIR"] = get_imod_path()
        if "LD_LIBRARY_PATH" in os.environ:
            os.environ["LD_LIBRARY_PATH"] = "{0}/qtlib:{0}/lib:{1}".format(
                get_imod_path(), os.environ["LD_LIBRARY_PATH"]
            )
        else:
            os.environ["LD_LIBRARY_PATH"] = "{0}/qtlib:{0}/lib".format(get_imod_path())

        os.environ["LD_LIBRARY_PATH"] = "{0}:{1}".format(
                os.environ["LD_LIBRARY_PATH"], '/usr/local/pkgs/fftw-3.3.10-nompi_hf0379b8_106/lib/'
            )

        os.environ["RELIONDIR"] = "{0}/relion-1.2".format(os.environ["PYP_DIR"])
        os.environ["JASPERDIR"] = "{0}/jasper".format(os.environ["PYP_DIR"])
        os.environ["MCR_CACHE_ROOT"] = os.environ[
            "PYP_SCRATCH"
        ]  # need to avoid problem with Matlab-related Jasper error
        if "PYTHONPATH" in os.environ:
            os.environ["PYTHONPATH"] = "{0}:{1}".format(
                os.environ["PYTHONPATH"], os.environ["PYTHONDIR"]
            )
        else:
            os.environ["PYTHONPATH"] = os.environ["PYTHONDIR"]


        current_directory = os.getcwd()
        job_name = None

        # if pyp was launched by the webserver, do some additional initialization
        if Web.exists:
            Web.init_env()
        else:
            # keep track of issued commands
            try:
                with open(".pyp_history", "a") as f:
                    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
                        "%Y/%m/%d %H:%M:%S "
                    )
                    f.write(timestamp + " ".join(sys.argv) + "\n")
            except:
                logger.error(f"Can't write to {os.getcwd()}/.pyp_history")
                trackback()
                logger.error("Failed to launch PYP")
                pass
        # daemon
        if "pypdaemon" in os.environ:

            del os.environ["pypdaemon"]

            # load existing parameters or from data_parent
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("-data_mode", "--data_mode")
            args, unknown = parser.parse_known_args()
            parent_parameters = vars(args)

            # parse arguments (same as pyp parser)
            parameters = project_params.load_parameters()
            args = project_params.parse_parameters(parameters,"stream",parent_parameters["data_mode"])

            # extract tilt-angles from dm4 header before compressing
            if 'stream_compress' in args and args['stream_compress'] != "none":
                output_dir = project_params.resolve_path(args["stream_transfer_target"])
                if output_dir and os.path.exists(output_dir):
                    data_dir = output_dir
                else:
                    data_dir = config["stream"]["target"]
                session_dir = os.path.join(data_dir, args["stream_session_group"], args["stream_session_name"])
                data_path = Path(project_params.resolve_path(args["data_path"]))
                args['movie_pattern'] = args['movie_pattern'].replace( data_path.suffix, '.' + args['stream_compress'] )
                project_params.save_pyp_parameters(args,session_dir)

            pyp_daemon.pyp_daemon(args)

        # daemon
        if "sess_img" in os.environ:

            del os.environ["sess_img"]

            # parse arguments (same as pyp parser)
            parameters = project_params.load_parameters("..")
            args = project_params.parse_parameters(parameters,"stream",parameters["data_mode"])

            pyp_daemon.pyp_daemon_process(args)

        # merge
        elif "sprmerge" in os.environ:

            del os.environ["sprmerge"]
            # job_name = "sprmerge"

            try:
                cwd = os.getcwd()
                spa_Tlog = {}

                if "PBS_O_WORKDIR" in os.environ:
                    if (
                        "swarm" in os.environ["PBS_O_WORKDIR"]
                        and not "swarm" in os.getcwd()
                    ):
                        os.chdir(os.environ["PBS_O_WORKDIR"])

                os.chdir(os.pardir)

                logger.info(f"Running merge from {os.getcwd()}")

                parameters = project_params.load_parameters()
                spr_merge(parameters)
                parameters["movie_force"] = parameters["ctf_force"] = parameters["detect_force"] = False
                project_params.save_pyp_parameters(parameters)

                spa_Tlog.update(timer.Timer.timers)
                spa_Tlog = {k:v for k, v in spa_Tlog.items() if v}
                json_file = cwd + "/mpi_%d" % parameters["slurm_tasks"] + "_sprmerge.json"
                with open(json_file, 'w') as fp:
                    json.dump(spa_Tlog, fp, indent=4,separators=(',', ': '))

                logger.info("PYP (sprmerge) finished successfully")
            except:
                trackback()
                logger.error("PYP (sprmerge) failed")
                pass
        # swarm
        elif "sprswarm" in os.environ:

            del os.environ["sprswarm"]
            job_name = "sprswarm"

            # cwd = os.getcwd()
            # args = project_params.parse_arguments("sprswarm")
            # parameters = project_params.load_pyp_parameters(path="../")

            # spr_swarm(args.path, args.file, args.debug, args.keep, args.skip)
            try:

                # clear local scratch and report free space
                clear_scratch(Path(os.environ["PYP_SCRATCH"]).parents[0])
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])

                cwd = os.getcwd()
                # spa_Tlog = {}
                args = project_params.parse_arguments("sprswarm")
                parameters = project_params.load_pyp_parameters(path="../")

                spr_swarm(args.path, args.file, args.debug, args.keep, args.skip)

                # spa_Tlog.update(timer.Timer.timers)
                # spa_Tlog = {k:v for k, v in spa_Tlog.items() if v}
                # json_file = cwd + "/mpi_%d" % parameters["slurm_tasks"] + "_sprswarm.json"
                # with open(json_file, 'w') as fp:
                #    json.dump(spa_Tlog, fp, indent=4,separators=(',', ': '))

                logger.info("PYP (sprswarm) finished successfully")
            except:
                trackback()
                logger.error("PYP (sprswarm) failed")
                pass

            # we are done, clear local scratch
            if os.path.exists(os.environ["PYP_SCRATCH"]):
                shutil.rmtree(os.environ["PYP_SCRATCH"])

        # swarm
        elif "tomoswarm" in os.environ:

            del os.environ["tomoswarm"]
            job_name = "tomoswarm"

            try:

                # clear local scratch and report free space
                clear_scratch(Path(os.environ["PYP_SCRATCH"]).parents[0])
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])

                args = project_params.parse_arguments("tomoswarm")

                tomo_swarm(args.path, args.file, args.debug, args.keep, args.skip)

                logger.info("PYP (tomoswarm) finished successfully")
            except:
                trackback()
                logger.error("PYP (tomoswarm) failed")
                pass

            # we are done, clear local scratch
            if os.path.exists(os.environ["PYP_SCRATCH"]):
                shutil.rmtree(os.environ["PYP_SCRATCH"])

        elif "tomomerge" in os.environ:

            try:
                del os.environ["tomomerge"]
                job_name = "tomomerge"

                if "PBS_O_WORKDIR" in os.environ:
                    if (
                        "swarm" in os.environ["PBS_O_WORKDIR"]
                        and not "swarm" in os.getcwd()
                    ):
                        os.chdir(os.environ["PBS_O_WORKDIR"])

                os.chdir(os.pardir)

                parameters = project_params.load_pyp_parameters()
                tomo_merge(parameters)
                # reset all flags for re-calculation
                parameters["movie_force"] = parameters["ctf_force"] = parameters["detect_force"] = parameters["tomo_vir_force"] = parameters["tomo_ali_force"] = parameters["tomo_rec_force"] = parameters["data_import"] = False
                project_params.save_pyp_parameters(parameters)
                logger.info("PYP (tomomerge) finished successfully")
            except:
                trackback()
                logger.error("PYP (tomomerge) failed")
                pass

        elif "tomoedit" in os.environ:

            del os.environ["tomoedit"]

            args = project_params.parse_arguments("tomoedit")

            if not (
                args.raw
                or args.ali
                or args.rec
                or args.vir
                or args.reg
                or args.seg
                or args.spk
                or args.clean
            ):
                raise Exception(
                    "You must specify at least one of -raw, -ali, -rec, -vir, -seg, -spk, -reg or -clean."
                )

            tomo_edit(
                args.startat,
                args.raw,
                args.ali,
                args.rec,
                args.reg,
                args.seg,
                args.vir,
                args.spk,
                args.skip,
                [args.clean, args.parx, args.dist, args.discard],
            )

        elif "boxedit" in os.environ:

            del os.environ["boxedit"]

            args = project_params.parse_arguments("boxedit")

            box_edit(args.skip, args.startat)

        elif "import_star" in os.environ:

            del os.environ["import_star"]

            try:
                parameters = parse_arguments("import_star")
                project_params.save_parameters(parameters)

                if parameters != 0:

                    if "import_read_star" in parameters.keys() and parameters["import_read_star"] and len(glob.glob("pkl/*")) == 0:

                        mode = parameters["data_mode"].lower()
                        if "spr" in mode:
                            set_up.prepare_spr_dir()
                        else:
                            set_up.prepare_tomo_dir()

                        dataset = parameters["data_set"]

                        globalmeta = pyp_metadata.GlobalMetadata(
                        dataset,
                        parameters,
                        imagelist="",
                        mode=mode,
                        getpickle=False,
                        parfile="",
                        path="./pkl"
                        )

                        starfile = project_params.resolve_path(parameters["import_refine_star"])

                        rln_path = project_params.resolve_path(parameters["import_relion_path"])

                        if "spr" in mode:
                            if "import_motion_star" in parameters and parameters["import_motion_star"] is not None:
                                motionstar = project_params.resolve_path(parameters["import_motion_star"])
                            else:
                                motionstar = ""
                            new_imagelist = globalmeta.SpaStar2meta(starfile, motionstar, rln_path=rln_path, linkavg=True)

                        else:
                            tomostar = project_params.resolve_path(parameters["import_tomo_star"])
                            new_imagelist = globalmeta.TomoStar2meta(tomostar, starfile, rln_path=rln_path)

                        # write new film file to PYP
                        filmname = dataset + ".films"
                        micrographs = dataset + ".micrographs"
                        filmarray = np.array(new_imagelist).reshape(-1, 1)
                        np.savetxt(filmname, filmarray, fmt="%s", comments="")
                        shutil.copy2(filmname, micrographs)

                        if "spr" in mode:
                            mag = parameters["scope_mag"]
                            globalmeta.star2par(starfile, mag=mag, path="frealign/")
                        else:
                            # update handedness
                            parameters["csp_ctf_handedness"] = True if globalmeta.micrograph_global["ctf_hand"].values[0] == -1.0 else False
                            project_params.save_parameters(parameters)

                        globalmeta.WritePickle(path="./pkl")

                        # generate image for display
                        binning = 8
                        input_file = glob.glob("mrc/*.*")[0]
                        output_file = os.path.join( os.environ["PYP_SCRATCH"], Path(input_file).name )
                        x, y, slices = get_image_dimensions(input_file)
                        if y > x:
                            rotate = "-rotate 90"
                        else:
                            rotate = ""
                        slice = math.floor( slices / 2 )
                        com = f"{get_imod_path()}/bin/newstack {input_file} {output_file} -bin {binning} -float 2 -secs {slice} {rotate}"
                        local_run.run_shell_command(com)
                        com = f"{get_imod_path()}/bin/mrc2tif -j {output_file} gain_corrected.jpg"
                        local_run.run_shell_command(com)
                        contrast_stretch("gain_corrected.jpg")

                        img2webp("gain_corrected.jpg","gain_corrected.webp")
                        os.remove("gain_corrected.jpg")

                logger.info("PYP (import_star) finished successfully")
            except:
                trackback()
                logger.error("PYP (import_star) failed")
                pass

        elif "export_session" in os.environ:

            del os.environ["export_session"]

            try:
                args = parse_arguments("export_session")

                if args != 0:

                    # assume we are in export directory
                    current_dir = os.getcwd()

                    session_path = args["data_parent"]
                    session_parameters = project_params.load_pyp_parameters(session_path)
                    data_set = session_parameters["data_set"]

                    micrographs = "{}.micrographs".format(data_set)
                    micrograph_list = [line.strip() for line in open(micrographs, "r")]

                    os.chdir(session_path)
                    pickle_files = glob.glob(os.path.join("pkl", "*.pkl"))
                    imagelist = [x.split("/")[-1].replace(".pkl", "") for x in pickle_files if x.split("/")[-1].replace(".pkl", "") in micrograph_list]

                    from pyp.inout.metadata import pyp_metadata

                    globalmeta = pyp_metadata.GlobalMetadata(
                        data_set,
                        session_parameters,
                        imagelist=imagelist,
                        mode=args["data_mode"],
                        getpickle=True,
                        parfile='',
                        path="./pkl"
                        )
                    output_dir = os.path.join( current_dir, "relion" )
                    os.makedirs( output_dir, exist_ok=True)
                    output = os.path.join( output_dir, data_set + ".star")

                    os.chdir(current_dir)
                    coords = False
                    globalmeta.weak_meta2Star(imagelist, output, session_path, coords=coords)

                logger.info("PYP (export_session) finished successfully")

            except:
                trackback()
                logger.error("PYP (export_session) failed")
                pass

        elif "export_star" in os.environ:

            del os.environ["export_star"]

            try:
                parameters = parse_arguments("import_star")

                if parameters != 0:

                    mode = parameters["data_mode"].lower()
                    iteration = parameters["refine_iter"]
                    micrographs = {}
                    all_micrographs_file = parameters["data_set"] + ".films"
                    with open(all_micrographs_file) as f:
                        index = 0
                        for line in f.readlines():
                            micrographs[line.strip()] = index
                            index += 1

                    parfile = "frealign/maps/" + parameters["data_set"] + "_r01" + "_%02d.par" % iteration
                    imagelist = list(micrographs.keys())

                    globalmeta = pyp_metadata.GlobalMetadata(
                        parameters["data_set"],
                        parameters,
                        imagelist=imagelist,
                        mode=mode,
                        getpickle=True,
                        parfile=parfile,
                        path="./pkl"
                        )

                    select = parameters["extract_cls"]
                    globalmeta.meta2Star(parameters["data_set"] + ".star", imagelist, select=select, stack="stack.mrc", parfile=parfile)

                logger.info("PYP (export_star) finished successfully")
            except:
                trackback()
                logger.error("PYP (export_star) failed")
                pass

        elif "csp" in os.environ:

            del os.environ["csp"]

            try:
                parameters = parse_arguments("refine")
 
                if parameters != 0:
                    if parameters["export_enable"]:

                        mode = parameters["data_mode"].lower()
                        iteration = parameters["refine_iter"]
                        micrographs = {}
                        all_micrographs_file = parameters["data_set"] + ".films"
                        with open(all_micrographs_file) as f:
                            index = 0
                            for line in f.readlines():
                                micrographs[line.strip()] = index
                                index += 1

                        par_input = project_params.resolve_path(parameters["export_parfile"])

                        if not os.path.exists(par_input):
                            try:
                                par_input = os.path.join(os.getcwd(), "frealign", "maps", parameters["data_set"] + "_r01_%02d" % parameters["refine_iter"] + ".par.bz2")
                                logger.info(f"Using parfile {par_input} as template for alignment")
                            except:
                                logger.error("Can find any available parfile to read alignment")

                        if par_input.endswith(".par"):
                            parfile = par_input
                        elif par_input.endswith(".bz2"):
                            parfile = frealign_parfile.Parameters.decompress_parameter_file(par_input, parameters["slurm_tasks"])
                        else:
                            logger.error("Can't recognize the parfile")
                            sys.exit()

                        imagelist = list(micrographs.keys())

                        globalmeta = pyp_metadata.GlobalMetadata(
                            parameters["data_set"],
                            parameters,
                            imagelist=imagelist,
                            mode=mode,
                            getpickle=True,
                            parfile=parfile,
                            path="./pkl"
                            )

                        select = parameters["extract_cls"]
                        globalmeta.meta2Star(parameters["data_set"] + ".star", imagelist, select=select, stack="stack.mrc", parfile=parfile)

                        logger.info("PYP (export_star) finished successfully")

                    else:
                        if "particle_rad" not in parameters.keys() and not "import_mode" in parameters.keys() or parameters["extract_box"] == 0:
                            logger.error("You need to pick particles before running refinement")
                            sys.exit()

                        if not parameters["refine_resume"]:
                            parameters["refine_iter"] = parameters["refine_first_iter"]

                        # normal csp procedure
                        iteration = parameters["refine_iter"]
                        if iteration < 2:
                            iteration = parameters["refine_iter"] = 2
                        project_params.save_parameters(parameters)

                        if parameters["data_mode"] == "spr":
                            set_up.prepare_spr_dir()
                        else:
                            set_up.prepare_tomo_dir()

                        # prepare directory structure
                        folders = [
                            "frealign",
                            "frealign/maps",
                            "frealign/log",
                        ]
                        null = [os.mkdir(f) for f in folders if not os.path.exists(f)]

                        if parameters["refine_iter"] == 2:

                            latest_parfile, latest_reference = None, None
                            # data_parent is None if running CLI
                            if "data_parent" in parameters and parameters["data_parent"] is not None:
                                latest_parfile, latest_reference = project_params.get_latest_refinement_reference(project_params.resolve_path(parameters["data_parent"]))

                            parameters["refine_model"] = latest_reference if project_params.resolve_path(parameters["refine_model"]) == "auto" else parameters["refine_model"]

                            # NOTE: spr does not really require a parfile first time we run csp
                            if parameters["refine_parfile"] is not None:
                                parameters["refine_parfile"] = latest_parfile if project_params.resolve_path(parameters["refine_parfile"]) == "auto" else parameters["refine_parfile"]

                            project_params.save_parameters(parameters)

                        if (
                            parameters["refine_model"] is None
                            or not Path(project_params.resolve_path(parameters["refine_model"])).exists() 
                        ):
                            logger.error(
                                f"Reference {parameters['refine_model']} does not exist"
                            )
                        else:
                            try:
                                data_set = parameters["data_set"]
                            except KeyError:
                                data_set = None

                            parxfile = f"frealign/{data_set}_frames_01.parx"
                            stackfile = f"frealign/{data_set}_frames_stack.mrc"

                            # resize reference and copy to frealign/maps
                            reference = (
                                "frealign/maps/" + parameters["data_set"] + "_r01_01.mrc"
                            )
                            if parameters["refine_iter"] == 2:
                                initial_model = project_params.resolve_path(
                                    parameters["refine_model"]
                                )
                                preprocess.resize_initial_model(
                                    parameters, initial_model, reference
                                )

                            films_csp = f"{data_set}.films_csp"
                            if os.path.exists(films_csp):
                                os.remove(films_csp)
                            if not os.path.isfile(stackfile) or not os.path.isfile(
                                parxfile
                            ):
                                csp_split(parameters, iteration)
                            else:
                                logger.warning(
                                    "CSPT files already exist. Please delete if you want to re-create them."
                                )
                                logger.info("\t %s", parxfile)
                                logger.info("\t %s", stackfile)

                        logger.info("PYP (csp) finished successfully")

            except:
                trackback()
                logger.error("PYP (csp) failed")
                pass

        elif "cspswarm" in os.environ:

            del os.environ["cspswarm"]
            # job_name = "cspswarm"

            try:

                # clear local scratch and report free space
                clear_scratch(Path(os.environ["PYP_SCRATCH"]).parents[0])
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])

                args = project_params.parse_arguments("cspswarm")

                working_path = os.path.join(os.environ["PYP_SCRATCH"], args.file)
                cwd = os.getcwd()
                csp_Tlog = {}
                # manage directories
                os.chdir(args.path)

                parameters = project_params.load_pyp_parameters()

                # pr = enable_profiler(parameters)

                csp_swarm(args.file, parameters, int(args.iter), args.skip, args.debug)

                csp_Tlog.update(timer.Timer.timers)
                csp_Tlog = {k:v for k, v in csp_Tlog.items() if v}
                json_file = cwd + "/mpi_%d" % parameters["slurm_tasks"] + "_cspswarm.json"
                with open(json_file, 'w') as fp:
                    json.dump(csp_Tlog, fp, indent=4,separators=(',', ': '))

                # disable_profiler(pr)
                logger.info("PYP (cspswarm) finished successfully")

            except:
                trackback()
                logger.error("PYP (cspswarm) failed")
                pass

        elif "classmerge" in os.environ:

            del os.environ["classmerge"]

            try:
                args = project_params.parse_arguments("classmerge")
                path = os.path.join(os.getcwd(), "..", "frealign", "scratch")
                particle_cspt.csp_class_merge(class_index=args.classId, input_dir=path)
                logger.info("PYP (classmerge) finished successfully")
            except:
                trackback()
                logger.error("PYP (classmerge) failed")
                pass

        elif "cspmerge" in os.environ:

            del os.environ["cspmerge"]

            try:
                os.chdir(os.environ["PBS_O_WORKDIR"] + "/..")
                parameters = project_params.load_pyp_parameters()

                cwd = os.getcwd()
                csp_Tlog = {}

                csp_merge(parameters)

                csp_Tlog.update(timer.Timer.timers)
                csp_Tlog = {k:v for k, v in csp_Tlog.items() if v}
                json_file = cwd + "/mpi_%d" % parameters["slurm_tasks"] + "_cspmerge.json"
                with open(json_file, 'w') as fp:
                    json.dump(csp_Tlog, fp, indent=4,separators=(',', ': '))

                # clean up local scratch
                if os.path.exists(os.environ["PYP_SCRATCH"]):
                    shutil.rmtree(os.environ["PYP_SCRATCH"])
                    logger.info("Deleted temporary files from " + os.environ["PYP_SCRATCH"])

                logger.info("PYP (cspmerge) finished successfully")

            except:
                trackback()
                logger.error("PYP (cspmerge) failed")
                pass

        elif "csp_local_merge" in os.environ:

            # merge multiple files in scratch
            del os.environ["csp_local_merge"]
            local_scratch = os.environ["PYP_SCRATCH"]

            try:
                args = project_params.parse_arguments("csp_local_merge")
                # for time
                cwd = os.getcwd()
                cspm_Tlog = {}

                os.chdir(args.path)

                # for time
                parameters = project_params.load_pyp_parameters()

                particle_cspt.merge_movie_files_in_job_arr(
                    args.stacks_files,
                    args.par_files,
                    args.ordering_file,
                    args.project_path_file,
                    args.output_basename,
                    args.save_stacks,
                )

                # for time
                cspm_Tlog.update(timer.Timer.timers)
                cspm_Tlog = {k:v for k, v in cspm_Tlog.items() if v}
                json_file = cwd + "/mpi_%d" % parameters["slurm_tasks"] + "_csp_local_merge.json"
                with open(json_file, 'w') as fp:
                    json.dump(cspm_Tlog, fp, indent=4,separators=(',', ': '))

                logger.info("PYP (csp_local_merge) finished successfully")

            except:
                trackback()
                logger.error("PYP (csp_local_merge) failed")
                pass

            # clean up local scratch
            if os.path.exists(local_scratch):
                shutil.rmtree(local_scratch)
                logger.info("Deleted temporary files from " + local_scratch)

        # cryolo_picking tomo
        elif "cryolo3d" in os.environ:

            del os.environ["cryolo3d"]
            """
            usage: cryolo_gui.py config [-h] [--train_image_folder TRAIN_IMAGE_FOLDER]
                                        [--train_annot_folder TRAIN_ANNOT_FOLDER]
                                        [--saved_weights_name SAVED_WEIGHTS_NAME]
                                        [-a {PhosaurusNet,YOLO,crYOLO}]
                                        [--input_size INPUT_SIZE [INPUT_SIZE ...]]
                                        [-nm {STANDARD,GMM}] [--num_patches NUM_PATCHES]
                                        [--overlap_patches OVERLAP_PATCHES]
                                        [--filtered_output FILTERED_OUTPUT]
                                        [-f {NONE,LOWPASS,JANNI}]
                                        [--low_pass_cutoff LOW_PASS_CUTOFF]
                                        [--janni_model JANNI_MODEL]
                                        [--janni_overlap JANNI_OVERLAP]
                                        [--janni_batches JANNI_BATCHES]
                                        [--pretrained_weights PRETRAINED_WEIGHTS]
                                        [--train_times TRAIN_TIMES]
                                        [--batch_size BATCH_SIZE]
                                        [--learning_rate LEARNING_RATE]
                                        [--nb_epoch NB_EPOCH]
                                        [--object_scale OBJECT_SCALE]
                                        [--no_object_scale NO_OBJECT_SCALE]
                                        [--coord_scale COORD_SCALE]
                                        [--class_scale CLASS_SCALE] [--debug]
                                        [--valid_image_folder VALID_IMAGE_FOLDER]
                                        [--valid_annot_folder VALID_ANNOT_FOLDER]
                                        [--log_path LOG_PATH]
                                        config_out_path boxsize

            usage: cryolo_gui.py train [-h] -c CONF -w WARMUP [-g GPU [GPU ...]]
                                    [-nc NUM_CPU] [--gpu_fraction GPU_FRACTION]
                                    [-e EARLY] [--fine_tune] [-lft LAYERS_FINE_TUNE]
                                    [--cleanup] [--seed SEED] [--warm_restarts]
                                    [--skip_augmentation]

            usage: cryolo_gui.py predict [-h] -c CONF -w WEIGHTS -i INPUT [INPUT ...] -o
                                        OUTPUT [-t THRESHOLD] [-g GPU [GPU ...]]
                                        [-d DISTANCE] [--minsize MINSIZE]
                                        [--maxsize MAXSIZE] [-pbs PREDICTION_BATCH_SIZE]
                                        [--gpu_fraction GPU_FRACTION] [-nc NUM_CPU]
                                        [--norm_margin NORM_MARGIN] [--monitor] [--otf]
                                        [--cleanup] [--skip] [--filament] [--nosplit]
                                        [--nomerging] [-fw FILAMENT_WIDTH]
                                        [-mw MASK_WIDTH] [-bd BOX_DISTANCE]
                                        [-mn MINIMUM_NUMBER_BOXES]
                                        [-sr SEARCH_RANGE_FACTOR] [--tomogram]
                                        [-tsr TRACING_SEARCH_RANGE]
                                        [-tmem TRACING_MEMORY] [-tmin TRACING_MIN_LENGTH]
                                        [-p PATCH] [--write_empty]
            """

            args = project_params.parse_arguments("cryolo3d")

            cryolo_3d(
                args.boxsize,
                args.flipyz,
                args.recdir,
                args.a,
                args.inputsize,
                args.lpcutoff,
                args.thr,
                args.tsr,
                args.tmem,
                args.tmin,
                args.skip,
            )

        elif "sprtrain" in os.environ:
            del os.environ["sprtrain"]
            try:

                # clear local scratch and report free space
                clear_scratch(Path(os.environ["PYP_SCRATCH"]).parents[0])
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])

                args = project_params.load_pyp_parameters()
                if args["detect_method"].startswith("topaz"):
                    topaz.sprtrain(args)
                else:
                    joint.sprtrain(args)
                logger.info("PYP (sprtrain) finished successfully")
            except:
                trackback()
                logger.error("PYP (sprtrain) failed")
                pass
        elif "tomotrain" in os.environ:
            del os.environ["tomotrain"]
            try:

                # clear local scratch and report free space
                clear_scratch(Path(os.environ["PYP_SCRATCH"]).parents[0])
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])

                args = project_params.load_pyp_parameters()
                joint.tomotrain(args)
                logger.info("PYP (tomotrain) finished successfully")
            except:
                trackback()
                logger.error("PYP (tomotrain) failed")
                pass
        # check gain reference
        elif "pypgain" in os.environ:

            try:
                # Attempt to apply gain reference and save result as jpg file: gain_corrected.jpg
                parameters = parse_arguments("import")

                if parameters is not None and "data_path" in parameters.keys():

                    all_files = glob.glob(
                        project_params.resolve_path(parameters["data_path"])
                    )

                    if len(all_files) > 0:

                        image_file = all_files[np.random.randint(0,high=len(all_files))]
                        logger.info("Selecting image for preview: " + image_file)
                        x, y, z = get_image_dimensions(image_file)
                        image_file_average = Path(image_file).name

                        gain_reference, gain_reference_file = get_gain_reference(
                            parameters, x, y
                        )

                        output_file = "gain_corrected_image.mrc"

                        if z > 1:
                            image_file_average = Path(image_file).stem + "_avg.mrc"
                            output, error = avgstack(
                                image_file, image_file_average, "/"
                            )
                            # os.remove(image_file_average + "~")
                        else:
                            shutil.copy( image_file, image_file_average)

                        # if using eer format, figure out binning factor
                        if image_file.endswith(".eer"):
                            gain_x, gain_y = gain_reference.shape
                            binning = int(x / gain_x)
                            if binning > 1:
                                logger.warning(f"Binning eer frames {binning}x to match gain reference dimensions")
                                com = f"{get_imod_path()}/bin/newstack {image_file_average} {image_file_average} -bin {binning}"
                                local_run.run_shell_command(com)

                        if parameters["gain_remove_hot_pixels"]:
                            preprocess.remove_xrays_from_file(Path(image_file_average).stem)

                        if gain_reference_file is not None:

                            com = '{0}/bin/clip multiply "{1}" "{2}" "{3}"; rm -f {3}~'.format(
                                get_imod_path(),
                                image_file_average,
                                gain_reference_file,
                                output_file,
                            )
                            output, error = local_run.run_shell_command(
                                com, verbose=False
                            )
                            os.remove(gain_reference_file)
                            if "error" in output.lower():
                                logger.error(output)
                                os.remove(output_file)
                                output_file = image_file_average
                                if "sizes must be equal" in output.lower():
                                    logger.error("Did you apply the correct transformation to the gain reference?")
                                raise Exception("Failed to apply gain reference")
                            else:
                                os.remove(image_file_average)
                        else:
                            output_file = image_file_average

                        binning = int(math.floor(x / 768))
                        com = f"{get_imod_path()}/bin/newstack {output_file} {output_file} -bin {binning} -float 2"
                        local_run.run_shell_command(com)
                        com = f"{get_imod_path()}/bin/mrc2tif -j {output_file} gain_corrected.jpg"
                        local_run.run_shell_command(com)
                        contrast_stretch("gain_corrected.jpg")

                        img2webp("gain_corrected.jpg","gain_corrected.webp")
                        [
                            os.remove(f)
                            for f in glob.glob("*.*")
                            if f != "gain_corrected.webp"
                        ]
                        # remove previously cached image
                        if os.path.exists("www/image.small.jpg"):
                            os.remove("www/image.small.jpg")
                logger.info("PYP (pypgain) finished successfully")
            except:
                trackback()
                logger.error("PYP (pypgain) failed")
                pass

        elif "clean" in os.environ:
            del os.environ["clean"]
            try:

                # clear local scratch and report free space
                clear_scratch(Path(os.environ["PYP_SCRATCH"]).parents[0])
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])

                parameters = parse_arguments("spr_tomo_map_clean")

                # prepare frealign directory
                os.makedirs("frealign", exist_ok=True)
                os.chdir("frealign")
                prepare_frealign_dir()
                os.chdir("..")

                if project_params.resolve_path(parameters["clean_parfile"]) == "auto":
                    reference_par_file = sorted(glob.glob( os.path.join(parameters["data_parent"],"frealign","maps","*_r01*.par*") ))
                    if len(reference_par_file) > 0:
                        parameters["clean_parfile"] = reference_par_file[-1]
                        parameters["refine_parfile"] = reference_par_file[-1]
                        parameters["refine_model"] = reference_par_file[-1].replace(".bz2","").replace(".par",".mrc")

                if parameters["clean_discard"]:
                    parfile_occ_zero = Path(os.getcwd(), "frealign", "maps", f"{parameters['data_set']}_r01_02.par.bz2")
                    parameters["clean_parfile"] = parfile_occ_zero if parfile_occ_zero.exists() else parameters["clean_parfile"]

                assert (Path(parameters["clean_parfile"]).exists()), f"{parameters['clean_parfile']} does not exist"

                # copy reconstruction to current frealign/maps
                filename_init = parameters["data_set"] + "_r01_01"
                parfile = project_params.resolve_path(parameters["clean_parfile"])
                reference = parfile.replace(".par.bz2", ".mrc").replace(".par", ".mrc")
                if os.path.exists(reference):
                    shutil.copy2(reference, Path("frealign", "maps", f"{filename_init}.mrc"))

                # do the actual cleaning
                parameters = particle_cleaning(parameters)

                # automatically run reconstruction using clean particles without any refinement 
                # use clean_parfile as refine_parfile
                parameters["refine_skip"] = True
                if not parameters["clean_class_selection"]:
                    parameters["refine_parfile"] = project_params.resolve_path(parameters["clean_parfile"])

                parameters["csp_refine_particles"] = False
                parameters["csp_refine_micrographs"] = False
                parameters["csp_refine_ctf"] = False
                parameters["csp_frame_refinement"] = False
                parameters["refine_iter"] = 2
                parameters["refine_first_iter"] = 2
                parameters["refine_maxiter"] = 2

                parameters["class_num"] = 1

                project_params.save_parameters(parameters)

                # run csp only if user wants to
                if parameters["clean_check_reconstruction"] and not parameters["clean_discard"]:
                    csp_split(parameters, parameters["refine_iter"])
                else:
                    with open("{}.films".format(parameters["data_set"])) as f:
                        files = [
                            line.strip() for line in f
                        ]
                        for filename in files:
                            save_refinement_to_website(filename, parameters["refine_iter"], 'slurm_verbose' in parameters and parameters['slurm_verbose'])

                logger.info("PYP (particle filtering) finished successfully")
            except:
                trackback()
                logger.error("PYP (particle filtering) failed")
                pass

        elif "mask" in os.environ:
            del os.environ["mask"]
            try:
                parameters = parse_arguments("spr_tomo_map_mask")
                parameters["refine_iter"] = 2
                reference = project_params.resolve_path(parameters["mask_model"])

                if reference == "auto":
                    maps = sorted(glob.glob(os.path.join(parameters["data_parent"],"frealign","maps","*_r01_??.mrc")))
                    if len(maps) > 0:
                        reference = maps[-1]

                os.makedirs("frealign/maps", exist_ok=True)

                if Web.exists:
                    name = os.path.split(os.getcwd())[-1]
                    name += "_r01_02"
                else:
                    name = Path(reference).stem + "_masked"
                masked_map =  os.path.join( "frealign", "maps", name + ".mrc" )

                cistem_mask_create(parameters, reference, masked_map)
                project_params.save_parameters(parameters)

                # save png file
                radius = (
                    float(parameters["particle_rad"])
                    / float(parameters["extract_bin"])
                    / float(parameters["data_bin"])
                    / float(parameters["scope_pixel"])
                )
                output_png = os.path.join( "frealign", "maps", name + "_map.png" )
                lim = frealign.build_map_montage( masked_map, radius, output_png )

                img2webp(output_png,output_png.replace(".png",".webp"),"-resize 1024x")

                # produce cropped version of map
                rec = mrc.read(masked_map)
                cropped_volume = rec[ lim:-lim, lim:-lim, lim:-lim ]
                mrc.write(cropped_volume, masked_map.replace(".mrc","_crop.mrc"))

                # send sharpened map to website
                output = {}
                output["def_rot_histogram"] = [[0]]
                output["def_rot_scores"] = [[0]]

                output["rot_hist"] = {}
                output["rot_hist"]["n"] = output["rot_hist"]["bins"] = [[0]]
                output["def_hist"] = {}
                output["def_hist"]["n"] = output["def_hist"]["bins"] = [[0]]
                output["scores_hist"] = {}
                output["scores_hist"]["n"] = output["scores_hist"]["bins"] = [[0]]
                output["occ_hist"] = {}
                output["occ_hist"]["n"] = output["occ_hist"]["bins"] = [[0]]
                output["logp_hist"] = {}
                output["logp_hist"]["n"] = output["logp_hist"]["bins"] = [[0]]
                output["sigma_hist"] = {}
                output["sigma_hist"]["n"] = output["sigma_hist"]["bins"] = [[0]]

                metadata = {}
                metadata["particles_total"] = metadata["particles_used"] = metadata["phase_residual"] = 0
                metadata["occ"] = metadata["logp"] = metadata["sigma"] = 0

                fsc = np.random.rand(10,1)
                save_reconstruction_to_website( name=Path(masked_map).stem, fsc=fsc, plots=output, metadata=metadata )
                logger.info("PYP (mask) finished successfully")
            except:
                trackback()
                logger.error("PYP (mask generation) failed")
                pass

        elif "postprocessing" in os.environ:

            try:
                parameters = parse_arguments("spr_tomo_post_process")

                if not os.path.exists("frealign"):
                    os.mkdir("frealign")
                if not os.path.exists("frealign/maps"):
                    os.mkdir("frealign/maps")

                if Web.exists:
                    name = os.path.split(os.getcwd())[-1] + "_r01_02"
                    output =  name 
                else:
                    name = Path(project_params.resolve_path(parameters["sharpen_input_map"])).stem.replace("_half1","")
                    output = name + "_postprocessing"
                output_map = output + "-masked.mrc"
                project_params.save_parameters(parameters)

                # set-up working area
                current_path = Path.cwd()

                working_path = Path(os.environ["PYP_SCRATCH"]) / name
                logger.info(f"Running on directory {working_path}")
                shutil.rmtree(working_path, "True")
                working_path.mkdir(parents=True, exist_ok=True)
                os.chdir(working_path)

                """
                model_file = sorted(glob.glob( os.path.join(parameters["data_parent"],"frealign","maps","*_r01*.mrc") ))
                if len(model_file) > 0:
                    parameters["sharpen_cistem_input_map"] = model_file[-1]
                else:
                    logger.error('Could not figure out input to map sharpening')

                postprocess.cistem_postprocess(parameters, output)
                """
                half1 = project_params.resolve_path(parameters["sharpen_input_map"])

                # get it from the previous block automatically (last iteration)
                if half1 == "auto":
                    half1_list = sorted(glob.glob(os.path.join(parameters["data_parent"],"frealign","maps","*_r01_half1.mrc")))
                    half1 = half1_list[-1] # we should only have 1 pair of half maps, unless we change in the future 

                half2 = half1.replace('half1', 'half2')

                if "sharpen_mask" in parameters and parameters["sharpen_mask"] != None and \
                    (project_params.resolve_path(parameters["sharpen_mask"]) == "auto" or os.path.isfile(project_params.resolve_path(parameters["sharpen_mask"]))):
                    mask_file = project_params.resolve_path(parameters["sharpen_mask"])
                    if mask_file == "auto":
                        os.chdir(current_path)
                        mask_file = project_params.get_mask_from_projects()
                        os.chdir(working_path)

                    mask = "--mask %s " % mask_file
                    automask = ""
                else:
                    mask = ""
                    automask_lp = parameters["sharpen_automask_lp"]

                    if parameters["sharpen_automask_threshold"] > 0:
                        automask_threshold = "--automask_threshold %.2f " % parameters["sharpen_automask_threshold"]
                        automask_fraction = ""
                        automask_sigma = ""
                    elif parameters["sharpen_automask_fraction"] > 0:
                        automask_threshold = ""
                        automask_fraction = "--automask_fraction %.2f " % parameters["sharpen_automask_fraction"]
                        automask_sigma = ""
                    else:
                        automask_threshold = ""
                        automask_fraction = ""
                        automask_sigma = "--automask_sigma %.1f " % parameters["sharpen_automask_sigma"]

                    automask = f"--automask --automask_input 0 --automask_lp {automask_lp} " +  f"{automask_threshold}{automask_fraction}{automask_sigma}"

                lowpass = parameters["sharpen_lowpass"]
                highpass = parameters["sharpen_highpass"]

                if parameters["sharpen_gaussian"]:
                    filter = f"--gaussian --lowpass {lowpass} --highpass {highpass} "
                else:
                    filter = f"--lowpass {lowpass} --highpass {highpass} "

                if parameters["sharpen_skip_fsc_weighting"]: 
                    skip_fsc_weighting = "--skip_fsc_weighting "
                else:
                    skip_fsc_weighting = ""

                if parameters["sharpen_apply_fsc2"]:
                    apply_fsc2 = "--apply_fsc2 "
                else:
                    apply_fsc2 = ""
                fsc = f"{skip_fsc_weighting} {apply_fsc2}"

                if parameters["sharpen_adhoc_bfac"] is not None:
                    auto_bfac = ""
                    adhoc_bfac = "--adhoc_bfac %i " % parameters["sharpen_adhoc_bfac"]
                else:
                    auto_bfac = "--auto_bfac %s " %  parameters["sharpen_auto_bfac"]
                    adhoc_bfac = ""
                bfac = f"{auto_bfac}{adhoc_bfac}"

                if parameters["sharpen_randomize_below_fsc"] < 1:
                    randomize_below_fsc = "--randomize_below_fsc %.2f " % parameters["sharpen_randomize_below_fsc"]
                    randomize_beyond = ""
                else:
                    randomize_below_fsc = ""
                    randomize_beyond = "--randomize_beyond %.2f " % parameters["sharpen_randomize_beyond"]
                randomize_phase = f"{randomize_below_fsc}{randomize_beyond} " 

                pixel_size = parameters["scope_pixel"] * parameters["extract_bin"]
                if parameters["sharpen_flip_x"]:
                    flip_x = "--flip_x "
                else:
                    flip_x = ""
                if parameters["sharpen_flip_y"]:
                    flip_y = "--flip_y "
                else:
                    flip_y = ""
                if parameters["sharpen_flip_z"]:
                    flip_z = "--flip_z "
                else:
                    flip_z = ""
                if "sharpen_mtf" in parameters and parameters["sharpen_mtf"] != None and os.path.isfile(project_params.resolve_path(parameters["sharpen_mtf"])):
                    mtf = "--mtf %s " % parameters["sharpen_mtf"]
                else:
                    mtf = ""
                if parameters["sharpen_plot_rhref"]:
                    refine_res_lim = "--refine_res_lim %.1f " % parameters["refine_rhref"]
                else:
                    refine_res_lim = ""

                comm_exe = os.environ["PYP_DIR"] + "/external/postprocessing/postprocessing.py "
                basic = f"{half1} {half2} {mask} --angpix {pixel_size} --out {output} {flip_x}{flip_y}{flip_z}{mtf}{refine_res_lim}--xml "
                comm = comm_exe + basic + bfac + filter + fsc + automask + randomize_phase
                local_run.run_shell_command(comm, verbose=False)
                if not os.path.exists(output_map):
                    raise Exception("Does the postprocessing block have enough RAM assigned (launch task)?")

                # produce map slices
                radius = (
                    float(parameters["particle_rad"])
                    / float(parameters["extract_bin"])
                    / float(parameters["data_bin"])
                    / float(parameters["scope_pixel"])
                )

                output_png = output + "_map.png"
                lim = frealign.build_map_montage( output_map, radius, output_png )

                output_path = Path(current_path) / "frealign" / "maps"
                output_path.mkdir(parents=True, exist_ok=True)

                img2webp(output_png, os.path.join(output_path, output_png.replace(".png",".webp")),options="-resize 1024x")

                # produce cropped version of map
                rec = mrc.read(output_map)
                cropped_volume = rec[ lim:-lim, lim:-lim, lim:-lim ]
                mrc.write(cropped_volume, output_map.replace(".mrc","_crop.mrc"))

                # set correct pixel size in mrc header
                command = """
%s/bin/alterheader << EOF
%s
del
%s,%s,%s
done
EOF
""" % (
        get_imod_path(),
        output_map.replace(".mrc","_crop.mrc"),
        pixel_size,
        pixel_size,
        pixel_size,
    )
                local_run.run_shell_command(command)

                # save useful maps and metadata to project directory
                shutil.move( output + "-unmasked.mrc", os.path.join( output_path, output + ".mrc") )
                shutil.move( output + "-masked_crop.mrc", os.path.join( output_path, output + "_crop.mrc") )
                shutil.copy( output + "_data.fsc", os.path.join( output_path, output + "_fsc.txt") )

                # send sharpened map to website
                plots = {}
                plots["def_rot_histogram"] = [[0]]
                plots["def_rot_scores"] = [[0]]

                plots["rot_hist"] = {}
                plots["rot_hist"]["n"] = plots["rot_hist"]["bins"] = [[0]]
                plots["def_hist"] = {}
                plots["def_hist"]["n"] = plots["def_hist"]["bins"] = [[0]]
                plots["scores_hist"] = {}
                plots["scores_hist"]["n"] = plots["scores_hist"]["bins"] = [[0]]
                plots["occ_hist"] = {}
                plots["occ_hist"]["n"] = plots["occ_hist"]["bins"] = [[0]]
                plots["logp_hist"] = {}
                plots["logp_hist"]["n"] = plots["logp_hist"]["bins"] = [[0]]
                plots["sigma_hist"] = {}
                plots["sigma_hist"]["n"] = plots["sigma_hist"]["bins"] = [[0]]

                metadata = {}
                metadata["particles_total"] = metadata["particles_used"] = metadata["phase_residual"] = 0
                metadata["occ"] = metadata["logp"] = metadata["sigma"] = 0

                # retrieve Part_FSC curve from cisTEM
                # part_fsc_file = glob.glob( os.path.join( project_params.resolve_path(parameters["data_parent"]), "frealign", "maps", "*_" + name.split("_")[-1] + "_statistics.txt") )[-1]
                # part_fsc = np.transpose( np.atleast_2d( np.append( 1, np.loadtxt( part_fsc_file, comments="C" )[:,4] ) ) )

                # only use frequency and FSC curves from fsc file
                masked_fsc = np.loadtxt(output + '_data.fsc', comments="#")[:,[0,2,3,4,5]]

                cutoff = fsc_cutoff(masked_fsc[:,[0,-1]], 0.143)
                logger.info(f"FINAL RESOLUTION (after mask correction) = {1/cutoff:.1f} A ({1/cutoff:.3f} A)")

                save_reconstruction_to_website( name, masked_fsc, plots, metadata )

                if not Web.exists:
                    # plot all curves
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 6))

                    ax.plot(1./masked_fsc[:, 0], masked_fsc[:, 1], label="Unmasked")
                    ax.plot(1./masked_fsc[:, 0], masked_fsc[:, 2], label="Masked")
                    ax.plot(1./masked_fsc[:, 0], masked_fsc[:, 3], label="Phase-randomized")
                    ax.plot(1./masked_fsc[:, 0], masked_fsc[:, 4], label="Corrected")

                    ax.plot(1./masked_fsc[:, 0], 0.143 * np.ones(masked_fsc[:, 1].shape), "k:")
                    ax.plot(1./masked_fsc[:, 0], 0.5 * np.ones(masked_fsc[:, 1].shape), "k:")
                    ax.plot(1./masked_fsc[:, 0], np.zeros(masked_fsc[:, 1].shape), "k")

                    legend = ax.legend(loc="upper right", shadow=True, fontsize=10)
                    ax.set_ylim(( min(-0.01, masked_fsc[:, 4].min()), 1.01))
                    ax.set_xlim((1./masked_fsc[0, 0], 1 * 1./masked_fsc[-1, 0]))
                    plt.title(f"FSC for {name}, Final resolution = {1/cutoff:.1f} A ({1/cutoff:.3f} A)")
                    plt.xlabel("Frequency (1/" + "\u00c5" + ")")
                    plt.ylabel("FSC")
                    plt.savefig( os.path.join( output_path, output + ".pdf") )

                shutil.rmtree(working_path)

                logger.info("PYP (postprocessing) finished successfully")

            except:
                trackback()
                logger.error("PYP (postprocessing) failed")
                pass

        # class selection
        elif "kselection" in os.environ:
            del os.environ["kselection"]

            args = project_params.parse_arguments("kselection")
            sel = args.selection
            selist = sel.split(",")
            selection = [int(x) for x in selist]
            iteration = args.iteration

            parameters = project_params.load_pyp_parameters()

            from pyp.inout.metadata.pyp_metadata import merge_par_selection
            merge_par_selection(
                selection,
                parameters,
                iteration,
                merge_align=args.merge_alignment
            )

        # split
        else:
            # initialize
            if (
                os.environ.get("PBS_O_WORKDIR")
                and not "frealign" in os.environ["PBS_O_WORKDIR"]
            ):
                os.chdir(os.environ["PBS_O_WORKDIR"])

            machinefile = "frealign/mpirun.mynodes"
            if os.environ.get("MYNODES"):
                shutil.copy(os.environ["MYNODES"], machinefile)
            elif os.environ.get("PBS_NODEFILE"):
                shutil.copy(os.environ["PBS_NODEFILE"], machinefile)

            logger.info(f"Running on directory {os.getcwd()}")

            parameters = parse_arguments("pre_process")

            if parameters != 0:

                # turn off csp mode
                parameters["csp_no_stacks"] = False

                if "extract_cls" not in parameters.keys():
                    parameters["extract_cls"] = 0

                # save configuration
                project_params.save_parameters(parameters)

                split(parameters)

                # clean up local scratch
                if os.path.exists(os.environ["PYP_SCRATCH"]):
                    shutil.rmtree(os.environ["PYP_SCRATCH"])

                logger.info("PYP (launch) finished successfully")

        if Path(current_directory).name == "swarm":
            folder = Path(current_directory).parents[0]
        else:
            folder = current_directory
        parameters = project_params.load_parameters(folder)
        if job_name and parameters and "slurm_verbose" in parameters and parameters["slurm_verbose"]:
            timers = timer.Timer().timers
            with open(Path(folder) / "swarm" / f"{job_name}.json", "w") as f:
                f.write(json.dumps(timers, indent=2))

    except:
        trackback()

        # clean up local scratch
        if "PYP_SCRATCH" in os.environ and os.path.exists(os.environ["PYP_SCRATCH"]):
            try:
                get_free_space(Path(os.environ["PYP_SCRATCH"]).parents[0])
                shutil.rmtree(os.environ["PYP_SCRATCH"])
                logger.info("Deleted temporary files from " + os.environ["PYP_SCRATCH"])
            except:
                pass

        sys.exit(1)
