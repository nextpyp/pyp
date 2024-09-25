import csv
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from pyp.inout.utils import load_results, save_results
from pyp.streampyp.web import Web
from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path
from pyp.utils import flatten, get_relative_path, symlink_relative
from pyp.inout.metadata import pyp_metadata

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def save_parameters_to_website(parameters):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    elif "data_set" in parameters:
        try:
            # actually send the micrograph to the website
            Web().write_parameters(parameter_id=parameters["data_set"], parameters=parameters)

            if 'slurm_verbose' in parameters and parameters['slurm_verbose']:
                logger.info("Parameters entered into database successfully")
        except:
            logger.error("Failed to enter parameters into database")
            raise

def save_micrograph_to_website(name,verbose=False):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # scan the CTF info
            ctf_path = "%s.ctf" % name
            ctf = None
            if os.path.exists(ctf_path):
                ctf = Web.CTF(*[float(x[0]) for x in csv.reader(open(ctf_path, "r"))])
            else:
                logger.warning("Cannot find ctf information to submit to database")

            # scan the AVGROT info
            avgrot_path = "%s_avgrot.txt" % name
            avgrot = None
            if os.path.exists(avgrot_path):
                avgrot = [Web.AVGROT(*x) for x in np.loadtxt(avgrot_path, comments="#").T]
            else:
                logger.warning("Cannot find avgrot information to submit to database")

            # scan motion info
            xf_path = "%s.xf" % name
            xf = None
            if os.path.exists(xf_path):
                xf = [Web.XF(*x) for x in np.loadtxt(xf_path, ndmin=2)]
            else:
                logger.warning("Cannot find xf information to submit to database")
            if len(xf) == 0:
                xf = None

            # scan particles info
            boxx_path = "%s.boxx" % name
            boxx = None
            if os.path.exists(boxx_path):
                boxx = [
                    Web.BOXX(x[0], x[1], x[2], x[3], int(x[4]), int(x[5]))
                    for x in np.loadtxt(boxx_path, ndmin=2)
                ]
            else:
                boxx = []
            if len(boxx) == 0:
                boxx = None

            # actually send the micrograph to the website
            Web().write_micrograph(name, ctf, avgrot, xf, boxx)

            if verbose:
                logger.info("Series %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter micrograph into database")
            raise

def save_tiltseries_to_website(name, metadata, verbose=False ):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # scan the CTF info
            ctf_path = "%s.ctf" % name
            ctf = None
            if os.path.exists(ctf_path):
                ctf = Web.CTF(*[float(x[0]) for x in csv.reader(open(ctf_path, "r"))])

            # scan the AVGROT info
            avgrot_path = "%s_avgrot.txt" % name
            avgrot = None
            if os.path.exists(avgrot_path):
                avgrot = [Web.AVGROT(*x) for x in np.nan_to_num(np.loadtxt(avgrot_path, comments="#").T)]

            # scan motion info
            xf_path = "%s.xf" % name
            xf = None
            if os.path.exists(xf_path):
                xf = [Web.XF(*x) for x in np.loadtxt(xf_path, ndmin=2)]

            # scan particles info
            boxx_path = "%s.boxx" % name
            boxx = None
            if os.path.exists(boxx_path):
                boxx = [
                    Web.BOXX(x[0], x[1], x[2], x[3], int(x[4]), int(x[5]))
                    for x in np.loadtxt(boxx_path, ndmin=2)
                ]
            else:
                boxx = []

            # actually send the tilt series to the website
            Web().write_tiltseries(name, ctf, avgrot, xf, boxx, metadata)

            if verbose:
                logger.info("Series %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter tilt-series into database")
            raise

def save_reconstruction_to_website(name, fsc, plots, metadata, verbose=False):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the reconstruction to the website
            Web().write_reconstruction(name, metadata, fsc, plots)

            if verbose:
                logger.info("Reconstruction %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter reconstruction into database")
            raise

def save_refinement_to_website(name, iteration, verbose=False):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the refinement to the website
            Web().write_refinement(name, iteration)
            if verbose:
                logger.info("Refinement %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter refinement into database")

def save_refinement_bundle_to_website(name, iteration, verbose=False):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the refinement to the website
            Web().write_refinement_bundle(name, iteration)

            if verbose:
                logger.info("Refinement bundle %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter refinement into database")

def save_classes_to_website(name, metadata, verbose=False):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the micrograph to the website
            Web().write_classes(name,metadata)

            if verbose:
                logger.info("Classes %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter classes into database")
            raise

def save_to_database_daemon(name, current_path, parameters):

    data_dir = "/hpc/group/cryoem/projects_mmc"
    try:
        # connect to the database
        with streampyp.metadb_daemon.open(timeout_ms=100000) as db:

            logger.info("\nConnected to database")
            # pick a session we want to work with
            group_id = current_path.split("/")[-2]
            session_id = parameters["data_set"]
            session_dir = os.path.join(data_dir, group_id, session_id)

            # (re-)create the session
            session = db.session(group_id, session_id)
            if not session.exists():
                logger.info("Session %s does not exist", session.session_id)
            else:
                # (re-)create the micrograph
                micrograph = session.micrograph(name)
                micrograph.create(int(round(time.time() * 1000)))

                logger.info("Updating micrograph id %s", name)
                # create micrograph if not already in list
                if not name in session.get_micrograph_ids():
                    session.append_micrograph_id(name)

                # scan the CTF info
                ctf_path = "%s.ctf" % name
                if os.path.exists(ctf_path):
                    ctf = streampyp.metadb_daemon.CTF(
                        *[float(x[0]) for x in csv.reader(open(ctf_path, "r"))]
                    )
                    micrograph.set_ctf(ctf)

                # scan the AVGROT info
                avgrot_path = "%s_avgrot.txt" % name
                if os.path.exists(avgrot_path):
                    avgrot = [
                        streampyp.metadb_daemon.AVGROT(*x)
                        for x in np.loadtxt(avgrot_path, comments="#").T
                    ]
                    micrograph.set_avgrot(avgrot)

                # scan motion info
                xf_path = "%s.xf" % name
                if os.path.exists(xf_path):
                    xf = [
                        streampyp.metadb_daemon.XF(*x)
                        for x in np.loadtxt(xf_path, ndmin=2)
                    ]
                    micrograph.set_xf(xf)

                # scan particles info
                boxx_path = "%s.boxx" % name
                if os.path.exists(boxx_path):
                    boxx = [
                        streampyp.metadb_daemon.BOXX(
                            x[0], x[1], x[2], x[3], int(x[4]), int(x[4])
                        )
                        for x in np.loadtxt(boxx_path, ndmin=2)
                    ]
                    micrograph.set_boxx(boxx)

                # update the 'updated' timestamp
                session.set_updated_now()

                logger.info("Database successfully updated")

    except:

        logger.error("Could not connect to database")
        type, value, traceback = sys.exc_info()
        sys.__excepthook__(type, value, traceback)


def load_config_files(data_set, project_path, working_path):
    """Load existing results from previous runs and standard project parameter files
    """
    config_files = [".pyp_config.toml", "{0}.micrographs", "{0}.films"]

    filelist = (str(project_path / f.format(data_set)) for f in config_files)

    load_results(filelist, project_path, working_path)


def load_spr_results(name, parameters, project_path, working_path, verbose=False):
    """Load existing results from previous runs and standard project parameter files
    into working_path."""

    initial_files = [
        "raw/{0}.xml",
        "mrc/{0}.mrc",
        "webp/{0}.webp",
        "pkl/{0}.pkl",
        "next/{0}.next"
    ]

    try:
        data_set = parameters["data_set"]
    except KeyError:
        # TODO: what to do when data_set is not present?
        data_set = ""

    file_patterns = [
        f"/ali/{data_set}_???_weights.txt",
        f"/ali/{name}_P????_frames_matches.png",
    ]

    filelist = []
    project_path_escape = Path(glob.escape(str(project_path)))
    filelist += (str(project_path / f.format(name)) for f in initial_files)
    filelist += flatten(sorted(glob.glob(str(project_path_escape / f))) for f in file_patterns)

    # do not retrieve local alignment parameters,
    # filelist += ' '.join( sorted( glob.glob( current_path + '/ali/{0}_P????_frames.xf'.format(name) ) ) )

    load_results(filelist, project_path, working_path, verbose)

    # convert to mrc
    if os.path.exists("{0}.tif".format(name)):
        command = "{0}/bin/newstack {1}.tif {1}.mrc; rm -f {1}.tif".format(
            get_imod_path(), name
        )
        run_shell_command(command, verbose=parameters["slurm_verbose"])

    # rename movie average
    if os.path.exists("{0}.mrc".format(name)):
        shutil.copy("{0}.mrc".format(name), "{0}.avg".format(name))


def save_spr_results(name, parameters, project_path, verbose = False):
    """Save spr swarm run results into original file path."""
    # TODO: reorganize in a similar way to load_spr_results
    files = dict()

    files["pkl"] = "{0}.pkl".format(name)

    files[
        "mrc"
    ] = "{0}.mrc {0}_DW.mrc {0}_DW.tif".format(
        name
    )

    # files['ali'] = '{0}_xray.mod {0}.xf {0}.prexgraw {0}.ccc {0}.blr {0}.mrc {0}_weights.txt {0}_P????_frames.xf {0}_P????_frames_ccc.png {0}_P????_frames.blr {0}_P????_frames_frc.png {0}_frames_matches.gif {0}_P0000_frames_weights_new.png {0}_field.pdf'.format(name)
    files[
        "webp"
    ] = "{0}.webp {0}_boxed.webp {0}_ctffit.webp".format(
        name
    )

    save_results(files, project_path, verbose)


def save_spr_results_lean(name, project_path, verbose=False):
    """Save spr swarm run results into original file path."""
    # TODO: reorganize in a similar way to load_spr_results
    files = dict()

    files["webp"] = "{0}.webp {0}_ctffit.webp".format(name)
    files["mrc"] = "{0}.mrc".format(name)
    files["pkl"] = "{0}.pkl".format(name)

    save_results(files, project_path,verbose)


def load_tomo_results(name, parameters, project_path, working_path, verbose):
    """Load existing results from previous runs and standard project parameter files
    into working_path."""

    initial_files = [
        "raw/{0}.rawtlt",
        "webp/{0}.webp",
        "webp/{0}_rec.webp",
        "webp/{0}_rec.png",
        "webp/{0}_ali.webp",
        "webp/{0}_sides.webp",
        "webp/{0}_raw.webp",
        "mrc/{0}.mrc",
        "mrc/{0}.rec",
        "next/{0}.next",
        "next/{0}_exclude_views.next",
        "next/virion_thresholds.next",
        "pkl/{0}.pkl",
    ]

    # no need to transfer composed tilt-series if re-doing frame alignment
    if 'movie_force' in parameters and parameters['movie_force']:
        initial_files.remove("mrc/{0}.mrc")
        initial_files.remove("mrc/{0}.rec")

    # no need to transfer tomogram if re-doing reconstruction
    elif 'tomo_rec_force' in parameters and parameters['tomo_rec_force']:
        initial_files.remove("mrc/{0}.rec")

    if parameters.get("tomo_ali_method") == "import" and os.path.exists(project_params.resolve_path(parameters["tomo_ali_import"])):
        # cp .tlt .xf 
        raw_tlt_file = name + ".rawtlt"
        tlt_file = name + ".tlt"
        xf_file = name + ".xf"
        external_tlt = os.path.join(project_params.resolve_path(parameters["tomo_ali_import"]), tlt_file)
        if os.path.exists(external_tlt):
            logger.info(f"Import tilt-angles from: {external_tlt}")
            shutil.copy2(external_tlt, working_path)
            # also copy as .rawtlt
            shutil.copy2(external_tlt, os.path.join(working_path,raw_tlt_file))
        else:
            logger.warning("No corresponding .tlt file from import path found for this tilt-series")

        external_xf = os.path.join(project_params.resolve_path(parameters["tomo_ali_import"]), xf_file)
        if os.path.exists(external_xf):
            logger.info(f"Import tilt-series alignments from: {external_xf}")
            shutil.copy2(external_xf, working_path)
        else:
            logger.warning("No corresponding .xf file from import path found for this tilt-series")

        # read metadata and save into pickle file
        data = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=False)
        data.loadFiles()

    if "tomo_spk_files" in parameters:
        if os.path.exists(project_params.resolve_path(parameters["tomo_spk_files"])):
            external_spk_file = os.path.join(
                project_params.resolve_path(parameters["tomo_spk_files"]), name + ".spk"
            )
            if os.path.exists(external_spk_file):
                shutil.copy2(external_spk_file, working_path)
            else:
                logger.warning("No particle coordinates from import path found for this tilt-series")
        else:
            logger.warning("Specified path for particle coordinates does not exist")

    file_patterns = [
        f"sva/{name}_vir????_cut.txt",
        f"raw/{name}.order",  # retrieve order file for dose weighting
        f"mrc/{name}_vir????_binned_nad.mrc",
        f"mrc/{name}_vir????_binned_nad_seg.mrc"
    ]

    filelist = []
    project_path_escape = Path(glob.escape(str(project_path)))
    filelist += (str(project_path / f.format(name)) for f in initial_files)
    filelist += flatten(sorted(glob.glob(str(project_path_escape / f))) for f in file_patterns)

    load_results(filelist, project_path, working_path, verbose)

    if "extract_files" in parameters.keys():
        spk_path = project_params.resolve_path(parameters["extract_files"])
        if os.path.exists(spk_path):
            spk_file = os.path.join(spk_path, name + ".spk")
            dest = os.path.join(project_path, "mod", name + ".spk")
            if os.path.exists(spk_file) and not os.path.exists(dest):
                print("Coping ", spk_file, " to ", dest)
                shutil.copy2(spk_file, dest)


def save_tomo_results(name, parameters, current_path, verbose=False):
    """Save tomo swarm run results into original file path."""
    # TODO: follow sprswarm -- refactor to function
    files = dict()

    files[
        "mrc"
    ] = "{0}.rec {0}_bin.mrc {0}_bin.ali {0}_vir????_binned_nad.mrc {0}_vir????_ccc_0.vtp {0}_vir????_binned_nad_seg.mrc".format(
        name
    )

    if parameters["movie_no_frames"] and os.path.exists(os.path.join(current_path,"raw",name+".mrc")) and not os.path.exists(os.path.join(current_path,"mrc",name+".mrc")):
        symlink_relative(
            os.path.join(current_path,"raw",name+".mrc"),
            os.path.join(current_path,"mrc",name+".mrc")
        )
    else:
        files["mrc"] = "{0}.mrc ".format(name) + files["mrc"]
    files[
        "webp"
    ] = "{0}_view.webp {0}_?D_ctftilt.webp {0}_raw.webp {0}_ali.webp {0}_sides.webp {0}_rec.webp {0}_vir????_binned_nad.webp".format(
        name
    )
    files[
        "sva"
    ] = "{0}_region_*.rec {0}_spk????.rec {0}_vir????_spk????.mrc {0}_vir????.txt {0}_vir????_cut.txt {0}_spk????.proj".format(
        name
    )
    files["pkl"] = "{0}.pkl".format(name)
    files["raw"] = "{0}.rawtlt {0}.order".format(name)

    save_results(files, current_path, verbose)


def save_tomo_results_lean(name, parameters, current_path, verbose):
    """Save tomo swarm run results into original file path."""
    # TODO: follow sprswarm -- refactor to function
    files = dict()

    files[
        "mrc"
    ] = "{0}.rec {0}_bin.ali {0}_vir????_binned_nad.mrc {0}_vir????_ccc_0.vtp {0}_vir????_binned_nad_seg.mrc".format(
        name
    )

    if parameters["movie_no_frames"] and os.path.exists(os.path.join(current_path,"raw",name+".mrc")) and not os.path.exists(os.path.join(current_path,"mrc",name+".mrc")):
        symlink_relative(
            os.path.join(current_path,"raw",name+".mrc"),
            os.path.join(current_path,"mrc",name+".mrc")
        )
    else:
        files["mrc"] = "{0}.mrc ".format(name) + files["mrc"]

    files[
        "webp"
    ] = "{0}.webp {0}_?D_ctftilt.webp {0}_raw.webp {0}_ali.webp {0}_sides.webp {0}_rec.webp {0}_rec.png {0}_vir????_binned_nad.webp".format(
        name
    )
    files[
        "sva"
    ] = "{0}_region_*.rec {0}_spk????.rec {0}_vir????_spk????.mrc {0}_vir????.txt {0}_vir????_cut.txt {0}_spk????.proj {0}_vir0000.rec".format(
        name
    )
    if parameters.get("tomo_ext_coords"):
        files["sva"] += " {0}.spk".format(name)

    files["pkl"] = "{0}.pkl".format(name)

    save_results(files, current_path, verbose=verbose)


def load_csp_results(name, parameters, project_path, working_path, verbose=False):
    """Load existing results from previous runs and standard project parameter files
    into working_path."""

    initial_files = [
        "ctf/{0}.ctf",
    ]

    file_patterns = [
        "box/{0}.box",  # needed by trajectory plotting after regularization 
        "box/{0}.boxx",
        "csp/{0}.allboxes",
        "csp/{0}_local.allboxes",
        "csp/{0}_boxes3d.txt"
    ]

    if "local" in parameters["extract_fmt"].lower():
        file_patterns.append(f"ali/{name}_*.xf")

    filelist = []
    project_path_escape = Path(glob.escape(str(project_path)))
    filelist += (str(project_path / f.format(name)) for f in initial_files)
    filelist += flatten(
        sorted(glob.glob(str(project_path_escape / f.format(name)))) for f in file_patterns
    )

    load_results(filelist, project_path, working_path, verbose=verbose)

    # this is not always strictly needed
    load_tomo_results(name, parameters, project_path, working_path, verbose=verbose)


def save_csp_results(name, parameters, current_path, verbose=False):
    """Save sp swarm run results into original file path."""
    # TODO: follow sprswarm -- refactor to function
    files = dict()
    # iteration = parameters["refine_iter"]
    # if iteration == 2:
    #     files[
    #     "csp"
    #     ] = "{0}.allboxes {0}_local.allboxes".format(name)
    # else:
    #     files[
    #         "csp"
    #     ] = "{0}.allboxes {0}_local.allboxes".format(name)
    files["csp"] = " {0}_local.webp {0}_*_P0000_combined.webp".format(name)

    save_results(files, current_path, verbose=verbose)
