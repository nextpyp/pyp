import csv
import glob
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
from pyp.system.utils import get_imod_path
from pyp.utils import flatten, symlink_relative
from pyp.inout.metadata import pyp_metadata

from pyp.system.logging import logger

def parameters_type_check(parameters):
    """
    Ensures that the types of the given parameters match the specifications defined in the configuration file.
    This function reads a TOML configuration file to retrieve the expected types for parameters. 
    It then checks the types of the provided parameters and converts them to the expected types if necessary. Supported types for conversion are 'int', 'float', and 'str'.
    Args:
        parameters (dict): A dictionary of parameters where keys are parameter names and values are their corresponding values.
    Returns:
        dict: The updated dictionary of parameters with corrected types where applicable.
    Notes:
        - The configuration file is expected to be located at "/opt/pyp/config/pyp_config.toml".
        - The configuration file should define parameter types under the "tabs" section.
        - Parameters with keys starting with "_" are ignored.
        - If a parameter's type does not match the expected type, it is converted and a trace log is generated.
    Raises:
        toml.TomlDecodeError: If the configuration file cannot be parsed.
        KeyError: If the expected structure of the configuration file is not met.
        ValueError: If a parameter cannot be converted to the expected type.
    """
    
    # read specification file
    import toml
    specifications = toml.load("/opt/pyp/config/pyp_config.toml")

    # make sure parameter types are up to date
    for t in specifications["tabs"].keys():
        if not t.startswith("_"):
            for p in specifications["tabs"][t].keys():
                if not p.startswith("_"):
                    ptype = specifications["tabs"][t][p]["type"]
                    if ptype in {'int','float','str'} and parameters.get(f"{t}_{p}") and type(parameters[f"{t}_{p}"]) != eval(ptype):
                        logger.trace(f"Converting parameter {t}_{p} with value {parameters[f'{t}_{p}']} from {type(parameters[f'{t}_{p}'])} to {ptype}")
                        if ptype == "int":
                            parameters[f"{t}_{p}"] = int(parameters[f"{t}_{p}"])
                        elif ptype == "float":
                            parameters[f"{t}_{p}"] = float(parameters[f"{t}_{p}"])
                        elif ptype == "str":
                            parameters[f"{t}_{p}"] = str(parameters[f"{t}_{p}"])
    return parameters

def save_parameters_to_website(parameters):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    elif "data_set" in parameters:
        try:
            parameters = parameters_type_check(parameters)

            # actually send the micrograph to the website
            Web().write_parameters(parameter_id=parameters["data_set"], parameters=parameters)

            logger.trace("Parameters entered into database successfully")
        except:
            logger.error("Failed to enter parameters into database")
            raise
    else:
        logger.error("No data_set field specified in parameters, cannot save to website")
        return

def save_micrograph_to_website(name):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # scan the CTF info
            ctf_path = "%s.ctf" % name
            if os.path.exists(ctf_path):
                ctf = Web.CTF(*[float(x[0]) for x in csv.reader(open(ctf_path, "r"))])
            else:
                ctf = []
                logger.warning("Cannot find ctf information to submit to database")
            if len(ctf) == 0:
               ctf = None

            # scan the AVGROT info
            avgrot_path = "%s_avgrot.txt" % name
            if os.path.exists(avgrot_path):
                avgrot = [Web.AVGROT(*x) for x in np.loadtxt(avgrot_path, comments="#").T]
            else:
                avgrot = []
                logger.warning("Cannot find avgrot information to submit to database")
            if len(avgrot) == 0:
                avgrot = None

            # scan motion info
            xf_path = "%s.xf" % name
            if os.path.exists(xf_path):
                xf = [Web.XF(*x) for x in np.loadtxt(xf_path, ndmin=2)]
            else:
                xf = []
                logger.warning("Cannot find xf information to submit to database")
            if len(xf) == 0:
                xf = None
 
            # scan particles info
            boxx_path = "%s.boxx" % name
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

            logger.trace("Series %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter micrograph into database")
            pass

def save_tiltseries_to_website(name, metadata):

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

            logger.trace("Series %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter tilt-series into database")
            raise

def save_reconstruction_to_website(name, fsc, plots, metadata):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the reconstruction to the website
            Web().write_reconstruction(name, metadata, fsc, plots)

            logger.trace("Reconstruction %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter reconstruction into database")
            raise

def save_refinement_to_website(name, iteration):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the refinement to the website
            Web().write_refinement(name, iteration)
            logger.trace("Refinement %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter refinement into database")

def save_refinement_bundle_to_website(name, iteration):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the refinement to the website
            Web().write_refinement_bundle(name, iteration)

            logger.trace("Refinement bundle %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter refinement into database")

def save_classes_to_website(name, metadata):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the micrograph to the website
            Web().write_classes(name,metadata)

            logger.trace("Classes %s entered into database successfully" % name)
        except:
            logger.error("Failed to enter classes into database")
            raise

def save_drgnmap_to_website(epoch):

    # if there's no website, don't bother saving anything
    if not Web.exists:
        return
    else:
        try:
            # actually send the reconstruction to the website
            Web().write_tomo_drgn_convergence(epoch)

            logger.trace("Drgn map entered into database successfully")
        except:
            logger.error("Failed to enter Drgn map into database")
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
                logger.info("Session %s does not exist" % session.session_id)
            else:
                # (re-)create the micrograph
                micrograph = session.micrograph(name)
                micrograph.create(int(round(time.time() * 1000)))

                logger.info("Updating micrograph id %s" % name)
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


def load_spr_results(name, parameters, project_path, working_path):
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

    load_results(filelist, project_path, working_path)

    # convert to mrc
    if os.path.exists("{0}.tif".format(name)):
        command = "{0}/bin/newstack -quiet '{1}.tif' '{1}.mrc'; rm -f '{1}.tif'".format(
            get_imod_path(), name
        )
        run_shell_command(command)

    # rename movie average
    if os.path.exists("{0}.mrc".format(name)):
        shutil.copy("{0}.mrc".format(name), "{0}.avg".format(name))


def save_spr_results(name, parameters, project_path, has_frames):
    """Save spr swarm run results into original file path."""
    # TODO: reorganize in a similar way to load_spr_results
    files = dict()

    files["pkl"] = [ '.pkl' ]

    # save frame averages only if raw data has frames
    if has_frames:
        files[
            "mrc"
        ] = [ '.mrc', '_DW.mrc', '_DW.tif' ]

    # files['ali'] = '{0}_xray.mod {0}.xf {0}.prexgraw {0}.ccc {0}.blr {0}.mrc {0}_weights.txt {0}_P????_frames.xf {0}_P????_frames_ccc.png {0}_P????_frames.blr {0}_P????_frames_frc.png {0}_frames_matches.gif {0}_P0000_frames_weights_new.png {0}_field.pdf'.format(name)
    files[
        "webp"
    ] = [ '.webp', '_boxed.webp', '_ctffit.webp' ]

    save_results(name, files, project_path)


def save_spr_results_lean(name, project_path, has_frames):
    """Save spr swarm run results into original file path."""
    # TODO: reorganize in a similar way to load_spr_results
    files = dict()

    files["webp"] = [ '.webp',  '_ctffit.webp' ]

    # save frame averages only if raw data has frames
    if has_frames:
        files["mrc"] = [ '.mrc' ]
    files["pkl"] = [ '.pkl' ]

    save_results(name, files, project_path)


def load_tomo_results(name, parameters, project_path, working_path):
    """Load existing results from previous runs and standard project parameter files
    into working_path."""

    initial_files = [
        "raw/{0}.rawtlt",
        "next/{0}.next",
        "next/{0}_exclude_views.next",
        "next/virion_thresholds.next",
        "pkl/{0}.pkl",
    ]

    #  transfer tomogram files if not re-doing reconstruction
    if not ( parameters.get('movie_force') or parameters.get('tomo_rec_force') or parameters.get('tomo_ali_force') ):
        initial_files.append("mrc/{0}.rec")
        initial_files.append("mrc/{0}_seg.rec")
        initial_files.append("mrc/{0}_den.rec")
        initial_files.append("mrc/{0}_half1.rec")
        initial_files.append("mrc/{0}_half2.rec")
        initial_files.append("webp/{0}.webp")
        initial_files.append("webp/{0}_rec.webp")
        initial_files.append("webp/{0}_rec.png")
        initial_files.append("webp/{0}_sides.webp")
    
    # transfer aligned tilt-series files if not re-doing alignment
    if not ( parameters.get('movie_force') or parameters.get('tomo_ali_force') ):
        initial_files.append("mrc/{0}_bin.ali")
        initial_files.append("webp/{0}_ali.webp")
        
    # transfer composed tilt-series if not re-doing frame alignment
    if not parameters.get('movie_force'):
        initial_files.append("mrc/{0}.mrc")
        initial_files.append("webp/{0}_raw.webp")

    if parameters.get("tomo_ali_method") == "import" and os.path.exists(project_params.resolve_path(parameters["tomo_ali_import"])) and parameters.get('tomo_ali_force'):
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
            raise Exception(f"No .tlt file found for {name} in import directory")

        external_xf = os.path.join(project_params.resolve_path(parameters["tomo_ali_import"]), xf_file)
        if os.path.exists(external_xf):
            logger.info(f"Import tilt-series alignments from: {external_xf}")
            shutil.copy2(external_xf, working_path)
        else:
            raise Exception(f"No .xf file found for {name} in import directory")

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

    load_results(filelist, project_path, working_path)

    if "extract_files" in parameters.keys():
        spk_path = project_params.resolve_path(parameters["extract_files"])
        if os.path.exists(spk_path):
            spk_file = os.path.join(spk_path, name + ".spk")
            dest = os.path.join(project_path, "mod", name + ".spk")
            if os.path.exists(spk_file) and not os.path.exists(dest):
                print("Copying ", spk_file, " to ", dest)
                shutil.copy2(spk_file, dest)


def save_tomo_results(name, parameters, current_path):
    """Save tomo swarm run results into original file path."""
    # TODO: follow sprswarm -- refactor to function
    files = dict()

    files[
        "mrc"
    ] = [ '.rec', '_bin.mrc', '_bin.ali', '_vir????_binned_nad.mrc', '_vir????_ccc_0.vtp', '_vir????_binned_nad_seg.mrc', '_half1.rec', '_half2.rec' ]

    if parameters["movie_no_frames"] and os.path.exists(os.path.join(current_path,"raw",name+".mrc")) and not os.path.exists(os.path.join(current_path,"mrc",name+".mrc")):
        symlink_relative(
            os.path.join(current_path,"raw",name+".mrc"),
            os.path.join(current_path,"mrc",name+".mrc")
        )
    else:
        files["mrc"].append('.mrc')
    files[
        "webp"
    ] = [ '_view.webp', '_?D_ctftilt.webp', '_raw.webp', '_ali.webp', '_sides.webp', '_rec.webp', '_vir????_binned_nad.webp' ]

    files[
        "sva"
    ] = [ '_region_*.rec', '_spk????.rec', '_vir????_spk????.mrc', '_vir????.txt', '_vir????_cut.txt', '_spk????.proj' ]

    files["pkl"] = [ '.pkl' ]
    if parameters.get("tomo_ali_export"):
        files["ali"] = [ '.tlt', '.xf' ]
    files["raw"] = [ '.rawtlt', '.order' ]

    save_results(name, files, current_path)


def save_tomo_results_lean(name, parameters, current_path):
    """Save tomo swarm run results into original file path."""
    # TODO: follow sprswarm -- refactor to function
    files = dict()

    files[
        "mrc"
    ] = [ '.rec', '_bin.ali', '_half1.rec', '_half2.rec' ]

    if parameters["movie_no_frames"] and os.path.exists(os.path.join(current_path,"raw",name+".mrc")) and not os.path.exists(os.path.join(current_path,"mrc",name+".mrc")):
        symlink_relative(
            os.path.join(current_path,"raw",name+".mrc"),
            os.path.join(current_path,"mrc",name+".mrc")
        )
    else:
        files["mrc"].append( '.mrc' )

    files[
        "webp"
    ] = [ '.webp', '_?D_ctftilt.webp', '_raw.webp', '_ali.webp', '_sides.webp', '_rec.webp', '_score.webp', '_rec.png', '_vir????_binned_nad.webp' ]
    
    # do not save virions and spikes during sessions
    if not Web.exists or len(parameters.get("micromon_block")) > 0:
        files[
            "sva"
        ] = [ '_region_*.rec', '_spk????.rec', '_vir????_spk????.mrc', '_vir????.txt', '_vir????_cut.txt', '_spk????.proj', '_vir0000.rec' ]
        
        files["mrc"] += [ '_vir????_binned_nad.mrc', '_vir????_ccc_0.vtp', '_vir????_binned_nad_seg.mrc' ]

    if parameters.get("tomo_ext_coords"):
        files["sva"].append( '.spk' )

    files["pkl"] = [ '.pkl' ]
    if parameters.get("tomo_ali_export"):
        files["ali"] = [ '.tlt', '.xf' ]

    save_results(name, files, current_path)

def load_csp_results(name, parameters, project_path, working_path):
    """Load existing results from previous runs and standard project parameter files
    into working_path."""

    initial_files = [
        "mrc/{0}.mrc",
        "pkl/{0}.pkl",
    ]

    filelist = []
    project_path_escape = Path(glob.escape(str(project_path)))
    filelist += (str(project_path_escape / f.format(name)) for f in initial_files)

    load_results(filelist, project_path, working_path)

def save_csp_results(name, parameters, current_path):
    """Save sp swarm run results into original file path."""
    # TODO: follow sprswarm -- refactor to function
    files = dict()
    files["csp"] = [ '_local.webp',  '_*_P0000_combined.webp' ] 

    save_results(name, files, current_path)
