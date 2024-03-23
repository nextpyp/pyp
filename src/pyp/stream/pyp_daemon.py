import csv
import datetime
import glob
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from traceback import FrameSummary

import numpy as np
from pyp.streampyp.web import Web
import pyp.streampyp.metadb_daemon
from pyp.analysis import plot
from pyp.inout.image import compress_and_delete
from pyp.inout.image import digital_micrograph as dm4
from pyp.system import project_params, slurm, user_comm, set_up
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_pyp_configuration, run_pyp
from pyp.system.utils import needs_gpu, get_gpu_queue
from pyp.utils import get_relative_path, movie2regex, timer
from pyp.inout.metadata import pyp_metadata

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def pyp_daemon(args):
    if "refine_model" in args.keys() and args["refine_model"] != None:
        print(args["refine_model"])
        model = project_params.resolve_path(args["refine_model"])
    else:
        model = None
    camera = args["stream_camera_profile"]
    scope = args["stream_scope_profile"]
    session = args["stream_session_name"]
    if "detect_rad" in args.keys():
        particle_rad = args["detect_rad"]
    else:
        particle_rad = None
    scope_pixel = args["scope_pixel"]

    config = get_pyp_configuration()

    # destination of raw data
    output_dir = project_params.resolve_path(args["stream_transfer_target"])
    if output_dir and os.path.exists(output_dir):
        data_dir = output_dir
    else:
        data_dir = config["stream"]["target"]
    session_dir = os.path.join(
        data_dir, args["stream_session_group"], args["stream_session_name"]
    )
    raw_dir = os.path.join(session_dir, "raw")
    swarm_dir = os.path.join(session_dir, "swarm")

    # movies = frames > 1

    # retrieve PYP parameters
    pyp_config_file = os.path.join(session_dir, ".pyp_config.toml")

    start_flag = os.path.join(session_dir, "pypd.start")
    stop_flag = os.path.join(session_dir, "pypd.stop")
    clear_flag = os.path.join(session_dir, "pypd.clear")
    restart_flag = os.path.join(session_dir, "pypd.restart")

    # populate directory structure
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(swarm_dir).mkdir(parents=True, exist_ok=True)

    # clean the flag in the beginning 
    [os.remove(f) for f in [stop_flag, restart_flag, clear_flag] if os.path.exists(f)]

    if not os.path.isfile(pyp_config_file):

        # load and set default PYP parameters
        parameters = args.copy()

        parameters["data_bin"] = 1
        parameters["movie_bin"] = 1
        parameters["movie_iters"] = 5
        parameters["detect_rad"] = 0
        if model and os.path.exists(model):
            parameters["extract_fmt"] = "frealign"
            parameters["refine_model"] = model
        else:
            parameters["extract_fmt"] = ""
        parameters["class_num"] = 0

        if scope in config["stream"].keys():
            if camera in config["stream"][scope].keys():
                if "gain_flipv" in config["stream"][scope][camera].keys():
                    parameters["gain_flipv"] = config["stream"][scope][camera][
                        "gain_flipv"
                    ]
                if "gain_fliph" in config["stream"][scope][camera].keys():
                    parameters["gain_fliph"] = config["stream"][scope][camera][
                        "gain_fliph"
                    ]
                if "gain_rotation" in config["stream"][scope][camera].keys():
                    parameters["gain_rotation"] = config["stream"][scope][camera][
                        "gain_rotation"
                    ]

        parameters["ctf_use_phs"] = False
        parameters["slurm_tasks"] = 7
        parameters["movie_ali"] = "unblur"
        parameters["detect_method"] = "auto"
        parameters["movie_weights"] = True
        parameters["ctf_max_res"] = 3.5
        parameters["extract_bin"] = 4
        parameters["class_bin"] = 1
        parameters["ctf_use_ast"] = True

        # Set parameters for tomography
        if "tomo" in args["data_mode"].lower():
            parameters["tomo_ali_fiducial"] = args["tomo_ali_fiducial"]
            parameters["data_mode"] = "tomo"
            parameters["extract_bin"] = 1
            parameters["ctf_min_res"] = 50
            parameters["ctf_max_res"] = 7.5
    else:
        # load and set default PYP parameters
        parameters = project_params.load_pyp_parameters(session_dir)

    parameters["data_set"] = session

    # run only one job per node
    if model != None and os.path.isfile(model):
        shutil.copy2(model, session_dir + "/frealign/" + session + "_01.mrc")
    if not particle_rad == None:
        parameters["detect_rad"] = particle_rad
        parameters["detect_method"] = args["detect_method"]
        radius = float(parameters["detect_rad"])
        if "extract_bin" in parameters.keys():
            if parameters["extract_bin"] == 0:
                boxsize = int(
                    4 * float(radius) / float(scope_pixel)
                )
            else:
                boxsize = int(
                    4 * float(radius) / float(scope_pixel) / float(parameters["extract_bin"])
                )
            if boxsize % 2 > 0:
                boxsize += 1

    project_params.save_pyp_parameters(parameters, session_dir)

    # set write access to group
    run_shell_command(
        "chmod g+w '%s'" % pyp_config_file, verbose=False
    )

    # SKIP: launch incremental 2D classification
    if True:
        # launch 2D classification and 3D refinement daemons
        if args["data_mode"] == "spr": # if "extract_box" in args.keys() and args["extract_box"] > 0:

            set_up.prepare_spr_daemon_dir()

            arguments = " -mode 1 -cutoff .75 -metric cc3m -maxiter 8 -daemon={0} -mindef 10000 -maxdef 35000".format(
                args["class2d_min"]
            )

            if args["slurm_class2d_queue"] == "None":
                queue = ""
            else:
                queue = args["slurm_class2d_queue"]

            slurm.submit_jobs(
                os.path.join( session_dir, "class2d"),
                run_pyp(command="fyp" + arguments, script=True),
                jobtype="sess_ref",
                jobname="pyp_sess_ref",
                queue=queue,
                scratch=0,
                threads=parameters["slurm_class2d_tasks"],
                memory=parameters["slurm_class2d_memory"],
                gres=parameters["slurm_class2d_gres"],
                account=parameters.get("slurm_class2d_account"),
                walltime=parameters["slurm_class2d_walltime"],
            )

    alreadysubmitted = []

    # raise start flag
    Path(start_flag).touch()

    # reset stop flag
    if os.path.exists(stop_flag):
        os.remove(stop_flag)

    logger.info("Entering loop %s", raw_dir)
    # Look for new and unprocesed data for a maximum of timeout days

    daemon_start_time = time.time()
    current_count = 0

    import toml

    while time.time() - daemon_start_time < datetime.timedelta(
        days=args["stream_session_timeout"]
    ).total_seconds() and not os.path.exists(stop_flag):

        restart_or_clean = False

        if os.path.exists(restart_flag):
            logger.info("Restart flag detected")

            # filelist = glob.glob( os.path.join( session_dir, "webp", "*.webp" ) ) 

            namelist = [os.path.basename(file).replace(".pkl", "") for file in glob.glob( os.path.join( session_dir, "pkl", "*.pkl" ) )]

            previous_parameters = parameters

            nextpyp_saved = "pypd.restart"
            new_parameters = project_params.load_parameters(path=session_dir, param_file_name=nextpyp_saved)

            # restart won't consider data path and data mode parameters which should reset whole session
            new_parameters["data_path"] = previous_parameters["data_path"]
            if "data_mode" not in new_parameters.keys():
                new_parameters["data_mode"] = previous_parameters["data_mode"] 
            if "data_set" not in new_parameters.keys():
                new_parameters["data_set"] = previous_parameters["data_set"] 

            # read specification file
            specifications = toml.load("/opt/pyp/config/pyp_config.toml")
            # figure out which parameters need to be added and set as default values
            for t in specifications["tabs"].keys():
                if not t.startswith("_"):
                    for p in specifications["tabs"][t].keys():
                        #if "copyToNewBlock" in specifications["tabs"][t][p] and not specifications["tabs"][t][p]["copyToNewBlock"]:
                        if f"{t}_{p}" not in new_parameters:
                            if "default" in specifications["tabs"][t][p]:
                                new_parameters[f"{t}_{p}"] = specifications["tabs"][t][p]["default"]

            parameters = project_params.parameter_force_check(previous_parameters, new_parameters, project_dir=session_dir)

            # update args
            for key in { k for k in args.keys() & parameters.keys() if args[k] != parameters[k] }:
                args[key] = parameters[key]

            if (
                parameters["movie_force"] or
                parameters["ctf_force"] or
                parameters["detect_force"] or
                parameters["tomo_rec_force"] or
                parameters["tomo_vir_force"] or
                parameters["tomo_ali_force"]
            ):
                # there is no longer need to clean the metadata since we are relying on _force parameters now
                # clean_pkl_items(parameters, namelist, session_dir)

                # create a flag for fyp restart
                Path(os.path.join(session_dir, "fypd.restart")).touch()

                alreadysubmitted = []

                # remove corresponding image files so refinement daemon can keep track of images that are ready to be processed
                if "spr" in args["data_mode"]:
                    [ os.remove(f) for f in glob.glob( os.path.join( session_dir, "mrc", "*.mrc" ) ) ]
                else:
                    [ os.remove(f) for f in glob.glob( os.path.join( session_dir, "mrc", "*.rec" ) ) ]

                restart_or_clean = True
            else:
                logger.info("Nothing changed in parameters, history files won't change")

            try:
                os.remove(restart_flag)
            except:
                logger.info("Cannot remove restart flag")


        if os.path.exists(clear_flag):
            logger.info("Clear flag detected")

            restart_or_clean = True

            alreadysubmitted = []
            previous_parameters = parameters
            nextpyp_saved = "pypd.clear"
            new_parameters = project_params.load_parameters(path=session_dir, param_file_name=nextpyp_saved)

            new_parameters["data_path"] = previous_parameters["data_path"]
            if "data_mode" not in new_parameters.keys():
                new_parameters["data_mode"] = previous_parameters["data_mode"] 
            if "data_set" not in new_parameters.keys():
                new_parameters["data_set"] = previous_parameters["data_set"] 

            # read specification file
            specifications = toml.load("/opt/pyp/config/pyp_config.toml")

            # figure out which parameters need to be added and set as default values
            for t in specifications["tabs"].keys():
                if not t.startswith("_"):
                    for p in specifications["tabs"][t].keys():
                        #if "copyToNewBlock" in specifications["tabs"][t][p] and not specifications["tabs"][t][p]["copyToNewBlock"]:
                        if f"{t}_{p}" not in new_parameters:
                            if "default" in specifications["tabs"][t][p]:
                                new_parameters[f"{t}_{p}"] = specifications["tabs"][t][p]["default"]

            different_values = {k for k in previous_parameters.keys() & new_parameters.keys() if previous_parameters[k] != new_parameters[k]}

            # path type re-evaluation
            if "gain_reference" in different_values:
                if project_params.resolve_path(parameters["gain_reference"]) == project_params.resolve_path(new_parameters["gain_reference"]):
                    different_values.remove("gain_reference")

            for key in different_values:
                parameters[key] = new_parameters[key]
                args[key] = new_parameters[key]
            try:
                filelist = glob.glob( os.path.join( session_dir, "ctf", "*.*" ) )
                filelist += glob.glob( os.path.join( session_dir, "mrc", "*.*" ) )
                filelist += glob.glob( os.path.join( session_dir, "pkl", "*.*" ) )
                # filelist += glob.glob( os.path.join( session_dir, "webp", "*.*" ) )
                # filelist += glob.glob( os.path.join( session_dir, "frealign", "maps", "*.*" ) )
                filelist += glob.glob( os.path.join( session_dir, "csp", "*.*" ) )
                filelist += glob.glob( os.path.join( session_dir, "sva", "*.*" ) )
                filelist += glob.glob( os.path.join( session_dir, "tomo", "*.*" ) )
                [ os.remove(f) for f in filelist ]
                os.remove(clear_flag)
                logger.info(f"Removed {len(filelist)} file(s)")

            except:
                logger.info("Cannot remove clear file")

            # create a flag for fyp restart
            Path(os.path.join(session_dir, "fypd.clear")).touch()

        # turn off "_force" parameters after cleaning/restart so next time won't trigger unnecessary procedure
        if restart_or_clean:
            '''
            for key in args:
                if "_force" in key and any(
                    ["ctf" in key, "drift" in key, "detect" in key, "ali" in key, "tomo_vir" in key, ]
                    ):
                    args[key] = False
                    parameters[key] = False
            '''
            project_params.save_parameters(args, path=session_dir)

        # collect existing images
        if not Web.exists:
            work_dir = os.getcwd()
            os.chdir(session_dir)

            # produce diagnostic plot
            try:
                plot.plot_dataset(parameters, current_count, work_dir, session_dir)
            except:
                pass

        tobesubmitted = []

        if True:
            # find list of un-processed data
            if parameters["movie_mdoc"] and "data_path_mdoc" in parameters and len(parameters["data_path_mdoc"]) > 0 and Path(project_params.resolve_path(parameters["data_path_mdoc"])).parents[0].exists():
                all_files = [ Path(s).name for s in glob.glob( os.path.join( session_dir, "raw", "*" + Path(project_params.resolve_path(parameters["data_path_mdoc"])).suffix ) ) ]
            else:
                all_files = [ Path(s).stem for s in glob.glob( os.path.join( session_dir, "raw", "*" + Path(project_params.resolve_path(parameters["data_path"])).suffix ) ) ]
            for f in all_files:
                if args["gain_reference"]:
                    isgain = f == Path(project_params.resolve_path(args["gain_reference"])).stem or "gain" in f.lower()
                else:
                    isgain = False

                if not isgain:

                    # check if transfer complete
                    condition = True

                    if parameters["movie_mdoc"]:
                        from pyp.preprocess import frames_from_mdoc
                        fileset = frames_from_mdoc([os.path.join( session_dir, "raw",f)], parameters)
                        for tilt in fileset:
                            # check if file finished transferring
                            signal_file = "%s/.%s" % (
                                raw_dir,
                                tilt[0],
                            )
                            if not os.path.exists(signal_file):
                                logger.warning(f"{signal_file} doesn't exist")
                                condition = False
                                break
                    else:
                        fileset = [Path(project_params.project_params.resolve_path(args["data_path"])).suffix]
                        if len(args["stream_transfer_fileset"]) > 0:
                            fileset.extend( args["stream_transfer_fileset"].split(",") )
                        for extension in fileset:
                            # check if file finished transferring
                            signal_file = "%s/.%s.%s" % (
                                raw_dir,
                                f,
                                extension.split(".")[-1],
                            )
                            if not os.path.exists(signal_file):
                                condition = False
                                break

                    if os.path.exists(os.path.join(raw_dir, f, ".tbz")) or args["stream_compress"] and os.path.exists(os.path.join(raw_dir, f, ".tif")):
                        condition = True

                    if "tomo" in args["data_mode"]:
                        # figure out tilt-series name
                        if args["movie_mdoc"]:
                            name = Path(f).stem.replace(".mrc", "")
                        elif not args["movie_no_frames"]:
                            regex = movie2regex(args["movie_pattern"].split(".")[0], filename="*")
                            r = re.compile(regex)
                            try:
                                name = re.match(r, f).group(1)
                            except:
                                raise Exception("Could not determine tilt-series name")
                        else:
                            name = f
                        # check that all files are present
                        if len(args["stream_transfer_fileset"]) == 0:
                            number_of_files = 0
                        else:
                            number_of_files = len(args["stream_transfer_fileset"].split(","))
                        if not args["movie_mdoc"]:
                            condition = len(glob.glob( os.path.join( raw_dir, "." + name + "*" ))) == args["stream_num_tilts"] * ( number_of_files + 1 )
                        condition_plus = condition and not os.path.isfile( os.path.join( session_dir, "mrc", name + ".rec" ) )
                    else:
                        name = f
                        condition_plus = condition and not os.path.isfile( os.path.join( session_dir, "mrc", name + ".mrc" ) )
                    if ( restart_or_clean or condition_plus ) and not name in tobesubmitted and not name in alreadysubmitted:
                        if args["slurm_verbose"]:
                            logger.info("Adding {0} to queue".format(name))
                        tobesubmitted.append(name)
                        if len(tobesubmitted) > 10:
                            break

                        # generate rawtlt file for this tilt-series
                        if "tomo" in args["data_mode"]:
                            tilts = args["stream_tilt_angles"].split(",") if "stream_tilt_angles" in args else 0
                            if len(tilts) > 1 and tilts != 0:
                                with open( os.path.join( raw_dir, name + ".rawtlt") ,'w') as rawtlt:
                                    for tilt in tilts:
                                        rawtlt.write( tilt + "\n" )

                            # generate order file for this tilt-series
                            orders = args["stream_tilt_order"].split(",") if "stream_tilt_order" in args else 0
                            if len(orders) > 1 and orders != 0:
                                with open( os.path.join( raw_dir, name + ".order") ,'w') as order_file:
                                    for order in orders:
                                        order_file.write( order + "\n" )

                        # cap number of jobs submitted
                        if args["slurm_bundle_size"] > 1 and len(tobesubmitted) == args["slurm_bundle_size"]:
                            break

        # Submit jobs to swarm
        if len(tobesubmitted) > 0:
            swarm_file = "{0}/swarm/pre_process_daemon_{1}.swarm".format(
                session_dir, time.time()
            )

            with open(swarm_file, "w") as f:
                if len(tobesubmitted) == 1:
                    f.write("#!/bin/bash\n")

                f.write(
                    "\n".join(
                        [
                            "export sess_img=sess_img; {0} --stream_file {1} > '{2}/log/{1}_pypd.log'".format(
                                run_pyp("pyp", script=True), s, session_dir,
                            )
                            for s in tobesubmitted
                        ]
                    )
                )

            run_shell_command("chmod u+x '{0}'".format(swarm_file),verbose=False)

            # submit jobs to batch system
            gpu = needs_gpu(parameters)
            if gpu:
                queue = get_gpu_queue(parameters)
            else:
                queue = parameters["slurm_queue"]

            id = slurm.submit_jobs(
                submit_dir=os.path.join(session_dir, "swarm"),
                command_file=swarm_file,
                jobtype="sess_img",
                jobname=args["data_mode"]+"_session",
                queue=queue,
                threads=args["slurm_tasks"],
                memory=args["slurm_memory"],
                gres=parameters["slurm_gres"],
                account=args.get("slurm_account"),
                walltime=args.get("slurm_merge_walltime"),
                tasks_per_arr=args.get("slurm_bundle_size"),
                csp_no_stacks=args.get("csp_no_stacks"),
                use_gpu=gpu,
            ).strip()

            alreadysubmitted.extend(tobesubmitted)

        time.sleep(10)

    if os.path.exists(stop_flag):
        logger.info("Stop flag detected " + stop_flag )
        try:
            os.remove(stop_flag)
            os.remove(start_flag)
        except:
            logger.info("Cannot remove stop file")
        try:
            os.remove(clear_flag)
        except:
            logger.info("No clear flag to be removed")
        try:
            os.remove(restart_flag)
        except:
            logger.info("No restart flag to be removed")


def pyp_daemon_process(args,):

    file = args["stream_file"]

    config = get_pyp_configuration()

    # destination of raw data
    output_dir = project_params.resolve_path(args["stream_transfer_target"])
    if output_dir and os.path.exists(output_dir):
        data_dir = output_dir
    else:
        data_dir = config["stream"]["target"]
    session_dir = os.path.join(
        data_dir, args["stream_session_group"], args["stream_session_name"]
    )
    raw_dir = os.path.join(session_dir, "raw")

    # movies = frames > 1

    # retrieve PYP parameters
    pyp_config_file = os.path.join(session_dir, ".pyp_config.toml")

    with timer.Timer("Copy mrc, par to scratch", text = "Setup and data compression took: {}", logger=logger.info):
        os.chdir(session_dir)

        parameters = project_params.load_pyp_parameters()

        # get file name
        name = os.path.basename(file)
        working_path = Path(os.environ["PYP_SCRATCH"]) / name
        shutil.rmtree(working_path, "True")
        try:
            os.mkdir(working_path)
        except:
            logger.info("Cannot create %s on node %s", working_path, socket.gethostname())
        os.chdir(working_path)

        # Format and store raw data
        '''
        if len( glob.glob( os.path.join( raw_dir, name + "*.tif") ) ) == 0:
            for i in glob.glob( os.path.join( raw_dir, name + "*" + parameters["stream_transfer_fileset"].split(",")[0])):
                compress_and_delete(
                    Path(i).with_suffix(""), "tif", parameters["stream_transfer_fileset"]
                )
        '''

        # extract tilt-angles from dm4 header before compressing
        data_path = Path(project_params.resolve_path(parameters["data_path"]))

        if data_path.suffix == ".dm4" and "tomo" in parameters["data_mode"].lower():
            tilt_angles = []
            for i in sorted( glob.glob( os.path.join( raw_dir, name + "*" + data_path.suffix ) ) ):
                dm = dm4.DigitalMicrographReader(i)
                tilt_angles.append( float(dm.get_tilt_angles()) )
            with open( os.path.join( raw_dir, name + ".rawtlt" ),'w' ) as f:
                for item in tilt_angles:
                    f.write("%s\n" % item)

        if 'stream_compress' in args and args['stream_compress'] != "none":
            for i in glob.glob( os.path.join( raw_dir, name + "*" + data_path.suffix ) ):
                compress_and_delete(
                    Path(i).with_suffix(""), args['stream_compress'], data_path.suffix
                )
        else:

            # simply remove signal files
            if "tomo" in parameters["data_mode"].lower():
                for fil in glob.glob( os.path.join( raw_dir, "." + name + "*") ):
                    if os.path.exists(fil):
                        os.remove(fil)
            else:
                try:
                    fil = glob.glob( os.path.join( raw_dir, "." + file + "*") )[0]
                    if os.path.exists(fil):
                        os.remove(fil)
                except:
                    pass

    # Start processing
    # make distinction between SPA and TOMO
    if not "tomo" in parameters["data_mode"].lower():
        # sprswarm
        from pyp_main import spr_swarm
        spr_swarm( session_dir, os.path.join("raw", file) )
    else:
        # tomoswarm
        from pyp_main import tomo_swarm
        tomo_swarm( session_dir, os.path.join("raw", name) )

    # clean-up
    shutil.rmtree(working_path, "True")


def clean_pkl_items(parameters, namelist, current_path):
    # clean items in pkl files for re-doing processing.
    if "spr" in parameters["data_mode"]:
        is_spr = True
    else:
        is_spr = False

    first = True
    if not is_spr:
        for name in namelist:
            # loading pkl file
            pklname = os.path.join(current_path, "pkl", name + ".pkl")
            if os.path.exists(pklname):
                metadata_object = pyp_metadata.LocalMetadata(pklname, is_spr=is_spr)
                metadata = metadata_object.data
                meta_update = False

                if "ctf_force" in parameters and parameters["ctf_force"]:
                    if first:
                        logger.info(
                            f"CTF parameters will be re-computed"
                        )
                    if "ctf" in metadata:
                        del metadata["ctf"]
                        meta_update = True

                if "tomo_ali_force" in parameters and parameters["tomo_ali_force"]:
                    if first:
                        logger.info(
                            f"Movie alignments will be re-computed"
                        )
                    if "ali" in metadata:
                        del metadata["ali"]
                        meta_update = True
                if "movie_force" in parameters and parameters["movie_force"]:
                    if first:
                        logger.info(
                            f"Movie drift parameters will be re-computed"
                        )
                    if "drift" in metadata:
                        del metadata["drift"]
                        meta_update = True
                if "tomo_vir_force" in parameters and parameters["tomo_vir_force"] or parameters["tomo_vir_method"] == "none":
                    if "vir" in metadata:
                        if first:
                            logger.info(
                                f"Virion parameters will be re-computed"
                            )
                        del metadata["vir"]
                        meta_update = True
                    [ os.remove(f) for f in glob.glob( os.path.join(current_path,"mrc",name+"_vir????_binned_nad.*") ) ]
                    [ os.remove(f) for f in glob.glob( os.path.join(current_path,"sva",name+"_vir*.*") ) ]
                if "detect_force" in parameters and parameters["detect_force"] or parameters["tomo_spk_method"] != "none":
                    if first:
                        logger.info(
                            f"Particle parameters will be re-computed"
                        )
                    if "box" in metadata:
                        del metadata["box"]
                        meta_update = True
                    [ os.remove(f) for f in glob.glob( os.path.join(current_path,"sva",name+"_vir*.*") ) ]
                    [ os.remove(f) for f in glob.glob( os.path.join(current_path,"mod",name+".spk") ) ]

                # update current pkl file
                if meta_update:
                    metadata_object.write()

                first = False

    else:
        for name in namelist:
            # loading pkl file
            pklname = os.path.join(current_path, "pkl", name + ".pkl")
            if os.path.exists(pklname):
                metadata_object = pyp_metadata.LocalMetadata(pklname, is_spr=is_spr)
                metadata = metadata_object.data
                meta_update = False

                if "ctf_force" in parameters and parameters["ctf_force"]:
                    if first:
                        logger.info(
                            f"CTF parameters will be re-computed"
                        )
                    if "ctf" in metadata:
                        del metadata["ctf"]
                        meta_update = True

                if "movie_force" in parameters and parameters["movie_force"]:
                    if first:
                        logger.info(
                            f"Movie drift parameters will be re-computed"
                        )
                    if "drift" in metadata:
                        del metadata["drift"]
                        meta_update = True

                if "detect_force" in parameters and parameters["detect_force"]:
                    if first:
                        logger.info(
                            f"Particle parameters will be re-computed"
                        )
                    if "box" in metadata:
                        del metadata["box"]
                        meta_update = True

                # update current pkl file
                if meta_update:
                    metadata_object.write()

                first = False


