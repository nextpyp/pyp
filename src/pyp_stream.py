#!/usr/bin/env python

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"

import matplotlib

matplotlib.use("Agg")

import argparse
import datetime
import fnmatch
import glob
import multiprocessing
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

from pyp.system import project_params, slurm
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_pyp_configuration, run_pyp, run_slurm, run_ssh
from pyp.system.user_comm import notify
from pyp.utils import get_relative_path, movie2regex, symlink_relative

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def transfer_multiprocessing(
    remove, remove_all, f, destination, session, camera, server, results, symlinks=False, pattern="",
):
    # print 'Processing %s' % f, destination
    condition = True
    try:

        filesize = os.path.getsize(f)

        start = time.time()

        # new_name = os.path.basename(f).replace(" ", "_")
        if remove or remove_all:

            move_to_destination(file=f, path=destination, server=server)

            if remove_all and Path(f).parents[0].stem == "Stack":
                # remove frame average
                for repeat in Path(f).parents[1].rglob(Path(f).stem + "*"):
                    try:
                        os.remove(repeat)
                    except:
                        pass

                # also remove averaged tilt-series saved by Latitude from DataImages/ folder

                # figure out tilt-series name first
                regex = movie2regex(pattern, filename="*")
                r = re.compile(regex)
                name = re.match(r, Path(f).name).group(1)
                try:
                    os.remove( os.path.join( Path(f).parents[1], name + Path(f).suffix ) )
                except:
                    pass

                # remove file
                try:
                    os.remove(f)
                except:
                    pass
        elif symlinks:

            link_to_destination(file=f, path=destination, server=server)

        else:

            copy_to_destination(file=f, path=destination, server=server)

        elapsed_time = time.time() - start
        secs_to_gbits = filesize * 8.0 / 1e9
        speed = secs_to_gbits / elapsed_time

        create_in_destination(
            file="." + os.path.split(f)[-1], server=server, path=destination
        )

    except Exception as e:

        logger.info(str(e))
        if "already exists" in str(e):
            logger.info("Clearing up truncated file %s %s", f, os.path.split(f)[-1])
            os.remove(os.path.join(destination, os.path.split(f)[-1]))
        condition = False
        pass

    send_results = dict()
    send_results["name"] = f
    send_results["condition"] = condition
    if condition:
        send_results["filesize"] = filesize
        send_results["speed"] = speed
    results.put(send_results)


def is_local(server):
    return True
    return server == socket.gethostname()


def parse_arguments():

    args = project_params.parse_arguments()

    # check arguments for particle picking
    if args.refine_extract_box:
        if not args.process_detect_rad:
            logger.error("-particle_rad is also required.")
            sys.exit()

    if "tomo" in args.session_mode.lower():
        if not args.process_gold_size:
            logger.error("-gold_size is also required")
            sys.exit()
        if not args.process_frames:
            logger.error("-frames is also required (highest positive tilt-angle)")
            sys.exit()
    else:
        args.process_frames = 7

    # check arguments if using 3D refinement
    if len(str(args.refine_model)) > 1:
        # if not args.particle_rad or not args.extract_box or not args.particle_sym or not args.particle_mw:
        if not args.process_detect_rad or not args.scope_pixel:
            logger.error(
                "-particle_rad -extract_box -particle_sym and -particle_mw are required."
            )
            sys.exit()

    if args.transfer_operation == "move":
        logger.warning(
            "Files will be removed once they have been successfully transferred"
        )

    config = get_pyp_configuration()

    if (
        args.scope_profile in config["stream"].keys()
        and "voltage" in config["stream"][args.scope].keys()
        or args.scope_voltage != 0
    ):
        pass
    else:
        logger.error("-scope_voltage is required.")
        sys.exit()

    if (
        args.scope_profile in config["stream"].keys()
        and args.scope_camera_profile in config["stream"][args.scope_profile].keys()
        and "pixel" in config["stream"][args.scope][args.scope_camera_profile].keys()
        or args.scope_pixel != 0
    ):
        pass
    else:
        logger.error("-scope_pixel is required.")
        sys.exit()

    if not args.transfer_source:
        if not args.scope_profile in config["stream"].keys():
            logger.error("Scope type must be properly configured.")
            sys.exit()

        if not args.scope_camera_profile in config["stream"][args.scope_profile].keys():
            logger.error("Camera type must be properly configured.")
            sys.exit()

    return args


def find_gain_reference(args, soures):

    # Latitude
    if len(glob.glob(sources.split(",")[0] + "/CameraSetup/*Gain*")) > 0:

        pattern = sources.split(",")[0] + "/CameraSetup/*Gain*"

    # SerialEM
    elif len(glob.glob(sources.split(",")[0] + "/Stack/Count*")) > 0:

        pattern = sources.split(",")[0] + "/Stack/Count*"

    # Other
    elif len(glob.glob(sources.split(",")[0] + "/Count*")) > 0:

        pattern = sources.split(",")[0] + "/Count*"

    else:

        logger.error("Cannot find gain reference file.")
        pattern = ""

    return pattern


def get_data_files(args, source):

    files = glob.glob(project_params.resolve_path(args["data_path"]))
    if args["data_path_mdoc"] != None:
        files = glob.glob(project_params.resolve_path(args["data_path_mdoc"])) + files
    if len(args["stream_transfer_fileset"]) > 0:
        original_files = files.copy()
        for extension in args["stream_transfer_fileset"].split(","):
            for f in original_files:
                files.append( f.replace( Path(f).suffix, extension ) )
    if False:
        if not args["data_path"] and "k3" in args["stream_camera_profile"]:

            if "tomo" in args["stream_session_mode"].lower():
                files = sorted(glob.glob(os.path.join(source, "Stack/*_*_??_*.tif*")))
            else:
                files = sorted(
                    glob.glob(os.path.join(source, "DataImages/Stack/*.dm4"))
                    + glob.glob(os.path.join(source, "Stack/*.tif*"))
                    + glob.glob(os.path.join(source, "*.tif*"))
                )
        elif args["data_path"]:
            files = sorted(
                glob.glob(os.path.join(args["data_path"], "*_*_??_*.tif*"))
                + glob.glob(os.path.join(args["data_path"], "*.tif*"))
                + glob.glob(os.path.join(args["data_path"], "*.dm4"))
            )
            files = sorted(
                glob.glob(os.path.join(args["data_path"], args["stream_transfer_fileset"]))
            )
        else:

            files = []
            for source in sources.split(","):
                for dirpath, dirnames, f in os.walk(source):
                    for f in fnmatch.filter(f, "FoilHole_*_Data_*.*"):
                        name = os.path.join(dirpath, f)
                        files.append(name)
                    for f in fnmatch.filter(f, "*.dm4"):
                        if "Stack" in dirpath:
                            name = os.path.join(dirpath, f)
                            files.append(name)
            files = sorted(files, reverse=True)

    # filter out gain reference and metadata
    clean_files = [
        f for f in files if not "gain" in f.lower() and f != args["gain_reference"]
    ]
    return clean_files


def get_target(file, path):

    # extract file name
    filename = os.path.split(file)[-1]

    # replace spaces with underscores
    filename = filename.replace(" ", "_")

    # form absolute target
    target = os.path.join(path, filename)

    return target


def is_image(file):

    return Path(file).suffix in [".tif", ".tiff", ".mrc", ".dm4", ".eer"]


def move_to_destination(file, server, path):

    target = get_target(file=file, path=path)

    if not is_image(file):
        return

    if file == target:
        logger.warning(f"Source and destination files coincide: {file} != {target}, skipping")
    else:
        if is_local(server):
            shutil.move(file, target)
        else:
            com = "scp -p {0} {1}:{2}".format(file, server, target)
            run_shell_command(com)
            os.remove(file)


def copy_to_destination(file, server, path, verbose=False):

    target = get_target(file=file, path=path)

    # logger.info('Copying ' + file + ' to ' + target )
    if is_image(file):
        if verbose:
            logger.info("Copying " + Path(file).name)

    if is_local(server):
        shutil.copy2(file, target)
    else:
        com = "scp -p '{0}' '{1}:{2}'".format(file, server, target)
        [output, error] = run_shell_command(com,verbose)


def link_to_destination(file, server, path, verbose=False):

    target = get_target(file=file, path=path)

    try:
        if verbose:
            logger.info("Linking " + file + " to " + target)
        symlink_relative(file, target)
    except:
        if verbose:
            if os.path.exists(target):
                logger.warning(f"{target} already exists")
            else:
                logger.error(f"Cannot create {target}")
            pass


def create_in_destination(file, server, path):

    target = get_target(file=file, path=path)

    # logger.info('Signaling ' + target )

    if is_local(server):
        with open(target, "w") as f:
            f.write(" ")

    else:

        com = "ssh {0} touch {1}".format(server, target)
        run_shell_command(com)


def remove_from_destination(file, server, path):

    target = get_target(file=file, path=path)

    if is_local(server):
        if os.path.exists(target):
            os.remove(target)
    else:
        com = "ssh {0} rm -f {1}".format(server, target)
        run_shell_command(com)


def create_paths(server, path):

    # we stay local
    if is_local(server):

        # create destination if doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)

        # set permissions
        com = "chmod g+w '{1}'".format(server, path)
        run_shell_command(com)

    # do over ssh
    else:

        com = "ssh {0} mkdir -p '{1}'".format(server, path)
        [output, error] = run_shell_command(com)

        # set permissions
        com = "ssh {0} chmod g+w '{1}'".format(server, path)
        run_shell_command(com)


def launch_preprocessing(args, autoprocess):

    if not args["stream_transfer_local"]:

        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        swarm_file = os.path.join(
            autoprocess,
            "{0}_{1}_{2}_daemon.swarm".format(
                time_stamp, args["stream_session_group"], args["stream_session_name"]
            ),
        )

        with open(swarm_file, "w") as f:

            f.write("#!/bin/bash\n")
            f.write(
                "#SBATCH --output=\"%s\"\n"
                % (os.path.join(target_path, swarm_file.replace(".swarm", ".out")))
            )
            f.write(
                "#SBATCH --error=\"%s\"\n"
                % (os.path.join(target_path, swarm_file.replace(".swarm", ".err")))
            )

            # just run pyp command
            pyp_command = "export pypdaemon=pypdaemon; {0} --data_mode {1}".format(
                run_pyp("pyp", script=True), args["data_mode"]
            )

            # f.write("{0} > {1}\n".format( pyp_command, os.path.join( target_path, swarm_file.replace('.swarm','.log') ) ) )
            f.write("{0}\n".format(pyp_command))

        run_shell_command(f"chmod u+x '{swarm_file}'")

        # transfer swarm file to remote server
        move_to_destination(file=swarm_file, server=server, path=target_path)

        # retrieve configuration
        config = get_pyp_configuration()

        # submit pre-processing
        if args["stream_camera_profile"] != "None":
            jobname = (
                args["stream_camera_profile"] + "_" + args["stream_session_name"][-4:]
            )
        else:
            jobname = args["stream_session_name"][-8:]

        if "slurm_daemon_queue" not in args or args["slurm_daemon_queue"] == "None":
            queue = ""
        else:
            queue = args["slurm_daemon_queue"]

        jobnumber = slurm.submit_jobs(
            target_path,
            swarm_file,
            jobtype="sess_pre",
            jobname="pyp_sess_pre",
            queue=queue,
            scratch=0,
            threads=args["slurm_daemon_tasks"],
            memory=args["slurm_daemon_memory"],
            gres=args["slurm_daemon_gres"],
            account=args.get("slurm_daemon_account"),
            walltime=args["slurm_daemon_walltime"],
            tasks_per_arr=1,
        )

        message = "none"

        if args["stream_transfer_remote"]:
            logger.info("Running daemon only")
            sys.exit()
    else:
        pyp_command = "none"
        jobnumber = "none"
        message = "none"

    return pyp_command, jobnumber, message


def resolve_sources(args):

    resolved_sources = ""

    if not args["data_path"]:
        for source in config["stream"][args["stream_scope_profile"]][
            args["stream_camera_profile"]
        ]["path"].split(","):

            logger.info(
                "Finding %s under %s" % (args["stream_session_name"], source)
            )

            pattern = "%s/**/%s" % (source, args["stream_session_name"])

            integrated = glob.glob(pattern, recursive=True)

            if len(integrated):
                resolved_sources += integrated[0]

        if len(resolved_sources):
            logger.info("Found " + resolved_sources)
        else:
            logger.warning("No matches found.")

    else:
        resolved_sources = project_params.resolve_path(args["data_path"])

    return resolved_sources


if __name__ == "__main__":

    # load existing parameters or from data_parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data_mode", "--data_mode")
    args, unknown = parser.parse_known_args()
    parent_parameters = vars(args)

    # parse arguments
    args = project_params.parse_parameters(0,"stream",parent_parameters["data_mode"])

    if (
        not "stream_transfer_target" in args.keys()
        or not "stream_session_name" in args.keys()
        or not "stream_session_group" in args.keys()
    ):
        logger.error(
            "You must specify non-empty values for -stream_transfer_target, -stream_session_name and -stream_session_group"
        )
        sys.exit()

    config = get_pyp_configuration()

    # destination of raw data
    if args["stream_transfer_target"]:
        target_path = os.path.join(
            project_params.resolve_path(args["stream_transfer_target"]),
            args["stream_session_group"],
            args["stream_session_name"],
        )
    elif "target" in config["stream"].keys():
        target_path = os.path.join(
            config["stream"]["target"],
            args["stream_session_group"],
            args["stream_session_name"],
        )
    else:
        message = "Please specify a target destination."
        raise Exception(message)

    if "extract_bnd" in args.keys() and isinstance(args["extract_bnd"], dict) and "ref" in args["extract_bnd"].keys():
        args["extract_bnd"] = args["extract_box"]

    if "extract_cls" not in args.keys():
        args["extract_cls"] = 0

    # remote server
    try:
        server = "{0}@{1}".format(os.environ["USER"], config["slurm"]["host"])
    except:
        server = None
        pass

    # create necessary folders
    create_paths(server, os.path.join(target_path, "raw"))
    create_paths(server, os.path.join(target_path, "txt"))
    create_paths(server, os.path.join(target_path, "log"))

    # create start flag
    start_flag = os.path.join(target_path, "streampyp.start")
    stop_flag = os.path.join(target_path, "streampyp.stop")
    restart_flag = os.path.join(target_path, "streampyp.restart")
    clear_flag = os.path.join(target_path, "streampyp.clear")

    Path(start_flag).touch()

    project_params.save_parameters(args, target_path)

    # figure out source directories
    sources = resolve_sources(args)

    # launch processing daemon pypd
    pyp_command, jobnumber, message = launch_preprocessing(args, target_path)

    # notify user if needed
    subject = (
        "Data on "
        + args["stream_session_group"]
        + "/"
        + args["stream_session_name"]
        + " ("
        + jobnumber
        + ")"
    )
    body = (
        "*** This is an automatically generated email ***\n\n"
        + "( export pypdaemon=pypdaemon && pyp_main.py "
        + " -group "
        + args["stream_session_group"]
        + " -session "
        + args["stream_session_name"]
        + " )\n\n"
        + "( export pypdaemon=pypdaemon && pyp_main.py "
        + " -group "
        + args["stream_session_group"]
        + " -session "
        + args["stream_session_name"]
        + " -particle_rad 75 -particle_sym D3 -particle_mw 300 -extract_box 384 -thresholds 1000,2500,7500,1000 -model /data/Livlab/autoprocess_d256/GDH/GDH_OG_20140612/frealign/20140826_011259_GDH_OG_20140612_01.mrc )\n"
    )
    body = body + "\n" + pyp_command + "\n\n" + message

    notify(subject, attach="", body=body, recipient=os.environ["USER"])

    if "gain_reference" in args and args["gain_reference"] != None and os.path.exists(project_params.resolve_path(args["gain_reference"])):
        copy_to_destination(
            server=server, file=project_params.resolve_path(args["gain_reference"]), path=os.path.join(target_path, "raw")
        )

    # keep track of transferred files
    scratch = os.path.join(target_path, "txt")
    Path(scratch).mkdir(parents=True, exist_ok=True)
    transferred_filename = os.path.join(
        scratch, "{0}_filelist_transferred.txt".format(args["stream_session_name"])
    )
    filelist_filename = os.path.join(
        scratch, "{0}_filelist.txt".format(args["stream_session_name"])
    )
    transfer_filename = os.path.join(
        scratch, "{0}_filelist_transfer.txt".format(args["stream_session_name"])
    )
    elapsed_time_file = os.path.join(
        target_path, "%s_speed.txt" % args["stream_session_name"]
    )

    # restart if needed
    if args["stream_transfer_restart"]:
        logger.warning("Restarting data transfer")
        for f in [transferred_filename, filelist_filename, transfer_filename]:
            try:
                if args["slurm_verbose"]:
                    logger.info("Deleting " + Path(f).name)
                os.remove(f)
            except:
                if args["slurm_verbose"]:
                    logger.error("Cannot delete " + f)
                pass
        remove_from_destination(
            file="%s_speed.txt" % args["stream_session_name"],
            server=server,
            path=target_path,
        )

    # remove stop flag if needed
    if os.path.exists(stop_flag):
        logger.info("Removing /stop flag " + stop_flag)
        try:
            os.remove(stop_flag)
        except:
            pass

    # run loop until timeout
    daemon_start_time = time.time()

    logger.info("Entering loop with timeout = %i day(s)" % args["stream_session_timeout"])

    while (
        time.time() - daemon_start_time
        < datetime.timedelta(days=args["stream_session_timeout"]).total_seconds() and not os.path.exists(stop_flag)
    ):

        # keep track of data transfer speed
        if os.path.isfile(elapsed_time_file):
            elapsed_time = np.loadtxt(elapsed_time_file,ndmin=1)
        else:
            elapsed_time = np.array([])

        files = get_data_files(args, sources)

        # overwrite file list if first time
        if os.path.exists(filelist_filename):
            os.remove(filelist_filename)

        with open(filelist_filename, 'a') as f:
            for file in files:

                # add only if older than specified age (in minutes)
                if os.path.exists(file) and (
                    os.stat(file).st_mtime
                    < time.time() - args["stream_transfer_age"] * 60
                ):
                    f.write(str(file) + "\n")

        # bundle files together to improve latency
        bundle = args["slurm_bundle_size"]

        # figure out remaining files
        if os.path.exists(transferred_filename):

            # already transferred files
            with open(transferred_filename) as t:
                words1 = set(t.read().split("\n"))

            # full list of files
            with open(filelist_filename) as t:
                words2 = set(t.read().split("\n"))

            # figure out files that need transferring
            uniques = [i for i in words2 if not i in words1]
            names = [os.path.split(f)[-1] for f in uniques]

            indices = [
                index
                for index, value in sorted(
                    enumerate(names), reverse=False, key=lambda x: x[1]
                )
                if len(value) > 0
            ][:bundle]

            with open(transfer_filename, "w") as t:
                if len(uniques) > 0:
                    for item in indices:
                        t.write(uniques[item] + "\n")
                else:
                    t.write("\n")

        elif os.path.exists(filelist_filename):

            # read list of files
            with open(filelist_filename) as f:
                files = f.read().split("\n")

            # extract movie names
            names = [Path(f).name for f in files]

            # sort based on movie name
            # print [files[index] for index, value in sorted(enumerate(names), reverse=True, key=lambda x: x[1]) if value > 1][1:bundle+1]
            # sys.exit()
            # indices = [index for index, value in sorted(enumerate(names), reverse=True, key=lambda x: x[1]) if value > 1][:bundle]

            indices = [
                index
                for index, value in sorted(
                    enumerate(names), reverse=False, key=lambda x: x[1]
                )
                if len(value) > 1
            ][0:bundle]
            with open(transfer_filename, "w") as f:
                for i in indices:
                    f.write(files[i] + "\n")

        pool = multiprocessing.Pool(processes=3)
        manager = multiprocessing.Manager()
        results = manager.Queue()

        with open(transfer_filename, "r") as f:
            current_files = f.read()

        for f in current_files.split("\n"):

            # skip empty entries
            if len(f) == 0:
                continue

            pool.apply_async(
                transfer_multiprocessing,
                args=(
                    args["stream_transfer_operation"] == "move",
                    args["stream_transfer_operation"] == "move" and args["stream_transfer_all"],
                    f,
                    os.path.join(target_path, "raw"),
                    args["stream_session_name"],
                    args["stream_camera_profile"],
                    server,
                    results,
                    args["stream_transfer_operation"] == "link",
                    args["movie_pattern"],
                ),
            )

        pool.close()
        pool.join()

        # bookkeeping
        while results.empty() == False:

            current = results.get()

            if current["condition"]:
                with open(transferred_filename, "a") as transferred:
                    transferred.write(current["name"] + "\n")

                # only use files larger than .05 GB for measuring speed
                if float(current["filesize"]) > 1024 * 1024 * 1024 * 0.01:
                    elapsed_time = np.append(
                        elapsed_time,
                        current["speed"] * len(current_files.split("\n")),
                    )

            continue

        if elapsed_time.size > 0:
            np.savetxt(elapsed_time_file, elapsed_time)

        if os.path.exists(restart_flag):
            logger.warning("Restart flag detected " + restart_flag)
            try:
                #shutil.rmtree(os.path.join(target_path, "txt"))
                #os.mkdir(os.path.join(target_path, "txt"))
                os.remove(restart_flag)
            except:
                raise Exception("Cannot delete " + restart_flag)

        if os.path.exists(clear_flag):
            logger.warning("Clear flag detected " + restart_flag)
            try:
                os.remove(clear_flag)
                # run_shell_command("rm -rf " + os.path.join(target_path,"ctf") )
                # run_shell_command("rm -rf " + os.path.join(target_path,"box") )
                # run_shell_command("rm -rf " + os.path.join(target_path,"ali") )
            except:
                raise Exception("Cannot delete " + clear_flag)

        # wait longer if there are no files to transfer
        if len(current_files.split("\n")[0]) == 0:
            time.sleep(30)
        else:
        	time.sleep(1)

    # check for stop flag
    if os.path.exists(stop_flag):
        logger.warning("Stop flag detected " + stop_flag)
        try:
            os.remove(stop_flag)
        except:
            raise Exception("Cannot delete " + stop_flag)
    else:
        logger.info("Wall time limit has been reached.")