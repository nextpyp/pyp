import glob
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np

import pyp.inout.image as imageio
from pyp.detect import tomo_subvolume_extract_is_required, tomo_vir_is_required, detect_gold_beads
from pyp import align, preprocess, merge
from pyp import ctf as ctf_mod
from pyp.inout.image import digital_micrograph as dm4
from pyp.inout.image import mrc
from pyp.inout.image.core import get_gain_reference, get_image_dimensions
from pyp.system import local_run, mpi, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path
from pyp.system.wrapper_functions import avgstack, cistem_rescale, cistem_resize
from pyp.system.project_params import resolve_path
from pyp.utils import get_relative_path, movie2regex, timer
from pyp.streampyp.logging import TQDMLogger

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def invert_contrast(name):
    command = "{0}/bin/newstack {1}.mrc {1}.mrc -multadd -1,0".format(
        get_imod_path(), name
    )
    local_run.run_shell_command(command)


@timer.Timer("remove_xrays", text="Removing hot pixels took: {}", logger=logger.info)
def remove_xrays_from_file(name,verbose=False):
    logger.info("Removing xrays")
    # Hot-pixel removal using IMOD's ccderaser
    command = "{0}/bin/ccderaser -input {1}.mrc -output {1}.st -find -points {1}_xray.mod -scan 4.50 -xyscan 128".format(
        get_imod_path(), name
    )
    local_run.run_shell_command(command,verbose=verbose)


def remove_xrays_from_movie_file(name, inplace=False):
    logger.info("Removing xrays")
    model = name + "_xray.mod"
    if not os.path.exists(model):

        # search for hot pixels in frame average
        #         command="""
        # %s/bin/avgstack << EOF
        # %s.mrc
        # %s_hpr.avg
        # /
        # EOF
        # """ % ( get_imod_path(), name, name )
        #         logger.info(command)
        #         [output,error] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE ).communicate()

        input_fname = f"{name}.tif"
        output_fname = f"{name}_hpr.avg"
        start_end_section = "/"
        output, error = avgstack(input_fname, output_fname, start_end_section)

        # Hot-pixel detection using IMOD's ccderaser
        command = "{0}/bin/ccderaser -input {1}_hpr.avg -output {1}_hpr.avg -find -points {2} -scan 4.50 -xyscan 128 -edge 64".format(
            get_imod_path(), name, model
        )
        local_run.run_shell_command(command)

    # correct stack in-place
    [output, error] = local_run.run_shell_command(
        "{0}/bin/imodinfo -a {1} | grep objects!".format(get_imod_path(), model)
    )

    if len(output) > 0:
        logger.info(output)
        logger.info("No hot pixels found.")
    else:
        command = "{0}/bin/ccderaser -input {1}.tif -output {1}.tif -model {2} -allsec /".format(
            get_imod_path(), name, model
        )
        local_run.run_shell_command(command)
        # remove backup file
        try:
            os.remove(name + ".mrc~")
        except:
            pass
        if "ERROR" in output:
            raise Exception(output)


def apply_alignment_to_frames(frame_name):
    # TODO: Why are we using linear interpolation and not using gain correction?
    com = "{0}/bin/newstack -input {1}.tif -output {1}.mrc -xform {1}.xf -linear".format(
        get_imod_path(), frame_name
    )
    local_run.run_shell_command(com,verbose=False)

    # remove xrays from movie frames
    preprocess.remove_xrays_from_movie_file(frame_name)

    input_fname = f"{frame_name}.mrc"
    output_fname = f"{frame_name}_avg.mrc"
    start_end_section = "/"

    output, error = avgstack(input_fname, output_fname, start_end_section)
    logger.info(output)


def preprocess_and_alignment_of_frames(
    frame_name, parameters, current_path, working_path, binning, gain_reference_file,
):

    # read tif files, gain correct, convert to mrc
    x, y, z = imageio.readMoviefileandsave(
        frame_name + ".tif", parameters, binning, gain_reference_file
    )

    if not any(
        x in parameters["movie_ali"]
        for x in ("tiltxcorr_average", "unblur", "skip", "relion")
    ):
        align.align_stack(
            frame_name, parameters,
        )
    else:
        align.align_stack_super(
            frame_name, parameters, current_path, working_path, parameters["movie_ali"],
        )

def frames_from_pattern(filename,name,pattern):
    """Generate list of frames from movie pattern

    Args:
        filename (str): absolute path to data
        name (str): tilt-series name
        pattern (str): movie-pattern

    Returns:
        _type_: _description_
    """
    _, file_format = os.path.splitext(pattern)
    regex = movie2regex(pattern,name)
    r = re.compile(regex)

    labels = ["TILTSERIES", "SCANORD", "ANGLE"]
    labels = [l for l in labels if pattern.find(l) >= 0]
    labels.sort(key=lambda x: int(pattern.find(x)))
    detected_movies = [f for f in sorted(os.listdir(Path(filename).parents[0])) if r.match(f)]

    return detected_movies

def read_tilt_series(
    filename, parameters, metadata, current_path=Path.cwd(), working_path=Path.cwd(), project_path=""
):

    binning = int(parameters["data_bin"])
    aligned_tilts = []

    data_path = Path(resolve_path(parameters["data_path"])).parent
    mdoc_path = Path(resolve_path(parameters["data_path_mdoc"])).parent if "data_path_mdoc" in parameters and parameters["data_path_mdoc"] != None else None
    project_raw_path = Path(filename).parent

    name = os.path.basename(filename)
    mdoc_pattern = "*.mdoc"

    mdocs = []
    if mdoc_path is not None:
        mdoc_pattern = Path(resolve_path(parameters["data_path_mdoc"])).name
        mdocs = list(mdoc_path.glob(str(mdoc_pattern)))
        mdocs = [str(file) for file in mdocs if str(file.name).replace(".mrc", "").replace(".mdoc", "") == name]

    if len(mdocs) == 0:
        # get the mdoc files from the path of raw data if it couldn't find them in mdoc path
        mdocs = list(data_path.glob(mdoc_pattern))
        mdocs = [str(file) for file in mdocs if str(file.name).replace(".mrc", "").replace(".mdoc", "") == name]

    # escape special character in case it contains [
    filename = glob.escape(filename)

    if os.path.isfile(filename + ".tbz"):
        if int(parameters["slurm_tasks"]) > 0:
            command = "pbzip2 -dc -p{0} {1}.tbz | tar xv".format(
                parameters["slurm_tasks"], filename
            )
        else:
            command = (
                "tar xvf " + filename + ".tbz --use-compress-prog=pbzip2".format(name)
            )
        local_run.run_shell_command(command)
    elif os.path.isfile(filename + ".tgz"):
        command = "tar xvfz " + filename + ".tgz".format(name)
        local_run.run_shell_command(command)
    elif not parameters["movie_no_frames"]:
        arguments = []
        if parameters["movie_mdoc"]:
            tilts = frames_from_mdoc(mdocs, parameters)
            for tilt_image in tilts:
                tilt_image_filename = tilt_image[0]
                if (project_raw_path / tilt_image_filename).exists():
                    arguments.append((os.path.join(project_raw_path, tilt_image_filename), "."))
                else:
                    raise Exception(f"{tilt_image_filename} indicated inside {name}.mdoc is not found in {project_raw_path}")
        else:
            for pattern in [ parameters["movie_pattern"], parameters["movie_pattern"].replace(Path(parameters["movie_pattern"]).suffix,"."+parameters["stream_compress"])]:
                detected_movies = frames_from_pattern(filename=filename,name=name,pattern=pattern)
                if len(detected_movies) > 0:
                    break
            for i in detected_movies:
                arguments.append((os.path.join(Path(filename).parents[0], i),"."))
        if len(arguments) > 0:
            mpi.submit_function_to_workers(shutil.copy2,arguments,verbose=True)
    elif os.path.exists(filename + ".mrc"):
        try:
            shutil.copy2(filename + ".mrc", ".")
        except:
            # ignore if file already exists
            pass
    elif os.path.exists(filename + ".tif") or os.path.exists(filename + ".tif.mdoc") or os.path.exists(filename + ".tiff") or os.path.exists(filename + ".tiff.mdoc"):
        for i in glob.glob(filename + ".tif") + glob.glob(filename + ".tif.mdoc") + glob.glob(filename + ".tiff") + glob.glob(filename + ".tiff.mdoc"):
            try:
                shutil.copy2(i, ".")
            except:
                # ignore if file already exists
                pass

    source = os.path.split(os.path.realpath(filename))[0]
    gain_pattern_fc3 = "{0}/*CountRef*".format("/".join(source.split("/")))
    if len(glob.glob(gain_pattern_fc3)) > 0:
        com = "cp {0} .".format(gain_pattern_fc3)
        local_run.run_shell_command(com)

    drift_metadata = {}

    if parameters["movie_no_frames"]:

        if os.path.isfile(name + ".dm4"):

            # raise flag for dm4 format
            open("isdm4", "a").close()

            # command = 'dm2mrc {0}.dm4 {0}.mrc'.format(name)
            # print command
            # commands.getoutput( command )

            # [ shared_image, image ] = readDMfile( '{0}.dm4'.format(name) )
            # mrc.write( image, '{0}.mrc'.format(name) )

            # extract tilt-angles
            with open("{0}.dm4".format(name), "rb") as f:
                dm = dm4.DigitalMicrographReader(f)
                [pixel_size, voltage, mag] = dm.get_info()
                [x, y, z, offset, dt] = dm.get_image_info()
                tilt_angles = dm.get_tilt_angles()

                # determine correct tilt-axis from header information
                tilt_axis = dm.get_tilt_axis_rotation()

            if not os.path.isfile("{0}.rawtlt".format(name)):
                # write to .rawtlt
                with open("{0}.rawtlt".format(name), "w") as f:
                    for item in tilt_angles:
                        f.write("%s\n" % item)

            dims = imageio.readDMfileandsave("{0}.dm4".format(name))

            # read and align intermediate frames
            if os.path.exists(filename + "_frames.tbz"):

                logger.info("Processing movie frames")

                # decompress frames
                if int(parameters["slurm_tasks"]) > 0:
                    command = "pbzip2 -dc -p{0} {1}_frames.tbz | tar xv".format(
                        parameters["slurm_tasks"], filename
                    )
                else:
                    command = (
                        "tar xvf "
                        + filename
                        + "_frames.tbz --use-compress-prog=pbzip2".format(name)
                    )
                local_run.run_shell_command(command)

                shifts = np.zeros([dims[-1]])

                # parallelize frame alignment
                if int(parameters["slurm_tasks"]) == 0:
                    pool = multiprocessing.Pool()
                else:
                    pool = multiprocessing.Pool(parameters["slurm_tasks"])

                # align frames in each tilt
                for tilt in range(dims[-1]):
                    logger.info("Aligning frames for tilt %f", tilt_angles[tilt])
                    frame_name = name + "_%04d" % tilt
                    d = imageio.readDMfileandsave(frame_name + ".dm4")
                    if not os.path.exists(frame_name + ".xf"):
                        pool.apply_async(align.align_stack, args=(frame_name, parameters))
                        # aligned_tilt = align.align_stack( frame_name, parameters )
                    else:
                        com = "{0}/bin/newstack -input {1}.mrc -output {1}.ali -xform {1}.xf -linear".format(
                            get_imod_path(), frame_name
                        )
                        local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])

                        #                     command="""
                        # %s/bin/avgstack << EOF
                        # %s.ali
                        # %s.avg
                        # /
                        # EOF
                        # """  % ( get_imod_path(), frame_name, frame_name )
                        #                     logger.info(command)
                        #                     [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                        input_fname = f"{name}.ali"
                        output_fname = f"{name}.avg"
                        start_end_section = "/"

                        output, error = avgstack(
                            input_fname, output_fname, start_end_section
                        )
                        logger.info(output)

                pool.close()
                pool.join()

                for tilt in range(dims[-1]):
                    frame_name = name + "_%04d" % tilt
                    s = np.loadtxt(frame_name + ".xf", dtype=float)[:, -2:]
                    shifts[tilt] = np.hypot(s[:, 0], s[:, 1]).sum()

                # compose drift-corrected tilt-series
                command = "{0}/bin/newstack {1}_????.avg {1}.mrc".format(
                    get_imod_path(), name
                )
                local_run.run_shell_command(command)

                # plot shifts as function of tilt angle
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
                # ax.plot( tilt_angles, shifts, 'r.-')
                ax.bar(tilt_angles, shifts, 0.1, label="Average drift")
                plt.savefig("{}_frames_xf.png".format(name))
                plt.close()

        elif os.path.isfile(name + ".mrc"):

            if not parameters["movie_mdoc"]:
                # .rawtlt and .order should be moved to the local scratch at this point
                assert Path(f"{name}.rawtlt").exists(), "Please provide .rawtlt file containing the initial tilt angles."
                assert Path(f"{name}.order").exists(), "Please provide .order file containing the acquisition order."

            elif len(mdocs) > 0 and parameters["movie_mdoc"]:

                tilts = frames_from_mdoc(mdocs, parameters)
                tilts.sort(key=lambda x: x[1])

                tilt_angles = [_[1] for _ in tilts]
                order = [_[-1] for _ in tilts]

                np.savetxt(f"{name}.rawtlt", tilt_angles, fmt="%.2f")
                np.savetxt(f"{name}.order", order, fmt="%d")

            else:
                raise Exception("Please either provide .rawtlt/.order files or .mdoc file(s) for initial tilt angles and acquisition order.")


            if not os.path.isfile("{0}.rawtlt".format(name)):
                # write to .rawtlt
                with open("{0}.rawtlt".format(name), "w") as f:
                    for item in sorted_tilts:
                        f.write("%s\n" % item[1])

            tilts = np.loadtxt("{0}.rawtlt".format(name))
            order = np.loadtxt("{0}.order".format(name))
            sorted_tilts = sorted(tilts)
            drift_metadata["tilts"] = [tilt for tilt in sorted_tilts]
            sorted_tilts = [("dummpy", tilt) for tilt in sorted_tilts]

            shifts = {}
            for i in range(tilts.size):
                shifts[i] = np.zeros([1, 2])

            x, y, z = get_image_dimensions(name + ".mrc")

            # sanity check if number of tilts derived from .rawtlt is correct
            assert (z == len(sorted_tilts)), f"{z} tilts in {name+'.mrc'} != {len(sorted_tilts)} from .rawtlt"
            assert (z == len(order)), f"{z} tilts in {name+'.mrc'} != {len(order)} from .order"

            pixel_size = parameters["scope_pixel"]
            voltage = parameters["scope_voltage"]
            mag = parameters["scope_mag"]
            tilt_axis = parameters["scope_tilt_axis"] - 90.0

            if "extract_fmt" in parameters.keys() and "frealign" not in parameters["extract_fmt"]:
                command = "{0}/bin/newstack {1}.mrc {1}.mrc -mode 1 -multadd 1,32768".format(
                    get_imod_path(), name
                )
                command = "{0}/bin/newstack {1}.mrc {1}.mrc -scale 0,32767 -mode 1".format(
                    get_imod_path(), name
                )
            else:
                command = "{0}/bin/newstack {1}.mrc {1}.mrc -mode 2".format(
                    get_imod_path(), name
                )
            local_run.run_shell_command(command)

            # read image dimensions
            [micrographinfo, error] = local_run.run_shell_command(
                "{0}/bin/header -size '{1}.mrc'".format(get_imod_path(), name), verbose=False
            )
            x, y, z = list(map(int, micrographinfo.split()))

            # separate tilted images for later per-tilt ctf estimation. 
            # UPDATE: only do this if we need to estimate the CTF
            if ctf_mod.is_required_3d(parameters) and not ctf_mod.is_done(metadata,parameters, name=name, project_dir=current_path):
                commands = []
                for idx in range(z):
                    command = "{0}/bin/newstack -secs {1} {2}.mrc {2}_{1:04d}.mrc".format(
                        get_imod_path(), idx, name, 
                    )
                    commands.append(command)

                mpi.submit_jobs_to_workers(commands, os.getcwd())

            docfile = filename + ".mrc.mdoc"
            if os.path.isfile(docfile):
                shutil.copy2(docfile, ".")

            # f = open("{0}.mrc".format(name), "rb")
            # headerbytes = f.read(1024)
            # headerdict = mrc.parseHeader(headerbytes)
            # x, y, z = headerdict["nx"], headerdict["ny"], headerdict["nz"]

            # pixel_size, voltage, mag, defocus, tilt_axis = imageio.readMRCheader(name + ".mrc")

        else:
            logger.error("Cannot read %s", filename)

    elif len(parameters["movie_pattern"]) > 0 or len(mdocs) == 1:
        # use either movie pattern OR mdoc file to find corresponding tilted images 

        pattern = parameters["movie_pattern"]
        metadata_from_mdoc = parameters["movie_mdoc"] 

        order_from_file = []
        tilt_angles_from_file = []

        if len(mdocs) == 0 or not metadata_from_mdoc:
            if "SCANORD" not in pattern:
                try:
                    order_from_file = [ _.strip() for _ in open(f"{name}.order", "r").readlines() ]
                except:
                    logger.warning(f"Cannot detect scanning order from filename and .order")
            if "ANGLE" not in pattern:
                try:
                    tilt_angles_from_file = [ _.strip() for _ in open(f"{name}.rawtlt", "r").readlines() ]
                except:
                    logger.warning(f"Cannot detect tilt angles from filename and .rawtlt")

            for pattern in [ parameters["movie_pattern"], parameters["movie_pattern"].replace(Path(parameters["movie_pattern"]).suffix,"."+parameters["stream_compress"])]:
                root_pattern, file_format = os.path.splitext(pattern)
                regex = movie2regex(pattern, name)
                r = re.compile(regex)
                r_mdoc = re.compile(regex.replace(file_format, ".mdoc"))

                # raise flag for tif format
                if parameters["movie_pattern"].endswith(".tif"):
                    open("istif", "a").close()

                # put all the tif files in a list and initialize their angles, scanord as zero
                labels = ["TILTSERIES", "SCANORD", "ANGLE"]
                labels = [l for l in labels if pattern.find(l) >= 0]
                labels.sort(key=lambda x: int(pattern.find(x)))
                detected_movies = [r.match(f) for f in sorted(os.listdir(".")) if r.match(f)]
                if len(detected_movies) > 0:
                    break

            if "SCANORD" not in pattern:
                assert (len(detected_movies) == len(order_from_file)), f"{len(detected_movies)} tilts matching {parameters['movie_pattern']} != {len(order_from_file)} from .order"
            if "ANGLE" not in pattern:
                assert (len(detected_movies) == len(tilt_angles_from_file)), f"{len(detected_movies)} tilts matching {parameters['movie_pattern']} != {len(tilt_angles_from_file)} from .rawtlt"

            tilts = [[f, 0, 0] for f in [r.group(0) for r in detected_movies]]
            if not len(tilts) > 0:
                raise Exception(
                    f"Cannot find tilted movies using movie_pattern {pattern}\nAvailable files: {sorted(os.listdir('.'))}"
                )

            index = 0

            dims = get_image_dimensions(tilts[0][0])

            for r, t in zip(detected_movies, tilts):

                # 1. try to extract tilt angles and scanning order from the filename
                # 2. if they do not exist, get from the user-provided files (.order, .rawtlt) 
                # 3. if user does not provide .rawtlt, extract tilt angles from header
                try:
                    pos_tiltangle = labels.index("ANGLE") + 1
                    t[1] = float(r.group(pos_tiltangle))
                except ValueError:
                    if len(tilt_angles_from_file) > 0: 
                        t[1] = float(tilt_angles_from_file[index])
                    elif t[0].endswith(".dm4"):
                        dm = dm4.DigitalMicrographReader(t[0])
                        t[1] = float(dm.get_tilt_angles())

                        command = f"{get_imod_path()}/bin/dm2mrc {t[0]} {t[0].replace('.dm4', '.mrc')}"
                        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])
                        t[0] = t[0].replace('.dm4', '.mrc')
                        file_format = ".mrc"
                    else:
                        raise Exception(f"Please at least provide a .rawtlt file for initial tilt angles")
                try:
                    pos_scanord = labels.index("SCANORD") + 1
                    t[2] = int(r.group(pos_scanord))
                except ValueError:
                    if len(order_from_file) > 0:
                        t[2] = int(float(order_from_file[index]))
                    else:
                        raise Exception(f"Please at least provide a .order file for scanning order")

                index += 1
        else:
            assert len(mdocs) > 0, f"Do not detect any mdoc files, please put mdoc files with movie frames"
            # use mdoc file to get corresponding tilted images
            tilts = frames_from_mdoc(mdocs, parameters)
            file_format = Path(tilts[0][0]).suffix

            dims = get_image_dimensions(tilts[0][0])

        pixel_size = parameters["scope_pixel"]
        voltage = parameters["scope_voltage"]
        mag = parameters["scope_mag"]
        tilt_axis = parameters["scope_tilt_axis"]

        # sort the list based on tilt angle
        sorted_tilts = sorted(tilts, key=lambda x: x[1])

        # write a file that contains frame filenames (sorted by tilt angle)
        with open("frame_list.txt", "w") as f:
            f.write("\n".join([f[0] for f in sorted_tilts]))

        # check if 0 is in the scanorder list
        if 0 not in [item[2] for item in sorted_tilts]:
            for item in sorted_tilts:
                item[2] -= 1

        if not os.path.isfile("{0}.rawtlt".format(name)):
            # write to .rawtlt
            with open("{0}.rawtlt".format(name), "w") as f:
                for item in sorted_tilts:
                    f.write(f"{item[1]}" + "\n")

        if not os.path.isfile("{0}.order".format(name)):
            # write to .order
            with open("{0}.order".format(name), "w") as f:
                for item in sorted_tilts:
                    f.write(f"{item[2]}" + "\n")

        # always assume frames
        if not os.path.isfile(name + ".mrc"):

            # parallelize frame alignment
            arguments = []

            # generate gain reference
            gain_reference, gain_reference_file = get_gain_reference(
                parameters, dims[0], dims[1],
            )

            # use local copy of gain reference and reset transformations since they were already applied
            if gain_reference_file:
                parameters["gain_reference"] = Path(os.getcwd()) / gain_reference_file
                parameters["gain_rotation"] = 0
                parameters["gain_flipv"] = False
                parameters["gain_fliph"] = False

            # align frames in each tilt
            isfirst = True
            t = timer.Timer(text="Gain correction + frame alignment took: {}", logger=logger.info)
            t.start()
            logger.info(f"Processing movie frames using: {parameters['movie_ali']}")
            import torch
            if torch.cuda.is_available() and 'motioncor' in parameters["movie_ali"]:
                with tqdm(desc="Progress", total=len(sorted_tilts), file=TQDMLogger()) as pbar:
                    for tilt in sorted_tilts:
                        frame_name = tilt[0].replace(file_format, "")
                        align.align_movie_frames( parameters, frame_name, file_format, isfirst)
                        pbar.update(1)
            else:
                # submit jobs to workers
                for tilt in sorted_tilts:
                    frame_name = tilt[0].replace(file_format, "")

                    arguments.append(
                        (
                            parameters,
                            frame_name,
                            file_format,
                            isfirst
                        )
                    )
                    isfirst = False

                mpi.submit_function_to_workers(
                    align.align_movie_frames, arguments, verbose=parameters["slurm_verbose"]
                )
            t.stop()

            # compose drift-corrected tilt-series
            aligned_tilts = [sorted_tilt[0].replace(file_format, ".avg") for sorted_tilt in sorted_tilts]
            aligned_tilts_str = " ".join(aligned_tilts)

            t = timer.Timer(text="Combine into one tilt-series took: {}", logger=logger.info)
            t.start()
            command = "{0}/bin/newstack {2} {1}.mrc".format(
                get_imod_path(), name, aligned_tilts_str
            )

            # suppress long log
            if parameters["slurm_verbose"]:
                logger.info(command)
            local_run.run_shell_command(command, verbose=False)

            # for per-tilt ctf estimation
            [os.rename(average, "%s_%04d.mrc"%(name, idx)) for idx, average in enumerate(aligned_tilts)]
            aligned_tilts = ["%s_%04d.mrc"%(name, idx) for idx, average in enumerate(aligned_tilts)]

            t.stop()

        shifts = {}
        shiftsmag = np.zeros([len(sorted_tilts)])

        for idx, tilt in enumerate(sorted_tilts):
            frame_name = tilt[0].replace(file_format, "")
            s = np.loadtxt(glob.glob(frame_name + "*.xf")[0], dtype=float, ndmin=2)[:, -2:]
            shifts[idx] = np.array( s[:, :2] )
            shiftsmag[idx] = np.hypot(s[:, 0], s[:, 1]).sum()

        # read image dimensions
        [micrographinfo, error] = local_run.run_shell_command(
            "{0}/bin/header -size '{1}.mrc'".format(get_imod_path(), name),verbose=False
        )
        x, y, z = list(map(int, micrographinfo.split()))
        drift_metadata["tilts"] = [tilt[1] for tilt in sorted_tilts]

    else:
        logger.error("Cannot read %s", filename)

    if metadata and metadata.get("drift"):
        drift_metadata["drift"] = {}
        if metadata.get("drift"):
            for i in metadata["drift"]:
                drift_metadata["drift"][i] = metadata["drift"][i].to_numpy()[:,-2:]
        elif metadata.get("web") and metadata.get("web").get("drift"):
            for i in metadata.get("web")["drift"]:
                drift_metadata["drift"][i] = metadata.get("web")["drift"][i]
    else:
        drift_metadata["drift"] = shifts

    if "eer" in parameters["data_path"] and parameters["movie_eer_reduce"] > 1:
        upsample = parameters["movie_eer_reduce"]
        logger.info("Aligned image pixel size upsampling to " + str(pixel_size / upsample))
        pixel_size /= upsample

    logger.info(
        "Unbinned tilt-series dimensions = [ %s, %s, %s ]",
        x,
        y,
        z
    )

    if parameters["tomo_ali_format"]:
        squarex = math.ceil(x / 512.0) * 512
        squarey = math.ceil(y / 512.0) * 512
    else:
        squarex = x
        squarey = y

    square = max(squarex, squarey)

    # only need squared tilt-series when: 
    # 1. tiltseries alignment is not done yet
    # 2. squared aligned tiltseries needs to be generated (for producing downsampled tomogram or subvolume/virion extraction)
    if ( not project_params.tiltseries_align_is_done(metadata)
        or not merge.tomo_is_done(name, os.path.join(project_path, "mrc")) 
        or ( parameters["tomo_vir_method"] != "none" and parameters["detect_force"] )
        or parameters["tomo_vir_force"] 
        or parameters["tomo_rec_force"]
        or tomo_subvolume_extract_is_required(parameters)
        or tomo_vir_is_required(parameters)
        or not ctf_mod.is_done(metadata,parameters, name=name, project_dir=project_path) ):

        imageio.tiltseries_to_squares(name, parameters, aligned_tilts, z, square, binning)

    if parameters["tomo_ali_square"]:
        x, y, z = square, square, parameters["tomo_rec_thickness"]

    pixel_size *= binning

    # invert contrast if needed
    if parameters["data_invert"]:
        preprocess.invert_contrast(name)

    logger.info(
        "Unbinned tomogram dimensions = [ %s, %s, %s ]",
        x,
        y,
        z
    )

    return [x, y, z, pixel_size, voltage, mag, tilt_axis, drift_metadata]


def resample_and_resize(input, output, scale, size):
    """Change pixel size and crop 3D volume.

    Parameters
    ----------
    input_filename : str
        Input mrc file
    output_filename : str
        Output mrc file
    scale : float
        Scaling factor
    size : int
        Size to crop volume
    """
    input_size = int(mrc.readHeaderFromFile(input)["nz"])
    scale = input_size * scale

    tmp_filename = input + "~"

    # first resample
    print(cistem_rescale(input, tmp_filename, scale)[0])

    if size != scale:
        # resize the putput
        print(cistem_resize(tmp_filename, output, size)[0])

        try:
            os.remove(tmp_filename)
        except:
            pass
    else:
        shutil.move(tmp_filename, output)


def resize_initial_model(mparameters, initial_model, frealign_initial_model):

    actual_pixel = (
        float(mparameters["scope_pixel"])
        * float(mparameters["data_bin"])
        * float(mparameters["extract_bin"])
    )
    model_box_size = int(mrc.readHeaderFromFile(initial_model)["nx"])
    model_pixel_size = float(mrc.readHeaderFromFile(initial_model)["xlen"]) / float(
        model_box_size
    )

    scaling = model_pixel_size / actual_pixel

    if (
        scaling < 0.99
        or scaling > 1.01
        or int(mparameters["extract_box"]) != model_box_size
    ):
        logger.warning(f"Rescaling reference {initial_model} {1/scaling:.2f}x to {model_pixel_size/scaling:.2f} A/pix")
        command = "{0}/bin/matchvol -size {1},{1},{1} -3dxform {3},0,0,0,0,{3},0,0,0,0,{3},0 '{4}' {2}".format(
            get_imod_path(), int(mparameters["extract_box"]), frealign_initial_model, scaling, initial_model,
        )
        local_run.run_shell_command(command,verbose=mparameters["slurm_verbose"])

    elif not initial_model == frealign_initial_model:
        shutil.copy2(initial_model, frealign_initial_model)



def frames_from_mdoc(mdoc_files: list, parameters: dict):
    """ Obtain filename, tilt angles, scanning order from mdoc files
       It is possible that one mdoc file per tilt-series or one mdoc file per tilted image

    Args:
        mdoc_files (list): List of mdoc files
        parameters (dict): PYP parameters

    Returns:
        list, list:
            list of detected frames where each entry is (filenames, tilt angle, order)
            dimension of images
    """
    tilt_angle = None
    frames_set = []

    DATETIMES = ["%y-%b-%d  %H:%M:%S", "%Y-%b-%d  %H:%M:%S", "%d-%b-%y  %H:%M:%S", "%d-%b-%Y  %H:%M:%S"]

    for file in mdoc_files:

        with open(file, 'r') as f:

            for line in f.readlines():
                # if line.startswith("ImageSize"):
                #     dims = list(map(int, line.split("=")[-1].strip().split()))

                if line.startswith("SubFramePath"):
                    # use the name of mdoc file as the filename of movie instead of the one in mdoc file, if there're multiple mdoc files
                    if "\\" in line:
                        frame = line.strip().split('\\')[-1] if len(mdoc_files) == 1 else str(file.stem).replace(".mdoc", "")
                    else:
                        frame = line.strip().split('/')[-1] if len(mdoc_files) == 1 else str(file.stem).replace(".mdoc", "")

                    # assert Path(frame).exists(), f"{frame} does not exist. Please check the filename in mdoc files is correct"
                    frames_set.append([frame, tilt_angle, None]) # append the metadata into the list

                elif line.startswith("TiltAngle"):
                    tilt_angle = float(line.split("=")[-1].strip())

                elif line.startswith("DateTime"):
                    time = line.split("=")[-1].strip()

                    for date_pattern in DATETIMES:
                        try:
                            data_output = datetime.datetime.strptime(time, date_pattern)
                            frames_set[-1][-1] = data_output
                            break
                        except:
                            continue

                    assert frames_set[-1][-1] is not None, f"{time} cannot be matched by the pattern. "

                elif line.startswith("RotationAngle"):
                    axis_angle = float(line.split("=")[-1].strip())
                    if parameters:
                        parameters["scope_tilt_axis"] = axis_angle

    # sort the frames by scanning orders
    frames_set = sorted(frames_set, key=lambda x: x[-1])
    order = 0
    for frame in frames_set:
        frame[-1] = order
        order += 1
    return frames_set


def regenerate_average_quick(
    filename, parameters, dims, frame_list
):

    binning = int(parameters["data_bin"])
    aligned_tilts = []

    name = os.path.basename(filename)

    # escape special character in case it contains [
    filename = glob.escape(filename)

    # generate gain reference
    _, gain_reference_file = get_gain_reference(
        parameters, dims[0], dims[1],
    )

    if gain_reference_file is not None:
        commands = []

        for movie in frame_list:
            com = '{0}/bin/clip multiply -m 2 {1} "{2}" {1}; rm -f {1}~'.format(
                get_imod_path(), movie, gain_reference_file,
            )
            commands.append(com)

        mpi.submit_jobs_to_workers(commands, os.getcwd())

    # regenerate average in each tilt
    t = timer.Timer(text="Gain correction + frame alignment took: {}", logger=logger.info)
    t.start()
    logger.info(f"Processing movie frames using existing alignments")
    arguments = []
    for movie in frame_list:
        m_name = movie.replace(".mrc", "")
        arguments.append((movie, m_name, parameters, "imod"))
  
    mpi.submit_function_to_workers(align.apply_alignments_and_average, arguments, verbose=parameters["slurm_verbose"])

    t.stop()

    # compose drift-corrected tilt-series
    aligned_tilts = [frame.replace(".mrc", ".avg") for frame in frame_list]
    aligned_tilts_str = " ".join(aligned_tilts)

    command = "{0}/bin/newstack {2} {1}.mrc".format(
        get_imod_path(), name, aligned_tilts_str
    )

    # suppress long log
    if parameters["slurm_verbose"]:
        logger.info(command)
    local_run.run_shell_command(command, verbose=False)

    # read image dimensions
    [micrographinfo, error] = local_run.run_shell_command(
        "{0}/bin/header -size '{1}.mrc'".format(get_imod_path(), name),verbose=False
    )
    x, y, z = list(map(int, micrographinfo.split()))
    
    if parameters["tomo_ali_format"]:
        squarex = math.ceil(x / 512.0) * 512
        squarey = math.ceil(y / 512.0) * 512
    else:
        squarex = x
        squarey = y

    square = max(squarex, squarey)

    # squared tilt-series: 
    if True:
        imageio.tiltseries_to_squares(name, parameters, aligned_tilts, z, square, binning)
 
    # invert contrast if needed
    if parameters["data_invert"]:
        preprocess.invert_contrast(name)


def erase_gold_beads(name, parameters, tilt_options, binning, zfact, x, y):
    """
    Erase gold beads and reconstruct tomograms
    """

    gold_mod = f"{name}_gold.mod"

    if not os.path.exists(gold_mod) and parameters["tomo_rec_force"]:
        # create binned aligned stack, if needed
        if not os.path.exists(f'{name}_bin.ali'):
            command = "{0}/bin/newstack -input {1}.ali -output {1}_bin.ali -mode 2 -origin -linear -bin {2}".format(
                get_imod_path(), name, binning
            )
            local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])

        detect_gold_beads(parameters, name, x, y, binning, zfact, tilt_options)

    if parameters["tomo_rec_erase_fiducials"]:

        if not os.path.exists(gold_mod):
            logger.error(f"Failed to erase gold becasue no fiducials were found in tomogram")
            return

        # save projected gold coordinates as txt file
        com = f"{get_imod_path()}/bin/model2point {gold_mod} {name}_gold_ccderaser.txt"
        local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])
        
        # calculate unbinned tilt-series coordinates
        with open(f"{name}_gold_ccderaser.txt") as f:
            gold_coordinates = np.array([line.split() for line in f.readlines() if '*' not in line and not "0.00" in line], dtype='f', ndmin=2)

        gold_coordinates[:,:2] *= binning
        np.savetxt(name + "_gold_ccderaser.txt",gold_coordinates)

        # convert back to imod model using one point per contour
        com = f"{get_imod_path()}/bin/point2model {name}_gold_ccderaser.txt {name}_gold_ccderaser.mod -scat -number 1"
        local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])

        # erase gold on (unbinned) aligned tilt-series
        erase_factor = parameters["tomo_rec_erase_factor"]
        if parameters["tomo_rec_erase_order"] == "noise":
            erase_order = -1
        elif parameters["tomo_rec_erase_order"] == "mean":
            erase_order = 0
        elif parameters["tomo_rec_erase_order"] == "first":
            erase_order = 1
        elif parameters["tomo_rec_erase_order"] == "second":
            erase_order = 2
        elif parameters["tomo_rec_erase_order"] == "third":
            erase_order = 3
        erase_iterations = parameters['tomo_rec_erase_iterations']

        com = f"{get_imod_path()}/bin/ccderaser -input {name}.ali -output {name}.ali -model {name}_gold_ccderaser.mod -expand {erase_iterations} -order {erase_order} -merge -exclude -circle 1 -better {parameters['tomo_ali_fiducial'] * erase_factor / parameters['scope_pixel']} -verbose"
        [ output, _ ] = local_run.run_shell_command(com,verbose=parameters["slurm_verbose"])
        if "The largest circle radius is too big for the arrays" in output:
            raise Exception("ccderaser error: The largest circle radius is too big for the arrays. Try reducing the Fiducial radius factor.")

        try:
            os.remove(name + "_gold_ccderaser.txt")
            os.remove(name + "_gold_ccderaser.mod")
        except:
            pass

        # re-calculate reconstruction using gold-erased tilt-series
        merge.reconstruct_tomo(parameters, name, x, y, binning, zfact, tilt_options, force=True)