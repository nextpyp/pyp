import glob
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy 

from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path, get_frealign_paths
from pyp.system.wrapper_functions import newstack
from pyp.utils import get_relative_path

from .. import metadata
from . import digital_micrograph as dm4
from . import mrc

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def write_out_relion_stack(name, current_path, particles):
    # write particle stack (negated per relion's convention)
    mrc.write(
        -particles,
        current_path / "relion/Particles/Micrographs/"
        + name
        + "_frames_particles.mrcs",
    )


def mrc2png(mrcfile, pngfile):
    data = mrc.read(mrcfile)
    writepng(data, pngfile)

def img2webp(pngfile, webpfile, options=""):
    command = f"{os.environ['IMAGICDIR']}/convert {pngfile} {options} -quality 100 -define webp:lossless=true {webpfile}"
    run_shell_command(command, verbose=False)

def mrc2webp(mrcfile, webpfile):
    pngfile = webpfile.replace(".webp",".png")
    data = mrc.read(mrcfile)
    writepng(data, pngfile)
    img2webp(pngfile,webpfile)
 
def writepng(data, pngfile):
    rescaled = (255.0 * (data - data.min()) / (data.max() - data.min())).astype(
        numpy.uint8
    )
    # flip to match mrc writeout
    from PIL import Image

    im = Image.fromarray(rescaled[::-1, :])
    # im = Image.fromarray(rescaled)
    im.save(pngfile)


def write_central_slices(map):
    """ This function takes one mrc map/volume and writes out its three 2D central slices 
        These three slices are populated in one montage
    Args:
        map (str): 3D map/volume 
    """
    rec = mrc.read(map)
    # intert contrast since the maps output from 3DAVG is white density
    rec *= -1
    z = rec.shape[0]
    # 3 * 1 montage
    montage = numpy.zeros([z, z * 3])

    # direction 1
    I = rec[z // 2, 0:z, 0:z]
    montage[0:z, 0:z] = (I - I.mean()) / I.std()
    I = rec[0:z, z // 2, 0:z]
    montage[0:z, z : z * 2] = (I - I.mean()) / I.std()
    I = rec[0:z, 0:z, z // 2]
    montage[0:z, z * 2 : z * 3] = (I - I.mean()) / I.std()

    return montage


def write_multiple_slices(map, num_slices_side, num_slices_top):
    """This function takes one mrc map/volume and writes out its slices from side and top views

    Args:
        map (str): 3D map/volume
        num_slices_side (int): 
        num_slices_top (int): 
    """
    rec = mrc.read(map)
    # intert contrast since the maps output from 3DAVG is white density
    rec *= -1
    z = rec.shape[0]
    side_montage, top_montage = (
        numpy.zeros([z, (z * num_slices_side)]),
        numpy.zeros([z, (z * num_slices_top)]),
    )

    # side view slices montage
    bound = math.floor(z / 4)
    for num in range(num_slices_side):
        slice = bound + int((z - bound * 2) / num_slices_side) * num
        I = rec[0:z, slice, 0:z]
        side_montage[0:z, z * num : z * (num + 1)] = (I - I.mean()) / I.std()
    # top view slices montage
    for num in range(num_slices_top):
        slice = bound + int((z - bound * 2) / num_slices_top) * num
        I = rec[slice, 0:z, 0:z]
        top_montage[0:z, z * num : z * (num + 1)] = (I - I.mean()) / I.std()

    return side_montage, top_montage


def png2gif(pattern, giffile):
    """convert series of pngs to animated gif file"""
    import matplotlib.animation as animation
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    # plt.clf()
    fig = plt.figure(frameon=False)
    plt.axis("off")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    ims = glob.glob(pattern)
    images = []
    for i in ims:
        im = plt.imshow(mpimg.imread(i), animated=True)
        images.append([im])
    ani = animation.ArtistAnimation(fig, images)
    ani.save(giffile, dpi=150, writer="imagemagick")


def mergeImagicFiles(inputlist, filename):

    stacklist = ["eman/" + line + "_phase_flipped_stack.mrc" for line in inputlist]

    mrc.merge(stacklist, filename)


def mergeMrcFiles(inputlist, filename):

    stacklist = [
        "frealign/" + line + "_stack.mrc"
        for line in inputlist
        if os.path.exists("frealign/" + line + "_stack.mrc")
    ]

    mrc.merge(stacklist, filename)


def mergeRelionFiles(inputlist, filename):

    stacklist = [
        "relion/Particles/Micrographs/" + line + "_particles.mrcs" for line in inputlist
    ]

    mrc.merge(stacklist, filename)


def get_gain_reference(parameters, x, y):

    """
    # look for gain reference in directory where links to raw data point to
    source = os.path.split(os.path.realpath(filename))[0]
    gain_pattern = "{0}/*Gain*".format("/".join(source.split("/")))
    gain_pattern_alt = "{0}/*_norm_*".format("/".join(source.split("/")))
    gain_pattern_fc3 = "{0}/*CountRef*".format("/".join(source.split("/")))
    """

    gain_reference_file = None
    gain_reference = numpy.ones([y, x])

    if (
        "gain_reference" in parameters.keys()
        and parameters["gain_reference"] is not None
        and len(project_params.resolve_path(parameters["gain_reference"])) > 0
    ):
        gain_file = project_params.resolve_path(parameters["gain_reference"])
        if os.path.isfile(gain_file):
            gain_reference_file = gain_file

    # turn off function to automatically search for gain reference
    if False and gain_reference_file is None:
        possible_patterns = ["*Gain*", "*gain*", "*_norm_*", "*CountRef_*"]
        for pattern in possible_patterns:
            gain_files = glob.glob(pattern)
            if len(gain_files) > 0 and os.path.exists(gain_files[-1]):
                logger.warning(
                    "Gain reference not explicitly provided, using %s"
                    % gain_reference_file
                )
                gain_reference_file = gain_files[-1]
                break

    if gain_reference_file is not None:
        # copy gain to local scratch
        try:
            shutil.copy2( gain_reference_file, "." )
            gain_reference_file = Path(gain_reference_file).name
        except:
            pass
        extension = Path(gain_reference_file).suffix
        if extension == ".dm4":
            [xr, yr, zr, headersizer, dtr] = dm4.DigitalMicrographReader(
                gain_reference_file
            ).get_image_info()
            with open(gain_reference_file, "rb") as fr:
                fr.seek(headersizer)
                gain_reference = numpy.reshape(
                    numpy.fromfile(fr, dtr, yr * xr), [yr, xr]
                )
        elif extension == ".mrc":
            gain_reference = mrc.read(gain_reference_file)
        elif extension == ".gain" or extension == ".tif" or extension == ".tiff":
            new_gain_reference_file = gain_reference_file.replace(extension,".mrc")
            com = f"{get_imod_path()}/bin/tif2mrc {gain_reference_file} {new_gain_reference_file}"
            run_shell_command(com)
            gain_reference = mrc.read(new_gain_reference_file)
        else:
            logger.warning("Can't recognize the gain reference extension")

        if "gain_fliph" in parameters.keys() and parameters["gain_fliph"]:
            logger.info("Applying horizontal flip to gain reference")
            gain_reference = numpy.fliplr(gain_reference)

        if "gain_flipv" in parameters.keys() and parameters["gain_flipv"]:
            logger.info("Applying vertical flip to gain reference")
            gain_reference = numpy.flipud(gain_reference)

        if (
            "gain_rotation" in parameters.keys()
            and abs(int(parameters["gain_rotation"])) > 0
        ):
            logger.info(
                "Rotating gain reference by %d degrees"
                % (90 * parameters["gain_rotation"])
            )
            gain_reference = numpy.rot90(gain_reference, parameters["gain_rotation"])

        logger.info(
            "Using gain reference: %s, shape: %s, min: %f, max: %f, mean: %f",
            gain_reference_file,
            gain_reference.shape,
            gain_reference.min(),
            gain_reference.max(),
            gain_reference.mean(),
        )

        gain_reference_file = "gain.mrc"
        mrc.write(gain_reference, os.path.join(os.getcwd(), gain_reference_file))

    return gain_reference, gain_reference_file


def read(filename, parameters, binning=1):
    if not os.path.isfile(filename):
        # figure out extension
        if os.path.isfile(filename + ".dm3"):
            extension = ".dm3"
        elif os.path.isfile(filename + ".dm4") or os.path.isfile(
            filename + "-0001.dm4"
        ):  # check for individual frames
            extension = ".dm4"
            if os.path.isfile(filename + "-0001.dm4"):
                filename = filename + "-0001"
        elif os.path.isfile(filename + ".tgz"):
            extension = ".tgz"
        elif os.path.isfile(filename + ".bz2"):
            extension = ".bz2"
        elif os.path.isfile(filename + ".tar.bz2"):
            extension = ".tar.bz2"
        elif os.path.isfile(filename + ".mrc.bz2"):
            extension = ".mrc.bz2"
        elif os.path.isfile(filename + ".tbz"):
            extension = ".tbz"
        elif os.path.isfile(filename + ".mrc"):
            extension = ".mrc"
        elif os.path.isfile(filename + ".tif"):
            extension = ".tif"
        elif os.path.isfile(filename + ".tiff"):
            extension = ".tiff"
        elif os.path.isfile(filename + ".eer"):
            extension = ".eer"
        else:
            logger.error("Cannot find {}".format(filename))
            return

        xmlfile = filename + ".xml"
        if os.path.isfile(xmlfile):
            shutil.copy2(xmlfile, ".")

        docfile = filename + "_sum.mrc.mdoc"
        if os.path.isfile(docfile):
            shutil.copy2(docfile, ".")

        filename = filename + extension

    extension = Path(filename).suffix

    if extension in [".tgz", ".bz2", ".tbz", ".tar.bz2", ".mrc.bz2"]:
        if extension == ".bz2" or extension == ".tbz":
            if int(parameters["slurm_tasks"]) > 0:
                command = "tar xvfj {0}".format(filename)
                command = "pbzip2 -dc -p{0} {1} | tar xv".format(
                    parameters["slurm_tasks"], filename
                )
            else:
                command = "tar xvf {0} --use-compress-prog=pbzip2".format(filename)
            [files, error] = run_shell_command(command)
            if "ERROR" in files:
                shutil.copy2(filename, os.getcwd())
                if int(parameters["slurm_tasks"]) > 0:
                    command = "pbzip2 -dc -p{0} {1} > {2}".format(
                        parameters["slurm_tasks"],
                        os.path.split(filename)[-1],
                        os.path.splitext(os.path.split(filename)[-1])[-2],
                    )
                else:
                    command = "bunzip2 -d %s" % os.path.split(filename)[-1]
                run_shell_command(command)
                files = os.path.splitext(os.path.split(filename)[-1])[-2]

            # handle zip files with paths
            if "/Images-Disc1" in files:
                if int(parameters["slurm_tasks"]) > 0:
                    command = (
                        "pbzip2 -dc -p%s %s | tar xv --transform='s/.*\///'"
                        % (parameters["slurm_tasks"], filename,)
                    )
                else:
                    command = (
                        "tar xvf %s --use-compress-prog=pbzip2 --transform='s/.*\///'"
                        % filename
                    )
                [files, error] = run_shell_command(command)
                run_shell_command("rm -rf Images-Disc1/",)

        else:
            [files, error] = run_shell_command("tar xvfz {0}".format(filename),)

        filenames = [
            s.split("/")[-1]
            for s in files.split()
            if ".mrc" in s or ".dm4" in s or ".tif" in s or ".tiff" in s
        ]

        if os.path.splitext(filenames[0])[0][-5:] == "-0001":
            filename = filenames[0]
        elif "_Fractions" in filenames[0]:
            if os.path.exists(filenames[0].replace("_Fractions", "")):
                filename = filenames[0]
            else:
                # rename frames files
                new_name = filenames[1].replace(".mrc", "_Fractions.mrc")
                os.rename(filenames[0], new_name)
                filename = new_name
        else:
            filename = filenames[-1]
        extension = os.path.splitext(filename)[1]

    else:
        # copy file to local scratch
        shutil.copy2(filename, os.getcwd())
        filename = os.path.split(filename)[-1]

    if extension in [".dm3", ".dm4"]:
        return readDMfileandsave(filename, parameters, binning)

    if extension == ".eer":
        new_name = filename.replace(".eer", ".mrc")
        command = f"{get_imod_path()}/bin/clip resize -es {parameters['movie_eer_reduce']} -ez {parameters['movie_eer_frames']} {filename} {new_name}"
        run_shell_command(command)
        extension = ".mrc"
        filename = new_name

    if extension in {".mrc", ".tif", ".tiff"}:
        return readMoviefileandsave(filename, parameters, binning)

    logger.error("Format not recognized: {0}".format(filename))


def read_from_matlab(filename):
    """Read file from matlab and return as 3D array
    
    Not currently used."""
    import struct

    with open(filename) as f:
        Nx = struct.unpack("i", f.read(4))[0]
        Ny = struct.unpack("i", f.read(4))[0]
        Nz = struct.unpack("i", f.read(4))[0]

        logger.info("%s %s %s", Nx, Ny, Nz)
        S = numpy.fromstring(f.read(), "double")

        logger.info(len(S))

        count = 0
        A = numpy.empty([Nz, Ny, Nx])
        for k in range(Nz):
            for j in range(Ny - 1, -1, -1):
                for i in range(Nx):
                    A[k, j, i] = S[count]
                    count += 1
    return A


def readDMfile(filename, parameters=0, binning=1):

    ## read DM header using IMOD (OBSOLETE)
    # header = commands.getoutput('dm3props %s "%s"' % ( filename[-1], filename ) )
    # x = int(header.split()[0])
    # y = int(header.split()[1])
    # z = int(header.split()[2])
    # headersize = int(header.split()[4])

    dm = dm4.DigitalMicrographReader(filename)

    # override binning if not in super-resolution
    if dm.get_image_info()[0] < 6096:
        binning = 1

    if parameters != 0:
        first = int(parameters["movie_first"])
        last = int(parameters["movie_last"])
        [scope_pixel, scope_voltage, scope_mag] = dm.get_info()
        if (
            parameters["scope_pixel"] == "0"
            or "auto" in parameters["scope_pixel"].lower()
        ):
            parameters["scope_pixel"] = str(scope_pixel)
        if (
            parameters["scope_voltage"] == "0"
            or "auto" in parameters["scope_voltage"].lower()
        ):
            parameters["scope_voltage"] = str(scope_voltage)
        if parameters["scope_mag"] == "0" or "auto" in parameters["scope_mag"].lower():
            parameters["scope_mag"] = str(scope_mag)
    else:
        first = 0
        last = -1

    [x, y, z, headersize, dt] = dm.get_image_info()

    # if data type depth is 1 assume data is not gain corrected
    if dt.itemsize == 1:

        if not os.path.exists(parameters["gain_reference"]):
            gain_reference_file = "K2-0001 2 Gain Ref. x1.m3.kv[300].dm4"
            if os.path.split(filename)[0]:
                gain_reference_file = (
                    os.path.split(filename)[0] + "/" + gain_reference_file
                )
            logger.warning(
                "Gain reference not explicitly provided, using %s"
                % gain_reference_file,
            )
        else:
            gain_reference_file = parameters["gain_reference"]

        # we also need a gain reference in this case

        [xr, yr, zr, headersizer, dtr] = dm4.DigitalMicrographReader(
            gain_reference_file
        ).get_image_info()
        fr = open(gain_reference_file, "rb")
        fr.seek(headersizer)
        gain_reference = numpy.reshape(numpy.fromfile(fr, dtr, yr * xr), [yr, xr])
        fr.close()

        logger.info(
            "Using gain reference: %s, shape: %s, min: %f, max: %f, mean: %f",
            gain_reference_file,
            gain_reference.shape,
            gain_reference.min(),
            gain_reference.max(),
            gain_reference.mean(),
        )

    # support for separate files for each frame
    if os.path.splitext(filename)[0][-5:] == "-0001":

        # figure out root name of current image
        root_name = os.path.splitext(filename)[0].split("-0")[0]

        # figure out number of frames from file list
        z = len(glob.glob(root_name + "-????.dm4"))

        if last >= 0 and last <= z:
            z = last

        if first < 0 and last <= z:
            first = 0

        logger.info("Extracting frames {} to {}".format(first, z))

        shared_image = multiprocessing.RawArray(
            "f", x / binning * y / binning * (z - first)
        )
        image = numpy.frombuffer(
            shared_image,
            dtype=numpy.float32,
            count=x / binning * y / binning * (z - first),
        )
        image = image.reshape([(z - first), y / binning, x / binning])

        for frame in range(first, z):

            # file for current frame (headersize changes so we have to read it each time)
            frame_file = root_name + "-%04d.dm4" % (frame + 1)
            dm = dm4.DigitalMicrographReader(frame_file)
            [xf, yf, zf, headersize, dt] = dm.get_image_info()
            f = open(frame_file, "rb")
            f.seek(headersize)

            if binning > 1:
                if dt.itemsize == 1:
                    image[frame - first, :, :] = (
                        (
                            gain_reference
                            * numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                        )
                        .reshape(y / binning, binning, x / binning, binning)
                        .mean(3)
                        .mean(1)
                    )
                else:
                    image[frame - first, :, :] = (
                        numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                        .reshape(y / binning, binning, x / binning, binning)
                        .mean(3)
                        .mean(1)
                    )
            else:
                if dt.itemsize == 1:
                    image[frame - first, :, :] = gain_reference * numpy.reshape(
                        numpy.fromfile(f, dt, y * x), [y, x]
                    )
                else:
                    image[frame - first, :, :] = numpy.reshape(
                        numpy.fromfile(f, dt, y * x), [y, x]
                    )
            sys.stdout.flush()

            f.close()

    # entire movie in single dm4 file
    else:

        ## support for range as fraction of total number of frames
        # if first > 0 and first < 1:
        #    first = int(first*z)
        # else:
        #    first = int(first)
        # if last >= 0:
        #    if last < 1:
        #        z = int(last*z)
        #    elif last <= z:
        #        z = int(last)

        if last >= 0 and last <= z:
            z = last

        if first < 0 and last <= z:
            first = 0

        logger.info("Extracting frames {} to {}".format(first, z))

        # open file
        f = open(filename, "rb")
        f.seek(headersize + first * y * x * dt.itemsize)

        shared_image = multiprocessing.RawArray(
            "f", x / binning * y / binning * (z - first)
        )
        image = numpy.frombuffer(
            shared_image,
            dtype=numpy.float32,
            count=x / binning * y / binning * (z - first),
        )
        image = image.reshape([(z - first), y / binning, x / binning])

        logger.info("Reading frame ")
        for frame in range(first, z):
            logger.info("\t %d", frame)
            if binning > 1:
                image[frame - first, :, :] = (
                    numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                    .reshape(y / binning, binning, x / binning, binning)
                    .mean(3)
                    .mean(1)
                )
            else:
                image[frame - first, :, :] = numpy.reshape(
                    numpy.fromfile(f, dt, y * x), [y, x]
                )
            sys.stdout.flush()
        f.close()
        logger.info("")

    return [shared_image, image]


def readDMfileandsave(filename, parameters=0, binning=1):

    dm = dm4.DigitalMicrographReader(filename)

    # override binning if not in super-resolution
    if dm.get_image_info()[0] < 6096:
        binning = 1

    if parameters != 0:
        first = int(parameters["movie_first"])
        last = int(parameters["movie_last"])
        [scope_pixel, scope_voltage, scope_mag] = dm.get_info()
        if parameters["scope_pixel"] == 0:
            parameters["scope_pixel"] = scope_pixel
        if parameters["scope_voltage"] == 0:
            parameters["scope_voltage"] = scope_voltage
        if parameters["scope_mag"] == 0:
            parameters["scope_mag"] = scope_mag
    else:
        first = 0
        last = -1

    [x, y, z, headersize, dt] = dm.get_image_info()

    # retrieve gain reference
    gain_reference, gain_reference_file = get_gain_reference(parameters, x, y)

    # support for separate files for each frame
    if os.path.splitext(filename)[0][-5:] == "-0001":

        # figure out root name of current image
        root_name = os.path.splitext(filename)[0].split("-0")[0]

        # figure out number of frames from file list
        z = len(glob.glob(root_name + "-????.dm4"))

        if last >= 0 and last <= z:
            z = last

        if first < 0 and last <= z:
            first = 0

        logger.info("Extracting frames {} to {}".format(first, z))

        # open cumulative output mrc file
        outputfile = root_name + ".mrc"
        with open(outputfile, "wb") as fout:

            logger.info("Reading frame ")
            for frame in range(first, z):
                logger.info("\t %d", frame)

                # file for current frame (headersize changes so we have to read it each time)
                frame_file = root_name + "-%04d.dm4" % (frame + 1)
                dm = dm4.DigitalMicrographReader(frame_file)
                [xf, yf, zf, headersize, dt] = dm.get_image_info()
                with open(frame_file, "rb") as f:

                    f.seek(headersize)

                    if binning > 1:
                        if dt.itemsize == 1:
                            image = (
                                (
                                    gain_reference
                                    * numpy.reshape(
                                        numpy.fromfile(f, dt, y * x), [y, x]
                                    )
                                )
                                .reshape(y / binning, binning, x / binning, binning)
                                .mean(3)
                                .mean(1)
                            )
                        else:
                            image = (
                                numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                                .reshape(y / binning, binning, x / binning, binning)
                                .mean(3)
                                .mean(1)
                            )
                    else:
                        if dt.itemsize == 1:
                            image = gain_reference * numpy.reshape(
                                numpy.fromfile(f, dt, y * x), [y, x]
                            )
                        else:
                            image = numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])

                # get rid of original frame to save space
                os.remove(root_name + "-%04d.dm4" % (frame + 1))

                # write out header of mrc file
                if frame == first:
                    h = mrc.newHeader()
                    mrc.updateHeaderDefaults(h)
                    mrc.updateHeaderUsingArray(h, image)
                    h["nz"] = h["mz"] = z - first
                    headerbytes = mrc.makeHeaderData(h)
                    fout.write(headerbytes)

                mrc.appendArray(image, fout)
                sys.stdout.flush()

            logger.info("")

    # entire movie in single dm4 file
    else:

        if last >= 0 and last <= z:
            z = last

        if first < 0 and last <= z:
            first = 0

        logger.info("Extracting frames {} to {}".format(first, z))

        # open cumulative output mrc file
        outputfile = os.path.split(filename.replace(".dm4", ".mrc"))[-1]
        with open(outputfile, "wb") as fout:

            for frame in range(first, z):

                # open file
                with open(filename, "rb") as f:
                    f.seek(headersize + frame * y * x * dt.itemsize)

                    if binning > 1:
                        image = (
                            (
                                gain_reference
                                * numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                            )
                            .reshape(y / binning, binning, x / binning, binning)
                            .mean(3)
                            .mean(1)
                        )
                    else:
                        image = gain_reference * numpy.reshape(
                            numpy.fromfile(f, dt, y * x), [y, x]
                        )

                # write out header of mrc file
                if frame == first:
                    h = mrc.newHeader()
                    mrc.updateHeaderDefaults(h)
                    mrc.updateHeaderUsingArray(h, image)
                    h["nz"] = h["mz"] = z - first
                    headerbytes = mrc.makeHeaderData(h)
                    fout.write(headerbytes)

                mrc.appendArray(image, fout)

                sys.stdout.flush()

    # cleanup
    # [ os.remove(f) for f in frame_list if os.path.exists(f) ]
    [os.remove(f) for f in glob.glob("%s*.dm4" % filename) if os.path.exists(f)]

    return numpy.array([x, y, z])


def readMRCfile(filename, parameters, binning):
    """Not currently used."""
    # extract info from .xml file
    xmlfile = filename.replace(".mrc", ".xml").replace("_frames", "")
    if os.path.isfile(xmlfile):
        scope_pixel, scope_voltage, scope_mag, defocus = metadata.readXMLfile(xmlfile)
        tilt_axis = 0
    else:
        (
            scope_pixel,
            scope_voltage,
            scope_mag,
            defocus,
            tilt_axis,
        ) = metadata.readMRCheader(filename)
    if parameters["scope_pixel"] == "0" or "auto" in parameters["scope_pixel"].lower():
        parameters["scope_pixel"] = str(scope_pixel)
    if (
        parameters["scope_voltage"] == "0"
        or "auto" in parameters["scope_voltage"].lower()
    ):
        parameters["scope_voltage"] = str(scope_voltage)
    if parameters["scope_mag"] == "0" or "auto" in parameters["scope_mag"].lower():
        parameters["scope_mag"] = str(scope_mag)

    first = int(parameters["movie_first"])
    last = int(parameters["movie_last"])

    f = open(filename, "rb")
    headerbytes = f.read(1024)
    headerdict = mrc.parseHeader(headerbytes)
    dt = headerdict["dtype"]
    if dt == "float32":
        dt = numpy.dtype("f4")
    elif dt == "int16":
        dt = numpy.dtype("i2")
    elif dt == "uint16":
        dt = numpy.dtype("u2")
    elif dt == "int8":
        dt = numpy.dtype("i1")
    elif dt == "uint8":
        dt = numpy.dtype("u1")
    else:
        logger.info("ERROR - mrc type {} not recognized.".format(dt))

    x = headerdict["nx"]
    y = headerdict["ny"]
    z = headerdict["nz"]

    if last >= 0 and last <= z:
        z = last

    if first < 0 and last <= z:
        first = 0

    logger.info("Extracting frames {} to {}".format(first, z))

    # move pointer to first frame
    total_header_size = 1024 + int(headerdict["nsymbt"])
    f.seek(total_header_size + first * y * x * dt.itemsize)

    # shared_image = multiprocessing.RawArray( "f", x*y*(z-first) )
    # image = numpy.frombuffer(shared_image, dtype=numpy.float32, count=x*y*(z-first))
    # image = image.reshape( [(z-first),y,x] )

    shared_image = multiprocessing.RawArray(
        "f", x / binning * y / binning * (z - first)
    )
    image = numpy.frombuffer(
        shared_image, dtype=numpy.float32, count=x / binning * y / binning * (z - first)
    )
    image = image.reshape([(z - first), y / binning, x / binning])

    logger.info("Reading frame ")
    for frame in range(first, z):
        logger.info("\t %d", frame)

        if binning > 1:
            image[frame - first, :, :] = (
                numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                .astype(numpy.float32)
                .reshape(y / binning, binning, x / binning, binning)
                .mean(3)
                .mean(1)
            )
        else:
            image[frame - first, :, :] = numpy.reshape(
                numpy.fromfile(f, dt, y * x), [y, x]
            ).astype(numpy.float32)
        sys.stdout.flush()
    f.close()
    logger.info("")
    return [shared_image, image]


def get_image_dimensions(name):
    
    assert Path(name), f"{name} does not exist."

    command = "{0}/bin/header -size {1}".format(get_imod_path(), name)
    [output, error] = run_shell_command(command, verbose=False)
    return [int(num) for num in output.split()]


def get_image_mode(name):

    command = "{0}/bin/header -mode {1}".format(get_imod_path(), name)
    [output, error] = run_shell_command(command, verbose=False)
    return int(output)


def readMoviefileandsave(filename, parameters, binning, gain_reference_file=None):
    # extract info from .xml file
    extension = Path(filename).suffix
    """
    xmlfile = (
        filename.replace(extension, ".xml")
        .replace("_frames", "")
        .replace("_Fractions", "")
    )

    docfile = filename.replace(extension, "_sum.mrc.mdoc")

    if os.path.isfile(xmlfile) and not "fractions" in xmlfile.lower():
        scope_pixel, scope_voltage, scope_mag, defocus = metadata.readXMLfile(xmlfile)
        tilt_axis = 0
    elif os.path.isfile(docfile):
        scope_pixel, scope_voltage, scope_mag, defocus = metadata.readMDOCfile(docfile)
    elif extension == ".mrc":
        (
            scope_pixel,
            scope_voltage,
            scope_mag,
            defocus,
            tilt_axis,
        ) = metadata.readMRCheader(filename)

    if parameters["scope_pixel"] == 0:
        parameters["scope_pixel"] = str(scope_pixel)
    if parameters["scope_voltage"] == 0 or parameters["scope_voltage"] == 0:
        parameters["scope_voltage"] = str(scope_voltage)
    if parameters["scope_mag"] == 0:
        parameters["scope_mag"] = str(scope_mag)
    """
    first = int(parameters["movie_first"]) - 1 
    last = int(parameters["movie_last"]) - 1
    
    x, y, z = get_image_dimensions(filename)

    threads = parameters["slurm_tasks"]
    env = "export OMP_NUM_THREADS={0}; export NCPUS={0}; IMOD_FORCE_OMP_THREADS={0}; ".format(threads)
    # mode = get_image_mode(filename)

    if last >= 0 and last <= z:
        z = last

    if first < 0 and last <= z:
        first = 0

    grouping = parameters["movie_group"]

    if os.path.split(filename)[0]:
        outputfile = os.path.split(filename)[1]
    else:
        outputfile = filename
    outputfile = outputfile.replace("_frames", "")
    outputfile = outputfile.replace("_Fractions", "")
    inputfile = filename
    outputfile = filename.replace(extension, ".mrc")

    # retrieve gain reference if needed
    if gain_reference_file == "gain.mrc":
        gain_reference = mrc.read("gain.mrc")
    else:
        gain_reference, gain_reference_file = get_gain_reference(parameters, x, y)

    from pyp import preprocess

    # remove x-rays (only for EPU's Mode 7 images)
    # TODO: further consolidate into a single function
    if parameters["gain_remove_hot_pixels"]:
        preprocess.remove_xrays_from_movie_file(Path(filename).stem)

    if grouping < 2:

        if first > 0 or z == last:
            output, error = newstack(
                filename, outputfile, threads, secs=f"{first}-{z - 1}", bin=binning, mode=2
            )
            mode = 2

        # apply gain reference and convert to mode 2
        if gain_reference_file is not None:

            com = env + '{0}/bin/clip multiply -m 2 {1} "{2}" {3}; rm -f {3}~'.format(
                get_imod_path(), inputfile, gain_reference_file, outputfile,
            )
            run_shell_command(com,parameters["slurm_verbose"])
            mode = 2
            if inputfile != outputfile:
                os.remove(inputfile)
        """
        elif mode <= 1:
            com = "{0}/bin/newstack {1} {2} -mode 2; rm {2}~".format(
                get_imod_path(), inputfile, outputfile,
            )
            run_shell_command(com)
            mode = 2
        elif inputfile != outputfile:
            com = "{0}/bin/newstack {1} {2} -mode 2; rm {2}~".format(
                get_imod_path(), inputfile, outputfile,
            )
            run_shell_command(com)
        """

    else:

        f = open(filename, "rb")

        # move pointer to first frame
        total_header_size = 1024 + int(headerdict["nsymbt"])
        f.seek(total_header_size + first * y * x * dt.itemsize)

        # open output mrc file
        fout = open(outputfile, "wb")

        group = 0
        global_frame = 0
        while group * grouping < z - first:

            # read first frame in group
            if binning > 1:
                if dt.itemsize == 1:
                    image = (
                        (
                            gain_reference
                            * numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                        )
                        .astype(numpy.float32)
                        .reshape(y / binning, binning, x / binning, binning)
                        .mean(3)
                        .mean(1)
                    )
                else:
                    image = (
                        numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                        .astype(numpy.float32)
                        .reshape(y / binning, binning, x / binning, binning)
                        .mean(3)
                        .mean(1)
                    )
            else:
                if dt.itemsize == 1:
                    image = gain_reference * numpy.reshape(
                        numpy.fromfile(f, dt, y * x), [y, x]
                    ).astype(numpy.float32)
                else:
                    image = numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x]).astype(
                        numpy.float32
                    )

            global_frame += 1
            local_frame = 1

            while local_frame < grouping and global_frame < z:

                if binning > 1:
                    if dt.itemsize == 1:
                        image = (
                            (
                                gain_reference
                                * numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                            )
                            .astype(numpy.float32)
                            .reshape(y / binning, binning, x / binning, binning)
                            .mean(3)
                            .mean(1)
                        )
                    else:
                        image = (
                            numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x])
                            .astype(numpy.float32)
                            .reshape(y / binning, binning, x / binning, binning)
                            .mean(3)
                            .mean(1)
                        )
                else:
                    if dt.itemsize == 1:
                        image = gain_reference * numpy.reshape(
                            numpy.fromfile(f, dt, y * x), [y, x]
                        ).astype(numpy.float32)
                    else:
                        image = numpy.reshape(
                            numpy.fromfile(f, dt, y * x), [y, x]
                        ).astype(numpy.float32)

                local_frame += 1
                global_frame += 1

            # open output mrc file and write out header
            if group == 0:
                h = mrc.newHeader()
                mrc.updateHeaderDefaults(h)
                mrc.updateHeaderUsingArray(h, image)
                h["nz"] = h["mz"] = int(math.ceil(1.0 * (z - first) / grouping))
                headerbytes = mrc.makeHeaderData(h)
                fout.write(headerbytes)

            mrc.appendArray(image / local_frame, fout)
            group += 1

        f.close()
        fout.close()

    if extension == ".mrc":
        f = open(outputfile, "rb")
        headerbytes = f.read(1024)
        f.close()
        headerdict = mrc.parseHeader(headerbytes)
        x = headerdict["nx"]
        y = headerdict["ny"]
        z = headerdict["nz"]

    return numpy.array([x, y, z])


def readTIFfileandsave(filename, parameters, binning):

    if False:
        new_filename = filename.replace(".tiff", ".mrc").replace(".tif", ".mrc")
        com = "{0}/bin/tif2mrc {1} {2}".format(get_imod_path(), filename, new_filename)
        run_shell_command(com)

        filename = new_filename
        com = "{0}/bin/newstack {1} {1} -mode 2".format(get_imod_path(), filename)
        run_shell_command(com)

    # extract info from .doc file
    extension = Path(filename).suffix
    docfile = filename.replace(extension, "_sum.mrc.mdoc")
    if os.path.isfile(docfile):
        scope_pixel, scope_voltage, scope_mag, defocus = metadata.readMDOCfile(docfile)
        if (
            parameters["scope_pixel"] == "0"
            or "auto" in parameters["scope_pixel"].lower()
        ):
            parameters["scope_pixel"] = str(scope_pixel)
        if (
            parameters["scope_voltage"] == "0"
            or "auto" in parameters["scope_voltage"].lower()
        ):
            parameters["scope_voltage"] = str(scope_voltage)
        if parameters["scope_mag"] == "0" or "auto" in parameters["scope_mag"].lower():
            parameters["scope_mag"] = str(scope_mag)

    return readMoviefileandsave(filename, parameters, binning)


def collate_and_compress(filename):

    logger.info("Compressing {}".format(filename))
    path = os.path.split(filename)[0]
    name = os.path.split(filename)[1]

    files_to_delete = []

    if os.path.exists("{0}/{1}_n0.raw".format(path, name)):
        frames = len(glob.glob(path + "/" + name + "_n*.raw"))
        for i in range(frames):
            f = name + "_n{}".format(i)
            com = "{0}/bin/raw2mrc -x 4096 -y 4096 -z 1 -c -t long -o 49 -f {1}/{2}.raw {2}_test.mrc".format(
                get_imod_path(), path, f
            )
            run_shell_command(com)
            com = "{0}/bin/newstack {1}_test.mrc {1}.mrc -rotate -90".format(
                get_imod_path(), f
            )
            run_shell_command(com)

            files_to_delete.extend(["{0}/{1}.raw".format(path, f)])

            files_to_delete.extend(["{0}.mrc".format(name)])
            files_to_delete.extend(["{0}.mrc".format(f)])
            files_to_delete.extend(["{0}_test.mrc".format(f)])

        com = "{0}/bin/newstack {1}_n?.mrc {1}_frames.mrc".format(get_imod_path(), name)
        run_shell_command(com)
    elif os.path.exists("{0}/{1}_n0.Mrc".format(path, name)):
        frames = len(glob.glob(path + "/" + name + "_n*.Mrc"))
        for i in range(frames):
            f = name + "_n{}".format(i)
            com = "{0}/bin/newstack {2}/{1}.Mrc {1}.mrc -rotate -90".format(
                get_imod_path(), f, path
            )
            run_shell_command(com)

            files_to_delete.extend(["{0}/{1}.Mrc".format(path, f)])
            files_to_delete.extend(["{0}/{1}.xml".format(path, f)])
            files_to_delete.extend(["{0}.xml".format(f)])
            files_to_delete.extend(["{0}.mrc".format(f)])
            shutil.copy2("{0}/{1}.xml".format(path, f), f + ".xml")

        files_to_delete.extend(["{0}.mrc".format(name)])

        com = "{0}/bin/newstack {1}_n*.mrc {1}_frames.mrc".format(get_imod_path(), name)
        run_shell_command(com)
    elif os.path.exists("{0}/{1}_frames_n0.mrc".format(path, name)):
        frames = len(glob.glob(path + "/" + name + "_frames_n*.mrc"))
        for i in range(frames):
            f = name + "_frames_n{}".format(i)
            com = "{0}/bin/newstack {2}/{1}.mrc {1}.mrc -rotate -90".format(
                get_imod_path(), f, path
            )
            run_shell_command(com)

            files_to_delete.extend(["{0}/{1}.mrc".format(path, f)])
            files_to_delete.extend(["{0}/{1}.xml".format(path, f)])
            files_to_delete.extend(["{0}.xml".format(f)])
            files_to_delete.extend(["{0}.mrc".format(f)])
            shutil.copy2("{0}/{1}.xml".format(path, f), f + ".xml")

        files_to_delete.extend(["{0}.mrc".format(name)])

        com = "{0}/bin/newstack {1}_frames_n*.mrc {1}_frames.mrc".format(
            get_imod_path(), name
        )
        run_shell_command(com)
    else:
        shutil.copy2("{0}/{1}_0.mrc".format(path, name), "{0}_frames.mrc".format(name))
        files_to_delete.extend(["{0}/{1}_0.mrc".format(path, name)])

    files_to_delete.extend(["{0}/{1}.mrc".format(path, name)])
    shutil.copy2("{0}/{1}.mrc".format(path, name), "{0}.mrc".format(name))

    # store results in compressed format
    if (
        os.path.exists("{0}.mrc".format(name))
        and os.path.exists("{0}_frames.mrc".format(name))
        and not os.path.exists("{0}/{1}.tbz".format(path, name))
    ):
        command = "tar cvf {0}/{1}.tbz --use-compress-prog=pbzip2 {1}.mrc {1}_n*.xml {1}_frames.mrc".format(
            path, name
        )
        [output, error] = run_shell_command(command)
        if output[0] == 0:
            command = "chmod 444 {0}/{1}.tbz".format(path, name)
            run_shell_command(command)
        else:
            logger.error(".bz2 compression failed on {0}".format(name))
            logger.info(output[1])
            return

    command = "tar tf {0}/{1}.tbz --use-compress-prog=pbzip2".format(path, name)
    logger.info("Testing compressed file")
    [output, error] = run_shell_command(command)
    if output[0] == 0:
        logger.info("Successful {0}.tbz".format(name))
        for fil in files_to_delete:
            if os.path.exists(fil):
                logger.info("Removing %s", fil)
                os.remove(fil)
                # remove signal files as well
                for i in glob.glob("." + os.path.split(fil)[-1].split(".")[0] + "*"):
                    logger.info("Removing %s", i)
                    os.remove(i)
    else:
        logger.error("{0}.tbz file not valid.".format(name))
        logger.error(output)
        com = "rm -f {0}/{1}.tbz".format(path, name)
        [output, error] = run_shell_command(com)
        logger.info(output[1])


def decompress(filename, threads):

    extension = Path(filename).suffix
    if extension == ".bz2" or extension == ".tbz":
        if threads > 0:
            command = "tar xvfj {0}".format(filename)
            command = "pbzip2 -dc -p{0} {1} | tar xv".format(
                threads, filename
            )
        else:
            command = "tar xvf {0} --use-compress-prog=pbzip2".format(filename)
        [files, error] = run_shell_command(command)
        if "ERROR" in files:
            shutil.copy2(filename, os.getcwd())
            if threads > 0:
                command = "pbzip2 -dc -p{0} {1} > {2}".format(
                    threads,
                    os.path.split(filename)[-1],
                    os.path.splitext(os.path.split(filename)[-1])[-2],
                )
            else:
                command = "bunzip2 -d %s" % os.path.split(filename)[-1]
            run_shell_command(command)
            files = os.path.splitext(os.path.split(filename)[-1])[-2]

        # handle zip files with paths
        if "/Images-Disc1" in files:
            if threads > 0:
                command = (
                    "pbzip2 -dc -p%s %s | tar xv --transform='s/.*\///'"
                    % (threads, filename,)
                )
            else:
                command = (
                    "tar xvf %s --use-compress-prog=pbzip2 --transform='s/.*\///'"
                    % filename
                )
            [files, error] = run_shell_command(command)
            run_shell_command("rm -rf Images-Disc1/",)

    else:
        [files, error] = run_shell_command("tar xvfz {0}".format(filename),)

    filenames = [
        s.split("/")[-1]
        for s in files.split()
        if ".mrc" in s or ".dm4" in s or ".tif" in s or ".tiff" in s
    ]

    if os.path.splitext(filenames[0])[0][-5:] == "-0001":
        filename = filenames[0]
    elif "_Fractions" in filenames[0]:
        if os.path.exists(filenames[0].replace("_Fractions", "")):
            filename = filenames[0]
        else:
            # rename frames files
            new_name = filenames[1].replace(".mrc", "_Fractions.mrc")
            os.rename(filenames[0], new_name)
            filename = new_name
    else:
        filename = filenames[-1]
    return filename

def compress_and_delete(filename, compression="tbz", fileset=""):

    logger.info("Compressing {}".format(filename))
    path = os.path.split(filename)[0]
    name = os.path.split(filename)[1]
    os.chdir(path)

    file_list = name + fileset.split(",")[0].split("*")[-1]
    tbz_file = Path(name).stem + "." + compression

    if not os.path.exists(tbz_file) and not ".tif" in file_list:
        if not "tif" in compression.lower():
            command = "tar cvf {0} --dereference --use-compress-prog=pbzip2 {1}".format(
                tbz_file, file_list
            )
        else:
            command = "{0}/bin/mrc2tif -c 5 -s {2} {1}".format(
                get_imod_path(), tbz_file, file_list
            )
        [output, error] = run_shell_command(command)
        if output[0] == 0 or not "ERROR" in output:
            command = "chmod 444 {0}".format(tbz_file)
            run_shell_command(command)
        else:
            logger.error(".bz2 compression failed creating file {0}".format(file_list))
            try:
                logger.info("Removing %s", tbz_file)
                os.remove(tbz_file)
            except:
                logger.exception("Could not delete file %s", tbz_file)
            logger.info(output[1])
            return

    if "tif" in tbz_file:
        if os.path.exists(tbz_file):
            if not "ERROR" in subprocess.check_output(
                get_imod_path() + "/bin/header " + tbz_file,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True,
            ):
                for i in glob.glob(file_list):
                    if file_list != tbz_file:
                        logger.info("Success. Deleting {0}".format(i))
                        os.remove(i)

                    # remove signal files as well
                    # for fil in glob.glob( '.' + i.split('.')[0][:-4] + '*' ):
                    for fil in glob.glob("." + name + "*"):
                        if os.path.exists(fil):
                            os.remove(fil)
                            logger.info("Deleting %s", fil)

    else:
        command = "tar tf {0} --use-compress-prog=pbzip2".format(tbz_file)
        logger.info("Testing compressed file before deleting")
        [output, error] = run_shell_command(command)
        if output[0] != 0 and file_list != "" and not "ERROR" in output and len(error) == 0:
            for i in glob.glob(file_list):
                logger.info("Success. Deleting {0}".format(i))
                os.remove(i)
                # remove signal files as well
                for fil in glob.glob("." + name + "*"):
                    if os.path.exists(fil):
                        os.remove(fil)
                        logger.info("Deleting %s", fil)
        else:
            logger.error("{0} file not valid. Keeping {1}".format(tbz_file, file_list))
            os.remove(tbz_file)
            logger.info(output[1])


def compress_images(input, output, cpus=1):

    os.environ["IMOD_FORCE_OMP_THREADS"] = str(cpus)

    command = "{0}/bin/mrc2tif -O 1 -P -s -c 8 {1} {2} && rm {1}".format(
        get_imod_path(), input, output
    )
    run_shell_command(command, verbose=False)


def tiltseries_to_squares(name, parameters, aligned_tilts, z, square, binning):

    commands = []
    if parameters["tomo_rec_square"]:
        square_enabled = True
    else:
        square_enabled = False
    squares = [ "%s_%04d_square.mrc"%(name, idx) for idx in range(z) ]
    if len(aligned_tilts) > 0:
        from_frames = True
        # make individual tilted images squares
        for tilt in aligned_tilts:
            if square_enabled:
                command = "{0}/bin/newstack {1} {2}_square.mrc -size {3},{3} -taper 1,1 -bin {4}".format(
                    get_imod_path(), tilt, tilt.replace(".mrc", ""), int(square / binning), binning
                )
            else:
                command = "{0}/bin/newstack {1} {2}_square.mrc -taper 1,1 -bin {3}".format(
                    get_imod_path(), tilt, tilt.replace(".mrc", ""), binning
                )
            commands.append(command)
    else:
        for tilt_idx in range(z):
            if square_enabled:
                command = "{0}/bin/newstack -secs {1} {2}.mrc {2}_{1:04d}_square.mrc -size {3},{3} -taper 1,1 -bin {4}".format(
                    get_imod_path(), tilt_idx, name, int(square / binning), binning
                )
            else:
                command = "{0}/bin/newstack -secs {1} {2}.mrc {2}_{1:04d}_square.mrc -taper 1,1 -bin {3}".format(
                    get_imod_path(), tilt_idx, name, binning
                )
            commands.append(command)

    from pyp.system import mpi
    mpi.submit_jobs_to_workers(commands, os.getcwd())
    
    command = "{0}/bin/newstack {1} {2}_square.mrc".format(
        get_imod_path(), " ".join(squares) , name
    )
    # suppress long log
    if parameters["slurm_verbose"]:
        logger.info(command)
    run_shell_command(command, verbose=False)
    os.rename("{0}.mrc".format(name), "{0}.raw.mrc".format(name))
    os.rename("{0}_square.mrc".format(name), "{0}.mrc".format(name))


def get_tilt_axis_angle(name, parameters):

    # figure out tilt-axis angle and store in metadata
    [output, _] = run_shell_command(
        "%s/bin/xf2rotmagstr %s.xf" % (get_imod_path(), name), 
        verbose=parameters["slurm_verbose"],
    )
    xf_rot_mag = output.split("\n")
    axis_mean = counter = 0
    for line in xf_rot_mag:
        if (
            "rot=" in line
            and line.split()[0] == "1:"
        ):
            axis, MAGNIFICATION = (
                float(line.split()[2][:-1]),
                float(line.split()[4][:-1]),
            )
            axis_mean += axis
            counter += 1

    logger.info("Detected TILT AXIS ANGLE = " + str(axis / counter))
    return axis / counter


def generate_aligned_tiltseries(name, parameters, tilt_metadata):

    # align unbinned data
    sec = 0 
    with open(f"{name}.xf", "r") as f:
        for line in f.readlines():
            if len(line) > 1:
                with open(f"{name}_{sec:04d}.xf", "w") as newf:
                    newf.write(line)
                sec += 1

    commands = [] 
    aligned_images = []
    for tilt in range(sec):
        command = "{0}/bin/newstack -input {1}_{2:04d}_square.mrc -output {1}_{2:04d}.ali -xform {1}_{2:04d}.xf -linear -taper 1,1 && rm {1}_{2:04d}_square.mrc {1}_{2:04d}.xf".format(
            get_imod_path(), name, tilt
        )
        commands.append(command)
        aligned_images.append("{0}_{1:04d}.ali".format(name, tilt))

    from pyp.system import mpi
    mpi.submit_jobs_to_workers(commands, os.getcwd())

    command = "{0}/bin/newstack {2} {1}.ali".format(
        get_imod_path(), name, " ".join(aligned_images)
    )
    # suppress long log
    if parameters["slurm_verbose"]:
        logger.info(command)
    run_shell_command(command, verbose=False)
    [os.remove(f) for f in aligned_images]


def cistem_mask_create(parameters: dict, model: str, output: str):
    """
        **   Welcome to CreateMask   **

                Version : 1.00
                Compiled : Dec 15 2022
        Library Version : 2.0.0-alpha--1--dirty
            From Branch : (HEAD
                    Mode : Interactive

    Input image/volume file name [ref.mrc]             :
    Output masked image/volume file name [mask.mrc]    :
    Pixel size of images (A) [1.7]                     :
    Outer radius of mask (A) [200]                     :
    Auto Estimate Binarization threshold? [No]         :
    Wanted initial binarization threshold [0.84]       :
    Low-pass filter resolution (A) [10.0]              :
    Re-Bin Value (0-1) [0.35]                          :
        
    """
    
    output_mask = "frealign/maps/mask.mrc"
    assert (os.path.exists(model)), f"{model} does not exist"
    assert ("particle_rad" in parameters), "Please provide particle radius"

    command = f"{get_imod_path()}/bin/header -pixel {model}"
    [stdo, stdr] = run_shell_command(command)
    model_pixel = float(stdo.split()[0])
    logger.info(f"{model} has pixel size {model_pixel}")

    # calculate normalized threshold
    # choose full map if _crop version specified
    if model.endswith("_crop.mrc"):
        model = model.replace("_crop.mrc",".mrc")
    density_map = mrc.read(model.replace(".mrc","_crop.mrc"))
    density_min = density_map.min()
    density_max = density_map.max()
    if 'mask_normalized' in parameters and parameters['mask_normalized']:
        threshold = density_min + parameters['mask_threshold'] * ( density_max - density_min )
        logger.info(f"Applying normalized density threshold of {threshold:.2f} for masking")
    else:
        threshold = parameters['mask_threshold']
        logger.info(f"Applying density threshold of {threshold:.2f} in range ({density_min:.2f},{density_max:.2f}) for masking")

    sharpenlogfile = "/dev/null"
    command = (
        "{0}/create_mask << eot >> {1} 2>&1\n".format(
            get_frealign_paths()["frealignx"], sharpenlogfile
        )
        + f"{model}\n"
        + f"{output_mask}\n"
        + f"{model_pixel}\n"
        + f"{parameters['particle_rad']}\n"
        + "No\n"
        + f"{threshold}\n"
        + f"{parameters['mask_lowpass']}\n"
        + "0.35\n"
        + "eot\n"
    )
    print(threshold)

    run_shell_command(command, verbose=parameters["slurm_verbose"])
    command = f"{get_imod_path()}/bin/alterheader -del {model_pixel},{model_pixel},{model_pixel} {output_mask}"
    run_shell_command(command, verbose=parameters["slurm_verbose"])

    assert (os.path.exists(output_mask)), f"Threshold falls outside range, try to use a different threshold"
    logger.info(f"Mask {output_mask} created successfully!")

    shape_mask_reference(model, output, output_mask, parameters)


def shape_mask_reference(model: str, masked_model: str, mask: str, parameters: dict):

        
    """
    Image format [M,S,I]?
    Input 3D map?
    Pixel size in A?
    Input 3D mask?
    Width of cosine edge to add (in pixel)?
    Weight for density outside mask?
    Low-pass filter outside (0=no, 1=Gauss, 2=cosine edge)?
    Cosine edge filter radius in A?
    Width of edge in pixels?
    Output masked 3D map?
    """

    frealign_paths = get_frealign_paths()
    mask_weight = parameters["mask_outside_weight"]
    apodization = parameters["mask_edge_width"]

    # copy model to local directory to prevent apply_mask from failing due to a limit in the lenght of the file name
    shutil.copy2( model, Path(model).name )
    model = Path(model).name

    command = """
%s/bin/apply_mask.exe << eot
M
%s
*
%s
%d
%s
2
10
10
%s
eot
""" % (
    frealign_paths["new"],
    model,
    mask,
    apodization,
    mask_weight,
    masked_model,
    )
    subprocess.Popen(command, shell=True, text=True).wait()
