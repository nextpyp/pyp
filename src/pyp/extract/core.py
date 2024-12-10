import math
import os
import random
import shutil
import string
import sys
from pathlib import Path
import numpy as np

import pyp.inout.image as imageio
from pyp.analysis.image import (
    downsample_stack,
    extract_background,
    fix_empty_particles_in_place,
    normalize_image,
)
from pyp.inout.image import (
    get_gain_reference,
    get_image_dimensions,
    mrc,
    readMoviefileandsave,
    readTIFfileandsave,
)
from pyp.inout.metadata import cistem_star_file
from pyp.system import mpi
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_frealign_paths, get_imod_path
from pyp.utils import get_relative_path, timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def extract_particles_old(
    image,
    boxes,
    radius,
    boxsize,
    binning,
    pixelsize,
    normalize=True,
    fixemptyframes=True,
    fromfile="",
    tofile="",
):

    frealign_paths = get_frealign_paths()

    if os.path.exists(fromfile):
        header = mrc.readHeaderFromFile(fromfile)
        nx, ny, frames = header["ny"], header["nx"], header["nz"]
    else:
        nx, ny, frames = image.shape[-2], image.shape[-1], image.ndim - 1

    number_of_particles = len(boxes)

    boxes.reverse()
    count = 0
    while len(boxes) > 0:
        box = boxes.pop()

        # Support for boxes that fall outside of micrograph bound
        # '''
        minx = miny = 0
        maxx = maxy = boxsize
        minX = box[1]
        maxX = box[1] + boxsize
        minY = box[0]
        maxY = box[0] + boxsize
        if minX < 0:
            # logger.warning("Particle %d falls outside X lower range %d", count, minX)
            minx = -minX
            minX = 0
        elif maxX >= nx:
            # logger.warning("Particle %d falls outside X upper range ( %d, %d)", count, maxX, nx)
            maxx = -(maxX - nx + 1)
            maxX = nx - 1
        if minY < 0:
            # logger.warning("Particle %d falls outside Y lower range %d", count, minY)
            miny = -minY
            minY = 0
        elif maxY >= ny:
            # logger.warning("Particle %d falls outside Y upper range ( %d, %d"), count, maxY, ny)
            maxy = -(maxY - ny + 1)
            maxY = ny - 1
        # pdb.set_trace()
        if frames == 1:
            # raw = np.squeeze( image[ box[1]:box[1]+boxsize, box[0]:box[0]+boxsize ] )
            if os.path.exists(fromfile):
                inside = np.squeeze(
                    mrc.read(fromfile)[int(minX) : int(maxX), int(minY) : int(maxY)]
                )
            else:
                inside = np.squeeze(image[int(minX) : int(maxX), int(minY) : int(maxY)])
            if min(inside.shape) > 0:
                raw = inside.mean() * np.ones([boxsize, boxsize])
                raw[int(minx) : int(maxx), int(miny) : int(maxy)] = inside
            else:
                logger.warning("Particle falls completely outside bounds:", box)
                raw = np.zeros([boxsize, boxsize])
        elif frames > 1:
            # raw = np.squeeze( image[ box[2], box[1]:box[1]+boxsize, box[0]:box[0]+boxsize ] )
            if os.path.exists(fromfile):
                inside = np.squeeze(
                    mrc.readframe(fromfile, box[2])[
                        int(round(minX)) : int(round(maxX)),
                        int(round(minY)) : int(round(maxY)),
                    ]
                )
            else:
                inside = np.squeeze(
                    image[
                        box[2],
                        int(round(minX)) : int(round(maxX)),
                        int(round(minY)) : int(round(maxY)),
                    ]
                )

            inside = inside.reshape(
                int(round(maxX)) - int(round(minX)), int(round(maxY)) - int(round(minY))
            )
            # inside = inside.reshape(int(round(maxX))-int(round(minX)), int(round(maxY))-int(round(minY)))

            if min(inside.shape) > 0:
                raw = inside.mean() * np.ones([boxsize, boxsize])
                # print raw[ int(minx):int(maxx), int(miny):int(maxy) ].shape, inside.shape
                # print miny, maxy, minY, maxY
                try:
                    # raw[ int(minx):int(maxx), int(miny):int(maxy) ] = inside
                    raw[
                        int(round(minx)) : int(round(maxx)),
                        int(round(miny)) : int(round(maxy)),
                    ] = inside
                except:
                    # print 'ERROR - Dimensions do not match', raw[ int(minx):int(maxx), int(miny):int(maxy) ].shape, inside.shape
                    logger.info(
                        "ERROR - Dimensions do not match",
                        raw[
                            int(round(minx)) : int(round(maxx)),
                            int(round(miny)) : int(round(maxy)),
                        ].shape,
                        inside.shape,
                    )
                    logger.info(minx, maxx, miny, maxy)
                    raw[
                        int(round(minx)) : int(round(maxx)),
                        int(round(miny)) : int(round(maxy)) - 1,
                    ] = inside
                    pass
                # raw[ int(round(minx)):int(round(maxx)), int(round(miny)):int(round(maxy)) ] = inside
            else:
                logger.warning(
                    "Particle falls completely outside bounds:",
                    minx,
                    maxx,
                    miny,
                    maxy,
                )
                raw = np.zeros([boxsize, boxsize])

        if fixemptyframes:
            raw = fix_empty_particles_in_place(raw)

        if len(tofile) > 0:
            if normalize:
                raw = normalize_image(raw, radius, pixelsize, binning)
            if count == 0:
                mrc.write(raw, tofile)
            else:
                mrc.append(raw, tofile)
        else:
            if count == 0:
                particles = np.empty(
                    [number_of_particles, boxsize, boxsize], dtype=np.float32
                )
            particles[count, :, :] = raw
        count += 1

    if binning > 1:

        method = "imod"
        # method = 'resample'

        # binning in reciprocal space
        if "real" in method:
            # real space binning
            particles_bin = np.empty(
                [
                    particles.shape[0],
                    int(particles.shape[1] / binning),
                    int(particles.shape[2] / binning),
                ]
            )
            particles_bin = (
                particles.reshape(
                    particles.shape[0],
                    1,
                    particles.shape[1] / binning,
                    binning,
                    particles.shape[2] / binning,
                    binning,
                )
                .mean(1)
                .mean(2)
                .mean(3)
            )
        elif "imod" in method:
            # IMOD's antialias filtering
            name = "".join(random.choice(string.ascii_letters) for i in range(20))
            if len(tofile) == 0:
                mrc.write(particles, name + "_unbinned.mrc")
                command = "{0}/bin/newstack -bin {1} -antialias 6 {2}_unbinned.mrc {2}_binned.mrc".format(
                    get_imod_path(), int(binning), name
                )
                run_shell_command(command)
                particles_bin = np.array(mrc.read(name + "_binned.mrc"), ndmin=3)
                os.remove(name + "_binned.mrc")
                os.remove(name + "_unbinned.mrc")
            else:
                command = "{0}/bin/newstack -bin {1} -antialias 6 {2} {3}_binned.mrc".format(
                    get_imod_path(), int(binning), tofile, name
                )
                run_shell_command(command)
                shutil.move("{0}_binned.mrc".format(name), tofile)
        elif "resample" in method:

            name = "".join(random.choice(string.ascii_letters) for i in range(20))
            mrc.write(particles, name + "_unbinned.mrc")
            command = """
%s/bin/resample_mp.exe << eot
%s_unbinned.mrc
%s_binned.mrc
no
no
%d
%d
eot
""" % (
                frealign_paths["new"],
                name,
                name,
                particles_bin.shape[1],
                particles_bin.shape[2],
            )
            # Input filename                         [input.mrc]
            # Output filename                       [output.mrc]
            # Is the input image a volume?                  [NO]
            # Real space binning?                           [NO]
            # New X dimension                              [100]
            # New Y dimension                              [100]
            [output, error] = run_shell_command(command)
            logger.info(output)

            particles_bin = np.array(mrc.read(name + "_binned.mrc"), ndmin=3)
            os.remove(name + "_binned.mrc")
            os.remove(name + "_unbinned.mrc")

        else:
            for count in range(particles.shape[0]):
                # binning by taking center of fft
                particles_bin[count, :, :] = abs(
                    np.fft.irfft2(
                        np.fft.fftshift(
                            np.fft.rfft2(
                                particles[count, :, :] - particles[count, :, :].min()
                            ),
                            0,
                        )[
                            particles.shape[1] / 2
                            - particles.shape[1] / 2 / binning : particles.shape[1] / 2
                            + particles.shape[1] / 2 / binning,
                            : particles.shape[2] / 2 / binning + 1,
                        ]
                    )
                )

        if len(tofile) == 0:
            particles = particles_bin

    # not currently using
    if False and normalize:
        normalized_stack = tofile.replace(".mrc", "_binned.mrc")
        for count in range(number_of_particles):
            # background_mean, background_std = extract_background( particles[count,:,:], radius, pixelsize * binning )
            if False and len(tofile) > 0:
                frame = mrc.readFrame(tofile, count)
            else:
                frame = particles[count, :, :]
            background_mean, background_std = extract_background(
                frame, radius, pixelsize * binning
            )
            frame -= background_mean
            if background_std > 0:
                frame /= background_std
            if len(tofile) > 0:
                if count == 0:
                    mrc.write(frame, normalized_stack)
                else:
                    mrc.append(normalized_stack, count)
            else:
                particles[count, :, :] = frame

        if len(tofile) > 0:
            shutil.move(normalized_stack, tofile)

    if len(tofile) > 0:
        return number_of_particles
    else:
        return particles


def extract_particles(
    input,
    output,
    boxes,
    radius,
    boxsize,
    binning,
    pixelsize,
    cpus,
    parameters,
    normalize=True,
    fixemptyframes=True,
    method="imod",
    is_tomo=False,
    use_frames=False,
):

    if is_tomo or use_frames:
        number_of_particles = extract_particles_mpi(
            input,
            output,
            boxes,
            radius,
            boxsize,
            binning,
            pixelsize,
            cpus,
            parameters,
            normalize,
            fixemptyframes,
            method,
            use_frames,
            extract=False,
        )
    else:
        number_of_particles = extract_particles_non_mpi(
            input,
            output,
            boxes,
            radius,
            boxsize,
            binning,
            binning,
            pixelsize,
            normalize,
            fixemptyframes,
            method,
        )

    return number_of_particles


def extract_particles_non_mpi(
    input,
    output,
    cistem_obj,
    radius,
    boxsize,
    image_binning,
    coordinate_binning,
    pixelsize,
    normalize=True,
    fixemptyframes=True,
    method="imod",
    use_frames=False,
    frames_list=[],
):
    """[summary]

    Parameters
    ----------
    input : str
        Input movie or tilt-series
    output : str
        File name of extracted particles
    boxes : list
        List of positions to extract
    radius : float
        Particle radius to use for normalization
    boxsize : int
        Size of box to extract particles in pixels
    image_binning : int
        Binning to apply to image data for extraction
    coordinate_binning : int
        Binning to apply to particle coordinates during extraction
    pixelsize : float
        Size of pixel in A
    normalize : bool, optional
        Normalize each extracted particle, by default True
    fixemptyframes : bool, optional
        Fix any empty frames, by default True
    method : str, optional
        Method to use for image downsampling, by default "imod". The only other option is "resample".

    Returns
    -------
    int
        Number of particles extracted
    """

    # downsample stack
    if image_binning > 1:
        binned_input = Path(os.environ["PYP_SCRATCH"]) / Path(input).name
        downsample_stack(input, binned_input, image_binning, method)
        input = str(binned_input)

    # read stack into memory
    if use_frames and len(frames_list):
        # read multiple tif tilt movies
        image = [mrc.read(im) for im in frames_list]
        nx, ny, frames = image[0].shape[-2], image[0].shape[-1], image[0].ndim - 1
    else:
        # read just one tiltseries
        if input[-4:] == ".mrc":
            image = mrc.read(input)
        else:
            image = mrc.read(input + ".mrc")
        nx, ny, frames = image.shape[-2], image.shape[-1], image.ndim - 1
    
    if isinstance(cistem_obj,cistem_star_file.ExtendedParameters) or isinstance(cistem_obj[0],cistem_star_file.Parameters):
        boxes_obj = cistem_obj[0] # only read 1 class

        x_coord_col = boxes_obj.get_index_of_column(cistem_star_file.ORIGINAL_X_POSITION)
        y_coord_col = boxes_obj.get_index_of_column(cistem_star_file.ORIGINAL_Y_POSITION)
        boxes = boxes_obj.get_data()[:, [x_coord_col, y_coord_col]].tolist()
    else:
        boxes = cistem_obj
    
    boxes.reverse()
    count = 0
    while len(boxes) > 0:
        box = boxes.pop()

        # Support for boxes that fall outside of micrograph bound
        # '''
        minx = miny = 0
        maxx = maxy = boxsize
        minX = math.floor(
            box[1] / float(coordinate_binning) - math.floor(boxsize / 2.0)
        )
        maxX = minX + boxsize
        minY = math.floor(
            box[0] / float(coordinate_binning) - math.floor(boxsize / 2.0)
        )
        maxY = minY + boxsize

        if minX < 0:
            # logger.warning("Particle %d falls outside X lower range %d", count, minX)
            minx = -minX
            minX = 0
        elif maxX >= nx:
            # logger.warning("Particle %d falls outside X upper range ( %d, %d)", count, maxX, nx)
            maxx = -(maxX - nx + 1)
            maxX = nx - 1

        if minY < 0:
            # logger.warning("Particle %d falls outside Y lower range %d", count, minY)
            miny = -minY
            minY = 0
        elif maxY >= ny:
            # logger.warning("Particle %d falls outside Y upper range ( %d, %d)", count, maxY, ny)
            maxy = -(maxY - ny + 1)
            maxY = ny - 1

        if frames == 1:
            inside = np.squeeze(image[int(minX) : int(maxX), int(minY) : int(maxY)])
            if min(inside.shape) > 0:
                raw = inside.mean() * np.ones([boxsize, boxsize])
                raw[int(minx) : int(maxx), int(miny) : int(maxy)] = inside
            else:
                logger.warning(
                    "Particle falls completely outside bounds:",
                    box[0] / coordinate_binning,
                )
                raw = np.zeros([boxsize, boxsize])
        elif False and frames > 1:
            if use_frames:
                # if frames, coordinates would be like = [ x, y, tilt, frame ]
                extraction = image[box[2]][
                    box[3],
                    int(round(minX)) : int(round(maxX)),
                    int(round(minY)) : int(round(maxY)),
                ]
            else:
                extraction = image[
                    box[2],
                    int(round(minX)) : int(round(maxX)),
                    int(round(minY)) : int(round(maxY)),
                ]
            inside = np.squeeze(extraction)

            inside = inside.reshape(
                int(round(maxX)) - int(round(minX)), int(round(maxY)) - int(round(minY))
            )

            if min(inside.shape) > 0:
                raw = inside.mean() * np.ones([boxsize, boxsize])
                try:
                    raw[
                        int(round(minx)) : int(round(maxx)),
                        int(round(miny)) : int(round(maxy)),
                    ] = inside
                except:
                    logger.info(
                        "ERROR - Dimensions do not match",
                        raw[
                            int(round(minx)) : int(round(maxx)),
                            int(round(miny)) : int(round(maxy)),
                        ].shape,
                        inside.shape,
                    )
                    logger.info(minx, maxx, miny, maxy)
                    raw[
                        int(round(minx)) : int(round(maxx)),
                        int(round(miny)) : int(round(maxy)) - 1,
                    ] = inside
                    pass
            else:
                logger.warning(
                    "Particle falls completely outside bounds: [%d, %d, %d, %d]"
                    % (minx, maxx, miny, maxy)
                )
                raw = np.zeros([boxsize, boxsize])

        if fixemptyframes:
            raw = fix_empty_particles_in_place(raw)
        if normalize:
            raw = normalize_image(raw, radius, pixelsize, coordinate_binning)

        # write out result
        if count == 0:
            mrc.write(raw, output)
        else:
            mrc.append(raw, output)
        count += 1

    return count


def extract_particles_mpi(
    input,
    output,
    boxes,
    radius,
    boxsize,
    binning,
    pixelsize,
    cpus,
    parameters,
    normalize=True,
    fixemptyframes=True,
    method="imod",
    use_frames=False,
    extract=True,
):
    """Particle frame extraction in parallel.

    Parameters
    ----------
    input : str
        Input movie or tilt-series
    output : str
        Output particle stack
    boxes : list
        List of position to extract
    radius : float
        Particle radius to use for background normalization
    boxsize : int
        Size of boxes to extract
    binning : int
        Downsample extracted particles by this factor
    pixelsize : float
        Pixel size of images
    cpus : int
        Number of cpus available to run program in parallel
    normalize : bool, optional
        Normalize extracted frames, by default True
    fixemptyframes : bool, optional
        Fix empty frames, by default True
    method : str, optional
        Method to use for image downsampling, by default "imod". The only other option available is "resample".

    Returns
    -------
    int
        Number of particles extracted
    """
    frames_list = []

    if use_frames:

        # get the dimension first
        dims = get_image_dimensions(input[0])

        # convert tif movies to mrc files
        commands = [] 
        for f in input:
            if ".tif" in f:
                com = "{0}/bin/newstack -mode 2 {1} {2}".format(
                    get_imod_path(), f, f.replace(".tiff", ".mrc").replace(".tif", ".mrc")
                )
                commands.append(com)
        if len(commands) > 0:
            mpi.submit_jobs_to_workers(commands, os.getcwd())
            
        input = [f.replace(".tiff", ".mrc").replace(".tif", ".mrc") for f in input]
        
        # remove x-ray hot pixels
        """
        arguments = []
        for f in input:
            arguments.append((f.strip(".mrc"), False))

        mpi.submit_function_to_workers(remove_xrays_from_movie_file, arguments)
        """

        # gain correction
        gain_reference, gain_reference_file = get_gain_reference(
            parameters, dims[0], dims[1]
        )

        if gain_reference_file is not None:

            commands = []

            for movie in input:
                com = '{0}/bin/clip multiply -m 2 {1} "{2}" {1}; rm -f {1}~'.format(
                    get_imod_path(), movie, gain_reference_file,
                )
                commands.append(com)

            mpi.submit_jobs_to_workers(commands, os.getcwd())

        # down-sample images
        arguments = []
        if binning > 1:
            for f in input:
                arguments.append((f, "frealign/" + f, binning, method))
            mpi.submit_function_to_workers(downsample_stack, arguments, verbose=parameters["slurm_verbose"])

            [os.remove(f) for f in input]
        else:
            for f in input:
                arguments.append((f, "frealign/" + f))
            mpi.submit_function_to_workers(os.rename, arguments, verbose=parameters["slurm_verbose"])


        # write a txt containing the filename of frames (for stackless CSP)
        frames_list = ["frealign/" + f for f in input]
        with open("frames_csp.txt", "w") as f:
            f.write("\n".join(frames_list))

    else:
        if binning > 1:
            downsample_stack(input, "frealign/" + input, binning, method)
        else:
            shutil.copy2(input, "frealign/" + input)

    number_of_particles = boxes[0].get_num_rows()# len(boxes)

    # bypass extraction
    if not extract:
        return number_of_particles

    if cpus > 1:
        chunks = max(math.ceil(len(boxes) / (cpus - 1)), 10000)
    else:
        chunks = len(boxes)
    split_boxes = [boxes[i : i + chunks] for i in range(0, len(boxes), chunks)]

    image_binning = 1
    coordinate_binning = binning

    movie_list = []
    arguments = []
    count = 0
    for chunk in split_boxes:
        chunk_output = output.replace(
            ".mrc",
            "_%08d_%08d.mrc" % (count, min(count + chunks, number_of_particles - 1)),
        )
        movie_list.append(chunk_output)
        arguments.append(
            (
                output,
                chunk_output,
                chunk,
                radius,
                boxsize,
                image_binning,
                coordinate_binning,
                pixelsize,
                normalize,
                fixemptyframes,
                method,
                use_frames,
                frames_list,
            )
        )
        count += chunks
    mpi.submit_function_to_workers(extract_particles_non_mpi, arguments, verbose=parameters["slurm_verbose"])

    if len(split_boxes) > 1:
        # merge stacks
        logger.info("Merging {} partial stacks into {}".format(len(movie_list), output))
        mrc.merge_fast(movie_list, output, remove=True)
    else:
        shutil.move(movie_list[0], output)

    return number_of_particles


def extract_stacks_particle_cspt(
    dataset, main_stackfile, ptlidx_regions_list, parx_object, mode="M", cpus=70
):
    """Extract particle stacks specifically for frame refinement, the naming follows {dataset}_region????_M??????.mrc

    Parameters
    ----------
    dataset : str
        The name of the dataset
    main_stackfile : str
        The path of the main parfile (particle stack for a tilt-series)
    ptlidx_regions_list : list[list]
        Nested list containing particle indexes in squares
    parx_object : Parameters
        Frealign parameter file object
    cpus : int
        The number of cpus/threads
    """

    root_stack = "/scratch/" + dataset + "_frames_CSP_01"
    root_micrograph_stack = root_stack + "_region????_M??????_stack.mrc"
    root_particle_stack = root_stack + "_region????_P??????_stack.mrc"

    command_list = []

    idx = 0
    for region in ptlidx_regions_list:

        if not len(region) > 0:
            continue

        if "M" in mode.upper():
            name_region = root_micrograph_stack.replace(
                "region????", "region%04d" % idx
            )

            local_frames = np.unique(parx_object.data[:, 19].astype("int"))

            for frame in local_frames:

                # find all lines corresponding to current micrograph
                indexes = np.argwhere(
                    (parx_object.data[:, 19] == frame)
                    & (np.isin(parx_object.data[:, 16], region))
                )

                # name of output stack
                name = name_region.replace("M??????", "M%06d" % frame)

                if not os.path.exists(name):

                    # work around newstack's -secs limitation
                    chunk_size = 50
                    list = indexes.astype("str").squeeze().tolist()

                    split_sections = ""

                    if len(indexes) == 0:
                        continue

                    elif len(indexes) == 1:
                        split_sections += " -secs " + list

                    else:
                        chunks = [
                            list[i : i + chunk_size]
                            for i in range(0, len(list), chunk_size)
                        ]
                        for chunk in chunks:
                            split_sections += " -secs " + ",".join(chunk)

                    # newstack command
                    command = "{0}/bin/newstack {1} {2} {3}".format(
                        get_imod_path(), main_stackfile, split_sections, name
                    )

                    command_list.append(command)

            idx += 1

        elif "P" in mode.upper():
            name_region = root_particle_stack.replace("region????", "region%04d" % idx)

            for particle in region:

                indexes = np.argwhere(parx_object.data[:, 16] == particle)

                name = name_region.replace("P??????", "P%06d" % particle)

                if not os.path.exists(name):
                    os.symlink(name.replace("_region%04d" % idx, ""), name)

            idx += 1

    if len(command_list) > 0:
        mpi.submit_jobs_to_workers(command_list, os.getcwd())
