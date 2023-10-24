import math
import os
import shutil

import numpy as np

from pyp.inout.image import mrc, write_out_relion_stack
from pyp.inout.metadata import frealign_parfile
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_frealign_paths, get_imod_path
from pyp.utils import get_relative_path, symlink_relative

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def bandpass(shape, radius1, sigma1, radius2, sigma2):
    """Return highpass filter to be multiplied with fourier transform."""
    # x = np.outer(
    #    np.cos(np.linspace(-math.pi/2., math.pi/2., shape[0])),
    #    np.cos(np.linspace(-math.pi/2., math.pi/2., shape[1])))
    # return (1.0 - x) * (2.0 - x)

    # this->bandPass.resize( this->size[0], this->size[1] / 2 + 1 );
    # Array< Pixel, Dim > radius( this->bandPass.shape() );
    #
    # typename Array< Pixel, Dim > :: iterator iter = radius.begin();
    # while ( iter != radius.end() ){
    #        TinyVector< Pixel, Dim > tmp( ( 1.0 * iter.position() - this->center ) / this->center );
    #        (*iter) = sqrt( sum( tmp * tmp ) );
    #        ++iter;
    # }

    x, y = np.mgrid[-shape[0] / 2 : shape[0] / 2, 0 : shape[1]]
    radius = np.fft.fftshift(np.hypot(x, y), 0)

    band = np.where(
        radius < radius1,
        np.exp(-np.square(radius1 - radius) / 2.0 / sigma1 / sigma1),
        1,
    )
    return np.where(
        radius > radius2,
        np.exp(-np.square(radius - radius2) / 2.0 / sigma2 / sigma2),
        band,
    )


def downsample_image(input, binning=1, method="Fourier"):

    x, y = input.shape

    if "real" in method.lower():

        output = (
            input.reshape(x / binning, binning, y / binning, binning).mean(1).mean(2)
        )

    elif "fourier" in method.lower():

        output = abs(
            np.fft.irfft2(
                np.fft.fftshift(np.fft.rfft2(input - input.min()), 0,)[
                    x / 2 - x / 2 / binning : x / 2 + x / 2 / binning,
                    : y / 2 / binning + 1,
                ]
            )
        )

    return output


def downsample_stack(input, output, binning, method):

    # IMOD's antialias filtering
    if "imod" in method.lower():

        os.environ["IMOD_DIR"] = get_imod_path()
        command = "{0}/bin/newstack -ftreduce {1} {2} {3}".format(
            get_imod_path(), binning, input, output
        )

    elif "resample" in method.lower():

        frealign_paths = get_frealign_paths()

        header = mrc.readHeaderFromFile(input)
        x = int(header["nx"])
        y = int(header["ny"])

        binned_x = int(x / binning)
        binned_y = int(y / binning)

        if input == output:
            tmp_output = input.replace(".mrc", "_tmp.mrc")
        else:
            tmp_output = output

        command = """
%s/bin/resample_mp.exe << eot
%s
%s
no
no
%d
%d
eot
""" % (
            frealign_paths["new"],
            input,
            tmp_output,
            binned_x,
            binned_y,
        )
        # Input filename                         [input.mrc]
        # Output filename                       [output.mrc]
        # Is the input image a volume?                  [NO]
        # Real space binning?                           [NO]
        # New X dimension                              [100]
        # New Y dimension                              [100]

    elif "fft" in method.lower():

        frames = mrc.readHeaderFromFile(input)["nz"]
        for frame in range(frames):
            image = mrc.readframe(input, frame)
            x, y = image.shape
            binned = abs(
                np.fft.irfft2(
                    np.fft.fftshift(np.fft.rfft2(image - image.min()), 0,)[
                        int(x / 2 - x / 2 / binning) : int(x / 2 + x / 2 / binning),
                        : int(y / 2 / binning + 1),
                    ]
                )
            )
            if frame == 0:
                mrc.write(binned, output)
            else:
                mrc.appendArray(binned)

    else:
        message = "Unknown downsampling method: ".format(method)
        raise Exception(message)

    if "fft" not in method.lower():
        run_shell_command(command)

        if "resample" in method.lower() and input == output:
            os.rename(tmp_output, output)


def bin_stack(input, output, binning, method, threads = 1):
    """Downsample image stack in x-y coordinates by specified binning factor.

    Parameters
    ----------
    input : str
        Input file name (.mrc format)
    output : str
        Output file name (.mrc format)
    binning : int
        Factor used to reduce image dimensions
    method : str
        Strategy used for binning: real, imod, or fourier
    """
    # real space binning
    if "real" in method:
        particles = mrc.read(input)
        particles_bin = np.empty(
            [
                particles.shape[0],
                particles.shape[1] / binning,
                particles.shape[2] / binning,
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
        mrc.write(particles_bin, output)

    # IMOD's antialias filtering
    elif "imod" in method:
        env = "export OMP_NUM_THREADS={0}; export NCPUS={0}; IMOD_FORCE_OMP_THREADS={0}; ".format(threads)
        if "antialias" in method:
            option = "-antialias 6"
        else:
            option = ""
        command = env + "{0}/bin/newstack -bin {1} {4} {2} {3}".format(
            get_imod_path(), int(binning), input, output, option
        )
        run_shell_command(command)
        # remove backup to save space
        try:
            os.remove(output + "~")
        except:
            pass

    # binning in reciprocal space
    elif "fourier" in method:
        if os.path.splitext(input)[-1] != ".mrc":
            symlink_relative(input, input + ".mrc")
            infile = input + ".mrc"
        else:
            infile = input

        if os.path.splitext(output)[-1] != ".mrc":
            outfile = output + ".mrc"
        else:
            outfile = output

        size = int(mrc.readHeaderFromFile(input)["ny"]) / binning

        frealign_paths = get_frealign_paths()

        command = """
%s/bin/resample_mp.exe << eot
%s
%s
no
no
%d
%d
eot
""" % (
            frealign_paths["new"],
            infile,
            outfile,
            size,
            size,
        )
        # Input filename                         [input.mrc]
        # Output filename                       [output.mrc]
        # Is the input image a volume?                  [NO]
        # Real space binning?                           [NO]
        # New X dimension                              [100]
        # New Y dimension                              [100]
        [resample_output, error] = run_shell_command(command)
        if infile != input:
            os.remove(infile)
        if outfile != output:
            shutil.move(outfile, output)


def contrast_stretch(input, output="", resize=100):
    if len(output) == 0:
        output = input
    """
    if '6.5.4' in commands.getoutput( '{0}/convert --version'.format( os.environ['IMAGICDIR'] ) ):
        commands.getoutput('{0}/convert {1} -resize {3}% -contrast-stretch 1%x2% {1}'.format(os.environ['IMAGICDIR'], input, output, resize ) )
    else:
        commands.getoutput('{0}/convert {1} -resize {3}% -contrast-stretch 1%x98% {2}'.format(os.environ['IMAGICDIR'], input, output, resize ) )
    """
    run_shell_command(
        "{0}/convert {1} -resize {3}% -contrast-stretch 1%x2% {2}".format(
            os.environ["IMAGICDIR"], input, output, resize
        ),
        verbose=False,
    )


def normalize_volume(image, radius=0, pixelsize=1):
    """Re-implementation of edge normalize for 3D volumes.
       Read an input mrc file and re-writes the result normalized (substract edge mean and scale by the inverse of the edge variance)

    Parameters
    ----------
    volume : volume
        Input volume
    radius : float
        Size of radius in A
    pixelsize : float
        Size of pixel in A

    Returns
    -------
    array
        Normalized volume
    """
    """Compute background region mask"""
    volume = mrc.read(image)
    boxsize = volume.shape[1]
    x, y, z = np.mgrid[0:boxsize, 0:boxsize, 0:boxsize] - boxsize // 2

    if radius == 0:
        radius = boxsize // 2

    if radius / pixelsize > boxsize / 2:
        logger.warning(
            "Particle radius falls outside box %f > %f",
            radius,
            boxsize // 2 * pixelsize,
        )
        radius = boxsize * pixelsize / 2
    condition = (
        np.square(x) + np.square(y) + np.square(y)
        > radius * radius / pixelsize / pixelsize
    )
    if condition.shape == volume.shape:
        background = np.extract(condition, volume)
        mean = background.mean()
        std = background.std()
        volume = (volume - mean) / std
        mrc.write(volume, image)
    else:
        logger.error("Shapes differ {0}, {1}".format(condition.shape, image.shape))
        return [0, 0]


def extract_background(image, radius, pixelsize):
    """Compute background region mask"""
    boxsize = image.shape[1]
    x, y = np.mgrid[0:boxsize, 0:boxsize] - boxsize // 2
    if radius / pixelsize > boxsize / 2:
        logger.warning(
            "Particle radius falls outside box %f > %f",
            radius,
            boxsize // 2 * pixelsize,
        )
        radius = boxsize * pixelsize / 2
    condition = np.hypot(x, y) > radius / pixelsize
    if condition.shape == image.shape:
        background = np.extract(condition, image)
        return [background.mean(), background.std()]
    else:
        logger.error("Shapes differ {0}, {1}".format(condition.shape, image.shape))
        return [0, 0]


def window(input, taper=100):
    # w0 = np.hamming(input.shape[0]).reshape(input.shape[0],1)
    # w1 = np.hamming(input.shape[1]).reshape(1,input.shape[1])

    w0 = np.ones(input.shape[0])
    w0[0:taper] = np.hanning(2 * taper)[0:taper]
    w0[-taper:] = np.hanning(2 * taper)[-taper:]

    w1 = np.ones(input.shape[1])
    w1[0:taper] = np.hanning(2 * taper)[0:taper]
    w1[-taper:] = np.hanning(2 * taper)[-taper:]

    m = input.mean()
    return (input - m) * (
        w0.reshape(input.shape[0], 1) * w1.reshape(input.shape[1], 1)
    ) + m


def compute_running_avg(particle, num_particles, num_frames, window_averaging):
    """Compute running frame averages."""
    frame_weights_width = int(
        math.floor(num_frames * window_averaging)
    )  # width of gaussian used for frame weighting
    if frame_weights_width % 2 == 0:
        frame_weights_width += 1
    all_weights = np.zeros([num_frames, num_frames])
    for i in range(num_frames):
        weights = np.exp(
            -pow((np.arange(num_frames) - float(i)), 2) / frame_weights_width
        )
        all_weights[i, :] = weights / weights.mean() / num_frames

    logger.info(
        "Now weighting frame average for particle %d of %d containing %d frames",
        particle,
        num_particles,
        num_frames,
    )


def normalize_frames(parameters, actual_pixel, indexes, particle_frames):
    background_mean, background_std = extract_background(
        particle_frames.mean(0),
        np.array(parameters["particle_rad"].split(","), dtype=float).max()
        * float(parameters["data_bin"]),
        actual_pixel * float(parameters["extract_bin"]),
    )
    stds = []
    for count in range(len(indexes)):
        particle_frames[count, :, :] = particle_frames[count, :, :] - background_mean
        background_meani, background_stdi = extract_background(
            np.squeeze(particle_frames[count, :, :]),
            np.array(parameters["particle_rad"].split(","), dtype=float).max()
            * float(parameters["data_bin"]),
            actual_pixel * float(parameters["extract_bin"]),
        )
        if background_stdi > 0:
            stds.append(background_stdi)

    std = np.array(stds).mean()
    for count in range(len(indexes)):
        particle_frames[count, :, :] /= std


def normalize_image(image, radius, pixelsize, binning):

    background_mean, background_std = extract_background(
        image, radius, pixelsize * binning
    )

    norm_image = image - background_mean

    if background_std > 0:
        norm_image /= background_std

    return norm_image


def per_particle_normalization_relion(
    parameters, name, current_path, actual_pixel, allparxs, particles
):
    local_particle = np.unique(np.asarray([int(f.split()[0]) - 1 for f in allparxs]))

    for particle in local_particle:
        indexes = frealign_parfile.Parameters.get_particle_indexes(allparxs, particle)
        if len(indexes) > 0:
            particle_frames = particles[indexes, :, :]
            normalize_frames(parameters, actual_pixel, indexes, particle_frames)

    write_out_relion_stack(name, current_path, particles)


def fix_empty_particles(scratch_stackfile, actual_number_of_particles, temp_stack):
    """Replace empty particle frames with Gaussian white noise."""
    empty_frames = 0
    for i in range(actual_number_of_particles):
        frame = mrc.readframe(scratch_stackfile, i)
        if (
            frame.min() == frame.max()
            or np.where(frame == np.median(frame), 0, 1).sum()
            < frame.shape[0] * frame.shape[1] * 0.01
        ):
            frame = np.random.normal(0, 0.1, [frame.shape[0], frame.shape[1]])
            frame -= frame.mean()
            frame /= np.std(frame)
            mrc.write(frame, temp_stack)
            command = "{0}/bin/newstack {1} {2} -replace {3}".format(
                get_imod_path(), temp_stack, scratch_stackfile, i
            )
            run_shell_command(command)
            empty_frames += 1

    if empty_frames > 0:
        logger.warning(
            "Detected %d mostly empty frames (substituted with random noise).",
            empty_frames,
        )


def fix_empty_particles_in_place(frame):
    """Replace empty particle frames with Gaussian white noise."""
    if (
        frame.min() == frame.max()
        or np.where(frame == np.median(frame), 0, 1).sum()
        < frame.shape[0] * frame.shape[1] * 0.01
    ):
        frame = np.random.normal(0, 0.1, [frame.shape[0], frame.shape[1]])
        frame -= frame.mean()
        frame /= np.std(frame)
    return frame


def dose_weight(args, parameters, imagefile, working_path, current_path):

    # find out what index is the 0-tilt
    zero_tilt = np.argmin(np.loadtxt(current_path + "/" + imagefile + ".order"))
    command = "{0}/bin/mtffilter -volt {1} -verbose 1 -dfixed {2} -bidir {3} {4}/{5}.mrc {4}/{5}.mrc".format(
        get_imod_path(),
        "%d" % float(parameters["scope_voltage"]),
        parameters["scope_dose_rate"],
        zero_tilt,
        working_path,
        args.file,
    )
    logger.info(command)
    os.system(command)
