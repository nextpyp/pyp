import glob
import math
import multiprocessing
import os
import random
import shutil
import string
import subprocess
import sys

import numpy as np
import scipy
from pathlib import Path

from pyp import utils
from pyp.analysis import plot
from pyp.analysis.image import bandpass, contrast_stretch
from pyp.inout.image import digital_micrograph as dm4
from pyp.inout.image import mrc, mrc2png, mrc2webp, writepng
from pyp.inout.image.core import get_image_dimensions
from pyp.system import local_run, mpi
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import (
    get_ctffind4_path,
    get_ctffind_tilt_path,
    get_frealign_paths,
    get_imod_path,
    get_tomoctf_path,
    imod_load_command,
    timeout_command,
)
from pyp.system.wrapper_functions import avgstack, tomo_ctf_grad
from pyp.utils import get_relative_path
from pyp.utils.timer import Timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def is_required_3d(parameters):
    return float(parameters["ctf_max_res"]) > 0

def is_done(metadata,parameters, name, project_dir="."):
    force = "ctf_force" in parameters and parameters["ctf_force"]
    if parameters["data_mode"] == "spr":
        return metadata != None and "ctf" in metadata and not force
    else:
        have_images = len(glob.glob(os.path.join(project_dir, "webp", name + "_?D_ctftilt.webp"))) == 2
        return metadata != None and "global_ctf" in metadata and "ctf" in metadata and not force and have_images

def ctffind4(power_spectra, parameters, meandef, tolerance=500):

    logfile = "ctffind4.log"

    ctffind_command = (
        "%s/ctffind --amplitude-spectrum-input --old-school-input --omp-num-threads=%s"
        % (get_ctffind4_path(), parameters["slurm_tasks"])
    )

    dstep = (
        float(parameters["extract_bin"])
        * float(parameters["scope_pixel"])
        * float(parameters["scope_mag"])
        / 10000.0
    )

    ctf_max_res = parameters["ctf_max_res"]

    command = """
%s > %s 2>&1 << EOF
%s
power.mrc
%s,%s,%s,%s,%s
%s,%s,%s,%s,%s,%s,%s
EOF
""" % (
        timeout_command(ctffind_command, 600, full_path=True),
        logfile,
        power_spectra,
        parameters["scope_cs"],
        parameters["scope_voltage"],
        parameters["scope_wgh"],
        parameters["scope_mag"],
        dstep,
        parameters["ctf_tile"],
        parameters["ctf_min_res"],
        ctf_max_res,
        meandef - tolerance,
        meandef + tolerance,
        parameters["ctf_fstep"],
        parameters["ctf_dast"],
    )
    [output, error] = local_run.run_shell_command(command)

    with open(logfile) as f:
        log = f.read()

        # parse output and return df1, df2, angast and CC
        return np.array(
            [line for line in log.split("\n") if "Final Values" in line][0].split()
        )[[0, 1, 2, 3]].astype("f")


def ctffind4_movie(movie, parameters, average=4):

    logfile = "ctffind4.log"
    if "ctffind5" in parameters['ctf_method']:
        ctffind5 = True
        ctffind_command = f"{get_frealign_paths()['cistem2']}/ctffind5"
    else:
        ctffind5 = False
        ctffind_command = f"{get_frealign_paths()['cistem2']}/ctffind"

    if os.path.exists(movie + ".avg"):
        source = movie + ".avg"
    elif os.path.exists(movie + ".mrc"):
        source = movie + ".mrc"
    os.symlink(source, "ctffind4.mrc")
    movie = "ctffind4"

    if parameters["ctf_use_ast"]:
        use_ast = f"yes\n{parameters['ctf_dast']}"
    else:
        use_ast = f"yes\n0"

    if parameters["ctf_known_ast"] > 0:
        known_ast = "Yes"
        use_restraint_ast = ""
        known_ast_value = f"{parameters['ctf_known_ast']}\n"
        known_ast_angle = f"{parameters['ctf_known_ast_angle']}"
    else:
        known_ast = "No"
        use_restraint_ast = "No"
        known_ast_value= ""
        known_ast_angle = ""

    if parameters["ctf_determine_thickness"]:
        determine_thickness = "Yes\nNo\nNo\n30.0\n3.0\nNo\nNo"
    else:
        determine_thickness = "No"

    exhaustive = "No"
    if parameters["ctf_phase_shift"]:
        phase_shift = f"Yes\n{parameters['ctf_min_rad']}\n{parameters['ctf_max_rad']}\n{parameters['ctf_ps_step']}"
        determine_tilt = ""
    else:    
        phase_shift = "No"
        if parameters["ctf_determine_tilt"]:
            determine_tilt = "Yes\n"
        else:
            determine_tilt = "No\n"
    

    
    # use frame average
    if mrc.readHeaderFromFile("ctffind4.mrc")["nz"] < 3:
        
        if ctffind5:

            """        
    **   Welcome to Ctffind   **

            Version : 5.0.2
        Compiled : Mar 28 2024
    Library Version : 2.0.0-alpha-295-b21db55-dirty
        From Branch : ctffind5_merge
            Mode : Interactive

Input image file name
[14sep05c_00024sq_00003hl_00002es.frames.mrc]      : 
Output diagnostic image file name
[diagnostic_output.mrc]                            : 
Pixel size [0.66]                                  : 
Acceleration voltage [300.0]                       : 
Spherical aberration [2.70]                        : 
Amplitude contrast [0.07]                          : 
Size of amplitude spectrum to compute [512]        : 
Minimum resolution [30.0]                          : 
Maximum resolution [5.0]                           : 
Minimum defocus [5000.0]                           : 
Maximum defocus [50000.0]                          : 
Defocus search step [100.0]                        : 
Do you know what astigmatism is present? [No]      : 
Slower, more exhaustive search? [No]               : 
Use a restraint on astigmatism? [No]               : 
Find additional phase shift? [No]                  : 
Determine sample tilt? [No]                        : 
Determine samnple thickness? [No]                  :Yes
Use brute force 1D search? [Yes]                   : 
Use 2D refinement? [Yes]                           : 
Low resolution limit for nodes [30.0]              : 
High resolution limit for nodes [3.0]              : 
Use rounded square for nodes? [No]                 : 
Downweight nodes? [No]                             : 
Do you want to set expert options? [No]            : Yes
Resample micrograph if pixel size too small? [Yes] : 
Target pixel size after resampling [1.4]           : 
Do you already know the defocus? [No]              : 
Weight down low resolution signal? [Yes]           : 
Desired number of parallel threads [1]             : 
            """
            # ctffind5
            command = f"""{timeout_command(ctffind_command, 600, full_path=True)} > {logfile} 2>&1 << EOF
{movie}.mrc
power.mrc
{parameters['scope_pixel'] * parameters['data_bin']}
{parameters['scope_voltage']}
{parameters['scope_cs']}
{parameters['scope_wgh']}
{parameters['ctf_tile']}
{parameters['ctf_min_res']}
{parameters['ctf_max_res']}
{parameters['ctf_min_def']}
{parameters['ctf_max_def']}
{parameters['ctf_fstep']}
{known_ast}
{exhaustive}
{use_restraint_ast}{known_ast_value}{known_ast_angle}
{phase_shift}
{determine_tilt}{determine_thickness}
Yes
Yes
1.4
No
Yes
{parameters["slurm_tasks"]}
EOF
"""
        else:
            # version 4.1
            if not parameters["ctf_use_phs"]:
                if True:
                    # cistem

                    command = """
    %s > %s 2>&1 << EOF
    %s.mrc
    power.mrc
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    no
    no
    %s
    no
    no
    no
    EOF
    """ % (
                        timeout_command(ctffind_command, 600, full_path=True),
                        logfile,
                        movie,
                        parameters["scope_pixel"] * parameters["data_bin"],
                        parameters["scope_voltage"],
                        parameters["scope_cs"],
                        parameters["scope_wgh"],
                        parameters["ctf_tile"],
                        parameters["ctf_min_res"],
                        parameters["ctf_max_res"],
                        parameters["ctf_min_def"],
                        parameters["ctf_max_def"],
                        parameters["ctf_fstep"],
                        use_ast,
                    )
            
                """
                # **   Welcome to Ctffind   **
                #      Version : 4.1.14
                #      Compiled : Mar 27 2020
                #  Mode : Interactive
                #  Input image file name [input.mrc]                  : ctffind4.mrc
                #  Input is a movie (stack of frames) [No]            : Yes
                #  Number of frames to average together [1]           : 4
                #  Output diagnostic image file name
                #  [diagnostic_output.mrc]                            : power.mrc
                #  Pixel size [1.0]                                   : 1.08
                #  Acceleration voltage [300.0]                       : 300
                #  Spherical aberration [2.70]                        : 2.7
                #  Amplitude contrast [0.07]                          : 0.07
                #  Size of amplitude spectrum to compute [512]        : 512
                #  Minimum resolution [30.0]                          : 15
                #  Maximum resolution [5.0]                           : 3.5
                #  Minimum defocus [5000.0]                           : 3500
                #  Maximum defocus [50000.0]                          : 50000
                #  Defocus search step [100.0]                        : 250
                #  Do you know what astigmatism is present? [No]      : No
                #  Slower, more exhaustive search? [No]               : No
                #  Use a restraint on astigmatism? [No]               : No
                #  Find additional phase shift? [No]                  : No
                #  Determine sample tilt? [No]                        : No
                #  Do you want to set expert options? [No]            : No
                """
            
            else:

                command = """
    %s > %s 2>&1 << EOF
    %s.mrc
    power.mrc
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    no
    no
    yes
    %s
    yes
    0.0
    3.15
    0.5
    no
    EOF
    """ % (
                    timeout_command(ctffind_command, 600, full_path=True),
                    logfile,
                    movie,
                    parameters["scope_pixel"] * parameters["data_bin"],
                    parameters["scope_voltage"],
                    parameters["scope_cs"],
                    parameters["scope_wgh"],
                    parameters["ctf_tile"],
                    parameters["ctf_min_res"],
                    parameters["ctf_max_res"],
                    parameters["ctf_min_def"],
                    parameters["ctf_max_def"],
                    parameters["ctf_fstep"],
                    parameters["ctf_dast"],
                )
        
                """
                # Input image file name
                # Output diagnostic filename [diagnostic_output.mrc]
                # Pixel size                                [0.3185]
                # Acceleration voltage                       [300.0]
                # Spherical aberration                         [2.7]
                # Amplitude contrast                          [0.07]
                # Size of power spectrum to compute            [512]
                # Minimum resolution                          [30.0]
                # Maximum resolution                             [3]
                # Minimum defocus                           [5000.0]
                # Maximum defocus                          [50000.0]
                # Defocus search step                        [500.0]
                # Do you know what astigmatism is present?      [no]
                # Slower, more exhaustive search?              [yes]
                # Use a restraint on astigmatism?              [yes]
                # Expected (tolerated) astigmatism           [200.0]
                # Find additional phase shift?                 [yes]
                # Minimum phase shift (rad)                    [0.0]
                # Maximum phase shift (rad)                   [3.15]
                # Phase shift search step                      [0.5]
                # Do you want to set expert options?             [y]
                """

    # use movie frames
    else:
        if True:

            # cistem

            command = """
%s > %s 2>&1 << EOF
%s.mrc
yes
%d
power.mrc
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
no
no
no
no
no
no
EOF
""" % (
                timeout_command(ctffind_command, 600, full_path=True),
                logfile,
                movie,
                average,
                parameters["scope_pixel"] * parameters["data_bin"],
                parameters["scope_voltage"],
                parameters["scope_cs"],
                parameters["scope_wgh"],
                parameters["ctf_tile"],
                parameters["ctf_min_res"],
                parameters["ctf_max_res"],
                parameters["ctf_min_def"],
                parameters["ctf_max_def"],
                parameters["ctf_fstep"],
            )

    # **   Welcome to Ctffind   **
    #      Version : 4.1.14
    #      Compiled : Mar 27 2020
    #  Mode : Interactive
    #  Input image file name [input.mrc]                  : ctffind4.mrc
    #  Input is a movie (stack of frames) [No]            : Yes
    #  Number of frames to average together [1]           : 4
    #  Output diagnostic image file name
    #  [diagnostic_output.mrc]                            : power.mrc
    #  Pixel size [1.0]                                   : 1.08
    #  Acceleration voltage [300.0]                       : 300
    #  Spherical aberration [2.70]                        : 2.7
    #  Amplitude contrast [0.07]                          : 0.07
    #  Size of amplitude spectrum to compute [512]        : 512
    #  Minimum resolution [30.0]                          : 15
    #  Maximum resolution [5.0]                           : 3.5
    #  Minimum defocus [5000.0]                           : 3500
    #  Maximum defocus [50000.0]                          : 50000
    #  Defocus search step [100.0]                        : 250
    #  Do you know what astigmatism is present? [No]      : No
    #  Slower, more exhaustive search? [No]               : No
    #  Use a restraint on astigmatism? [No]               : No
    #  Find additional phase shift? [No]                  : No
    #  Determine sample tilt? [No]                        : No
    #  Do you want to set expert options? [No]            : No

    # make sure ctffind runs in parallel
    if not "ctffind5"  in parameters["ctf_method"]:
        command = (
            "export OMP_NUM_THREADS={0}; export NCPUS={0}; ".format(
                parameters["slurm_tasks"]
            )
            + command
        )
    
    [ output, error ] = local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])

    try:
        if parameters["slurm_verbose"]:
            with open(logfile, "r") as f:
                ctffind4 = f.read()
                logger.info(ctffind4)
        if not ctffind5:
            # parse output and return df1, df2, angast and CC
            return np.loadtxt("power.txt", comments="#", dtype="f")[[1, 2, 3, 5, 6]]
        else:
            # df1, df2, angast, phaseshift (rad), CC, fit resolution (A), tilt axis, tilt angle, sample thickness (A) 
            return np.loadtxt("power.txt", comments="#", dtype="f")[[1, 2, 3, 5, 6, 7, 8, 9]]
    except:
        logger.error("ctffind failed, aborting.")
        logger.info(error)
        logger.info(output)
        sys.exit(0)


# Run CTFFIND4 using power spectra
def ctffind4_power(power_spectra, parameters, meandef=0, tolerance=500):

    if meandef > 0:
        mindefocus = meandef - tolerance
        maxdefocus = meandef + tolerance
    else:
        mindefocus = parameters["ctf_min_def"]
        maxdefocus = parameters["ctf_max_def"]

    # convert half spectrum to full size FFT as expected by CTFFIND4
    tilesize = int(parameters["ctf_tile"])
    A = np.fft.fftshift(power_spectra, axes=0)
    C = np.zeros([tilesize, tilesize])
    C[:, tilesize // 2 :] = A[:, :-1]
    C[1:, 1 : tilesize // 2] = A[:, :-1][::-1, ::-1][:-1, :-1]
    C[0, :] = C[1, :]
    C[:, 0] = C[:, 1]
    # TODO: write the following out to wrapper function
    spectra = "ctffind4.mrc"
    mrc.write(C, spectra)

    logfile = "ctffind4.log"

    ctffind_command = "ctffind-4.1.5/bin/ctffind --amplitude-spectrum-input"

    command = """
%s > %s 2>&1 << EOF
%s
power.mrc
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
no
no
yes
%s
no
no
EOF
""" % (
        timeout_command(ctffind_command, 600),
        logfile,
        spectra,
        float(parameters["extract_bin"]) * float(parameters["scope_pixel"]),
        parameters["scope_voltage"],
        parameters["scope_cs"],
        parameters["scope_wgh"],
        parameters["ctf_tile"],
        parameters["ctf_min_res"],
        parameters["ctf_max_res"],
        mindefocus,
        maxdefocus,
        parameters["ctf_fstep"],
        parameters["ctf_dast"],
    )

    # Input image file name
    # Input is a movie (stack of frames)           [yes]
    # Number of frames to average together           [4]
    # Output diagnostic filename [diagnostic_output.mrc]
    # Pixel size                                [0.3185]
    # Acceleration voltage                       [300.0]
    # Spherical aberration                         [2.7]
    # Amplitude contrast                          [0.07]
    # Size of power spectrum to compute            [512]
    # Minimum resolution                          [30.0]
    # Maximum resolution                             [3]
    # Minimum defocus                           [5000.0]
    # Maximum defocus                          [50000.0]
    # Defocus search step                        [500.0]
    # Do you know what astigmatism is present?      [no]
    # Slower, more exhaustive search?              [yes]
    # Use a restraint on astigmatism?              [yes]
    # Expected (tolerated) astigmatism           [200.0]
    # Find additional phase shift?                  [no]
    # Do you want to set expert options?             [y]

    # print command
    [output, error] = local_run.run_shell_command(command)

    os.remove(spectra)

    try:
        # parse output and return df1, df2, angast and CC
        return np.loadtxt("power.txt", comments="#", dtype="f")[[1, 2, 3, 5]]
    except:
        logger.error("CTFFIND4 failed to run.")
        logger.info(error)
        sys.exit()
        pass


@Timer("ctf estimation", text="CTF estimation took: {}", logger=logger.info)
def ctffind4_quad(name, aligned_average, parameters, save_ctf=False, movie=0):

    # x = mrc.readHeaderFromFile(name + ".mrc")["nx"]
    # y = mrc.readHeaderFromFile(name + ".mrc")["ny"]
    # z = mrc.readHeaderFromFile(name + ".mrc")["nz"]
    x, y, z = get_image_dimensions(name + ".avg")

    counts = 0.0
    
    ctf = ctffind4_movie(name, parameters)

    df1 = ctf[0]
    df2 = ctf[1]
    df = (df1 + df2) / 2.0
    angast = ctf[2]
    ccc = cc = ctf[3]
    cccc = ctf[4]

    mrc2webp("power.mrc", name + "_ctffit.webp")
    mrc2png("power.mrc", "ctffind3.png")
    local_run.run_shell_command(
        "{0}/convert ctffind3.png -gravity Center -crop 100%x+0+0 ctffind3.png".format(
            os.environ["IMAGICDIR"]
        ),
        verbose=parameters["slurm_verbose"],
    )
    local_run.run_shell_command(
        '{0}/convert ctffind3.png -pointsize 20 -fill white -weight 700  -undercolor "#0008" -annotate +10+30 Defocus1={1} -pointsize 20 -fill white -weight 700 -undercolor "#0008" -annotate 0x0+10+65 Defocus2={2} -pointsize 20 -fill white -weight 700  -undercolor "#0008" -annotate 0x0+10+100 Angast={3} -pointsize 20 -fill white -weight 700  -undercolor "#0008" -annotate 0x0+10+135 CCC={4} ctffind3.png'.format(
            os.environ["IMAGICDIR"],
            "%.2f" % df1,
            "%.2f" % df2,
            "%.2f" % angast,
            "%.4f" % ccc,
        ),
        verbose=parameters["slurm_verbose"],
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="darkgrid")
    f, axarr = plt.subplots(2, figsize=(6, 6))

    ctfprof = np.loadtxt("power_avrot.txt", comments="#")
    os.rename("power_avrot.txt", name + "_avgrot.txt")

    # determine index of ctf_min_res
    min_res = np.argmin(np.fabs(ctfprof[0, :] - 1 / float(parameters["ctf_min_res"])))

    five_res = min_res + ctfprof[0, min_res:].size // 2

    # 1D CTF FIT
    axarr[0].set_title(name + "\n", fontsize=10)
    axarr[0].plot(
        ctfprof[0, min_res:five_res],
        ctfprof[2, min_res:five_res],
        color="r",
        linewidth=1,
        label="Radial",
    )
    axarr[0].plot(
        ctfprof[0, min_res:five_res],
        ctfprof[3, min_res:five_res],
        color="g",
        linewidth=1,
        label="CTF",
    )
    axarr[0].plot(
        ctfprof[0, min_res:five_res],
        ctfprof[4, min_res:five_res],
        color="b",
        linewidth=1,
        label="Quality",
    )
    axarr[0].set_xlim(ctfprof[0, min_res], ctfprof[0, five_res])
    axarr[0].set_ylim(0, 1.2)

    # 1D CTF FIT HIGH-RESOLUTION
    axarr[1].plot(
        ctfprof[0, five_res:],
        ctfprof[2, five_res:],
        color="r",
        linewidth=1,
        label="Radial average",
    )
    axarr[1].plot(
        ctfprof[0, five_res:],
        ctfprof[3, five_res:],
        color="g",
        linewidth=1,
        label="CTF fit",
    )
    axarr[1].plot(
        ctfprof[0, five_res:],
        ctfprof[4, five_res:],
        color="b",
        linewidth=1,
        label="Quality fit",
    )
    axarr[1].legend(
        loc="upper right", frameon=True, facecolor="whitesmoke", framealpha=1
    )

    axarr[1].set_xlim(ctfprof[0, five_res], ctfprof[0, -1])
    ulimit = min(
        1.2,
        max(
            max(ctfprof[2, five_res:].max(), ctfprof[3, five_res:].max()),
            ctfprof[2, five_res:].max(),
        ),
    )
    axarr[1].set_ylim(0, ulimit)

    # draw 3A marker
    if ctfprof[0, -1] > 1 / 3.0:
        axarr[0].plot((1 / 3.0, 1 / 3.0), (0, 1.2), "k-.")
        axarr[1].plot((1 / 3.0, 1 / 3.0), (0, ulimit), "k-.")
    plt.savefig("{0}_CTFprof.png".format(name), bbox_inches="tight")
    plt.close()

    # compose png

    plot.plot_spr_ctf(name, parameters["slurm_verbose"])

    ctf_params = np.array(
        [
            df,
            cc,
            df1,
            df2,
            angast,
            ccc,
            x,
            y,
            z,
            float(parameters["scope_pixel"]),
            float(parameters["scope_voltage"]),
            float(parameters["scope_mag"]),
            cccc,
            counts,
        ]
    )
    np.savetxt("{}.ctf".format(name), ctf_params)

    return ctf_params


def ctffind_spr_local_estimate(parameters, name, allboxes, local_particle, box_size):
    # calculate power spectra of all particles
    tilesize = int(box_size)
    binning = 1
    power_spectra = np.zeros([tilesize, tilesize // 2 + 1, len(local_particle)])
    for particle in local_particle:
        power_spectra[:, :, particle] = periodogram_averaging_from_file(
            name + "_stack", particle, tilesize, binning
        )

    ctf = np.loadtxt(name + ".ctf")
    ctf_local = np.zeros([len(local_particle), 4])

    for particle in local_particle:

        distances = scipy.spatial.distance.cdist(
            np.array(allboxes)[: len(local_particle), :],
            [np.array(allboxes)[particle, :]],
        )
        sigma = 1000

        # for proteasome
        sigma /= 2

        # increase sigma if we don't have enough particles to average
        minimum_particles_to_average = 15
        if (
            np.where(distances < 2 * sigma, 1, 0).sum()
            < minimum_particles_to_average / 3
        ):

            if distances.size > minimum_particles_to_average:

                while (
                    np.where(distances < 2 * sigma, 1, 0).sum()
                    < minimum_particles_to_average / 3
                ):

                    sigma *= 1.25

        weights = np.exp(-distances / sigma)
        weights /= weights.sum()

        # calculate weighted power spectrum for current particle
        particle_power = np.zeros(power_spectra[:, :, 0].shape)
        for i in local_particle:
            particle_power += power_spectra[:, :, i] * weights[i]
        particle_power /= len(local_particle)

        # default to global estimation if we don't have enough particles
        parameters["ctf_tile"] = tilesize
        if distances.size >= minimum_particles_to_average:
            logger.info(
                f"Local CTF estimation - Using sigma {sigma:5.0f} for particle {particle}"
            )
            ctf_local[particle, :] = ctffind4_power(particle_power, parameters, ctf[0])
        else:
            if particle == 0:
                logger.warning(
                    f"Using global CTF estimation, too few particles = {distances.size} < {minimum_particles_to_average}"
                )
            ctf_local[particle, :] = ctf[[2, 3, 4, 5]]

        if particle == 0:
            try:
                with open("ctffind4.log", "r") as f:
                    logger.info(f.read())
            except:
                pass

    np.savetxt(name + "_ctf.txt", ctf_local)

    return ctf_local


def ctffind_tilt_multiprocessing(
    name, parameters, counter, tilt_angle, tilt_axis, meandef=0, tolerance=2000
):

    current_name = name + "_%04d" % counter
    current = os.getcwd()

    os.mkdir(current + "/" + current_name)
    os.chdir(current + "/" + current_name)

    os.symlink( f"../{current_name}.mrc", f"{current_name}.mrc" )

    [df1, df2, angast, cc, res] = run_ctffind_tilt(
        current_name, parameters, tilt_angle, tilt_axis, meandef, tolerance, False
    )
    os.chdir(current)

    shutil.rmtree(current_name)
    os.remove(f"{current_name}.mrc")
    filename = current_name + ".txt"
    np.savetxt(filename, np.array([counter, df1, df2, angast, cc, res]).astype("float"))
    return


@Timer("ctf estimation", text="CTF estimation took: {}", logger=logger.info)
def ctffind_tomo_estimate(name, parameters):

    ctffind_tilt_args = []

    # figure out common CTF estimation parameters
    actual_pixel = float(parameters["scope_pixel"]) * float(parameters["data_bin"])
    dstep = actual_pixel * float(parameters["scope_mag"]) / 10000.0

    # check that ctf_max_res is not
    if float(parameters["ctf_max_res"]) > 2.75 * actual_pixel:
        ctf_max_res = float(parameters["ctf_max_res"])
    else:
        ctf_max_res = 2.75 * actual_pixel

    rawtilt = name + ".rawtlt"
    if os.path.isfile(rawtilt):
        tiltang = np.loadtxt(rawtilt, dtype='float', ndmin=0)
        centralz = np.argmin(abs(tiltang.ravel()))
        start_end_section = f"{int(centralz) + 1},{int(centralz) + 1}"
        logger.info("Central z-slice for initial CTF estimation is " + start_end_section)
    else:
        z = mrc.readHeaderFromFile(name + ".mrc")["nz"]
        start_end_section = f"{int(z/2)},{int(z/2)}"

    input_fname = f"{name}.mrc"
    output_fname = f"{name}_avg.mrc"

    output, error = avgstack(input_fname, output_fname, start_end_section)

    # 2D CTF
    ctf = ctffind4_movie(name + "_avg", parameters)

    df1 = ctf[0]
    df2 = ctf[1]
    angast = ctf[2]
    ccc = cc = ctf[3]
    cccc = ctf[4]
    mrc2png("power.mrc", name + "_ctffit.png")
    mrc2png("power.mrc", "ctffind3.png")

    os.rename("power_avrot.txt", name + "_avgrot.txt")

    if df1 > 0:
        meandef = (df1 + df2) / 2.0
        min_defocus = max(5000, meandef - 10000)
        max_defocus = meandef + 10000
        df = meandef
    else:
        min_defocus = parameters["ctf_min_def"]
        max_defocus = parameters["ctf_max_def"]
        df = (min_defocus + max_defocus) / 2

    if False and not parameters["ctf_use_ast"]:

        tilts = np.loadtxt("%s.tlt" % name)
        if not os.path.exists("%s.def" % name):

            if int(parameters["slurm_tasks"]) == 0:
                pool = multiprocessing.Pool()
            else:
                pool = multiprocessing.Pool(int(parameters["slurm_tasks"]))

            manager = multiprocessing.Manager()
            results = manager.Queue()

            counter = 0
            for tilt in tilts:
                pool.apply_async(
                    tomoctffind_multiprocessing,
                    args=(
                        name,
                        parameters,
                        dstep,
                        ctf_max_res,
                        min_defocus,
                        max_defocus,
                        counter,
                        tilt,
                        results,
                    ),
                )
                counter += 1
            pool.close()
            pool.join()

            # collate results
            defocuses = np.empty([len(tilts), 3])
            while results.empty() == False:
                current = results.get()
                defocuses[int(current[0]), :] = [
                    tilts[int(current[0])],
                    current[1],
                    current[2],
                ]
            logger.info(defocuses.shape)
            logger.info(defocuses)

            np.savetxt("{}.def".format(name), defocuses, fmt="%f\t")
        else:
            defocuses = np.loadtxt("%s.def" % name)

        # custom visualization
        import matplotlib.pyplot as plt

        f, axarr = plt.subplots(3, figsize=(6, 6))
        ctfprof = np.loadtxt("{}_CTFprof.txt".format(name))
        axarr[0].plot(tilts, defocuses[:, 1], color="r")
        axarr[0].plot(tilts, df * np.ones(defocuses[:, 1].shape), color="g")
        axarr[0].plot(tilts, (df - 1000) * np.ones(defocuses[:, 1].shape), "k--")
        axarr[0].plot(tilts, (df + 1000) * np.ones(defocuses[:, 1].shape), "k--")
        axarr[0].set_xlim(tilts[0] - 1, tilts[-1] + 1)
        axarr[0].set_ylim(df - 5000, df + 5000)
        axarr[0].set_title(
            "{0}\nDefocus = {1}, CC = {2}".format(name, df, cc), fontsize=12
        )
        axarr[0].set_ylabel("Defocus (A)")
        axarr[1].plot(tilts, defocuses[:, 2], color="b")
        axarr[1].set_xlim(tilts[0] - 1, tilts[-1] + 1)
        axarr[1].set_ylabel("CC")
        axarr[2].plot(ctfprof[:, 1], ctfprof[:, 2], color="r")
        axarr[2].plot(ctfprof[:, 1], ctfprof[:, 3], color="g")
        axarr[2].set_ylim(0, 1.2)
        axarr[2].set_xlim(ctfprof[0, 1], ctfprof[-1, 1])
        axarr[2].set_ylabel("AU")
        plt.savefig("{0}_CTFprof.png".format(name))
        plt.close()

    else:

        tilts = np.loadtxt("%s.tlt" % name)
        if not os.path.exists("%s.def" % name) or not os.path.exists("%s.pickle" % name):

            mean_df = (df1 + df2) / 2.0
            tolerance = math.fabs(df1 - df2) / 2.0 + 10000

            counter = 0
            arguments = []
            for tilt_angle in tilts:

                # get parameters from IMOD's affine transformation
                tilt_axis = float(
                    [
                        line.split("\n")
                        for line in subprocess.check_output(
                            "%s/bin/xf2rotmagstr %s" % (get_imod_path(), name + ".xf"),
                            stderr=subprocess.STDOUT,
                            shell=True,
                            text=True,
                        ).split("\n")
                        if "rot=" in line
                    ][counter][0].split()[2][:-1]
                )
                arguments.append(
                    (
                        name,
                        parameters,
                        counter,
                        tilt_angle,
                        -90 + tilt_axis,
                        mean_df,
                        tolerance,
                    )
                )

                counter += 1

            ctffind_tilt_args = arguments

        else:
            defocuses = np.loadtxt("%s.def" % name)

    x = mrc.readHeaderFromFile(name + ".mrc")["nx"]
    y = mrc.readHeaderFromFile(name + ".mrc")["ny"]
    z = mrc.readHeaderFromFile(name + ".mrc")["nz"]

    counts = 0

    return [
            np.array(
                [
                    df,
                    cc,
                    df1,
                    df2,
                    angast,
                    ccc,
                    x,
                    y,
                    z,
                    float(parameters["scope_pixel"]),
                    float(parameters["scope_voltage"]),
                    float(parameters["scope_mag"]),
                    cccc,
                    counts,
                ]
            ),
            ctffind_tilt_args
            ]



def plot_ctffind_tilt(name, parameters, ctf):

    df = ctf[0]
    cc = ctf[1]

    tilts = np.loadtxt("%s.tlt" % name)
    defocuses = np.empty([len(tilts), 6])
    counter = 0
    for tilt_angle in tilts:
        filename = name + "_%04d.txt" % counter
        current = np.loadtxt(filename)
        defocuses[int(current[0]), :] = current
        defocuses[int(current[0]), 0] = tilts[int(current[0])]
        counter += 1

    np.savetxt("{}.def".format(name), defocuses, fmt="%f\t")

    # custom visualization
    new_profile_file = "%s_%04d_ctffind4_avrot.txt" % (
        name,
        math.floor(len(tilts) / 2),
    )
    if os.path.exists("{}_CTFprof.txt".format(name)):
        ctfprof_file = "{}_CTFprof.txt".format(name)
    elif os.path.exists(new_profile_file):
        ctfprof_file = new_profile_file
    else:
        logger.error("Cannot find ctf profile for ploting")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="darkgrid")
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(4, figsize=(6, 6))
    ctfprof = np.genfromtxt(
        ctfprof_file, dtype="f", delimiter=[6, 14, 14, 14, 14, 14],
    )
    # df variation
    axarr[0].plot(
        tilts, np.mean(defocuses[:, 1:2], axis=1), color="r", label="df1+df2"
    )
    axarr[0].plot(
        tilts, df * np.ones(defocuses[:, 1].shape), color="g", label="mean"
    )
    # axarr[1].plot(tilts,np.fabs(defocuses[:,1]-defocuses[:,2]), color='b',label='|df1-df2|')
    axarr[0].plot(tilts, (df - 1000) * np.ones(defocuses[:, 1].shape), "k--")
    axarr[0].plot(tilts, (df + 1000) * np.ones(defocuses[:, 1].shape), "k--")
    axarr[0].set_xlim(tilts[0] - 1, tilts[-1] + 1)
    axarr[0].set_ylim(df - 5000, df + 5000)
    axarr[0].set_title(
        "{0}\nDefocus = {1}, CC = {2}".format(name, df, cc), fontsize=12
    )
    axarr[0].set_ylabel("Defocus (A)")
    axarr[1].plot(tilts, defocuses[:, -2], color="b")
    axarr[1].set_xlim(tilts[0] - 1, tilts[-1] + 1)
    axarr[1].set_ylabel("CC")
    axarr[2].plot(tilts, defocuses[:, -1], color="m")
    axarr[2].set_xlim(tilts[0] - 1, tilts[-1] + 1)
    axarr[2].set_ylim(4, 20)
    axarr[2].set_ylabel("Resolution")
    axarr[3].plot(tilts, np.fabs(defocuses[:, 1] - defocuses[:, 2]), color="c")
    axarr[3].set_xlim(tilts[0] - 1, tilts[-1] + 1)
    axarr[3].set_ylabel("|DF1-DF2|")
    plt.savefig("{0}_CTFprof.png".format(name))
    plt.close()

    # produce 2D map of power spectra
    files = sorted(glob.glob("*_ctffind4.mrc"))
    tile = int(parameters["ctf_tile"])
    if len(files) > 0:
        A = np.empty([len(files), tile, tile])
        count = 0
        for f in files:
            A[count, :, :] = mrc.read(f)
            count += 1
        cols = int( math.ceil( math.sqrt(count) ) )
        writepng(plot.contact_sheet(A, cols), name + "_2D_ctftilt.png")
        command = "/usr/bin/convert {0}_2D_ctftilt.png {0}_2D_ctftilt.webp".format(name)
        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

        # produce 2D map of 2D profiles
        command = "/usr/bin/montage -geometry 384x384+0+0 {0}_????_ctftilt.png {0}_1D_ctftilt.webp".format(
            name
        )
        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])


@Timer(
    "ctf correction", text="CTF correction took: {}", logger=logger.info
)
def ctffind_tomo_correct(parameters, name, tltfile, tilt_angles, nprocs):
    """Correct TOMO ctf parameters."""
    if parameters["extract_box"] == 0:
        ################################################################
        # Check defocus gradient direction using TOMOCTF's tomoctfgrad #
        ################################################################

        # Use average of three consecutive images around +30 degree-tilt to check gradient defocus
        idx = np.abs(tilt_angles - 30).argmin()
        max_idx = len(tilt_angles) - 2
        if idx > max_idx:
            idx = max_idx

        command = "{0}/bin/newstack {1}.ali {1}_defocus_gradient_range.mrc -secs {2}-{3}".format(
            get_imod_path(), name, idx - 1, idx + 1
        )
        local_run.run_shell_command(command)

        input_fname = f"{name}_defocus_gradient_range.mrc"
        output_fname = f"{name}_defocus_gradient.mrc"
        start_end_section = "/"

        output, error = avgstack(input_fname, output_fname, start_end_section)

        with open("{0}_defocus_gradient.tlt".format(name), "w") as f:
            f.write(str(tilt_angles[idx]))

        binning = 1
        dstep = (
            binning
            * float(parameters["scope_pixel"])
            * float(parameters["scope_mag"])
            / 10000.0
        )
        if float(parameters["ctf_max_res"]) > 2.75 * binning * float(
            parameters["scope_pixel"]
        ):
            ctf_max_res = float(parameters["ctf_max_res"])
        else:
            ctf_max_res = 2.75 * binning * float(parameters["scope_pixel"])
        ctf_max_res *= 1.0

        # retrieve defocus if available
        if os.path.exists("{}.ctf".format(name)):
            ctf = np.loadtxt("{}.ctf".format(name))
            ctf_min_def = ctf[0] - 2500
            ctf_max_def = ctf[0] + 2500
        else:
            ctf_min_def = parameters["ctf_min_def"]
            ctf_max_def = parameters["ctf_max_def"]

        tilt_series_filename = f"{name}_defocus_gradient.mrc"
        tilt_angles_filename = f"{name}_defocus_gradient.tlt"
        output_diag_filename = f"{name}_defocus_gradient_diagnostic.mrc"

        tile_size = 256

        scope_cs = f"{parameters['scope_cs']}"
        scope_voltage = f"{parameters['scope_voltage']}"
        scope_wgh = f"{parameters['scope_wgh']}"
        scope_mag = f"{parameters['scope_mag']}"

        ctf_min = 0.5
        ctf_min_res = f"{parameters['ctf_min_res']}"

        ctf_min_def = str(ctf_min_def)
        ctf_max_def = str(ctf_max_def)
        fstep = 250.0

        [output, error] = tomo_ctf_grad(
            tilt_series_filename,
            tilt_angles_filename,
            output_diag_filename,
            tile_size,
            scope_cs,
            scope_voltage,
            scope_wgh,
            scope_mag,
            dstep,
            ctf_min,
            ctf_min_res,
            ctf_max_res,
            ctf_min_def,
            ctf_max_def,
            fstep,
        )
        logger.info(output)

        if "NOT" in output:
            logger.warning(
                "Measured defocus gradient NOT consistent with specified tilt angles!"
            )
        else:
            logger.info(
                "Measured defocus gradient IS consistent with specified tilt angles."
            )

    # Further correct for CTF
    load_imod_cmd = imod_load_command()
    command = "{0} export PATH=$IMOD_DIR/bin:$PATH; {1}/bin/CTFphaseflipstack.exe {2}.ali {2}.pf.ali {3} {2}.param {4}".format(
        load_imod_cmd, get_tomoctf_path(), name, tltfile, nprocs
    )
    local_run.run_shell_command(command)
    shutil.move("{0}.pf.ali".format(name), "{0}.ali".format(name))


def periodogram_averaging(
    image, tilesize, binning=1, interp="real", power=2, order="before"
):
    """
    Periodogram averaging from image.
    image    : input image
    tilesize : size of tiles
    binning  : downsampling factor
    interp   : controls interpolation method
    power    : use a power of the power spectra modulo
    order    : controls whether binning is done before or after tile extraction
    """

    # bin image before tiling
    if binning > 1 and "before" in order:
        if "real" in interp:
            if True:
                name = "".join(random.choice(string.ascii_letters) for i in range(20))
                mrc.write(image, name + ".mrc")
                command = "{0}/bin/newstack -bin 2 -antialias 6 {1}.mrc {1}_out.mrc".format(
                    get_imod_path(), name
                )
                local_run.run_shell_command(command)
                small = mrc.read("{0}_out.mrc".format(name))
                os.remove("{0}.mrc".format(name))
                os.remove("{0}_out.mrc".format(name))
            else:
                small = (
                    image.reshape(
                        image.shape[0] / binning,
                        binning,
                        image.shape[1] / binning,
                        binning,
                    )
                    .mean(3)
                    .mean(1)
                )
        else:
            small = abs(
                np.fft.irfft2(
                    np.fft.fftshift(np.fft.rfft2(image - image.min()), 0)[
                        int(image.shape[0] / 2 - image.shape[0] / 2 / binning) : int(
                            image.shape[0] / 2 + image.shape[0] / 2 / binning
                        ),
                        : int(image.shape[1] / 2 / binning + 1),
                    ]
                )
            )

            # re-normalize to zero mean and unit variance
            small = (small - np.average(small)) / np.std(small)
        binning = 1
    else:
        small = image

    tiles = 0
    fft = np.zeros([tilesize, tilesize // 2 + 1])

    for x in range(
        0,
        1 + (small.shape[0] // (tilesize * binning // 2) - 1) * tilesize * binning // 2,
        tilesize * binning // 2,
    ):
        for y in range(
            0,
            1
            + (small.shape[1] // (tilesize * binning // 2) - 1)
            * (tilesize * binning // 2),
            tilesize * binning // 2,
        ):

            # take care of last column of tiles
            if x + tilesize * binning > small.shape[0]:
                xfix = small.shape[0] - tilesize * binning
            else:
                xfix = x

            # take care of last row of tiles
            if y + tilesize * binning > small.shape[1]:
                yfix = small.shape[1] - tilesize * binning
            else:
                yfix = y

            if binning == 1:
                fft += np.power(
                    abs(
                        np.fft.rfft2(
                            small[xfix : xfix + tilesize, yfix : yfix + tilesize]
                        )
                    ),
                    power,
                )
            else:
                # take central part of unbinned FFT
                fft += np.power(
                    abs(
                        np.fft.fftshift(
                            np.fft.fftshift(
                                np.fft.rfft2(
                                    small[
                                        xfix : xfix + tilesize * binning,
                                        yfix : yfix + tilesize * binning,
                                    ]
                                ),
                                0,
                            )[
                                tilesize * binning // 2
                                - tilesize // 2 : tilesize * binning // 2
                                + tilesize // 2,
                                : tilesize // 2 + 1,
                            ],
                            0,
                        )
                    ),
                    power,
                )

            tiles += 1
    fft /= tiles
    fft[0, 0] = 0
    fft /= fft.max()
    return fft


def periodogram_averaging_from_file(name, frame, tilesize, binning, results=0):

    # read single frame from file
    image = mrc.readframe(name + ".mrc", frame)

    # compute power spectra
    if results:
        results.put(periodogram_averaging(image, tilesize, binning, "fourier", 1))
        return
    else:
        return periodogram_averaging(image, tilesize, binning, "fourier", 1)


def refineCtftilt(
    imagefile,
    parameters,
    ctffind_exe,
    mindefocus,
    maxdefocus,
    tilt_angle,
    tilt_axis,
    angle_tolerance,
    axis_tolerance,
    angle_step,
    axis_step,
):

    best_angle, best_axis, best_df1, best_df2, best_angast, best_ccc, best_res = (
        tilt_angle,
        tilt_axis,
        0.0,
        0.0,
        0.0,
        -math.inf,
        math.inf,
    )


    rootname = Path(imagefile).stem
    best_output_spectra = rootname + "_ctffind4.mrc"
    best_avrot = rootname + "_ctffind4_avrot.txt"
    best_txt = Path(best_output_spectra).stem + ".txt"

    if parameters["ctf_use_ast"]:
        use_ast = f"yes\n{parameters['ctf_dast']}"
    else:
        use_ast = f"yes\n0"

    """
    # Determine tilt geometry

Input image file name [tilt10_30.mrc]
Output diagnostic image file name [tilt10_30_output.mrc]
Pixel size [2.1]
Acceleration voltage [300]
Spherical aberration [2.70]
Amplitude contrast [0.07]
Size of amplitude spectrum to compute [512]
Minimum resolution [50]
Maximum resolution [10]
Minimum defocus [5000.0]
Maximum defocus [50000.0]
Defocus search step [250]
Do you know what astigmatism is present? [No]
Slower, more exhaustive search? [No]
Use a restraint on astigmatism? [No]
Find additional phase shift? [No]
Determine sample tilt? [Yes]
If initial tilt statistics is known? [Yes]
Known tilt axis [-94.48]
Known tilt angle [29.51]
Do you want to set expert options? [Yes]
Resample micrograph if pixel size too small? [No]
Do you already know the defocus? [No]
Desired number of parallel threads [1]
    """

    """
    # Do NOT determine tilt geometry

Input image file name [12.mrc]                     :
Output diagnostic image file name
[diagnostic_output.mrc]                            :
Pixel size [1.35]                                  :
Acceleration voltage [300.0]                       :
Spherical aberration [2.70]                        :
Amplitude contrast [0.07]                          :
Size of amplitude spectrum to compute [512]        :
Minimum resolution [30.0]                          :
Maximum resolution [10]                            :
Minimum defocus [5000.0]                           :
Maximum defocus [50000.0]                          :
Defocus search step [100.0]                        :
Do you know what astigmatism is present? [No]      :
Slower, more exhaustive search? [Yes]              :
Use a restraint on astigmatism? [No]               :
Find additional phase shift? [No]                  :
Determine sample tilt? [No]                        :
Do you want to set expert options? [Yes]           :
Resample micrograph if pixel size too small? [Yes] :
Do you already know the defocus? [No]              :
Desired number of parallel threads [1]             :
    """

    output_spectra_tilt = rootname + "_tilt.mrc"
    logfile_tilt = rootname + "_tilt.log"
    avrot_tilt = Path(output_spectra_tilt).stem + "_avrot.txt"

    output_spectra_notilt = rootname + "_notilt.mrc"
    logfile_notilt = rootname + "_notilt.log"
    avrot_notilt = Path(output_spectra_notilt).stem + "_avrot.txt"

    command_determine_tilt = """
%s > %s 2>&1 << EOF
%s.mrc
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
No
No
%s
No
Yes
Yes
%s
%s
Yes
Yes
No
1
EOF
""" % (
        timeout_command(ctffind_exe, 600, full_path=True),
        logfile_tilt,
        imagefile,
        output_spectra_tilt,
        float(parameters["data_bin"]) * float(parameters["scope_pixel"]),
        parameters["scope_voltage"],
        parameters["scope_cs"],
        parameters["scope_wgh"],
        parameters["ctf_tile"],
        parameters["ctf_min_res"],
        parameters["ctf_max_res"],
        mindefocus,
        maxdefocus,
        parameters["ctf_fstep"],
        use_ast,
        tilt_angle,
        tilt_axis,
    )

  
    command_not_determine_tilt = """
%s > %s 2>&1 << EOF
%s.mrc
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
No
No
No
No
No
Yes
Yes
No
1
EOF
""" % (
        timeout_command(ctffind_exe, 600, full_path=True),
        logfile_notilt,
        imagefile,
        output_spectra_notilt,
        float(parameters["data_bin"]) * float(parameters["scope_pixel"]),
        parameters["scope_voltage"],
        parameters["scope_cs"],
        parameters["scope_wgh"],
        parameters["ctf_tile"],
        parameters["ctf_min_res"],
        parameters["ctf_max_res"],
        mindefocus,
        maxdefocus,
        parameters["ctf_fstep"],
    )

    if "ctffind5"  in parameters["ctf_method"]:
        ctffind_command = f"{get_frealign_paths()['cistem2']}/ctffind5"

        determine_tilt = "Yes"
        determine_thickness = "Yes"
        # ctffind5
        ctffind5_command = f"""{timeout_command(ctffind_command, 800, full_path=True)} > {logfile_notilt} 2>&1 << EOF
{imagefile + ".mrc"}
{output_spectra_notilt}
{parameters['scope_pixel'] * parameters['data_bin']}
{parameters['scope_voltage']}
{parameters['scope_cs']}
{parameters['scope_wgh']}
{parameters['ctf_tile']}
{parameters['ctf_min_res']}
{parameters['ctf_max_res']}
{parameters['ctf_min_def']}
{parameters['ctf_max_def']}
{parameters['ctf_fstep']}
No
No
No
No
{determine_tilt}
{determine_thickness}
No
No
{30.0}
{3.0}
No
No
No
EOF
        """
    else:
        ctffind5_command = ""

    # We run two different per-tilt ctf estimation on the same image, and choose the best one based on
    # estimated resolution and cc
    for estimation in [command_determine_tilt, ctffind5_command]: #, command_not_determine_tilt]:

        command = estimation
        
        if not len(command):
            continue

        if estimation == command_determine_tilt:
            output_spectra = output_spectra_tilt
            logfile = logfile_tilt
            avrot = avrot_tilt
        elif estimation == ctffind5_command:
            output_spectra = output_spectra_notilt
            logfile = logfile_notilt
            avrot = avrot_notilt
        else:
            raise Exception("Do not recognize the ctffind command")

        # suppress long log
        [output, error] = local_run.run_shell_command(command, verbose=False)

        assert Path(logfile).exists(), f"{logfile} does not exist. CTFFIND_TILT fails to run."

        with open(logfile, 'r') as f:
            estimated_tilt_axis, estimated_tilt_angle = None, None
            for line in f.readlines():
                if "Tilt_axis, tilt angle" in line:
                    estimated_tilt_axis, estimated_tilt_angle = line.split(":")[1].replace("degrees", "").replace(" ", "").split(",")
                    estimated_tilt_axis, estimated_tilt_angle = float(estimated_tilt_axis), float(estimated_tilt_angle)
                    with open(f"../{imagefile}_handedness.txt", "w") as f:
                        f.write(f"{estimated_tilt_angle} {estimated_tilt_axis}")
                    break
            assert estimated_tilt_angle is not None and estimated_tilt_axis is not None, "Ctffind_tilt logfile does not contain estimated tilt geometry. Please check. "

        if parameters["slurm_verbose"] and imagefile.endswith('_0000'):
            with open(logfile) as f:
                logger.info(f.read())
            if len(error) > 0:
                logger.error(error)

        df1, df2, angast, ccc, est_res = np.loadtxt(
            output_spectra.replace(".mrc", ".txt"), comments="#", dtype="f"
        )[[1, 2, 3, 5, 6]]

        if est_res < best_res:
            best_ccc, best_res = ccc, est_res
            best_df1, best_df2, best_angast = df1, df2, angast

            shutil.move(output_spectra, best_output_spectra)
            shutil.move(avrot, best_avrot)
            shutil.move(output_spectra.replace(".mrc", ".txt"), best_txt)

        else:
            os.remove(output_spectra)
            os.remove(avrot)

    return [best_df1, best_df2, best_angast, best_ccc, best_res, best_angle, best_axis]


def run_ctffind_tilt(
    image_file, parameters, tilt_angle, tilt_axis, meandef=0, tolerance=500, refine=True
):

    if meandef > 0:
        mindefocus = meandef - tolerance
        maxdefocus = meandef + tolerance
    else:
        mindefocus = parameters["ctf_min_def"]
        maxdefocus = parameters["ctf_max_def"]

    output_spectra = image_file + "_ctffind4.mrc"

    logfile = "../" + image_file + "_ctffind4.log"

    ctffind_command = f"{get_ctffind_tilt_path()}/ctffind_tilt"

    if refine:
        # the locality and the step size can be changed
        angle_tolerance, axis_tolerance = 5.0, 5.0
        angle_step, axis_step = 1.0, 1.0
    else:
        angle_tolerance, axis_tolerance = 0.0, 0.0
        angle_step, axis_step = 1.0, 1.0

    # refine CTF by locally searching the best tilt axis and tilt angle
    [
        best_df1,
        best_df2,
        best_angast,
        best_ccc,
        best_res,
        best_angle,
        best_axis,
    ] = refineCtftilt(
        image_file,
        parameters,
        ctffind_command,
        mindefocus,
        maxdefocus,
        tilt_angle,
        tilt_axis,
        angle_tolerance,
        axis_tolerance,
        angle_step,
        axis_step,
    )

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="darkgrid")
    f, axarr = plt.subplots(1, figsize=(6, 6))

    ctfprof = np.loadtxt(image_file + "_ctffind4_avrot.txt", comments="#")

    # determine index of ctf_min_res
    min_res = np.argmin(np.fabs(ctfprof[0, :] - 1 / float(parameters["ctf_min_res"])))

    five_res = int(min_res + ctfprof[0, min_res:].size / 2)

    # 1D CTF FIT
    df1, df2, angast, ccc, est_res = np.loadtxt(
        output_spectra.replace(".mrc", ".txt"), comments="#", dtype="f"
    )[[1, 2, 3, 5, 6]]

    axarr.set_title(
        image_file + " (CC=%0.2f, Res=%0.2f A)" % (ccc, est_res), fontsize=10
    )
    axarr.plot(
        ctfprof[0, min_res:five_res],
        ctfprof[3, min_res:five_res],
        color="g",
        linewidth=3,
        label="DF1={0}, DF2={1}".format(df1, df2),
    )
    axarr.plot(
        ctfprof[0, min_res:five_res],
        ctfprof[2, min_res:five_res],
        color="r",
        linewidth=3,
        label="Angast={0}".format(angast),
    )
    axarr.plot(
        ctfprof[0, min_res:five_res],
        ctfprof[4, min_res:five_res],
        color="b",
        linewidth=3,
        label="Quality",
    )
    axarr.set_xlim(ctfprof[0, min_res], ctfprof[0, five_res])
    axarr.set_ylim(0, 1.2)
    axarr.set_xlabel("DF1=%.2d, DF2=%.2f, ANGAST=%.2f" % (df1, df2, angast))
    # axarr.legend()
    axarr.set_xlim(
        1.0 / float(parameters["ctf_min_res"]), 1.25 / float(parameters["ctf_max_res"])
    )
    plt.savefig("../" + image_file + "_ctftilt.png")
    plt.close()

    shutil.move(output_spectra, Path(os.getcwd()).parent)
    shutil.move(image_file + "_ctffind4_avrot.txt", Path(os.getcwd()).parent)

    try:
        # parse output and return df1, df2, angast and CC
        return np.loadtxt(
            output_spectra.replace(".mrc", ".txt"), comments="#", dtype="f"
        )[[1, 2, 3, 5, 6]]
    except Exception as error:
        logger.exception("CTFFIND4 failed to run.")
        logger.info(error)
        pass

def run_tomoctffind(name, parameters, dstep, ctf_max_res, min_defocus, max_defocus):

    # 1D CTF using TOMOCTFFIND
    command = """
%s > %s_tomoctffind.log 2>&1 << EOF
%s_power.mrc
%s_CTF_diagnostic.mrc
%s,%s,%s,%s,%s
0.0,%s,%s
%s,%s,%s
EOF
""" % (
        timeout_command(
            f"{get_tomoctf_path()}/bin/tomoctffind.exe", 600, full_path=True
        ),
        name,
        name,
        name,
        parameters["scope_cs"],
        parameters["scope_voltage"],
        parameters["scope_wgh"],
        parameters["scope_mag"],
        dstep,
        parameters["ctf_min_res"],
        ctf_max_res,
        min_defocus,
        max_defocus,
        parameters["ctf_fstep"],
    )
    local_run.run_shell_command(command)

    f = open("%s_tomoctffind.log" % name, "r")
    ctffind = f.read()
    f.close()

    if "CC" in ctffind:
        ctf = [
            line.split()[3:]
            for line in open("%s_tomoctffind.log" % name)
            if "CC)" in line
        ]
        df, cc = ctf[0][0], ctf[0][1]
    else:
        logger.error("tomoctffind failed")
        df = cc = 0

    # compile output
    if df > 0:
        shutil.move("CTFPROFILE", "%s_CTFprof.txt" % name)
        command = "{0}/convert -rotate 90 -gravity center -extent 100x143% CTFPROFILE.ps CTFprof.png".format(
            os.environ["IMAGICDIR"]
        )
        local_run.run_shell_command(command)
        shutil.move("tomoctf.param", "%s.param" % name)

        command = '{0}/convert CTFprof.png -pointsize 20 -undercolor white -undercolor "#0008" -annotate 0x0+51+48 {1} -pointsize 20 -undercolor white -undercolor "#0008" -annotate 0x0+51+650 Defocus={2} -pointsize 20 -undercolor white -undercolor "#0008" -annotate 0x0+300+650 CC={3} CTFprof.png'.format(
            os.environ["IMAGICDIR"], name, df, cc
        )
        local_run.run_shell_command(command)

        # custom visualization
        import matplotlib.pyplot as plt

        f, axarr = plt.subplots(2, figsize=(6, 6))

        # ctfprof = np.loadtxt('{}_CTFprof.txt'.format(name))

        f = open("{}_CTFprof.txt".format(name))
        ctfprof = np.genfromtxt(f, delimiter=(6, 14, 14, 14, 14, 14), dtype="f")
        f.close()

        axarr[0].plot(ctfprof[:, 1], ctfprof[:, 4], color="r")
        axarr[0].plot(ctfprof[:, 1], ctfprof[:, 5], color="g")
        axarr[0].set_title(
            "{0}\nDefocus = {1}, CC = {2}".format(name, df, cc), fontsize=12
        )
        axarr[1].plot(ctfprof[:, 1], ctfprof[:, 2], color="r")
        axarr[1].plot(ctfprof[:, 1], ctfprof[:, 3], color="g")
        axarr[1].set_ylim(0, 1.2)
        plt.savefig("{0}_CTFprof.png".format(name))
        plt.close()

    return np.array([df, cc]).astype("float")


def tomoctffind_multiprocessing(
    name,
    parameters,
    dstep,
    ctf_max_res,
    min_defocus,
    max_defocus,
    counter,
    tilt,
    results,
):

    current_name = name + "_%04d" % counter
    current = os.getcwd()

    os.mkdir(current + "/" + current_name)
    os.chdir(current + "/" + current_name)

    if not os.path.exists(current + "/" + current_name + ".dm4"):
        com = "{0}/bin/newstack ../{1}.ali {2}.ali -secs {3}".format(
            get_imod_path(), name, current_name, counter
        )
        local_run.run_shell_command(com)
        f = open(current + "/" + current_name + "/" + current_name + ".tlt", "w")
        f.write(str(tilt))
        f.close()
    else:
        logger.info("Using individual frames")
        # use movie frames
        com = "{0}/bin/dm2mrc {1}/{2}.dm4 {2}.ali".format(
            get_imod_path(), current, current_name
        )
        local_run.run_shell_command(com)
        f = open(current + "/" + current_name + "/" + current_name + ".tlt", "w")
        for frames in range(mrc.readHeaderFromFile(current_name + ".ali")["nz"]):
            f.write(str(tilt) + "\n")
        f.close()
    run_tomops(current_name, parameters, dstep, 10000)
    [df, cc] = run_tomoctffind(
        current_name, parameters, dstep, ctf_max_res, min_defocus, max_defocus
    )
    logger.info(f"counter={counter}, tilt={tilt}, df={df}, cc={cc}")
    os.chdir(current)

    shutil.rmtree(current_name)
    logger.info(np.array([counter, df, cc]).astype("float"))
    results.put(np.array([counter, df, cc]).astype("float"))
    return


def run_tomops(name, parameters, dstep, defocus_tolerance=2000):

    if os.path.exists(name + "_clean.ali") and os.path.exists(name + "_clean.tlt"):
        name_clean = name + "_clean"
    else:
        name_clean = name

    # Power spectra from region within defocus tolerance
    command = """
%s >& %s_tomops.log << EOF
0
%s.ali
%s.tlt
%s_power.mrc
%s
%s,%s
%s
EOF
""" % (
        timeout_command(f"{get_tomoctf_path()}/bin/tomops.exe", 600, full_path=True),
        name,
        name_clean,
        name_clean,
        name,
        defocus_tolerance,
        parameters["scope_mag"],
        dstep,
        parameters["ctf_tile"],
    )
    local_run.run_shell_command(command)

    # produce output
    avg_power = mrc.read("%s_power.mrc" % name)
    bpf = bandpass(avg_power[:, 0:-1].shape, 35, 5, 1000, 1)
    writepng(np.fft.fftshift(avg_power[:, 0:-1] * bpf, 0), "power.png")
    # commands.getstatusoutput('{0}/convert power.png -contrast-stretch 1%x98% power.png'.format( os.environ['IMAGICDIR'] ) )
    contrast_stretch("power.png")
    writepng(
        np.flipud(np.fliplr(np.fft.fftshift(avg_power[:, 0:-1] * bpf, 0))),
        "power_avg.png",
    )
    # commands.getstatusoutput('{0}/convert power_avg.png -contrast-stretch 1%x98% power_avg.png'.format( os.environ['IMAGICDIR'] ) )
    contrast_stretch("power_avg.png")
    local_run.run_shell_command(
        "%s/montage power_avg.png power.png -geometry +0+0 power.png"
        % os.environ["IMAGICDIR"],
        verbose=parameters["slurm_verbose"],
    )

    # 2D CTF USING CTFFIND3
    shutil.copy2("%s_power.mrc" % name, "power_for_ctffind3.mrc")

    return

def detect_handedness(name: str, tiltang_file: Path, xf_file: Path, angle_to_detect: float = 30.0, tilt_axis_error: float = 90.0, tilt_angle_error = 10.0):
    """detect_handedness Detect tilt handedness by checking the tilt geometry estimated by ctffind_tilt

    Parameters
    ----------
    name : str
        Name of tilt-series
    tiltang_file : Path
        Path to tilt angle file (*.tlt)
    xf_file : Path
        Path to tilt alignment file (*.xf)
    angle_to_detect : float, optional
        Tilt angle to detect the handedness, usually angle between 10 to 50 or -10 to -50 works better, by default 30.0
    tilt_axis_error : float, optional
        Tolerance of estimated tilt axis angle, by default 90.0
    tilt_angle_error : float, optional
        Tolerance of estimated tilt angle, by default 10.0
    """
    assert tiltang_file.exists(), f"Tilt angle file ({tiltang_file}) does not exist. "
    assert xf_file.exists(), f"Tiltseries alignment file ({xf_file}) does not exist. "
    
    FLIP = True
    NO_FLIP = False

    tilt_angles = np.loadtxt(tiltang_file, ndmin=1, dtype=float)
    tilt_angles_modified = tilt_angles - angle_to_detect
    index = np.argmin(abs(tilt_angles_modified.ravel()))

    tilt_axis = float(
                    [
                        line.split("\n")
                        for line in subprocess.check_output(
                            "%s/bin/xf2rotmagstr %s" % (get_imod_path(), name + ".xf"),
                            stderr=subprocess.STDOUT,
                            shell=True,
                            text=True,
                        ).split("\n")
                        if "rot=" in line
                    ][index][0].split()[2][:-1]
                )
    tilt_axis = -90 + tilt_axis
    tilt_angle = tilt_angles[index]

    # produced by running ctffind_tilt
    estimated_tilt = Path(f"{name}_{index:04d}_handedness.txt")

    # NOTE: tilt angle is always positive, check tilt axis
    if estimated_tilt.exists():
        with open(estimated_tilt, "r") as f:
            estimated_tilt_angle, estimated_tilt_axis = f.read().strip().split()
            estimated_tilt_angle, estimated_tilt_axis = float(estimated_tilt_angle), float(estimated_tilt_axis)
            # logger.info(f"{estimated_tilt_axis}, {tilt_axis}, {estimated_tilt_angle}, {tilt_angle}")

            handedness = NO_FLIP
            if abs(estimated_tilt_angle - abs(tilt_angle)) < tilt_angle_error:
                if tilt_angle > 0:
                    if abs(estimated_tilt_axis - tilt_axis) < tilt_axis_error:
                        handedness = FLIP
                else:
                    if abs(estimated_tilt_axis - tilt_axis) > tilt_axis_error:
                        handedness = FLIP
                return handedness
            else:
                logger.warning(f"Estimated tilt-angle ({estimated_tilt_angle}) is very different from expected value ({tilt_angles[index]}). Skipping handedness detection for this tilt")
    else:
        logger.warning(f"{estimated_tilt} does not exist. Skipping detecting handedness using tilt angle {angle_to_detect}... ")

    return None


def detect_handedness_tilt_range(name: str, tilt_angles: np.ndarray, lower_tilt: float = 10.0, upper_tilt: float = 50.0): 
    """ Detect the tilt handedness using multiple tilts 
        Lower bound and upper bound should be all positive. 
        For example, using lower_tilt == 10 and upper_tilt == 50, images within +10 to +50 and -10 to -50 will be used.  

    Args:
        name (str): Name of tilt-series
        tilt_angles (np.ndarray): tilt angles
        lower_tilt (float, optional): Lower tilt in the range. Defaults to 10.0.
        upper_tilt (float, optional): Upper tilt in the range. Defaults to 50.0.
    """

    lower_tilt = abs(lower_tilt)
    upper_tilt = abs(upper_tilt)
    assert lower_tilt <= upper_tilt, f"Lower tilt ({lower_tilt}) needs to be <= upper tilt ({upper_tilt})"
    logger.info(f"Using tilts between {lower_tilt} and {upper_tilt} to determine CTF handedness")
    candidates = []
    angle_used = 0

    for angle in tilt_angles:
        if (lower_tilt <= angle and angle <= upper_tilt) or (-upper_tilt <= angle and angle <= -lower_tilt):
            angle_used += 1 
            candidates.append(detect_handedness(name=name, 
                                                tiltang_file=Path(f"{name}.tlt"), 
                                                xf_file=Path(f"{name}.xf"), 
                                                angle_to_detect=angle,
                                                ))
    # remove tilted images that can be used 
    candidates = [_ for _ in candidates if _ is not None]
    if len(candidates) > 0:

        # report how many tilts are consistent with inverson/no-inversion
        true_count = candidates.count(True)
        false_count = candidates.count(False)
        logger.info(f"From a total of {angle_used} tilt images used for CTF handedness detection, {true_count} indicate that inversion is required, and {false_count} that is not")

        candidates.sort() # False is the first element after sorting
        median = candidates[math.floor(len(candidates)/2)]
        handedness = "" if median is True else "NOT "
        logger.warning(f"Invert CTF handedness option should {handedness}be selected during refinement")

    else:
        logger.warning("Not enough tilts to detect CTF handedness")