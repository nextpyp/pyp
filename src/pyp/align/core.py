from concurrent.futures import thread
import glob
from logging import raiseExceptions
import math
import multiprocessing
import os
import random
import shutil
import socket
import copy
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import scipy

from pyp import extract, merge, preprocess
from pyp.analysis import fit, plot
from pyp.analysis.geometry import transformations as vtk
from pyp.analysis.image import (
    bin_stack,
    contrast_stretch,
    extract_background,
    normalize_volume,
)
from pyp.analysis.scores import per_frame_scoring
from pyp.inout.image import mrc, writepng, img2webp, get_gain_reference
from pyp.inout.image.core import get_image_dimensions
from pyp.inout.metadata import (
    csp_extract_coordinates,
    csp_spr_swarm,
    frealign_parfile,
    get_non_empty_lines_from_par,
    get_particles_from_par,
    cistem_star_file,
)
from pyp.inout.metadata.frealign_parfile import ParameterEntry, Parameters
from pyp.refine.csp.particle_cspt import (
    merge_alignment_parameters,
    prepare_particle_cspt,
)
from pyp.refine.frealign import frealign
from pyp.system import mpi, project_params
from pyp.system.local_run import create_csp_split_commands, run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.set_up import initialize_classification, prepare_frealign_dir
from pyp.system.utils import (
    get_frealign_paths,
    get_imod_path,
    get_aretomo_path,
    get_summovie_path,
    get_unblur_path,
    get_unblur2_path,
    get_motioncor3_path,
    get_gpu_id,
    imod_load_command,
)
from pyp.system.wrapper_functions import avgstack
from pyp.utils import get_relative_path, symlink_force, symlink_relative
from pyp.utils.timer import Timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


@Timer("align_movies", text="Alignment took: {}", logger=logger.info)
def align_frames(parameters, name, current_path, working_path):
    if not any(
        x in parameters["movie_ali"]
        for x in ("tiltxcorr_average", "unblur", "skip", "relion")
    ):
        # align image stack (save into .avg)
        aligned_average = align_stack(name, parameters)
    else:
        # further refinement
        # default, unblur, none
        aligned_average = align_stack_super(
            name,
            parameters,
            current_path,
            working_path,
            parameters["movie_ali"],
            "-nearest",
            int(parameters["movie_group"]) - 1,
        )

    # get rid of aligned micrograph stack
    try:
        os.remove(name + ".ali")
    except:
        pass

    return aligned_average


def align_frame_to_reference(name, frame, input, tiltxcorr_options, frames, results):
    frame_name = "%06d" % frame

    # extract frame from movie
    com = "{0}/bin/newstack {1}.{4} {1}_{2}.mrc -secs {3}; rm -f {1}_{2}.mrc~".format(
        get_imod_path(), name, frame_name, frame, input
    )
    run_shell_command(com)

    # extract current frame from average
    avg = mrc.read("{0}.avg".format(name))
    this_frame = mrc.read("{0}_{1}.mrc".format(name, frame_name))
    cavg = (avg * frames - this_frame) / (frames - 1)
    mrc.write(cavg, "{0}_{1}.avg".format(name, frame_name))

    # use current average as reference
    com = "{0}/bin/newstack {1}_{2}.avg {1}_{2}.mrc {1}_{2}_tiltxcorr.mrc; rm -f {1}_{2}_tiltxcorr.mrc~".format(
        get_imod_path(), name, frame_name
    )
    run_shell_command(com,verbose=False)

    # compute shifts
    com = "{0}/bin/tiltxcorr -input {1}_{2}_tiltxcorr.mrc -output {1}_{2}_cumulative.prexg {3}".format(
        get_imod_path(), name, frame_name, tiltxcorr_options
    )
    run_shell_command(com,verbose=False)

    values = [float(cci.split()[-1]) for cci in output.split("\n") if "shifts" in cci]
    results.put([frame, values[0]])

    # extract frame shifts
    com = "tail -n +2 {0}_{1}_cumulative.prexg > {0}_{1}_cumulative.tmp; mv {0}_{1}_cumulative.tmp {0}_{1}_cumulative.prexg".format(
        name, frame_name
    )
    run_shell_command(com,verbose=False)

    # cleanup
    list = "{0}_{1}.mrc {0}_{1}.avg {0}_{1}_tiltxcorr.mrc".format(frame_name, frame)
    for i in list.split():
        if os.path.exists(i):
            os.remove(i)


def align_movie_frames_multiprocessing(
    allboxes,
    allparxs,
    parameters,
    local_parameters,
    particle,
    name,
    current_path,
    working_path,
    imagefile,
    apply=False,
):

    actual_pixel = (
        local_parameters["scope_pixel"]
        * local_parameters["data_bin"]
        * local_parameters["extract_bin"]
    )

    box_size = local_parameters["extract_box"]

    sname = name + "_P%06d_frames" % particle

    stack_file = sname + "_unbinned.mrc"
    if not os.path.exists(stack_file):

        # dummy image
        # image = np.empty([0])

        # extract frames from current particle only
        local_indexes = [
            boxes
            for boxes, line in zip(allboxes, allparxs)
            if float(line.split()[15]) == particle
        ]

        # local_particles = extract.extract_particles( image, local_indexes, np.array(parameters['particle_rad'].split(','),dtype=float).max() * float(parameters['data_bin']), int(parameters['extract_box']) * int(parameters['extract_bin']), float(parameters['extract_bin']), actual_pixel, False, name + '.mrc' )

        # extract particle frames in unbinned resolution
        local_particles = extract.extract_particles(
            input=os.path.join(current_path, imagefile),
            output=stack_file,
            boxes=local_indexes,
            radius=parameters["particle_rad"] * parameters["data_bin"],
            boxsize=parameters["extract_box"] * parameters["extract_bin"],
            binning=1,
            pixelsize=actual_pixel,
            cpus=1,
            parameters=local_parameters,
            normalize=False,
            fixemptyframes=True,
            method="imod",
            is_tomo=False,
            use_frames=False,
        )

        # post per-particle normalization
        local_particles = normalize_particles(
            parameters, actual_pixel, np.array(mrc.read(stack_file), ndmin=3)
        )

        # write particle stack
        mrc.write(local_particles, stack_file)

        if int(parameters["extract_bin"]) == 1:
            try:
                os.symlink(stack_file, sname + ".mrc")
            except:
                pass

        else:
            if not apply or not os.path.exists(sname + ".mrc"):
                bin_stack(
                    stack_file, sname + ".mrc", parameters["extract_bin"], "imod",
                )
            else:
                # leave original intact
                pass

    if float(parameters["extract_bin"]) <= 1:
        interpolation = "-nearest"
    else:
        interpolation = "-antialias 6"

    # do frame alignment
    align_stack_super(
        sname,
        local_parameters,
        current_path,
        working_path,
        method=parameters["movie_ali"],
        interpolation=interpolation,
        interval=int(parameters["movie_group"]) - 1,
        apply=apply,
    )

    # cleanup
    keep = False
    if not keep and apply:
        [
            os.remove(i)
            for i in glob.glob(sname + "*.*")
            if not ".xf" in i
            and not "ccc" in i
            and not "blr" in i
            and not "matches.png" in i
            and not "weights" in i
            and not ".avg" in i
            and not "_weights.txt" in i
            and not "P0000_frames_weights_new.png" in i
            and not "_stack.mrc" in i
            and not "_r01_02.par" in i
            and not "_r01_02_used.par" in i
        ]


# TODO: this should be moved elsewhere
def normalize_particles(parameters, actual_pixel, local_particles):
    background_mean, background_std = extract_background(
        local_particles.mean(0),
        parameters["particle_rad"] * parameters["data_bin"],
        actual_pixel * parameters["extract_bin"],
    )
    stds = []
    local_particles -= background_mean
    for count in range(local_particles.shape[0]):
        background_meani, background_stdi = extract_background(
            np.squeeze(local_particles[count, :, :]),
            parameters["particle_rad"] * parameters["data_bin"],
            actual_pixel * parameters["extract_bin"],
        )
        if background_stdi > 0:
            stds.append(background_stdi)
    std = np.array(stds).mean()
    for count in range(local_particles.shape[0]):
        local_particles[count, :, :] /= std
    return local_particles


def align_spr_local(
    parameters, name, current_path, working_path, allboxes, allparxs, imagefile,
):
    # this is a dummy to avoid reading the entire movie into memory

    # create copy of parameters and reset movie_bin
    local_parameters = parameters.copy()
    local_parameters["data_bin"] = parameters["data_bin"] * parameters["extract_bin"]
    local_parameters["movie_bin"] = 1

    local_particle = np.unique(np.asarray([int(f.split()[15]) for f in allparxs]))

    # creating the initial model
    # if os.path.exists(parameters["class_ref"]):
    #    frealign.create_initial_model(
    #        parameters, actual_pixel, box_size, local_parameters
    #    )

    # skip initialization if alignments exist
    if os.path.exists(name + "_P0000_frames.xf"):
        iterations = 0
    else:
        iterations = 1

    clean_shifts = align_spr_local_inner(
        parameters,
        name,
        current_path,
        working_path,
        allboxes,
        allparxs,
        local_particle,
        local_parameters,
        iterations,
        imagefile,
    )

    if iterations == 0:
        # retrieve clean shifts from existing .xf files
        frames = len(
            [
                boxes
                for boxes, line in zip(allboxes, allparxs[0])
                if float(line.split()[15]) == 0
            ]
        )
        clean_shifts = np.zeros([len(local_particle), frames, 6])
        for particle in local_particle:
            xf_file = name + "_P%06d_frames.xf" % particle
            clean_shifts[particle, :, :] = np.loadtxt(xf_file, dtype="f")
    else:
        # reload particle coordinates using local alignments

        logger.info("Re-loading particle coordinates using local drifts")

        os.chdir(current_path)
        allboxes = allparxs = []

        # [allboxes, allparxs] = csp_spr_swarm(name.replace("_r01",""), parameters)

        [os.remove(f) for f in glob.glob(os.path.join(working_path, "*.allparxs"))]

        [allboxes, allparxs] = csp_extract_coordinates(
            name.replace("_r01", ""),
            parameters,
            working_path,
            current_path,
            skip=False,
            only_inside=False,
            use_frames=True,
            use_existing_frame_alignments=True,
        )

        totalboxes = len(allboxes)
        os.chdir(working_path)

        # force re-extraction of particles

        [os.remove(f) for f in glob.glob(name + "_P??????_frames_unbinned.mrc")]

    micrograph_drift = np.round(np.loadtxt(name.replace("_r01", "") + ".xf", ndmin=2))

    logger.info("#### Apply final shifts and weights ####\n")

    arguments = []

    # apply final shifts and weights
    for particle in local_particle:
        arguments.append(
            (
                allboxes,
                allparxs,
                parameters,
                local_parameters,
                particle,
                name,
                current_path,
                working_path,
                imagefile,
                True,
            )
        )

    mpi.submit_function_to_workers(align_movie_frames_multiprocessing, arguments, verbose=parameters["slurm_verbose"])

    # Collate periodogram averages
    """
    if 'frealign' in parameters['movie_ali'].lower():
        aligned_averages = np.empty( [ len(local_particle), int(box_size), int(box_size) ] )
    else:
        aligned_averages = np.empty( [ len(local_particle), int(parameters['extract_box']), int(parameters['extract_box']) ] )

    while ( results.empty() == False ):
        t = results.get()
        aligned_averages[ t[0], :, : ] = t[1]
    """

    aligned_average_files = []
    particle_filenames = []

    for particle in local_particle:
        averaged_file = name + "_P%06d_frames.avg" % particle
        aligned_average_files.append(averaged_file)
        particle_filenames.append(name + "_P%06d_frames" % particle)
        if not os.path.exists(averaged_file):
            logger.warning(f"{averaged_file} not found. Retrying.")
            align_movie_frames_multiprocessing(
                allboxes,
                allparxs,
                parameters,
                local_parameters,
                particle,
                name.replace("_r01", ""),
                current_path,
                working_path,
                imagefile,
                True,
            )
            if not os.path.exists(averaged_file):
                logger.error(f"{averaged_file} not found.")

    per_frame_scoring(
        parameters, name, current_path, allboxes, allparxs, particle_filenames
    )

    # back to regular frame refinement step: merge into stack
    mrc.merge(aligned_average_files, name.replace("_r01", "") + "_stack.mrc")

    return (
        allboxes,
        local_particle,
        clean_shifts,
        micrograph_drift,
        aligned_average_files,
    )


def align_spr_local_inner(
    parameters,
    name,
    current_path,
    working_path,
    allboxes,
    allparxs,
    local_particle,
    local_parameters,
    iterations,
    imagefile,
):
    """Perform local csp spr alignment using particle frame alignment paradigm."""
    for iteration in range(iterations):

        ###########################################
        # compute all noisy particle trajectories #
        ###########################################

        arguments = []

        for particle in local_particle:
            arguments.append(
                (
                    allboxes,
                    allparxs,
                    parameters,
                    local_parameters,
                    particle,
                    name,
                    current_path,
                    working_path,
                    imagefile,
                )
            )

        mpi.submit_function_to_workers(align_movie_frames_multiprocessing, arguments, verbose=parameters["slurm_verbose"])

        ####################################
        # regularize particle trajectories #
        ####################################

        logger.info("Regularizing particle trajectories")

        # 1. read noisy shifts
        # frames = len(np.unique(film_arr[:, scanor_col]))
        frames = len(
            [
                boxes
                for boxes, line in zip(allboxes, allparxs[0])
                if float(line.split()[15]) - 1 == 0
            ]
        )
        num_particles = len(local_particle)

        total_shift_dimensions = 6

        noisy_shifts = np.zeros((num_particles, frames, total_shift_dimensions))
        for particle in local_particle:
            xf_file = name + "_P%06d_frames.xf" % particle
            noisy_shifts[particle, :, :] = np.loadtxt(xf_file, dtype="f")

            # save noisy trajectories
            xf_noisy_file = name + "_P%06d_frames_noisy.xf" % particle
            np.savetxt(xf_noisy_file, noisy_shifts[particle, :, :], fmt="%13.7f")

        # np.save(name + '_allshifts', noisy_shifts)
        # np.save(name + '_allboxes', np.array(allboxes))

        # 2. denoise trajectories
        clean_shifts = np.zeros(noisy_shifts.shape)
        clean_shifts[:, :, :4] = noisy_shifts[:, :, :4]

        # round global translation for consistency
        micrograph_drift = np.round(
            np.loadtxt(name.replace("_r01", "") + ".xf", ndmin=2)
        )

        for particle in local_particle:

            distances = scipy.spatial.distance.cdist(
                np.array(allboxes)[: len(local_particle), :],
                [np.array(allboxes)[particle, :]],
            )

            # retrieve frealign config
            if os.path.exists(
                os.path.split(parameters["refine_parfile"])[0] + "/../frealign.config"
            ):
                fparameters = project_params.load_fyp_parameters(
                    os.path.split(parameters["class_par"])[0] + "/../"
                )
                spatial_sigma = float(
                    project_params.param(fparameters["spatial_sigma"], 2)
                )
                time_sigma = float(project_params.param(fparameters["time_sigma"], 2))

                # rotational_method = project_params.param(fparameters["rotreg_method"])
                translational_method = project_params.param(
                    fparameters["transreg_method"], 2
                )
            else:
                spatial_sigma = 250
                time_sigma = 21

                # rotational_method = ""
                translational_method = ""

            particle_spatial_sigma = spatial_sigma

            # At a minimum, average this number of particles
            minimum_particles_to_average = 5

            # increase sigma if we don't have enough particles to average
            if (
                np.where(distances < 2 * particle_spatial_sigma, 1, 0).sum()
                < minimum_particles_to_average
            ):
                if distances.size > minimum_particles_to_average:

                    while (
                        np.where(distances < 2 * particle_spatial_sigma, 1, 0).sum()
                        < minimum_particles_to_average
                    ):
                        particle_spatial_sigma *= 1.25

            weights = np.exp(-distances / particle_spatial_sigma)

            if iteration == 0:
                if distances.size >= minimum_particles_to_average:
                    logger.info(
                        f"Local drift estimation - Using sigma {particle_spatial_sigma:5.0f} for particle {particle}"
                    )
                elif particle == 0:
                    logger.warning(
                        f"Using global drift correction, too few particles: {distances.size} < {minimum_particles_to_average}"
                    )

            weights = np.exp(-distances / particle_spatial_sigma)
            weights /= weights.sum()

            # regularize X-coordinate for this particle
            clean_shifts[particle, :, -1] = np.multiply(
                noisy_shifts[:, :, -1], weights
            ).sum(axis=0)

            # regularize Y-coordinate for this particle
            clean_shifts[particle, :, -2] = np.multiply(
                noisy_shifts[:, :, -2], weights
            ).sum(axis=0)

            # compute global trajectory

            if distances.size > minimum_particles_to_average:
                clean_shifts[particle, :, -2:] += micrograph_drift[:, -2:]
            else:
                # revert to global drift values
                clean_shifts[particle, :, -2:] = micrograph_drift[:, -2:]

            # fit global particle trajectory
            #### use methods from analysis.fit.regularize_film
            if "spline" in translational_method.lower():
                logger.info("using normal trans refine")
                clean_shifts[particle, :, :] = fit.fit_spline_trajectory(
                    clean_shifts[particle, :, :]
                )
            elif "xd" in translational_method.lower():
                logger.info("using XD trans refine")
                clean_shifts[particle, :, :] = fit.fit_spline_trajectory(
                    clean_shifts[particle, :, :], k=5, factor=0.6
                )
            # using AB method
            # TODO: crazy bug here -- TestCSP_SPR fails if this branch isn't commented out
            elif "ab" in translational_method.lower():
                logger.info("using AB trans refine")
                # clean_shifts[particle, :, -2] = fit.fit_spatial_trajectory_1D_new(clean_shifts[particle, :, -2], time_sigma)
                # clean_shifts[particle, :, -1] = fit.fit_spatial_trajectory_1D_new(clean_shifts[particle, :, -1] , time_sigma)
                logger.info("after second clean shift in AB trans refine")

            """
            method = parameters["movie_ali"]
            if "spline" in method.lower():
                clean_shifts[particle, :, :] = fit.fit_spline_trajectory(
                    clean_shifts[particle, :, :]
                )
            elif "poly" in method.lower():
                clean_shifts[particle, :, :] = fit.fit_poly_trajectory(
                    clean_shifts[particle, :, :], 2
                )
            elif "line" in method.lower():
                clean_shifts[particle, :, :] = fit.fit_poly_trajectory(
                    clean_shifts[particle, :, :], 1
                )
            """

            clean_shifts[particle, :, -2:] -= micrograph_drift[:, -2:]

        logger.info("Save updated particle trajectories")

        # 3. save regularized trajectories
        for particle in local_particle:
            xf_file = name + "_P%06d_frames.xf" % particle
            np.savetxt(xf_file, clean_shifts[particle, :, :], fmt="%13.7f")

        # update micrograph weights from all particle weights
        # particle weighting
        if os.path.exists(name + "_P0000_frames_scores.txt"):

            frames = len(
                [
                    boxes
                    for boxes, line in zip(allboxes, allparxs)
                    if float(line.split()[15]) - 1 == 0
                ]
            )
            all_weights = np.zeros([len(local_particle), frames])

            counter = 0
            for particle in local_particle:

                data_set_weights = parameters["data_set"] + "_%03d_weights.txt" % frames

                if len(local_particle) < 10 and os.path.exists(data_set_weights):

                    # default to global weights if not enough particles
                    if counter == 0 and iteration == 0:
                        logger.warning(
                            f"Not enough particles in micrograph, using global frame weights {data_set_weights}"
                        )

                    all_weights[counter, :] = np.loadtxt(data_set_weights)

                elif os.path.exists(name + "_P%06d_frames_scores.txt" % particle):

                    if len(local_particle) < 10 and counter == 0 and iteration == 0:
                        logger.warning(
                            f"Not enough particles in micrograph but cannot find global frame weights {data_set_weights}"
                        )

                    all_weights[counter, :] = np.loadtxt(
                        name + "_P%06d_frames_scores.txt" % particle
                    )

                else:

                    raise Exception(
                        f"File not found {name}_P{particle:04d}_frames_scores.txt"
                    )

                counter += 1

            np.savetxt(name + "_weights.txt", all_weights)
            for particle in local_particle:
                os.remove(name + "_P%06d_frames_scores.txt" % particle)

    return clean_shifts


def get_csp_command():
    return os.path.join(os.environ["PYP_DIR"], "external/CSP/csp")


def frealign_refinement(new_par_file, name, parameters, fp, target, working_path):
    # evaluate scores using mode 1, mask 0,0,0,0,0 (MOVE INTO FUNCTION)

    # figure out number of particles
    frames = get_particles_from_par(new_par_file)

    # go to frealign directory and create directory structure
    os.chdir("frealign")
    prepare_frealign_dir()

    # create new frealign parameters with updated dataset field
    fp_local = fp.copy()
    fp_local["dataset"] = name

    # create all neccesary symlinks
    os.symlink("../../" + new_par_file, "scratch/" + name + "_r01_02.par")
    os.symlink("../../" + target, "scratch/" + name + "_r01_01.mrc")
    os.symlink("../" + name + "_stack.mrc", name + "_stack.mrc")
    # run refinement
    frealign.split_refinement(parameters, fp_local, 1, 1, frames, 2, fp["metric"])

    # compose extended .parx file (MOVE INTO FUNCTION)
    long_file_name = "../../" + new_par_file
    short_file_name = name + "_r01_02.par_%07d_%07d" % (1, frames)

    concatenate_par_files(long_file_name, short_file_name, fp)

    # save alignments to working directory (remove comments and first column)
    com = "grep -v C {0} | cut -c8- > {1}".format(
        long_file_name, os.path.join(working_path, name + ".allparxs")
    )
    run_shell_command(com)
    # shutil.copy2( new_par_file, os.path.join( working_path, name + '_csp_alignments.parx' ) )

    # go back to working directory
    os.chdir(working_path)


def concatenate_par_files(long_file_name, short_file_name, mp):
    long_file = [line for line in open(long_file_name) if not line.startswith("C")]

    # if just switching to frealignx or new
    if len(long_file[0]) == 426 or len(long_file[0]) == 421:
        lwidth = 137
    else:
        lwidth = 145

    short_file = [line for line in open(short_file_name) if not line.startswith("C")]

    comments = [line for line in open(short_file_name) if line.startswith("C")]

    if "frealignx" in mp["refine_metric"].lower():
        width = 137
        columns = 17
        header = frealign_parfile.EXTENDED_FREALIGNX_PAR_HEADER
    else:
        width = 145
        columns = 16
        header = frealign_parfile.EXTENDED_NEW_PAR_HEADER

    if (
        len(long_file[0].split()) > columns
        and not len(short_file[0].split()) > columns
        and not "star" in short_file_name
    ):

        logger.info(
            f"Merging {Path(long_file_name).name} with {Path(short_file_name).name} into {long_file_name}"
        )

        with open(long_file_name, "w") as f:
            # add header first
            [f.write(line) for line in header]

            for i, j in zip(short_file, long_file):
                # f.write(i[: width - 1] + j[lwidth - 1 :])
                f.write(i[:-1] + j[lwidth - 1 :])
            # add possible statistics for the tail
            [f.write(line) for line in open(short_file_name) if line.startswith("C")]

def shape_mask_reference(mp, i, previous, target, mask=None):

    if mask or "refine_maskth" in mp.keys():
        mask_path = project_params.resolve_path(project_params.param(mp["refine_maskth"], i)) if not mask else mask
        if os.path.isfile(mask_path):
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

            # see if mask is already apodized
            mask = mrc.read(mask_path)
            if np.where(np.logical_and(mask > 0.0, mask < 1.0), 1, 0).sum() > 0:
                apodization = 0
            else:
                apodization = mp["mask_edge_width"] if "mask_edge_width" in mp else 3

            if os.path.exists(mask_path):
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
                previous,
                mask_path,
                apodization,
                mp["mask_outside_weight"] if "mask_outside_weight" in mp else 0.0,
                target,
            )
            else:
                command = """
%s/bin/apply_mask.exe << eot
M
%s
*
%s
%d
0
0.0
%s
eot
""" % (
                frealign_paths["new"],
                previous,
                project_params.param(mp["refine_maskth"], i + 1),
                apodization,
                target,
            )
            run_shell_command(command, verbose=mp["slurm_verbose"])
    
    elif not previous == target:

        # no masking
        shutil.copy2(previous, target)


def prepare_to_run_frealign(
    mp, fp, new_name, parxfile, dataset, current_class, current_path, working_path, i
):
    # write parx file by pre-pending particle index
    frealign_parfile.Parameters.csp_merge_parameters([new_name], parxfile)

    shutil.copy2(
        parxfile, "maps/%s_frames_CSP_01.parx" % dataset,
    )

    # reference to use for refinement
    previous = mp["refine_model"]
    if not os.path.exists(previous):
        previous = os.path.join(current_path, previous)

    target = "maps/%s_frames_CSP_r%02d_%02d.mrc" % (dataset, current_class, i + 1)

    # shape masking
    # target = masking(previous)
    shape_mask_reference(fp, i + 1, previous, target)

    # copy to local scratch
    shutil.copy2(target, os.environ["PYP_SCRATCH"])

def csp_run_refinement(
    alignment_parameters, 
    parameters,
    dataset,
    name,
    current_class,
    iteration,
    current_path,
    use_frames=False,
):
    # do further initialization for csp
    # csp expects all configuration files to be on local scratch
    local_scratch = os.path.join(os.environ["PYP_SCRATCH"])

    frame_tag = "_local" if use_frames else ""

    if current_class > 1:
        try:
        # os.remove(os.path.join("frealign", "maps", "%s_frames_CSP_01.parx" % dataset))
            os.remove(os.path.join("frealign", "scratch", "%s_frames_CSP_01.mrc" % dataset))
            os.remove(os.path.join(local_scratch, "%s_frames_CSP_01.mrc" % dataset))
            os.remove(os.path.join(local_scratch, "_frames_CSP_01.mrc"))
        except:
            pass

    parameter_file = f"frealign/maps/{name}.cistem"
    extended_parameter_file = parameter_file.replace(".cistem", "_extended.cistem")

    # sync projection occ with particle occ
    alignment_parameters.sync_particle_occ()
    alignment_parameters.to_binary(output=parameter_file)

    # reconstruction
    source = os.path.join(
        os.getcwd(), "frealign", "scratch", "%s.mrc" % (name)
    )

    # link needed by csp
    target1 = os.path.join(local_scratch, "%s_frames_CSP_01.mrc" % dataset)

    # TODO: why is this one needed when running frontend?
    target2 = os.path.join(local_scratch, "_frames_CSP_01.mrc")

    # link needed by frealign evaluation
    target3 = os.path.join("frealign", "scratch", "%s_frames_CSP_01.mrc" % dataset)

    symlink_force(source, target1)
    symlink_force(source, target2)
    symlink_force(source, target3)

    # get number of tilts and particles
    ptlind_list = alignment_parameters.get_extended_data().get_particle_list() 
    scanord_list = alignment_parameters.get_extended_data().get_tilt_list() 
    frame_list = np.sort(np.unique(alignment_parameters.get_data()[:, alignment_parameters.get_index_of_column(cistem_star_file.FIND)])) 

    cpus = int(parameters["slurm_tasks"])

    use_images_for_refinement_min = project_params.param(
        parameters["csp_UseImagesForRefinementMin"], iteration
    )
    use_images_for_refinement_max = project_params.param(
        parameters["csp_UseImagesForRefinementMax"], iteration
    )

    csp_refine_min, csp_refine_max = parameters["csp_UseImagesForRefinementMin"], parameters["csp_UseImagesForRefinementMax"]
    only_evaluate = False
    patch_refinement = False

    if "spr" in parameters["data_mode"].lower():
        grids = [int(number.strip()) for number in parameters["csp_Grid_spr"].split(",")]
        assert (len(grids) == 2), f"Grids for single-particle region-based refinement should have two dimensions. (i.e, 4,4)"
    else:
        grids = [int(number.strip()) for number in parameters["csp_Grid"].split(",")]
        assert (len(grids) == 3), f"Grids for tomography region-based refinement should have three dimensions (i.e., 8,8,2)."

    # we need -2 in ALL CASES to extract particles for later refinement or reconstruction
    if current_class > 1:
        csp_modes = []
    else:
        csp_modes = [-2] 
        if 'spr' in parameters["data_mode"].lower() and (parameters["csp_refine_ctf"] and not use_frames):
            csp_modes = []

    # check if we wanna do patch-based refinement 
    if 'tomo' in parameters["data_mode"].lower():
        for i, numGrid in enumerate(grids):
            if numGrid > 1: 
                patch_refinement = True
            elif numGrid <= 0:
                grids[i] = 1 # it shouldn't be 0 
    else:
        for i, numGrid in enumerate(grids):
            if numGrid <= 0:
                grids[i] = 1 # it shouldn't be 0 

    # map CSP modes into codes
    if 'tomo' in parameters["data_mode"].lower():
        if parameters["csp_refine_micrographs"]:
            if patch_refinement:
                csp_modes += [5]
            else:
                csp_modes += [3]
        if parameters["csp_refine_particles"]:
            if patch_refinement:  
                csp_modes += [7] 
            else: 
                csp_modes += [2] 
        if parameters["csp_refine_ctf"]:
            csp_modes += [4]

        if use_frames and parameters["csp_frame_refinement"]:
            csp_modes += [5]
    else:
        if parameters["csp_refine_ctf"] and not use_frames:
            csp_modes += [4]
        if use_frames:
            if parameters["csp_produce_running_average"]:
                csp_modes += [-2.1]
            if parameters["csp_frame_refinement"]:
                csp_modes += [5]
            if not parameters["refine_skip"] and parameters["class_num"] == 1:
                csp_modes += [3]

    for i in range(1):

        prev_alignment_parameters = copy.deepcopy(alignment_parameters)

        starting_num_projections = alignment_parameters.get_num_rows()
        starting_num_particles = alignment_parameters.get_extended_data().get_num_tilts()
        starting_num_tilts = alignment_parameters.get_extended_data().get_num_tilts()

        ###############################
        # The available modes in CSPT #
        ###############################
        # Mode -1 - Skip refinement
        # Mode 0 - Micrograph angles ( tilt angle, axis angle )
        # Mode 1 - Particle angles ( PSI, THETA, PHI )
        # Mode 2 - Particle shifts ( pX, pY, pZ )
        # Mode 3 - Micrograph shifts ( mX, mY )     ## note: global
        # Mode 4 - Micrograph defocus ( defocus offset ) under construction

        for mode in csp_modes:

            outputs_pattern = "_??????_??????"

            if mode == -2:
                t = Timer(text="Particle extraction (mode -2) took: {}", logger=logger.info)
                t.start()
            elif mode == 3:
                t = Timer(text="Micrograph refinement (mode 3) took: {}", logger=logger.info)
                t.start()
            else:
                t = Timer(text="Particle refinement (mode 2) took: {}", logger=logger.info)
                t.start()

            # parx_object = Parameters.from_file(new_par_file)
            merged_stack = "frealign/%s_stack.mrc" % (name.split("_r")[0])
            frame_refinement = False

            logger.info(f"Running CSPT (mode {mode}) using exposures {use_images_for_refinement_min} to {use_images_for_refinement_max}")

            if not use_frames:
                if mode != -2 and mode != 7 and mode != 2 and mode != 4:
                    parameters["csp_UseImagesForRefinementMin"], parameters["csp_UseImagesForRefinementMax"] = 0, -1
                else:
                    parameters["csp_UseImagesForRefinementMin"], parameters["csp_UseImagesForRefinementMax"] = csp_refine_min, csp_refine_max

            project_params.save_parameters(parameters)

            if mode in (0, 3):
                if "spr" in parameters["data_mode"].lower():
                    commands, count, movie_list = create_csp_split_commands(
                        get_csp_command(),
                        parameter_file,
                        mode,
                        cpus,
                        name,
                        merged_stack,
                        ptlind_list,
                        scanord_list,
                        frame_list,
                        parameters,
                        use_frames,
                    )
                else:
                    commands, count, movie_list = create_csp_split_commands(
                        get_csp_command(),
                        parameter_file,
                        mode,
                        cpus,
                        name,
                        merged_stack,
                        ptlind_list,
                        scanord_list,
                        frame_list,
                        parameters,
                        use_frames,
                    )

            elif mode in (-1, -2, -2.1, 1, 2):
                commands, count, movie_list = create_csp_split_commands(
                    get_csp_command(),
                    parameter_file,
                    mode,
                    cpus,
                    name,
                    merged_stack,
                    ptlind_list,
                    scanord_list,
                    frame_list,
                    parameters, 
                    use_frames,
                )
            elif mode in (4, 5, 6, 7, 8):
                if mode in (4, 5, 6):
                    MODE = "M"
                elif mode in (7, 8):
                    MODE = "P"

                split_parx_list = prepare_particle_cspt(
                    name[:-4], parameter_file, alignment_parameters, parameters, grids=grids, use_frames=use_frames
                )
                if not split_parx_list:
                    logger.error("Mode %d stops running." % mode)
                    continue

                outputs_pattern = "_region????_??????_??????"

                if mode == 5:
                    if use_frames and parameters["csp_frame_refinement"]: 
                        frame_refinement = True
                    mode = 3  # micrograph trans

                elif mode == 6:
                    mode = 0  # micrograph rot

                elif mode == 7:
                    mode = 2  # particle trans

                elif mode == 8:
                    mode = 1  # particle rot

                elif mode == 4:
                    pass

                commands, count, movie_list = create_csp_split_commands(
                    get_csp_command(),
                    split_parx_list,
                    mode,
                    cpus,
                    name,
                    merged_stack,
                    ptlind_list,
                    scanord_list,
                    frame_list,
                    parameters,
                    use_frames,
                    cutoff=parameters["csp_RefineProjectionCutoff"]
                )

            else:
                message = "Unknown mode" + str(mode)
                logger.error(message)
                raise Exception(message)

            if mode in (-2, -2.1):
                extract_only = True
            else:
                extract_only = False

            if not (extract_only and current_class > 1): 
                mpi.submit_jobs_to_workers(commands, os.getcwd(), verbose=parameters["slurm_verbose"])

            time.sleep(3)

            if extract_only and current_class == 1:
                # first check if every stack exists since we discard some particles in parfile
                merged_stack = "frealign/%s_stack.mrc" % (name.split("_r")[0]) if mode == -2 else "frealign/%s_stack_weighted_average.mrc" % (name.split("_r")[0])
                movie_list = [stack for stack in movie_list if os.path.exists(stack)]

                try:
                    mrc.merge_fast(movie_list,merged_stack,remove=True)
                except:
                    raise Exception("Particle extraction fails. Perhaps your slurm_memory is not enough. (currently %d G)" % (parameters["slurm_memory"]))
                [
                    os.remove(f)
                    for f in glob.glob(
                        "%s_csp%s.log" % (name.split("_r")[0], outputs_pattern)
                    )
                ]
                t.stop()

                # stop here if extracting particle stacks
                continue


            # collate log files
            for f in sorted(
                glob.glob("*_csp_*.log")
            ):
                with open(f) as log:
                    lines = log.read()
                    if len(lines) > 0:
                        # pretty display
                        if parameters["slurm_verbose"]:
                            logger.info("\n" + lines)

                os.remove(f)

            # merge updated parameter files
            if not extract_only:
                alignment_parameters = merge_alignment_parameters(parameter_file, mode, outputs_pattern)
                alignment_parameters.update_particle_score(tind_range=[use_images_for_refinement_min, use_images_for_refinement_max])
                alignment_parameters.to_binary(output=parameter_file)

            # clean-up intermediate results after merge
            [
                os.remove(f)
                for f in set(glob.glob(f"frealign/maps/{name}{outputs_pattern}*.cistem") 
                + glob.glob(f"frealign/maps/*region*.cistem"))
            ]

            # regularize particle trajectories if we're doing particle frame refinement
            if frame_refinement and not only_evaluate:

                if parameters["csp_rotreg"] or parameters["csp_transreg"]:
                    
                    fit.regularize(filename=name.split("_r")[0], 
                                   prev_alignment_parameters=prev_alignment_parameters, 
                                   alignment_parameters=alignment_parameters, 
                                   parameters=parameters)
                    # NOTE: regularize() function updates the shifts in "alignment_parameters", so we just need to write a new file so that csp can read it
                    alignment_parameters.to_binary(output=parameter_file)

                    csp_modes.append(5)
                    parameters["csp_UseImagesForRefinementMin"] = int(max(scanord_list)) + 1
                    parameters["csp_UseImagesForRefinementMax"] = int(max(scanord_list)) + 1
                    only_evaluate = True 

                    project_params.save_parameters(parameters)

            elif only_evaluate:
                parameters["csp_UseImagesForRefinementMin"], parameters["csp_UseImagesForRefinementMax"] = csp_refine_min, csp_refine_max
                project_params.save_parameters(parameters)
                only_evaluate = False
            
            t.stop()

            num_projections = alignment_parameters.get_num_rows()
            num_particles = alignment_parameters.get_extended_data().get_num_tilts()
            num_tilts = alignment_parameters.get_extended_data().get_num_tilts()
            
            assert (starting_num_projections == num_projections), f"Number of projections (in {parameter_file}) before and after refinement differ: {starting_num_projections} != {num_projections}"
            assert (starting_num_particles == num_particles), f"Number of particles (in {extended_parameter_file}) before and after refinement differ: {starting_num_particles} != {num_particles}"
            assert (starting_num_tilts == num_tilts), f"Number of tilts (in {extended_parameter_file}) before and after refinement differ: {starting_num_tilts} != {num_tilts}"
            
    return Path().cwd() / parameter_file


def csp_run_spa_refinement(
    parxfile,
    parameters,
    dataset,
    name,
    current_class,
    iteration,
    current_path,
    working_path,
    allboxes,
    allparxs,
    imagefile,
):

    # do further initialization for csp
    if current_class == 1:

        # csp expects all configuration files to be on local scratch
        local_scratch = os.path.join(os.environ["PYP_SCRATCH"], name.split("_r01")[0])

        shutil.copy2(
            parxfile,
            os.path.join("frealign", "maps", "%s_frames_CSP_01.parx" % dataset),
        )

        # reconstruction
        source = os.path.join(
            os.getcwd(), "frealign", "scratch", "%s_%02d.mrc" % (name, iteration - 1)
        )

        # link needed by csp
        target = os.path.join(local_scratch, "initial_model.mrc")
        symlink_force(source, target)

    # allparxs = get_non_empty_lines_from_par(parxfile)

    (allboxes, _, _, _, _,) = align_spr_local(
        parameters, name, current_path, working_path, allboxes, allparxs, imagefile,
    )

    return os.path.join(os.getcwd(), parxfile)

@Timer(
    "postprocess_after_refinement", text="Refinement took: {}", logger=logger.info
)
def postprocess_after_refinement(
    new_par_file, name, mp, current_class, current_path, working_path, iteration,
):
    """Run frealign refinement.

    Parameters
    ----------
    new_par_file : str
        Frealign parameter file
    name : str
        Dataset name
    parameters : Parameter
        PYP parameters
    mp : Parameter
        PYP parameters
    current_class : int
        Class number
    current_path : str
        Project path
    working_path : str
        Working directory
    """
    # post-processing: evaluate scores using mode 1, mask 0,0,0,0,0 (MOVE INTO FUNCTION)

    # figure out number of particles
    # frames = get_particles_from_par(new_par_file)
    stackfile = os.path.join(working_path, "frealign", name + "_stack.mrc")
    frames = mrc.readHeaderFromFile(stackfile)["nz"]

    # create symlinks in scratch folder
    # new_par_file_class = new_par_file.replace("_r01_","_r%02d_" % current_class)
    # os.symlink( os.path.join( os.getcwd(), new_par_file_class ), os.path.join( "frealign", "scratch", name + "_r%02d_%02d.par" % ( current_class, iteration ) ) )

    # go to frealign directory
    os.chdir("frealign")

    # create new frealign parameters with updated dataset field
    mp_local = mp.copy()
    mp_local["refine_dataset"] = name
    if (
        not "local" in mp_local["extract_fmt"]
        and "spr" == mp_local["data_mode"]
        and int(mp_local["refine_mode"]) == 0
    ):
        mp_local["refine_mode"] = "4"
    else:
        mp_local["refine_mode"] = "1"

    new_name = name + "_r%02d" % current_class

    global_dataset_name =  mp_local["data_set"]
    global_decompressed_foler = os.path.join(current_path, "frealign", "maps", global_dataset_name + "_r%02d_%02d" % (current_class, iteration - 1))
    global_stat_file = os.path.join(global_decompressed_foler, global_dataset_name + "_r%02d_stat.cistem" % current_class)
    # create symlinks in scratch folder
    symlink_relative(
        new_par_file, os.path.join("scratch", f"{new_name}.cistem")
    )

    if os.path.exists(global_stat_file):
        symlink_relative(
            global_stat_file, os.path.join("scratch", f"{new_name}_stat.cistem")
        )

    # reset occupancies if not doing classification
    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(mp["class_num"], iteration))

    # update alignment mask according to settings used in classification
    if classes > 1:
        if "spr" in mp_local["data_mode"]:
            if iteration % mp_local["class_refineeulers"] == 0:
                refine_eulers = "1,1,1,"
            else:
                refine_eulers = "0,0,0,"
            if iteration % mp_local["class_refineshifts"] == 0:
                refine_shifts = "1,1"
                mp_local["refine_mode"] = mp["refine_mode"] = "1"
            else:
                refine_shifts = "0,0"
            mp_local["refine_mask"] = refine_eulers + refine_shifts
        else: # tomo never use refine3d refine for classification
            mp_local["refine_mask"] = "0,0,0,0,0"
    else: # updating refine mask by user
        refine_mask = []
        if mp_local["refine_refine_angle_phi"]:
            refine_mask.append("1")
        else:
            refine_mask.append("0")
        if mp_local["refine_refine_angle_theta"]:
            refine_mask.append("1")
        else:
            refine_mask.append("0")
        if mp_local["refine_refine_angle_psi"]:
            refine_mask.append("1")
        else:
            refine_mask.append("0")
        if mp_local["refine_refine_shiftx"]:
            refine_mask.append("1")
        else:
            refine_mask.append("0")
        if mp_local["refine_refine_shifty"]:
            refine_mask.append("1")
        else:
            refine_mask.append("0")

        mp_local["refine_mask"] = ",".join(refine_mask)

    # run refinement (produces short_file_name in the scratch folder)
    frealign.split_refinement(
        mp_local,
        current_class,
        current_path,
        1,
        frames,
        iteration,
        project_params.param(mp["refine_metric"], iteration),
    )

    # save log file to frealign directory (only if not in mode=0)
    if True or int(project_params.param(mp["refine_mode"], iteration)):
        with open(glob.glob(os.path.join(current_path, "*.micrographs"))[0]) as f:
            micrographs = f.read().split("\n")
        if name == micrographs[0]:
            shutil.copy2(
                glob.glob("*_msearch_n.log_*")[0],
                os.path.join(
                    current_path,
                    "frealign",
                    "log",
                    "%s_r%02d_%02d_msearch.log"
                    % (mp["data_set"], current_class, iteration),
                ),
            )
            # send output to user interface
            if 'slurm_verbose' in mp and mp['slurm_verbose']:
                with open(glob.glob("*_msearch_n.log_*")[0]) as f:
                    logger.info(f.read())
        # remove log files
        if not mp["refine_debug"]:
            [os.remove(f) for f in glob.glob("*_msearch_n.log_*")]

    # output_refine3d = new_name + "_%07d_%07d.cistem" % (1, frames)
    output_refine3d = new_name + "_refined.cistem" # current dir is frealign/scratch/
    
    # compose extended .parx file
    # long_file_name = os.path.join("maps", new_par_file)
    # short_file_name = new_name + "_%02d.par_%07d_%07d" % (iteration, 1, frames,)
    # remove outliers after refine3d
    # frealign_parfile.Parameters.remove_outliers(short_file_name, "score", 0, 100)
    # concatenate_par_files(long_file_name, short_file_name, mp)
    
    newpar_obj = cistem_star_file.Parameters.from_file(output_refine3d)
    newpar_obj.modify_outliers_in_column(newpar_obj.get_index_of_column(cistem_star_file.SCORE), min=0, max=100)

    if classes == 1:    
        col_index = newpar_obj.get_index_of_column(cistem_star_file.OCCUPANCY)
        newpar_obj.modify_projdata_by_column(col_index, 100)

        logger.info("Resetting occupancies to 100.0 since classes = 1")

    os.remove(new_par_file)
    # replace the cistem scratch folder
    shutil.copy2( output_refine3d, new_par_file)

    """
    # here we reformat columns to force the standard format (even if that means columns will be joint)
    input = np.array(
        [line for line in open(long_file_name) if not line.startswith("C")]
    )

    if "frealignx" in project_params.param(mp["refine_metric"], iteration).lower():
        scores = 15
        occ = 12
    else:
        scores = 14
        occ = 11

    comments = [line for line in open(long_file_name) if line.startswith("C")]

    (
        fieldwidths,
        fieldstring,
        _,
        _,
    ) = frealign_parfile.Parameters.format_from_parfile(long_file_name)
    

    with open(long_file_name, "w") as f:

        f.write("".join(comments))

        for line in input:
            values = np.array(line.split(), dtype="f")
            if occ > 0 and mp["data_mode"] == "spr":
                values[occ] = 100
            # truncate scores to prevent overflow
            # if values[scores] > 9999:
            #     values[scores] = 0
            #     values[scores + 1] = 0
            f.write(fieldstring % tuple(values))
    """

    # go back to working directory
    os.chdir(working_path)

@Timer(
    "csp_refinement", text="CSP Total time elapsed: {}", logger=logger.info
)
def csp_refinement(
    mp,
    name,
    current_path,
    working_path,
    use_frames,
    allparxs,
    iteration,
):
    """Unified stack-less single-particle refinement.

    Parameters
    --------- 
    mp : Parameters
        PYP parameters
    name : str
        Name of series
    current_path : Path
        Location of project directory
    working_path : Path
        Location of working scratch directory
    """

    # local frealign directory
    local_frealign_folder = os.path.join(working_path, "frealign")

    # go to local working directory
    os.chdir(working_path)

    dataset = mp["data_set"]

    remote_frealign_dir = os.path.join(current_path, "frealign")

    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(mp["class_num"], iteration))
    
    # parx file not needed.
    """
    # write parx file for class=1 by pre-pending particle index (frealign/maps/name_r01_01.parx -> parxfile)
    with Timer(
        "particle re-index", text = "Pre-pending index to par took: {}", logger=logger.info
    ):
        frealign_parfile.Parameters.csp_merge_parameters(
            ["maps/" + name + "_r01_%02d" % (iteration - 1)],
            parxfile,
            "frealignx" in project_params.param(mp["refine_metric"], iteration),
        )
    """

    # copy maps to working maps directory
    [
        shutil.copy2(f, os.path.join(local_frealign_folder, "maps"))
        for f in glob.glob(
            os.path.join(
                remote_frealign_dir,
                "maps",
                "%s_r??_%02d.mrc" % (dataset, iteration - 1),
            )
        )
    ]

    # not being used
    """
    if classes > 1:

        # frealign/name_r??_01.parx -> frealign/name_r??_01.par
        for class_index in range(classes):

            current_class = class_index + 1

            # write parx file by pre-pending particle index (frealign/name_r01_01.parx -> parxfile)
            class_parxfile = (
                "maps/" + name + "_r%02d_%02d" % (current_class, iteration - 1)
            )
            new_class_parxfile = parxfile.replace("_r01_", "_r%02d_" % (current_class))

            if current_class > 1:
                this_parfile = os.path.join("frealign", class_parxfile)
                if os.path.exists(this_parfile):
                    frealign_parfile.Parameters.csp_merge_parameters(
                        [class_parxfile],
                        new_class_parxfile,
                        "frealignx"
                        in project_params.param(mp["refine_metric"], iteration),
                    )
                else:
                    logger.error(
                        "Class initialization failed. %s does not exist." % this_parfile
                    )

            # also create .par file needed to calculate occupancies
            shutil.copy2(
                new_class_parxfile, new_class_parxfile.replace(".parx", ".par")
            )
    """
    # prepare, run and post-process after refinement (do this for each class)
    for class_index in range(classes):

        current_class = class_index + 1
        logger.info(
            "Running refinement for class {} of {}".format(current_class, classes)
        )

        new_name = name + "_r%02d" % current_class

        previous = os.path.join(
            local_frealign_folder,
            "maps",
            dataset + "_r%02d_%02d.mrc" % (current_class, iteration - 1),
        )

        if not os.path.exists(previous):
            message = "Cannot find " + previous
            logger.error(message)
            raise Exception(message)

        # output map from this iteration
        target = new_name + ".mrc"
        target = os.path.join(local_frealign_folder, "scratch", target)

        # copy initial model to local frealign folder
        shutil.copy2(previous, target)

        # apply shape masking if needed
        shape_mask_reference(mp, iteration, target, target)

        is_tomo = "tomo" in mp["data_mode"].lower()

        # class_parxfile = parxfile.replace("_r01_", "_r%02d_" % current_class)

        # Everything that modifies parameter files falls here
        allparxs[class_index].update_pixel_size(mp["scope_pixel"] * mp["extract_bin"]) # data_bin?
        parameter_file = Path().cwd() / "frealign" / "maps" / f"{new_name}.cistem"

        # execute refinement
        if is_tomo:

            parameter_file = csp_run_refinement(
                allparxs[class_index],
                mp,
                dataset,
                new_name,
                current_class,
                iteration,
                current_path,
                use_frames,
            )

            allparxs[class_index] = cistem_star_file.Parameters.from_file(str(parameter_file))

        elif use_frames or mp["csp_refine_ctf"]: # run csp for only refine ctf or frame refinement 

            parameter_file = csp_run_refinement(
                allparxs[class_index],
                mp,
                dataset,
                new_name,
                current_class,
                iteration,
                current_path,
                use_frames,
            )

            allparxs[class_index] = cistem_star_file.Parameters.from_file(str(parameter_file))

        else:
            # we need parameter file on disk for spr
            allparxs[class_index].to_binary(str(parameter_file))

        # run frealign refinement
        if (classes > 1 or not use_frames) and not mp["refine_skip"]:

            postprocess_after_refinement(
                str(parameter_file),
                name,
                mp,
                current_class,
                current_path,
                working_path,
                iteration,
            )

    # write out the stack file and par file into a txt for later processing
    with open(os.path.join(os.environ["PYP_SCRATCH"], "stacks.txt"), "a") as f:
        f.write(os.path.join(name, "frealign/" + name + "_stack.mrc\n"))

    # save the first class name here only
    with open(os.path.join(os.environ["PYP_SCRATCH"], "pars.txt"), "a") as f:
        f.write(str(parameter_file).replace(f"_r{classes:02d}", f"_r01") + "\n")

    # if the project directory file is not written
    project_dir_file = os.path.join(os.environ["PYP_SCRATCH"], "project_dir.txt")
    if not os.path.exists(project_dir_file):
        with open(project_dir_file, "w") as f:
            f.write(str(current_path))

    # save fp and mp into the main slurm job folder
    project_params.save_pyp_parameters(mp, "..")

    # find film number for this micrograph to figure out particle alignments
    try:
        with open(os.path.join(current_path, mp["data_set"] + ".films")) as x:
            series = [
                num
                for num, line in enumerate(x, 1)
                if "{}".format(name) == line.strip()
            ][0] - 1
        # write the overall order of the files
        with open(os.path.join(os.environ["PYP_SCRATCH"], "ordering.txt"), "a") as f:
            f.write("{}\n".format(series))
    except:
        raise Exception("ERROR - Cannot find film number for " + name)


def align_stack(name, parameters, interpolation="-linear"):
    if os.path.isfile("{}.xf".format(name)):

        # apply binning factor
        t = np.loadtxt("%s.xf" % name, ndmin=2)
        binning = int(parameters["data_bin"])
        if binning > 1:
            t[:, -2:] /= binning
        first = int(parameters["movie_first"])
        if first > 0:
            t = t[
                first:,
            ]
        np.savetxt("%s_actual.xf" % name, t, fmt="%13.7f")

        # generate aligned stack with latest alignment parameters
        command = "{0}/bin/newstack -nearest -xform {1}_actual.xf {1}.mrc {1}.ali; rm -f {1}.ali~".format(
            get_imod_path(), name
        )
        # command = '{0}/bin/newstack -linear -xform {1}_actual.xf {1}.mrc {1}.ali'.format(get_imod_path(),name)
        run_shell_command(command)

    else:

        frames = mrc.readHeaderFromFile(name + ".mrc")["nz"]

        if frames > 1:

            x = mrc.readHeaderFromFile(name + ".mrc")["nx"]
            # add an additional factor of binning if images are super-resolution
            if x > 6096:
                movie_binning_factor = 2
            else:
                movie_binning_factor = 1
            movie_binning = int(parameters["movie_bin"]) * movie_binning_factor

            # use binned stack
            run_shell_command(
                "{0}/bin/newstack {3} {1}.mrc {1}.bin -bin {2} -mode 2; rm -f {1}.bin~".format(
                    get_imod_path(), name, movie_binning, interpolation
                )
            )

            middle_frame = frames // 2
            # middle_frame = 38
            logger.info(
                f"Computing all transformations with respect to frame {middle_frame}"
            )

            tiltxcorr_options = "-first {0} -increment 1.0 -nostretch -binning 1 -shift 10,10 -rotation 0.000000 -radius1 0.010000 -sigma1 0.030000 -radius2 0.100000 -sigma2 0.030000 -border 64,64 -taper 256,256 -iterate 5".format(
                1 - middle_frame
            )

            # align all frames to frame one
            [output, error] = run_shell_command(
                "{0}/bin/tiltxcorr -input {1}.bin -output {1}_first.prexf {2}".format(
                    get_imod_path(), name, tiltxcorr_options
                ), verbose = False
            )
            # extract ccc values
            indexes = [
                float(cci.split()[1].replace(",", ""))
                for cci in output.split("\n")
                if "shifts" in cci
            ]
            values = [
                float(cci.split()[-1]) for cci in output.split("\n") if "shifts" in cci
            ]
            a = np.array([indexes, values])
            b = a[:, np.argsort(a[0])]
            ccc = np.concatenate(
                (
                    b[:, : middle_frame - 1],
                    np.array([middle_frame, 1]).reshape(2, 1),
                    b[:, middle_frame - 1 :],
                ),
                axis=1,
            )
            np.savetxt("{0}.ccc".format(name), ccc[1, :])

            # convert to global shifts with respect to middle frame
            run_shell_command(
                "{0}/bin/xftoxg -nfit 0 -ref {1} -input {2}_first.prexf -goutput {2}_first.prexg".format(
                    get_imod_path(), middle_frame, name
                ), verbose = False
            )

            # generate aligned stack
            run_shell_command(
                "{0}/bin/newstack {2} -xform {1}_first.prexg {1}.bin {1}.ali -mode 2 -multadd 1,0".format(
                    get_imod_path(), name, interpolation
                ), verbose = False
            )

            error = 1
            iteration = 0
            while error > 1e-5 and iteration < int(parameters["movie_iters"]):

                # re-align stack to cumulative frame average
                [output, error] = run_shell_command(
                    "{0}/bin/tiltxcorr -input {1}.ali -output {1}_cumulative.prexf -cumulative {2}".format(
                        get_imod_path(), name, tiltxcorr_options
                    )
                )
                indexes = [
                    float(cci.split()[1].replace(",", ""))
                    for cci in output.split("\n")
                    if "shifts" in cci
                ]
                values = [
                    float(cci.split()[-1])
                    for cci in output.split("\n")
                    if "shifts" in cci
                ]
                a = np.array([indexes, values])
                b = a[:, np.argsort(a[0])]
                ccc = np.concatenate(
                    (
                        b[:, : middle_frame - 1],
                        np.array([middle_frame, 1]).reshape(2, 1),
                        b[:, middle_frame - 1 :],
                    ),
                    axis=1,
                )
                np.savetxt("{0}.ccc".format(name), ccc[1, :])

                # convert to global shifts with respect to middle frame
                run_shell_command(
                    "{0}/bin/xftoxg -nfit 0 -ref {1} -input {2}_cumulative.prexf -goutput {2}_cumulative.prexg".format(
                        get_imod_path(), middle_frame, name
                    ),
                )

                # concatenate with latest transform
                run_shell_command(
                    "{0}/bin/xfproduct {1}_first.prexg {1}_cumulative.prexg {1}.prexg".format(
                        get_imod_path(), name
                    ), verbose = False
                )

                # generate aligned stack with latest alignment parameters
                run_shell_command(
                    "{0}/bin/newstack {2} -xform {1}.prexg {1}.bin {1}.ali".format(
                        get_imod_path(), name, interpolation
                    ), verbose = False
                )

                # update current transform
                run_shell_command("mv {0}.prexg {0}_first.prexg".format(name),verbose=parameters["slurm_verbose"])

                # newerror = abs( np.loadtxt('{0}_cumulative.prexf'.format(name))[:,4:5] ).max()
                translations = np.loadtxt("{0}_cumulative.prexf".format(name))
                newerror = np.hypot(translations[:, -2], translations[:, -1]).max()
                if iteration > 0 and newerror >= error:
                    logger.info(
                        "Error did not decrease, this will be the last iteration"
                    )
                    error = 0
                else:
                    error = newerror

                iteration += 1
                logger.info(
                    "Max detected shift change at iteration {0} is {1}\t\t({2})".format(
                        iteration, newerror, ccc[1, :].mean()
                    )
                )

                # doing iterations does not help matters?
                # break

            shutil.move("{0}_first.prexg".format(name), "{0}.xf".format(name))

            # undo movie_binning
            if movie_binning > 1:
                t = np.loadtxt("%s.xf" % name, ndmin=2)
                t[:, -2:] *= movie_binning
                np.savetxt("%s.xf" % name, t, fmt="%13.7f")
                run_shell_command(
                    "{0}/bin/newstack -nearest -xform {1}.xf {1}.mrc {1}.ali".format(
                        get_imod_path(), name
                    )
                )

            # save .xf file without binning
            binning = int(parameters["data_bin"])
            if binning > 1:
                t = np.loadtxt("%s.xf" % name, ndmin=2)
                t[:, -2:] *= binning
                np.savetxt("%s.xf" % name, t, fmt="%13.7f")

        else:

            # single frame case
            f = open("{0}.xf".format(name), "w")
            for i in range(frames):
                f.write(
                    """   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n"""
                )
            f.close()
            shutil.copy2("{}.mrc".format(name), "{}.ali".format(name))

    if True:
        # average aligned tilt-series
        #         command="""
        # %s/bin/avgstack << EOF
        # %s.ali
        # %s.avg
        # /
        # EOF
        # """  % ( get_imod_path(), name, name )
        #         logger.info(command)
        #         [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        input_fname = f"{name}.ali"
        output_fname = f"{name}.avg"
        start_end_section = "/"

        output, error = avgstack(input_fname, output_fname, start_end_section)
        logger.info(output)

    pixel = float(parameters["data_bin"]) * float(parameters["scope_pixel"])

    # set correct pixel size in mrc header because ctffind3 uses this value
    command = """
%s/bin/alterheader << EOF
%s.avg
del
%s,%s,%s
done
EOF
""" % (
        get_imod_path(),
        name,
        pixel,
        pixel,
        pixel,
    )
    run_shell_command(command)

    # average aligned stack and save
    aligned_average = mrc.read("{}.avg".format(name))

    # save bin8 image png
    if aligned_average.shape[0] > 6096:
        binning = 4
    elif aligned_average.shape[0] < 2048:
        binning = 1
    else:
        binning = 2

    small = (
        aligned_average.reshape(
            aligned_average.shape[0] // binning,
            binning,
            aligned_average.shape[1] // binning,
            binning,
        )
        .mean(3)
        .mean(1)
    )
    writepng(small, "{}_small.png".format(name))
    # commands.getoutput('{0}/convert {1}_small.png -resize 50% -contrast-stretch 1%x98% {1}.jpg'.format(os.environ['IMAGICDIR'],name))
    contrast_stretch(name + "_small.png", name + ".jpg", 50)

    # bin image
    size = min(aligned_average.shape[0], aligned_average.shape[1]) * int(
        parameters["data_bin"]
    )
    binning = int(math.floor(size / 512))
    new_size = binning * 512 // int(parameters["data_bin"])
    view = aligned_average[0:new_size, 0:new_size]
    small = view.reshape(512, view.shape[0] // 512, 512, -1).mean(3).mean(1)
    writepng(small, "image.png")

    # commands.getoutput('{0}/convert image.png -contrast-stretch 1%x98% image.png'.format( os.environ['IMAGICDIR'] ) )
    contrast_stretch("image.png")

    # tmp_files = '{0}_first.prexf {0}_cumulative.prexf {0}_cumulative.prexg {0}.ali {0}.ali~'.format(name)
    # for file in tmp_files.split():
    #    os.remove(file)

    return aligned_average


def apply_alignments_and_average(input_name, name, parameters, method="imod"):

    aligned_fname = f"{name}.ali"
    output_fname = f"{name}.avg"

    t = np.loadtxt(name + ".xf", ndmin=2)

    first_frame = 1
    if int(parameters["movie_last"]) <=0:
        last_frame = t.shape[0]
    else:
        last_frame = int(parameters["movie_last"])
    
    threads = parameters["slurm_tasks"]
    env = "export OMP_NUM_THREADS={0}; export NCPUS={0}; IMOD_FORCE_OMP_THREADS={0}; ".format(threads)
    
    command = env + "{0}/bin/newstack -mode 2 {1} {2}.mrc && rm {1}~".format(
        get_imod_path(), input_name, name
    )
    run_shell_command(command)

    if method == "imod" and not parameters["movie_weights"]:

        command = env + "{0}/bin/newstack -nearest -xform {2}.xf {2}.mrc {2}.ali; rm -f {2}.ali~".format(
            get_imod_path(), input_name, name
        )
        run_shell_command(command)

        if parameters["data_bin"] > 1:
            command = env + "{0}/bin/newstack -ftreduce {3} {2}.ali {2}.ali; rm -f {2}.ali~".format(
                get_imod_path(), input_name, name, parameters["data_bin"]
            )
            run_shell_command(command)

        start_end_section = f"{first_frame-1},{last_frame-1}"

        output, error = avgstack(aligned_fname, output_fname, start_end_section)

        os.remove(aligned_fname)

    else:

        # convert movie to mrc if needed
        if Path(input_name).suffix != ".mrc":
            command = "{0}/bin/tif2mrc {1} {2}.mrc; rm -f {2}.mrc~".format(
                get_imod_path(), input_name, name
            )
            run_shell_command(command)
            input_name = name + ".mrc"

        actual_pixel = str(parameters["scope_pixel"])

        # convert .xf to shifts file (apparently summovie likes shifts in pixels)
        shifts = np.zeros([2, t.shape[0]])
        shifts[0, :] = t[:, -2]
        shifts[1, :] = t[:, -1]
        np.savetxt(name + "_shifts.txt", shifts, fmt="%f")

        if parameters["movie_weights"]:
            weighted = "YES\n%s\n%s\n0" % (
                str(parameters["scope_dose_rate"]),
                str(parameters["scope_voltage"]),
            )
        else:
            weighted = "NO"

        # summovie
        command = """
%s/%s << EOF
%s.tif
%d
%s_avg.mrc
%s_shifts.txt
%s_frc.txt
%d
%d
%s
%s
YES
EOF
""" % (
            get_summovie_path(),
            "/bin/sum_movie_openmp_7_17_15.exe",
            name,
            t.shape[0],
            name,
            name,
            name,
            first_frame,
            last_frame,
            actual_pixel,
            weighted,
        )

        """
        Input stack filename                [my_movie.mrc] :
        Number of frames per movie                     [1] :
        Output aligned sum file           [my_aligned.mrc] :
        Input shifts file                  [my_shifts.txt] :
        Output FRC file                       [my_frc.txt] :
        First frame to sum                             [1] :
        Last frame to sum                              [1] :
        Pixel size of images (A)                     [1.0] :
        Apply dose filter?                            [no] :
        Dose per frame (e/A^2)                       [1.0] :
        Acceleration voltage (kV)                  [300.0] :
        Pre-exposure Amount(e/A^2)                   [0.0] :
        Restore noise power after filtering?         [yes] :
        """

        run_shell_command(command)
        
        if parameters["data_bin"] > 1:
            bin_stack(name + "_avg.mrc", name + ".avg", parameters["data_bin"], "imod")
        else:
            shutil.move(name + "_avg.mrc", name + ".avg")

@Timer("align_stack_super", text="Alignment took: {}", logger=logger.info)
def align_stack_super(
    name,
    parameters,
    current_path,
    working_path,
    method="",
    interpolation="-linear",
    interval=0,
    apply=False,
    **kwargs,
):
    micrograph_name = name.split("_r01_")[0]
    micrograph_name = name.split("_P0")[0]
    threads = parameters["slurm_tasks"]

    if True:
        if Path(name + ".mrc").exists():
            input_name = str(Path(name + ".mrc"))
        elif Path(name + "_unbinned.mrc").exists():
            input_name = str(Path(name + "_unbinned.mrc"))
        elif Path(name + ".tif").exists():
            input_name = str(Path(name + ".tif"))
        else:
            error = f"{name}: file format not recognized"
            raise Exception(error)

        x, y, frames = get_image_dimensions(input_name)

        if frames > 1:

            # add an additional factor of binning if images are super-resolution
            if x > 6096 and not "unblur" in method:
                movie_binning_factor = 2
            else:
                movie_binning_factor = 1
            movie_binning = int(parameters["movie_bin"]) * movie_binning_factor

            stack_binning = float(parameters["data_bin"]) * movie_binning

            # use binned stack as seed (aligned if applicable)

            """
            if os.path.exists( name + '.ali' ):
                # make previous aligned stack the current starting stack
                os.rename( name + '.ali', name + '.bin' )
                # shutil.copy( name + '.ali', name + '.bin' )
            else:
            """

            if os.path.isfile("{}.xf".format(name)):

                # average aligned stack and save
                apply_alignments_and_average(input_name, name, parameters)

                aligned_average = mrc.read(name + ".avg")

                return aligned_average

                # apply transform first and then do binning
                if True:
                    if not apply:
                        com = "{0}/bin/newstack {1}_unbinned.mrc {1}_unbinned.bin -linear -xform {1}.xf".format(
                            get_imod_path(), name
                        )
                        # print com
                        subprocess.check_output(
                            com, stderr=subprocess.STDOUT, shell=True, text=True
                        )
                        # com = '{0}/bin/newstack {3} {1}.bin {1}.bin -bin {2} -mode 2'.format( get_imod_path(), name, int(stack_binning), interpolation )
                        # commands.getoutput( com )
                        bin_stack(
                            name + "_unbinned.bin", name + ".bin", stack_binning, "imod"
                        )
                        try:
                            os.remove(name + "_unbinned.bin")
                        except:
                            pass
                    else:
                        # apply residual shifts after discretization
                        xf = np.loadtxt("{}.xf".format(name))
                        # XD: seems like there's rounding of the shifts when frame refinement extract
                        xf[:, -2:] -= np.round(xf[:, -2:])
                        np.savetxt("{}_stack.xf".format(name), xf)
                        com = "{0}/bin/newstack {1}_unbinned.mrc {1}_unbinned.bin -linear -mode 2 -xform {1}_stack.xf".format(
                            get_imod_path(), name, stack_binning
                        )
                        subprocess.check_output(
                            com, stderr=subprocess.STDOUT, shell=True, text=True
                        )
                        bin_stack(
                            name + "_unbinned.bin", name + ".bin", stack_binning, "imod"
                        )
                        os.remove(name + "_unbinned.bin")
                elif not apply:
                    # elif True:
                    xf = np.loadtxt("{}.xf".format(name))
                    xf[:, -2:] /= stack_binning
                    np.savetxt("{}_stack.xf".format(name), xf)
                    com = "{0}/bin/newstack {3} {1}.mrc {1}.bin -bin {2} -mode 2 -xform {1}_stack.xf".format(
                        get_imod_path(), name, movie_binning, interpolation
                    )
                    run_shell_command(com)
                    # bin_stack( name + '.mrc', name + '.bin', stack_binning, 'imod' )
                else:
                    com = "{0}/bin/newstack {3} {1}.mrc {1}.bin -bin {2} -mode 2".format(
                        get_imod_path(), name, movie_binning, interpolation
                    )
                    run_shell_command(com)
                    # bin_stack( name + '.mrc', name + '.bin', stack_binning, 'imod' )

            else:
                if movie_binning > 1:
                    # com = '{0}/bin/newstack {1}.mrc {1}.bin -bin {2} -mode 2'.format( get_imod_path(), name, movie_binning )
                    # print com
                    # commands.getoutput( com )
                    if os.path.exists(name + "_unbinned.mrc"):
                        bin_stack(
                            name + "_unbinned.mrc",
                            name + ".bin",
                            stack_binning * movie_binning,
                            "imod",
                            threads=threads,
                        )
                    else:
                        bin_stack(input_name, name + ".bin", movie_binning, "imod", threads=threads)
                elif stack_binning > 1:
                    if os.path.exists(name + "_unbinned.mrc"):
                        bin_stack(
                            name + "_unbinned.mrc", name + ".bin", stack_binning, "imod", threads=threads,
                        )
                        try:
                            os.symlink(name + "_unbinned.mrc", name + ".bin")
                        except:
                            pass
                    else:
                        # bin_stack( name + '.mrc', name + '.bin', stack_binning, 'imod' )
                        try:
                            os.symlink(input_name, name + ".bin")
                        except:
                            pass
                else:
                    try:
                        os.symlink(input_name, name + ".bin")
                    except:
                        pass

            middle_frame = frames / 2

            # collapse frames into (interval) groups

            intervals = frames
            if interval > 1:
                run_shell_command("rm {0}_????.avg".format(name))
                intervals = 0
                # for n in range(interval,frames-interval):
                for n in range(0, frames, interval):
                    #                     command="""
                    # %s/bin/avgstack << EOF
                    # %s.bin
                    # %s_%06d.avg
                    # %d,%d
                    # EOF
                    # """  % ( get_imod_path(), name, name, n, n, n + interval - 1 )
                    # # """  % ( get_imod_path(), name, name, n, n - interval, n + interval )
                    #                     [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                    input_fname = f"{name}.bin"
                    output_fname = f"{name}_{n:04d}.avg"
                    start_end_section = f"{n},{n}"

                    output, error = avgstack(
                        input_fname, output_fname, start_end_section
                    )
                    intervals += 1

                # compose decimated movie
                run_shell_command(
                    "{0}/bin/newstack {1}_????.avg {1}.bin".format(
                        get_imod_path(), name
                    )
                )

            # set pixel size depending whether we are using particles or micrographs
            actual_pixel = (
                float(parameters["scope_pixel"])
                * float(parameters["data_bin"])
                * movie_binning
            )

            # skip alignment operation
            if "skip" in method:

                # write identity matrix for null shifts
                f = open("{0}.xf".format(name), "w")
                for i in range(frames):
                    f.write(
                        """   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n"""
                    )
                f.close()
                shutil.copy2("{}.bin".format(name), "{}.ali".format(name))

            elif "unblur" in method:

                if "movie_weights" in parameters.keys() and parameters["movie_weights"]:
                    weighted = "YES\n%s\n%s\n0" % (
                        parameters["scope_dose_rate"],
                        parameters["scope_voltage"],
                    )
                else:
                    weighted = "NO"

                if not os.path.exists(name + "_bin.mrc"):
                    os.symlink(name + ".bin", name + "_bin.mrc")

                x_bin, y_bin, frames_bin = get_image_dimensions(name + "_bin.mrc")

                # skip alignment if fewer than 4 frames (unblur requires 4 frames at least)
                if frames_bin < 4:
                    #                         command = """
                    # %s/bin/avgstack << EOF
                    # %s_bin.mrc
                    # %s.avg
                    # /
                    # EOF
                    # """  % (get_imod_path(), name, name)
                    #                         # print command
                    #                         [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

                    input_fname = f"{name}_bin.mrc"
                    output_fname = f"{name}.avg"
                    start_end_section = "/"

                    output, error = avgstack(
                        input_fname, output_fname, start_end_section
                    )

                    # identity transform
                    rot2D = np.zeros([frames_bin, 6])
                    for i in range(rot2D.shape[0]):
                        rot2D[i, :] = np.array([1, 0, 0, 1, 0, 0])
                    np.savetxt("{0}.xf".format(name), rot2D, fmt="%13.7f")

                else:

                    # actual_pixel = float(parameters['scope_pixel']) * float(parameters['data_bin']) * movie_binning

                    # if 'local' in parameters['extract_fmt']:
                    #    actual_pixel *= float( parameters['extract_bin'] )

                    # if "nospline" in method:
                    #     executable = "unblur_1.0/build/unblur_no_smoothing"
                    # else:
                    #     executable = "unblur_1.0/build/unblur_spline"

                    executable = "bin/unblur_openmp_7_17_15.exe"

                    if "bf" in method:
                        bfactor = method.split("bf")[-1]
                    else:
                        bfactor = "800"

                    tmp_directory = name
                    os.makedirs(tmp_directory)
                    os.chdir(tmp_directory)

                    # run unblur
                    command = """
%s/%s << EOF
../%s_bin.mrc
%d
../%s_avg.mrc
../%s_shifts.txt
%f
%s
NO
YES
../%s_frc.txt
0
200.0
%s
1
1
0.1
10
no
no
EOF
""" % (
                        get_unblur_path(),
                        executable,
                        name,
                        intervals,
                        name,
                        name,
                        actual_pixel,
                        weighted,
                        name,
                        bfactor,
                    )
                    # Input stack filename                [my_movie.mrc]
                    # Number of frames per movie                    [38]
                    # Output aligned sum file       [my_aligned_sum.mrc]
                    # Output shifts file                 [my_shifts.txt]
                    # Pixel size of images (A)                       [1]
                    # Apply Dose filter?                            [NO]
                    # Exposure per frame (e/A^2)                   [1.0]
                    # Acceleration voltage (kV)                  [300.0]
                    # Pre-exposure amount(e/A^2)                   [0.0]
                    # Save Aligned Frames?                          [NO]
                    # Set Expert Options?                           [NO]
                    # Output FRC file                       [my_frc.txt]
                    # Minimum shift for initial search (Angstroms) [2.0]
                    # Outer radius shift limit (Angstroms)       [200.0]
                    # B-factor to apply to images (A^2)           [1500]
                    # Half-width of central vertical line of Fourier mask [1]
                    # Half-width of central horizontal line of Fourier mask [1]
                    # Termination shift threshold                  [0.1]
                    # Maximum number of iterations                  [10]
                    # Restore Noise Power?                         [YES]
                    # Verbose Output?                               [NO]

                    # make sure unblur runs in parallel if we are in spr mode
                    if parameters["data_mode"] == "tomo":
                        cpus = 1
                    else:
                        cpus = parameters["slurm_tasks"]
                    command = (
                        "export OMP_NUM_THREADS={0}; export NCPUS={0}; ".format(cpus)
                        + command
                    )
                    with Timer(
                        "unblur_com", text="unblur took: {}", logger=logger.info
                        ):
                        [output, error] = run_shell_command(command)

                    # go back to parent directory and cleanup
                    os.chdir("..")
                    shutil.rmtree(tmp_directory)

                    # parse unblur's output
                    average_x_shift = float(
                        [
                            line
                            for line in output.split("\n")
                            if "Average X Shift" in line
                        ][0].split()[-2]
                    )
                    average_y_shift = float(
                        [
                            line
                            for line in output.split("\n")
                            if "Average Y Shift" in line
                        ][0].split()[-2]
                    )
                    score = float(
                        [line for line in output.split("\n") if "Final Score" in line][
                            0
                        ].split()[-1]
                    )

                    # save metadata
                    np.savetxt(
                        name + ".blr",
                        np.array([average_x_shift, average_y_shift, score]),
                        fmt="%13.7f",
                    )
                    # convert shifts to .xf file
                    shifts = np.loadtxt(name + "_shifts.txt", comments="#")
                    xfshifts = np.zeros((shifts.shape[1], 6))
                    xfshifts[:, 0] = 1
                    xfshifts[:, 3] = 1
                    xfshifts[:, 4] = shifts[0, :] / actual_pixel
                    xfshifts[:, 5] = shifts[1, :] / actual_pixel
                    np.savetxt(name + ".xf", xfshifts, fmt="%13.7f")

                    frc = np.loadtxt(name + "_frc.txt", comments="#")
                    import matplotlib.pyplot as plt

                    plt.clf()
                    plt.plot(frc[0, :], frc[1, :], label="Score = %f" % score)
                    plt.savefig(name + "_frc.png")
                    plt.close()

                    # maximum displacement
                    error = np.hypot(xfshifts[:, -2], xfshifts[:, -1]).max()

            elif "relion" in method:

                # construct star file with single micrograph

                starfile = name + ".star"

                with open(starfile, "w") as f:
                    f.write("data_\n")
                    f.write("loop_\n")
                    f.write("_rlnMicrographMovieName\n")
                    f.write(name + ".mrc\n")

                if float(parameters["movie_first"]) == 0:
                    movie_first = 1
                if float(parameters["movie_last"]) == -1:
                    movie_last = 0

                com = "/dscrhome/ab690/code/relion-3.0_beta/build/bin/relion_run_motioncorr --i {0}.star --o MotionCorr --first_frame_sum {1} --last_frame_sum {2} --use_own --j {3} --bin_factor 1 --bfactor {4} --angpix {5} --voltage {6} --dose_per_frame {7} --preexposure 0 --patch_x {8} --patch_y {8} --gain_rot 0 --gain_flip 0 --save_noDW".format(
                    name,
                    movie_first,
                    movie_last,
                    parameters["slurm_tasks"],
                    parameters["movie_bfactor"],
                    parameters["scope_pixel"],
                    parameters["scope_voltage"],
                    parameters["scope_dose_rate"],
                    parameters["movie_patches"],
                )

                if (
                    "movie_weights" in parameters
                    and "t" in parameters["movie_weights"].lower()
                ):
                    com += " --dose_weighting"

                run_shell_command(com)

                # os.remove( starfile )

                relion_shifts_file = "MotionCorr/" + name + ".star"

                relion_shifts = (
                    np.array(
                        open(relion_shifts_file).read().split("loop_")[1].split()[6:-1]
                    )
                    .reshape([frames, 3])
                    .astype("f")[:, 1:]
                )

                if (
                    "movie_weights" in parameters
                    and "t" in parameters["movie_weights"].lower()
                ):
                    shutil.copy2("MotionCorr/" + name + ".mrc", name + "_DW.mrc")
                    shutil.copy2("MotionCorr/" + name + "_noDW.mrc", name + ".avg")
                else:
                    shutil.copy2("MotionCorr/" + name + ".mrc", name + ".avg")

                # convert shifts to .xf file
                xfshifts = np.zeros((relion_shifts.shape[0], 6))
                xfshifts[:, 0] = 1
                xfshifts[:, 3] = 1
                xfshifts[:, 4] = relion_shifts[:, 0]
                xfshifts[:, 5] = relion_shifts[:, 1]
                np.savetxt(name + ".xf", xfshifts, fmt="%13.7f")

                # convert to global shifts with respect to middle frame
                # commands.getoutput('{0}/bin/xftoxg -nfit 0 -ref {1} -input {2}.xg -goutput {2}.xf'.format(get_imod_path(),middle_frame,name))

                # maximum displacement
                error = np.hypot(xfshifts[:, -2], xfshifts[:, -1]).max()

            # alignment to average using tiltxcorr
            elif "tiltxcorr" in method:

                # tiltxcorr implementation of frame alignment to average
                # differentiate between whole frame alignment and local alignment
                if x > 2048:
                    tiltxcorr_options = "-first 0 -increment 1 -nostretch -binning 1 -shift 10,10 -rotation 0.000000 -radius1 0.010000 -sigma1 0.030000 -radius2 0.100000 -sigma2 0.030000 -border 64,64 -taper 256,256 -iterate 5"
                else:
                    tiltxcorr_options = "-first 0 -increment 1 -nostretch -binning 1 -shift 3,3 -rotation 0.000000 -radius1 0.00000 -sigma1 0.00000 -radius2 0.1000 -sigma2 0.0250000 -border 32,32 -taper 64,64"

                error = 1
                iteration = 0
                scores = []
                while error > 1e-4 and iteration < int(parameters["movie_iters"]):

                    if iteration > 0 or os.path.isfile("{}.xf".format(name)):
                        input = "ali"
                        if os.path.isfile("{}.xf".format(name)):
                            t = np.loadtxt("%s.xf" % name, ndmin=2)
                            t[:, -2:] /= movie_binning
                            np.savetxt("%s.xf" % name, t, fmt="%13.7f")
                            com = "{0}/bin/newstack {2} -xform {1}.xf -mode 2 -multadd 1,0 {1}.bin {1}.ali".format(
                                get_imod_path(), name, interpolation
                            )
                            run_shell_command(com)
                            shutil.move(
                                "{0}.xf".format(name), "{0}_first.prexg".format(name)
                            )
                    else:
                        input = "bin"

                    # current aligned average
                    #                     command="""
                    # %s/bin/avgstack << EOF
                    # %s.%s
                    # %s.avg
                    # /
                    # EOF
                    # """  % ( get_imod_path(), name, input, name )
                    #                     # print command
                    #                     [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                    input_fname = f"{name}.{input}"
                    output_fname = f"{name}.avg"
                    start_end_section = "/"

                    output, error = avgstack(
                        input_fname, output_fname, start_end_section
                    )

                    # cleanup
                    com = "rm {0}_????_cumulative.prexg".format(name)
                    run_shell_command(com)

                    manager = multiprocessing.Manager()
                    results = manager.Queue()

                    # use 4 cores max
                    maxcount = 4
                    if int(parameters["slurm_tasks"]) > 0:
                        maxcount = min(maxcount, int(parameters["slurm_tasks"]))

                    arguments = []

                    for frame in range(intervals):
                        arguments.append(
                            (
                                name,
                                frame,
                                input,
                                tiltxcorr_options,
                                1.0 * intervals,
                                results,
                            )
                        )

                    mpi.submit_function_to_workers(align_frame_to_reference, arguments, verbose=parameters["slurm_verbose"])

                    ccc = np.zeros([intervals, 1])

                    while results.empty() == False:
                        t = results.get()
                        ccc[t[0]] = t[1]
                    np.savetxt("{0}.ccc".format(name), ccc)

                    # collate all shifts
                    com = "cat {0}_????_cumulative.prexg > {0}_cumulative.prexg".format(
                        name
                    )
                    run_shell_command(com)

                    if not os.path.exists(name + "_first.prexg"):
                        shutil.copy2(
                            "{0}_cumulative.prexg".format(name),
                            "{0}.prexg".format(name),
                        )
                    else:
                        # concatenate with latest transform
                        com = "{0}/bin/xfproduct {1}_first.prexg {1}_cumulative.prexg {1}.prexg".format(
                            get_imod_path(), name
                        )
                        run_shell_command(com,verbose = False)

                    run_shell_command("cp {0}.prexg {0}.prexgraw".format(name))

                    # fit spline to trajectory
                    if "spline" in method:
                        shifts = np.loadtxt(name + ".prexg")
                        shifts = fit.fit_spline_trajectory(shifts)
                        np.savetxt(name + ".prexg", shifts)

                    # generate aligned stack with latest alignment parameters
                    com = "{0}/bin/newstack {2} -xform {1}.prexg -mode 2 -multadd 1,0 {1}.bin {1}.ali".format(
                        get_imod_path(), name, interpolation
                    )
                    run_shell_command(com,verbose=False)

                    # update current transform
                    run_shell_command("mv {0}.prexg {0}_first.prexg".format(name), verbose=parameters["slurm_verbose"])

                    # newerror = abs( np.loadtxt('{0}_cumulative.prexg'.format(name))[:,4:5] ).max()
                    translations = np.loadtxt("{0}_cumulative.prexg".format(name))
                    newerror = np.hypot(translations[:, -2], translations[:, -1]).max()
                    if iteration > 0 and newerror >= error:
                        logger.info(
                            "Error did not decrease, this will be the last iteration"
                        )
                        error = 0
                    else:
                        error = newerror

                    iteration += 1
                    scores.append(ccc.mean())
                    logger.info(
                        "Max detected shift change at iteration {0} is {1}\t\t({2})".format(
                            iteration, newerror, ccc.mean()
                        )
                    )

                np.savetxt("{0}.ddd".format(name), np.array(scores))

                shutil.move("{0}_first.prexg".format(name), "{0}.xf".format(name))

            # if os.path.exists( name + '.xf' ):
            #    plot.plot_trajectory( name )

            if "frealign" in method:

                # print 'FREALIGN based particle frame alignment'

                particle_index = int(name[-11:-7])

                # PARAMETERS
                frame_weights_width = 15.0  # width of gaussian used for frame weighting
                frame_weights_width = int(
                    math.floor(frames * 0.4)
                )  # width of gaussian used for frame weighting
                # NTSR1
                # frame_weights_width = int( math.floor( frames * 2 ) )  # width of gaussian used for frame weighting
                if frame_weights_width % 2 == 0:
                    frame_weights_width += 1
                frame_weights_step = False  # use step-like weights for frame weighting
                frealign_iters = 2  # number of frealign iterations
                low_res = 75  # lowest resolution used for refinement
                high_res_refi = 3  # maximum resolution to use for FREALIGN refinements
                high_res_eval = (
                    3  # maximum resolution to use for FREALIGN score evaluations
                )
                rrec = 2  # resolution of reconstruction (not actually used)
                spread = (
                    1 * actual_pixel
                )  # radius of xy-shift distribution to be used as priors (in pixels)
                repeats = 1  # number of alignment trials to try for each frame (3)
                iterations = 1  # number of times to iterate alignment (12)
                metric = "cc3m -fboost T"  # metric to use for alignment
                metric_weights = "cc3m -fboost T"  # metric to use for alignment
                intervals = 4  # bands to evaluate frame weights
                cores_for_frealign = 1  # cores for FREALIGN processing

                # load most recent FREALIGN parameters to ensure consistency
                if True:
                    # fparameters = project_params.load_fyp_parameters(
                    #    os.path.split(parameters["refine_parfile"])[0] + "/../"
                    # )
                    maxiter = parameters["refine_maxiter"]
                    low_res = float(
                        project_params.param(parameters["refine_rlref"], maxiter)
                    )
                    high_res_refi = high_res_eval = float(
                        project_params.param(parameters["refine_rhref"], maxiter)
                    )
                    # logger("high_res_refine", float( project_params.param( fparameters['rhref'], maxiter ) ))
                    metric = (
                        str(project_params.param(parameters["refine_metric"], maxiter))
                        + " -fboost "
                        + str(
                            project_params.param(parameters["refine_fboost"], maxiter)
                        )
                    )
                    # metric_weights = metric
                    # print 'Retrieving FREALIGN compatible FREALIGN parameters: rhlref = %.2f, rhref = %.2f, metric = %s' % ( low_res, high_res_refi, metric )
                else:
                    logger.warning(
                        "Could not find FREALIGN parameters to insure consistency"
                    )

                frealign_path = "frealign_" + name
                try:
                    os.mkdir(frealign_path)
                except:
                    pass

                try:
                    os.mkdir(frealign_path + "/maps")
                except:
                    pass

                # find film number for this micrograph to figure out particle alignments
                try:
                    with open(
                        os.path.join(current_path, parameters["data_set"] + ".films")
                    ) as x:
                        series = [
                            num
                            for num, line in enumerate(x, 1)
                            if "{}".format(micrograph_name) == line.strip()
                        ][0] - 1
                except:
                    raise Exception(
                        "ERROR - Cannot find film number for " + micrograph_name
                    )

                refine_par_file = project_params.resolve_path(
                    parameters["refine_parfile"]
                )
                refine_model_file = project_params.resolve_path(
                    parameters["refine_model"]
                )
                if not os.path.exists(refine_par_file):
                    raise Exception(
                        "ERROR - Cannot find parameter file " + refine_par_files
                    )

                # figure out FREALIGN refinement parameters for this particle
                ref = [
                    line
                    for line in open(refine_par_file)
                    if not line.startswith("C")
                    and line.split()[7] == "{}".format(series)
                ][particle_index]

                particles_in_micrograph = len(
                    [
                        line
                        for line in open(refine_par_file)
                        if not line.startswith("C")
                        and line.split()[7] == "{}".format(series)
                    ]
                )

                particle_par = [float(item) for item in ref.split()]
                (
                    number,
                    psi,
                    theta,
                    phi,
                    sx,
                    sy,
                    mag,
                    mfilm,
                    df1,
                    df2,
                    angast,
                    occ,
                    logp,
                    sigma,
                    mscore,
                    change,
                ) = particle_par[:16]

                initial_shifts = np.array([sx, sy])

                import re

                micrograph_name = re.sub(r"_P\d+_frames", "", name)

                logger.info("## Extracting parameters in frealign format##")
                mod_parameters = parameters.copy()
                mod_parameters["extract_fmt"] = "frealign"
                os.chdir(current_path)
                [allboxes, allparxs] = csp_spr_swarm(
                    micrograph_name.split("_r01")[0], mod_parameters
                )

                os.chdir(working_path)

                # using parx
                # NEED TO CHANGE to non-hardcoded variables: idx 6 because we don't have the row index at idx 0; idx 15 for particle index; idx 18 for scanord
                particle_frames_parxs = np.array(
                    [
                        item.split()
                        for item in [
                            line
                            for line in allparxs
                            if line.split()[6] == "{}".format(series)
                            and line.split()[15] == "{}".format(particle_index)
                        ]
                    ],
                    dtype="float",
                )
                # output_fname = 'particle_frames.parx'
                # particle_parxs =  [ float(item) for item in [ line for line in allparxs if line.split()[6] == '{}'.format(series) and line.split()[15] == '{}'.format(particle_index)][0].split() ]

                # assert particle_par[1:] == particle_parxs[:len(particle_par)-1], "ERROR: par file and parxs parameters from csp_spr_swarm do not match!"

                # up to `change`, use the updated values from particle

                for rows in range(particle_frames_parxs.shape[0]):
                    particle_frames_parxs[rows, :15] = particle_par[1:16]
                # psi, theta, phi, sx, sy, mag, mfilm, df1, df2, angast, occ, logp, sigma, mscore, change, local_particle, tilt, dose, scan_order, confidence, ptl_CCX, axis, norm0, norm1, norm2, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, ppsi, ptheta, pphi = particle_parxs

                # assert local_particle == particle_index, "ERROR: particle in parx file does not match particle_index"

                # create FREALIGN parameter file for refinement
                frealign_parameter_file = frealign_path + "/" + name + "_01.par"
                parfile = "%s/maps/%s_r01_%02d.par" % (
                    frealign_path,
                    name,
                    frealign_iters,
                )
                local_model = os.path.join( os.getcwd(), frealign_path, "%s_01.mrc" % name)
                if not os.path.exists(local_model):
                    symlink_relative(
                        os.path.join(os.getcwd(), os.path.split(refine_model_file)[-1]),
                        local_model,
                    )
                """
                model_copy = os.getcwd() + '/%s_01.mrc' % ( name )
                #  provide a model for later scoring
                if not os.path.exists( model_copy ):
                    shutil.copy2(local_model, model_copy)
                """

                # print 'Using reference', parameters['class_ref']

                # assume we are always using scores from FREALIGN_v9
                field = 14

                # use cropped version of frames (patch based uses larger box size!)
                # shutil.copy( name + '_cropped.bin', name + '.bin' )

                # start first iteration from global alignments
                # shutil.copy( name + '.bin', name + '.ali' )

                if os.path.exists(name + ".ali"):
                    os.remove(name + ".ali")
                os.symlink(name + ".bin", name + ".ali")

                boxsize = int(mrc.readHeaderFromFile(name + ".ali")["nx"])

                # build weights for frame averaging
                all_weights = np.zeros([frames, frames])
                for i in range(frames):
                    weights = np.exp(
                        -pow((np.arange(frames) - float(i)), 2) / frame_weights_width
                    )
                    # apply hard threshold if using simple running averages
                    if frame_weights_step:
                        weights = np.where(weights > 0.5, 1, 0)
                    all_weights[i, :] = weights / weights.mean() / frames

                # save weights of central frame for display purposes
                center_frame_weights = all_weights[frames // 2, :]

                binFactor = actual_pixel / float(parameters["scope_pixel"])

                # retrieve shifts from previous iteration
                if os.path.exists(name + ".xf"):
                    # read transform and convert to A (from super-resolution pixels)
                    shifts = np.loadtxt(name + ".xf")
                    shifts[:, -2:] *= float(parameters["scope_pixel"])
                    # print name, 'LOADED SHIFTS IN A', shifts[ 0, -2: ]
                else:
                    # use identity transformation
                    shifts = np.zeros([frames, 6])
                    shifts[:, 0] = shifts[:, 3] = 1
                previous_shifts = np.copy(shifts)
                starting_previous_shifts = np.copy(shifts)

                # if '_P0000_' in name:
                #    print name, ', loading shifts in A', shifts[:1,-2:]

                # calculate local shifts using FREALIGN engine
                if not apply:

                    # iterative refinement
                    for iter in range(iterations):

                        # print 'previous', previous_shifts[0,-2:]

                        if repeats > 1:

                            """
                            # create weighted stack
                            merge.weight_stack( name + '.ali', name + '.mov', all_weights )
                            # take multiple repeats
                            mrc.merge( [ name + '.mov' ] * repeats, frealign_path + '/' + name + '_stack.mrc' )
                            """

                            merge.weight_stack(
                                name + ".ali",
                                frealign_path + "/" + name + "_stack.mrc",
                                all_weights,
                            )
                            # take multiple repeats
                            mrc.repeatFile(
                                frealign_path + "/" + name + "_stack.mrc", repeats
                            )

                        else:
                            # create weighted stack
                            merge.weight_stack(
                                name + ".ali",
                                frealign_path + "/" + name + "_stack.mrc",
                                all_weights,
                            )

                        # prior distribution of shifts in Angstroms

                        # for faster version only
                        frealign_parameter_file = (
                            frealign_path + "/maps/" + name + "_r01_02.par"
                        )

                        with open(frealign_parameter_file, "w") as f:
                            # write out header (including resolution table)
                            [
                                f.write(line)
                                for line in open(refine_par_file)
                                if line.startswith("C")
                            ]
                            for i in range(frames * repeats):
                                # use current position as one of the candidates
                                if i < frames:
                                    dx = sx
                                    dy = sy
                                # use uniform distribution of shifts of given spread
                                else:
                                    dx = sx + random.uniform(-spread, spread)
                                    dy = sy + random.uniform(-spread, spread)
                                # logp = 0; sigma = .5
                                f.write(
                                    frealign_parfile.NEW_PAR_STRING_TEMPLATE
                                    % (
                                        i + 1,
                                        psi,
                                        theta,
                                        phi,
                                        dx,
                                        dy,
                                        float(mag),
                                        0,
                                        df1,
                                        df2,
                                        angast,
                                        100.0,
                                        logp,
                                        sigma,
                                        0,
                                        0,
                                    )
                                )
                                f.write("\n")

                        frealign_parx_file = (
                            frealign_path + "/maps/" + name + "_r01_02.parx"
                        )

                        with open(frealign_parx_file, "w") as f:
                            # write out header (including resolution table)
                            [
                                f.write(line)
                                for line in open(refine_par_file)
                                if line.startswith("C")
                            ]
                            for repeat in range(repeats):
                                for i in range(frames):
                                    # use current position as one of the candidates
                                    if (1 + repeat) * i < frames:
                                        dx = sx
                                        dy = sy
                                    # use uniform distribution of shifts of given spread
                                    else:
                                        dx = sx + random.uniform(-spread, spread)
                                        dy = sy + random.uniform(-spread, spread)
                                    # XD: change scanord to i
                                    # logp = 0; sigma = .5
                                    (
                                        psi,
                                        theta,
                                        phi,
                                        sx,
                                        sy,
                                        mag,
                                        mfilm,
                                        df1,
                                        df2,
                                        angast,
                                        occ,
                                        logp,
                                        sigma,
                                        mscore,
                                        change,
                                        local_particle,
                                        tilt,
                                        dose,
                                        scan_order,
                                        confidence,
                                        ptl_CCX,
                                        axis,
                                        norm0,
                                        norm1,
                                        norm2,
                                        a00,
                                        a01,
                                        a02,
                                        a03,
                                        a04,
                                        a05,
                                        a06,
                                        a07,
                                        a08,
                                        a09,
                                        a10,
                                        a11,
                                        a12,
                                        a13,
                                        a14,
                                        a15,
                                        ppsi,
                                        ptheta,
                                        pphi,
                                    ) = particle_frames_parxs[i]
                                    f.write(
                                        frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE
                                        % (
                                            i + 1,
                                            psi,
                                            theta,
                                            phi,
                                            dx,
                                            dy,
                                            float(mag),
                                            0,
                                            df1,
                                            df2,
                                            angast,
                                            100.0,
                                            logp,
                                            sigma,
                                            0,
                                            0,
                                            local_particle,
                                            tilt,
                                            dose,
                                            scan_order,
                                            confidence,
                                            ptl_CCX,
                                            axis,
                                            norm0,
                                            norm1,
                                            norm2,
                                            a00,
                                            a01,
                                            a02,
                                            a03,
                                            a04,
                                            a05,
                                            a06,
                                            a07,
                                            a08,
                                            a09,
                                            a10,
                                            a11,
                                            a12,
                                            a13,
                                            a14,
                                            a15,
                                            ppsi,
                                            ptheta,
                                            pphi,
                                        )
                                    )
                                    f.write("\n")

                        # assert that the original par file is a subset of the parx file
                        frealign_parx = frealign_parfile.Parameters.from_file(
                            frealign_parx_file
                        ).data
                        frealign_par = frealign_parfile.Parameters.from_file(
                            frealign_parameter_file
                        ).data
                        assert np.allclose(
                            frealign_parx[:, :16], frealign_par
                        ), "original par file is not a subset of the parx file"

                        # copy expanded parx file to main movie directory
                        # shutil.copy2( frealign_parameter_file, name + '_r01_02.par')
                        shutil.copy2(frealign_parx_file, name + "_r01_02.par")

                        if True:

                            # faster version

                            # save statistics if using v9.11
                            # XD testing: don't do first
                            # if 'new' in metric:
                            #     com = """grep -A 10000 "C  NO.  RESOL  RING RAD" {0}""".format( frealign_parameter_file ) + """ | grep -v RESOL | grep -v Average | grep -v Date | grep C | awk '{if ($2 != "") printf "%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f\\n", $2, $3, $4, $6, $7, $8, $9}' > """ + frealign_path + "/maps/statistics_r01.txt"
                            #     commands.getoutput(com)

                            # call FREALIGN directly to improve performance
                            mp = parameters.copy()
                            mp["data_bin"] = 1
                            # fp = fparameters.copy()
                            mp["refine_mode"] = "1"
                            if "fboost" in metric.lower():
                                mp["refine_fboost"] = "T"
                            else:
                                mp["refine_fboost"] = "F"
                            mp["refine_mask"] = "0,0,0,1,1"
                            mp["refine_rlref"] = "{}".format(low_res)
                            mp["refine_rhref"] = "{}".format(high_res_refi)
                            mp["refine_dataset"] = name

                            local_model = os.path.join( os.getcwd(), frealign_path, "maps", "%s_r01_01.mrc" % name)
                            if not os.path.exists(local_model):
                                symlink_relative(
                                    os.path.join(os.getcwd(), os.path.split(refine_model_file)[-1]),
                                    local_model
                                )
                            # shutil.copy( frealign_parameter_file, frealign_path + '/maps/' + name + '_r01_02.par' )
                            import re

                            model_copy = os.getcwd() + "/%s_r01_01.mrc" % (
                                re.sub(r"P\d+_", "", name)
                            )
                            #  provide a model for later scoring
                            if not os.path.exists(model_copy):
                                shutil.copy2(local_model, model_copy)

                            os.chdir(frealign_path + "/maps")

                            # def mrefine_version( mp, fp, first, last, i, ref, name, ranger, logfile, scratch, metric ):
                            # logger.info("printing mp")
                            # logger.info(mp)
                            # logger.info("printing fp")
                            # logger.info(fp)
                            # logger.info("frames, repeats")
                            # logger.info(frames, repeats)
                            # logger.info("metric")
                            # logger.info(metric)

                            command = frealign.mrefine_version(
                                parameters=mp,
                                first=1,
                                last=frames * repeats,
                                iter=2,
                                ref=1,
                                name=name + "_r01_02",
                                ranger="",
                                logfile="log.txt",
                                scratch="",
                                metric=metric,
                            )
                            run_shell_command(command)
                            os.chdir("../..")

                            # parse parameter file
                            input = np.array(
                                [
                                    line.split()
                                    for line in open(parfile + "_")
                                    if not line.startswith("C")
                                ],
                                dtype=float,
                            )

                        # convert to standarized image shifts
                        current_shifts = np.zeros([frames, 6])
                        current_shifts[:, 0] = current_shifts[:, 3] = 1
                        scores = np.zeros([frames])
                        for i in range(frames):
                            # collate repeats and find best candidate for each frame according to score
                            candidates = input[i::frames, :]
                            best = candidates[
                                candidates[:, field] == candidates[:, field].max()
                            ]
                            current_shifts[i, -2:] = best[0, 4:6] - initial_shifts
                            scores[i] = best[0, field]

                        # subtract common offset
                        # current_shifts[:,-2:] -= initial_shifts

                        # keep track of cummulative shifts across iterations
                        if iter > 0:
                            shifts[:, -2:] = (
                                previous_shifts[:, -2:] - current_shifts[:, -2:]
                            )
                            ## check if average score has increased
                            # if scores.mean() > previous_scores.mean():
                            #    # convert incremental shifts to global translations
                            #    shifts[:,-2:] = previous_shifts[:,-2:] + current_shifts[:,-2:]
                            # else:
                            #    print 'FREALIGN scores did not decrease in this iteration (%8.3f < %8.3f). Stopping.' % ( scores.mean(), previous_scores.mean() )
                            #    break
                        else:
                            current_shifts[:, -2:] *= -1
                            shifts = np.copy(current_shifts)

                        previous_scores = np.copy(scores)
                        previous_shifts = np.copy(shifts)

                        # np.savetxt( '{0}.ccc'.format(name), scores )

                        # global_shifts = np.loadtxt( micrograph_name + '.xf' )

                        # convert FREALIGN shifts in A to stack size pixels
                        binned_shifts = np.copy(shifts)
                        binned_shifts[:, -2:] += starting_previous_shifts[:, -2:]
                        # if '_P0000_' in name:
                        #    print name, ', saving shifts in binned A', binned_shifts[:1,-2:]
                        binned_shifts[:, -2:] /= actual_pixel
                        np.savetxt(name + ".xf", binned_shifts, fmt="%13.7f")

                        # if '_P0000_' in name:
                        #    print name, ', saving shifts in binned pixels', binned_shifts[:1,-2:]
                        """
                        # convert FREALIGN shifts in A to stack size pixels
                        unbinned_shifts = np.copy( shifts )
                        unbinned_shifts[:,-2:] += starting_previous_shifts[:,-2:]
                        unbinned_shifts[:,-2:] /= float( parameters['scope_pixel'] )
                        np.savetxt( name + '_unbinned.xf', unbinned_shifts, fmt='%13.7f' )
                        """

                        # apply local alignments

                        # apply shifts to unbinned stack (new)
                        if True:
                            if not apply:
                                unbinned_shifts = np.copy(binned_shifts)
                                unbinned_shifts[:, -2:] *= stack_binning

                                np.savetxt(
                                    name + "_unbinned.xf", unbinned_shifts, fmt="%13.7f"
                                )
                                com = "{0}/bin/newstack {1}_unbinned.mrc {1}_unbinned.ali -linear -xform {1}_unbinned.xf".format(
                                    get_imod_path(), name
                                )
                                run_shell_command(com)
                            # com = '{0}/bin/newstack {3} {1}.ali {1}.ali -bin {2} -mode 2 -multadd 1,0'.format( get_imod_path(), name, stack_binning, interpolation )
                            bin_stack(
                                name + "_unbinned.ali",
                                name + ".ali",
                                stack_binning,
                                "imod",
                            )
                            os.remove(name + "_unbinned.ali")
                        # apply shifts to binned stack (old)
                        else:

                            # if int(parameters['extract_bin']) > 1 or movie_binning > 1:
                            #    com = '{0}/bin/newstack -antialias 6 -xform {1}.xf -mode 2 -multadd 1,0 {1}.bin {1}.ali'.format(get_imod_path(),name)
                            # else:
                            #    com = '{0}/bin/newstack -nearest -xform {1}.xf -mode 2 -multadd 1,0 {1}.bin {1}.ali'.format(get_imod_path(),name)
                            com = "{0}/bin/newstack {3} {1}.mrc {1}.ali -bin {2} -mode 2 -multadd 1,0 -xform {1}.xf".format(
                                get_imod_path(), name, movie_binning, interpolation,
                            )
                            # com = '{0}/bin/newstack {2} {1}.mrc {1}.ali -mode 2 -multadd 1,0 -xform {1}.xf'.format( get_imod_path(), name, interpolation )
                            # print com
                            subprocess.check_output(
                                com, stderr=subprocess.STDOUT, shell=True, text=True
                            )

                        # commands.getoutput( 'touch %s_iteration_%02d_weights_score_%08.3f' % ( name, iter, scores.mean() ) )

                        # weight frames
                        last_iteration = iter == (iterations - 1)
                        if last_iteration:
                            fweights = np.empty([0])
                            blend = np.empty([0])
                        # merge.apply_weights( name + '.ali', micrograph_name + '.xf', binFactor, 50, 5.0, 0, fweights, blend, previous_scores, name + '_weighted.ali', last_iteration )
                        # shutil.move( name + '_weighted.ali', name + '.ali' )

                        if "P0000" in name:
                            # print 'Average score %s (iteration %d) = %8.3f' % ( name, iter, scores.mean() )
                            logger.info(
                                "Average score %s = %8.3f" % (name, scores.mean())
                            )

                    # write identity matrix if we are skipping shifts refinement
                    if iterations == 0:
                        if not os.path.exists("{0}.xf".format(name)):
                            f = open("{0}.xf".format(name), "w")
                            for i in range(frames):
                                f.write(
                                    """   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n"""
                                )
                            f.close()
                            previous_shifts = np.zeros([frames, 6])
                            previous_shifts[:, 0] = previous_shifts[:, 3] = 1
                        shutil.copy2("{}.bin".format(name), "{}.ali".format(name))
                        scores = np.ones([frames])

                    # save/update scores to file
                    np.savetxt(name + "_scores.txt", scores)

                else:
                    previous_shifts = np.zeros([frames, 6])
                    if os.path.exists(micrograph_name + "_weights.txt"):
                        """
                        # use global weights if available and if low particle count
                        if os.path.exists( parameters['data_set'] + '_weights.txt' ) and particles_in_micrograph < 20:
                            previous_scores = np.loadtxt( parameters['data_set'] + '_weights.txt' )
                        else:
                        """
                        previous_scores = np.loadtxt(
                            micrograph_name + "_weights.txt", ndmin=2
                        )[particle_index, :]
                    else:
                        previous_scores = np.ones([frames])
                    scores = previous_scores

                ###################################
                # end of local particle alignment #
                ###################################

                # frequency dependent weights
                weight_matrix = np.empty([0])

                ###########################
                # done evaluating weights #
                ###########################

                # create diagnosis png's
                if apply:
                    weights_power = 2
                    if os.path.exists(micrograph_name + "_weights.txt"):

                        # use global weights if available and if few particles or specifically requested
                        if os.path.exists(parameters["data_set"] + "_weights.txt") and (
                            particles_in_micrograph < 20
                            or (
                                "movie_weights" in parameters
                                and "global" in parameters["movie_weights"].lower()
                            )
                        ):
                            micrograph_weights = (
                                np.loadtxt(parameters["data_set"] + "_weights.txt")
                                ** weights_power
                            )
                        else:
                            micrograph_weights = (
                                np.loadtxt(
                                    micrograph_name + "_weights.txt", ndmin=2
                                ).mean(axis=0)
                                ** weights_power
                            )

                        """
                        if not 'global' in parameters['movie_weights'].lower():
                            micrograph_weights = np.loadtxt( micrograph_name + '_weights.txt', ndmin=2 ).mean(axis=0)**weights_power
                        else:
                            micrograph_weights = np.loadtxt( parameters['data_set'] + '_weights.txt' )**weights_power
                        """
                        micrograph_weights = (
                            micrograph_weights - micrograph_weights.min()
                        ) / (micrograph_weights.max() - micrograph_weights.min())
                    else:
                        micrograph_weights = np.empty([0])

                    # aligned_average = merge.apply_weights( name + '.ali', micrograph_name + '.xf', 2, 250, 0.0000001, 0.5, fweights, blend, np.empty([0]), '', True )
                    # mrc.write( aligned_average, name + '.avg' )

                    if True:
                        global_weighted_average = np.zeros([boxsize, boxsize])
                        counter = 0.0
                        if frames < 5:
                            start_frame = 1
                        else:
                            start_frame = 3
                        end_frame = min(12, frames - 1)
                        for i in range(start_frame, end_frame):
                            global_weighted_average += mrc.readframe(name + ".mrc", i)
                            counter += 1
                        global_weighted_average /= counter

                        # retrieve most recent scores
                        unaligned_scores = np.loadtxt(
                            micrograph_name + "_weights.txt", ndmin=2
                        )[particle_index, :]

                    # 3. evaluate patch-based alignment result
                    if os.path.exists(name + "_patch.ali"):
                        # build weighted particle averages
                        merge.weight_stack(
                            name + "_patch.ali",
                            frealign_path + "/" + name + "_stack.mrc",
                            all_weights,
                        )

                        # launch FREALIGN refinement (skip reconstruction, score evaluation only)
                        command = "cd '{0}'; export MYCORES={1}; echo {2} > `pwd`/mynode; export MYNODES=`pwd`/mynode; {3}/bin/fyp -dataset {4} -iter 2 -maxiter {5} -metric {6} -mode 1 -mask 0,0,0,0,0 -cutoff -1 -rlref {7} -rhref {8} -rrec {9} -fmatch F".format(
                            frealign_path,
                            cores_for_frealign,
                            socket.gethostname(),
                            os.environ["PYP_DIR"],
                            name,
                            frealign_iters,
                            metric,
                            low_res,
                            high_res_refi,
                            rrec,
                        )
                        run_shell_command(command)

                        input = np.array(
                            [
                                line.split()
                                for line in open(parfile)
                                if not line.startswith("C")
                            ],
                            dtype=float,
                        )
                        patch_scores = input[:, field]

                        # produced global weighted average
                        patch_weighted_average = np.zeros([boxsize, boxsize])
                        for i in range(frames):
                            patch_weighted_average += patch_scores[i] * mrc.readframe(
                                name + "_patch.ali", i
                            )
                        patch_weighted_average /= patch_scores.sum()

                    ##########################################################
                    # Evaluate all scores using independent resolution range #
                    ##########################################################

                    # store: aligned frames, aligned weighted framesi, unaligned average, weighed global average, local average and weighted local average
                    images = 3
                    if os.path.exists(name + "_patch.ali"):
                        images += 2
                    elif iterations > 0:
                        images += 2

                    new_weighted_movie = np.zeros([images, boxsize, boxsize])

                    # weighted frames (most current weights)
                    # if iterations > 0:
                    #    new_weighted_movie[ :frames, :, :] = weighted_movie[ :, :, : ]
                    # else:
                    #    new_weighted_movie[ :frames, :, :] = original_weighted_movie

                    if os.path.exists(name + "_patch.ali"):
                        logger.info("patch ali avail")
                        # compute average without reading entire movie
                        for i in range(frames):
                            new_weighted_movie[-images, :, :] += mrc.readframe(
                                name + "_patch.ali", i
                            )
                        new_weighted_movie[-images, :, :] /= frames
                        new_weighted_movie[-images + 1, :, :] = patch_weighted_average
                        images -= 2

                    if iterations > 0:
                        # aligned frame average (local)
                        for i in range(frames):
                            new_weighted_movie[-images, :, :] += mrc.readframe(
                                name + ".ali", i
                            )
                        new_weighted_movie[-images, :, :] /= float(frames)

                        # aligned and weighted frame average
                        for i in range(frames):
                            new_weighted_movie[-images + 1, :, :] += previous_scores[
                                i
                            ] * mrc.readframe(name + ".ali", i)
                        new_weighted_movie[-images + 1, :, :] /= previous_scores.sum()
                        images -= 2

                    # original frame average (global alignment)
                    for i in range(frames):
                        new_weighted_movie[-images, :, :] += mrc.readframe(
                            name + ".mrc", i
                        )
                    new_weighted_movie[-images, :, :] /= float(frames)

                    # unaligned weighted average
                    new_weighted_movie[-images + 1, :, :] = global_weighted_average

                    """
                    # aligned average of frames
                    if os.path.exists( name + '.avg' ):
                        new_weighted_movie[ -images+2, :, :] = mrc.read( name + '.avg' )
                    else:
                        for i in range(frames):
                            new_weighted_movie[ -images+2, :, :] += mrc.readframe( name + '.ali', i )
                        new_weighted_movie[ -images+2, :, :] /= float(frames) 
                    """

                    # over write with weighted average in fourier space
                    # new_weighted_movie[ -images+2, :, :] = merge.apply_weights( name + '.bin', micrograph_name + '.xf', 1, 50, 5.0, 0, np.empty([0]), np.empty([0]), unaligned_scores, '', True )

                    # binning factor between global unbinned shifts and local binned shifts
                    # new_weighted_movie[ -images+2, :, :] = merge.apply_weights( name + '.ali', micrograph_name + '.xf', binFactor, 250, 0.1, 0.5, fweights, blend, unaligned_scores, '', True )
                    # new_weighted_movie[ -images+2, :, :] = merge.apply_weights( name + '.ali', micrograph_name + '.xf', binFactor, 250, 0.1, 0.5, fweights, blend, [], '', True )
                    # new_weighted_movie[ -images+2, :, :] = merge.apply_weights( name + '.bin', micrograph_name + '.xf', binFactor, 250, 0.1, 0.5, np.empty([0]), np.empty([0]), '', '', True )

                    # def merge.apply_weights( filename, shifts, binFactor = 2, delta = 250, deltaF = 0.0000001, radiK = 0.5, fweights = np.empty([0]), blend = np.empty([0]), scores = np.empty([0]), output_stack = '', diagnostics = False ):
                    # micrograph_weights = np.zeros( frames )
                    # micrograph_weights[3:12] = 1
                    # merge.apply_weights( filename, shifts, binFactor = 2, delta = 250, deltaF = 0.0000001, radiK = 0.5, fweights = np.empty([0]), blend = np.empty([0]), scores = np.empty([0]), output_stack = '', diagnostics = False ):
                    # new_weighted_movie[ -images+2, :, :] = merge.apply_weights( name + '.ali', micrograph_name + '.xf', binFactor, 100, 0.1, 0.0, np.empty([0]), np.empty([0]), micrograph_weights, '', True )
                    new_weighted_movie[-images + 2, :, :] = merge.apply_weights(
                        name + ".ali",
                        micrograph_name + ".xf",
                        binFactor,
                        12,
                        0.5,
                        0.0,
                        np.empty([0]),
                        np.empty([0]),
                        micrograph_weights ** 2,
                        "",
                        True,
                    )

                    # save final result
                    # XD: here's where particle frame alignment decides to apply weights or not
                    if (
                        "movie_weights" in parameters.keys()
                        and "scores" in parameters["movie_weights"].lower()
                    ):
                        # use local alignments (score weighted)
                        if "_P0000_" in name:
                            logger.info("Using score-based weights")
                        mrc.write(new_weighted_movie[1, :, :], name + ".avg")
                    elif "movie_weights" in parameters and (
                        parameters["movie_weights"].lower() == "t"
                        or "true" in parameters["movie_weights"].lower()
                    ):
                        # use local alignments (weighted)
                        if "_P0000_" in name:
                            logger.info("Using micrograph-based computed weights")
                        mrc.write(new_weighted_movie[-1, :, :], name + ".avg")
                    else:
                        # use local alignment (full exposure)
                        if "_P0000_" in name:
                            logger.info("Using full exposure")
                        mrc.write(new_weighted_movie[0, :, :], name + ".avg")

                    # XD: temporary delete
                    # if os.path.exists( name + '_matches.pdf' ) or os.path.exists( name + '_matches.png' ):
                    #     return

                    # XD: save stack for reconstruct3d later
                    # output_stack = name + '_stack.mrc'
                    shutil.copy2(name + ".ali", name + "_stack.mrc")
                    os.symlink(name + "_r01_02.par", name + "_r01_02_used.par")

                    # save stack for FREALIGN score evaluation
                    mrc.write(
                        new_weighted_movie, frealign_path + "/" + name + "_stack.mrc"
                    )

                    # this file is gone by the end of this
                    # perhaps save the .par file at the end of the apply = False loop
                    # frealign_paramter_file is frealign_20200110_BG505.F14_VRC01_A006_G000_H040_D007_P0000_frames/20200110_BG505.F14_VRC01_A006_G000_H040_D007_P0000_frames_01.par
                    with open(frealign_parameter_file, "w") as f:
                        # write out header (including resolution table)
                        [
                            f.write(line)
                            for line in open(refine_par_file)
                            if line.startswith("C")
                        ]
                        for i in range(new_weighted_movie.shape[0]):
                            # use uniform distribution of shifts of given spread
                            f.write(
                                frealign_parfile.NEW_PAR_STRING_TEMPLATE
                                % (
                                    i + 1,
                                    psi,
                                    theta,
                                    phi,
                                    sx,
                                    sy,
                                    float(mag),
                                    0,
                                    df1,
                                    df2,
                                    angast,
                                    100.0,
                                    logp,
                                    sigma,
                                    0,
                                    0,
                                )
                            )
                            f.write("\n")

                    # launch FREALIGN refinement (skip reconstruction, full search)

                    # We have to use metric cc3m to generate the fmatch images and that imposes a limit on RHREF (Unrealistic number error)
                    if high_res_eval < 2:
                        high_res_eval = 2.0

                    # faster version

                    '''
                    # # save statistics if using v9.11
                    # if 'new' in metric:
                    #     com = """grep -A 10000 "C  NO.  RESOL  RING RAD" {0}""".format( frealign_parameter_file ) + """ | grep -v RESOL | grep -v Average | grep -v Date | grep C | awk '{if ($2 != "") printf "%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f\\n", $2, $3, $4, $6, $7, $8, $9}' > """ + frealign_path + "/maps/statistics_r01.txt"
                    #     commands.getoutput(com)


                    # XD testing: set metric_weights to metric
                    metric_weights = metric

                    # call FREALIGN directly to improve performance
                    mp = parameters.copy()
                    mp['data_bin'] = 1
                    fp = fparameters.copy()
                    fp['mode'] = '1'
                    if 'fboost' in metric_weights.lower():
                        fp['fboost'] = 'T'
                    else:
                        fp['fboost'] = 'F'
                    fp['mask'] = '0,0,0,1,1'
                    fp['rlref'] = '{}'.format( low_res )
                    fp['rhref'] = '{}'.format( high_res_eval )
                    fp['fmatch'] = 'T'
                    fp['maxiter'] = frealign_iters
                    fp['dataset'] = name

                    local_model = os.getcwd() + '/%s/maps/%s_r01_01.mrc' % ( frealign_path, name )
                    if not os.path.exists( local_model ):
                        os.symlink( os.getcwd() + '/' + os.path.split( parameters['class_ref'] )[-1], local_model )
                    # shutil.copy( frealign_parameter_file, frealign_path + '/maps/' + name + '_r01_02.par' )

                    os.chdir( frealign_path + '/maps' )

                    import pdb; pdb.set_trace()

                    # def mrefine_version( mp, fp, first, last, i, ref, name, ranger, logfile, scratch, metric ):
                    command = frealign.mrefine_version( mp, fp, 1, frames*repeats, 2, 1, name + '_r01_02', '', 'log_score.txt', '', metric_weights )
                    if '_P0000' in name:
                        print 'Score evaluations for graphical output:', command
                    process = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    logger.info(stderr)
                    logger.info(stdout)
                    os.chdir('../..')
                    '''

                    # parse parameter file

                    command = "cd '{0}'; export MYCORES={1}; echo {2} > `pwd`/mynode; export MYNODES=`pwd`/mynode; {3}/bin/fyp -dataset {4} -iter 2 -maxiter {5} -metric {6} -mode 1 -mask 0,0,0,1,1 -cutoff -1 -rlref {7} -rhref {8} -rrec {9} -fmatch T".format(
                        frealign_path,
                        cores_for_frealign,
                        socket.gethostname(),
                        os.environ["PYP_DIR"],
                        name,
                        frealign_iters,
                        metric_weights,
                        low_res,
                        high_res_eval,
                        rrec,
                    )
                    logger.info("skipping")

                    logger.info(command)
                    # code, output = commands.getstatusoutput(command)
                    # logger.info(code)
                    # logger.info(output)

                    # pdb.set_trace()
                    input = np.array(
                        [
                            line.split()
                            for line in open(parfile + "_")
                            if not line.startswith("C")
                        ],
                        dtype=float,
                    )
                    new_scores = input[:, field]

                    # print 'Score with weights = %.2f, without weights %.2f' % ( input[-1,field], input[-2,field] )

                    local_score = input[0, field]
                    global_score = input[2, field]
                    if True or local_score > global_score:
                        # use weighted average as result
                        logger.info(
                            "Using local average as result for %s (Local score = %.2f, global score %.2f)"
                            % (name, local_score, global_score)
                        )
                    else:
                        logger.info(
                            "Local average did not produce a better score for %s (Local score = %.2f, global score %.2f). Using global average."
                            % (name, local_score, global_score)
                        )
                        if True:
                            # write identity matrix for null shifts
                            f = open("{0}.xf".format(name), "w")
                            for i in range(frames):
                                f.write(
                                    """   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n"""
                                )
                            f.close()
                            shutil.copy2("{}.bin".format(name), "{}.ali".format(name))
                            # aligned_average = merge.apply_weights( name + '.ali', micrograph_name + '.xf', 2, 250, 0.0000001, 0.5, fweights, blend )
                            # mrc.write( aligned_average, name + '.avg' )
                            # mrc.write( global_weighted_average, name + '.avg' )

                    ################## done evaluating weights

                    matches = "%s/maps/%s_r01_%02d_match_unsorted.mrc" % (
                        frealign_path,
                        name,
                        frealign_iters,
                    )
                    if os.path.exists(matches):
                        A = mrc.read(matches)
                        M = plot.contact_sheet(A, new_weighted_movie.shape[0])
                        writepng(M, "{0}_matches.png".format(name))
                        command = "convert -resize 200% {0}_matches.png {0}_matches.png".format(
                            name
                        )
                        run_shell_command(command)

                    # composite plot
                    import matplotlib.pyplot as plt

                    c = np.linspace(0, unaligned_scores.size - 1, unaligned_scores.size)
                    my_dpi = 200.0
                    fig, ax = plt.subplots(
                        1, 1, figsize=(1128 / my_dpi, 560 / my_dpi), dpi=my_dpi
                    )
                    unaligned_scores = (unaligned_scores - unaligned_scores.min()) / (
                        unaligned_scores.max() - unaligned_scores.min()
                    )
                    ax.plot(
                        c, unaligned_scores, "g.-", label="%dA (global)" % high_res_refi
                    )
                    if iterations > 0 and scores.min() < scores.max():
                        scores = (scores - scores.min()) / (scores.max() - scores.min())
                        ax.plot(c, scores, "r.-", label="%dA (local)" % high_res_refi)

                        combined_weights = 1.0 * np.arange(frames) / frames + 1 - scores
                        combined_weights = (
                            combined_weights - combined_weights.min()
                        ) / (combined_weights.max() - combined_weights.min())
                        # ax.plot( c, 1 - combined_weights, 'b.-', label='combined' )

                    else:
                        combined_weights = (
                            1.0 * np.arange(frames) / frames + 1 - unaligned_scores
                        )
                        combined_weights = (
                            combined_weights - combined_weights.min()
                        ) / (combined_weights.max() - combined_weights.min())
                        # ax.plot( c, 1 - combined_weights, 'b.-', label='combined' )

                    # plot radiation term
                    # ax.plot( c, 1.0 - 1.0 * np.arange(frames) / frames, 'm.-', label='radiation' )

                    # ax.plot( c, new_scores - new_scores.mean(), 'b.-', label='%dA (local)' % high_res_eval )
                    if os.path.exists(name + "_patch.ali"):
                        patch_scores = (patch_scores - patch_scores.min()) / (
                            patch_scores.max() - patch_scores.min()
                        )
                        ax.plot(
                            c, patch_scores, "m.-", label="%dA (patch)" % high_res_eval
                        )
                    for i in range(weight_matrix.shape[0]):
                        weights = weight_matrix[i, :]
                        if weights.max() > weights.min():
                            weights = (weights - weights.min()) / (
                                weights.max() - weights.min()
                            )
                        ax.plot(
                            c,
                            weights,
                            ":",
                            linewidth=1,
                            label="%4.1fA (band)" % (1.0 / float(centers[i])),
                        )
                    # scale = ( unaligned_scores - unaligned_scores.mean() ).max()
                    ax.plot(c, center_frame_weights, "k.-", label="weights")

                    # plot micrograph weights as reference
                    if os.path.exists(micrograph_name + "_weights.txt"):
                        # micrograph_weights = np.loadtxt( micrograph_name + '_weights.txt' )[::4,:].mean(axis=0)**2
                        # micrograph_weights = ( micrograph_weights - micrograph_weights.min() ) / ( micrograph_weights.max() - micrograph_weights.min() )
                        ax.plot(c, micrograph_weights, "b.-", label="image")

                    ax.set_xlim((-0.5, unaligned_scores.size - 0.5))
                    ax.legend(fontsize=8, loc="upper right")
                    ax.set_title("Scores")
                    plt.savefig("{}_diagnostic.png".format(name))
                    plt.close()

                    if os.path.exists(name + "_patch.ali") and os.path.exists(
                        name + "_xf.png"
                    ):
                        shutil.copy(name + "_xf.png", name + "_patch_xf.png")

                    # XD testing: plot both noisy and not noisy trajectories
                    noisy_traj = np.loadtxt(name + "_noisy.xf")
                    reg_traj = np.loadtxt(name + ".xf")
                    output_name = "{}_xf.png".format(name)

                    plot.plot_trajectory_raw(
                        reg_traj, output_name=output_name, noisy=noisy_traj
                    )

                    # plot.plot_trajectory( name + "_noisy")
                    # plot.plot_trajectory( name )

                    plot.plot_trajectory(micrograph_name)

                    # also plot global trajectory
                    micrograph_drift = np.round(np.loadtxt(micrograph_name + ".xf"))
                    particle_drift = np.loadtxt(name + ".xf")
                    total_drift = micrograph_drift + particle_drift
                    total_drift[:, [0, 3]] /= 2
                    np.savetxt(name + "_global.xf", total_drift)
                    plot.plot_trajectory(name + "_global")

                    if os.path.exists(name + "_patch_xf.png"):
                        com = "montage {0}_diagnostic.png {0}_patch_xf.png {1}_xf.png {0}_xf.png -geometry +0+0 -tile 4x1 -geometry +0+0 {0}_diagnostic.png".format(
                            name, micrograph_name
                        )
                    elif iterations > 0:
                        com = "montage {0}_diagnostic.png {1}_xf.png {0}_xf.png {0}_global_xf.png -geometry +0+0 -tile 4x1 -geometry +0+0 {0}_diagnostic.png".format(
                            name, micrograph_name
                        )
                    else:
                        com = ""
                        com = "montage {0}_diagnostic.png {1}_xf.png -geometry +0+0 -tile 2x1 -geometry +0+0 {0}_diagnostic.png".format(
                            name, micrograph_name
                        )
                    if len(com) > 0:
                        run_shell_command(com, verbose=parameters["slurm_verbose"])

                    com = "montage {0}_matches.png {0}_diagnostic.png {0}_weights.png -tile 1x3 -geometry +0+0 {0}_matches.png".format(
                        name
                    )
                    com = "montage {0}_matches.png {0}_diagnostic.png {0}_weights.png {0}_weights_new.png -tile 1x4 -geometry +0+0 {0}_matches.png".format(
                        name
                    )
                    run_shell_command(com, verbose=parameters["slurm_verbose"])

                    try:
                        shutil.rmtree("frealign_" + name)
                    except:
                        logger.warning(
                            "Could not delete folder %s/frealign_%s"
                            % (os.getcwd(), name)
                        )
                        pass

                    # collate and save weights
                    # all_weights = np.vstack( ( unaligned_scores, scores, combined_weights, 1.0 - 1.0 * np.arange(frames) / frames ) )
                    # np.savetxt( name + '_weights.txt', all_weights )
                    # np.savetxt( name + '_weights.txt', scores )

                    ##############################

            # done with alignment

            if movie_binning > 1 or interval > 0:
                t = np.loadtxt("%s.xf" % name, ndmin=2)
                t[:, -2:] *= movie_binning
                np.savetxt("%s.xf" % name, t, fmt="%13.7f")

                if os.path.exists(name + ".prexgraw"):
                    traw = np.loadtxt("%s.prexgraw" % name, ndmin=2)
                    traw[:, -2:] *= movie_binning
                    np.savetxt("%s.prexgraw" % name, traw, fmt="%13.7f")

            # save .xf file without binning
            binning = int(parameters["data_bin"])
            if binning > 1 and not apply:
                t = np.loadtxt("%s.xf" % name, ndmin=2)
                t[:, -2:] *= binning
                np.savetxt("%s.xf" % name, t, fmt="%13.7f")

        else:

            # single frame case
            f = open("{0}.xf".format(name), "w")
            for i in range(frames):
                f.write(
                    """   1.0000000   0.0000000   0.0000000   1.0000000       0.000       0.000\n"""
                )
            f.close()
            shutil.copy2("{}.mrc".format(name), "{}.ali".format(name))

    # compute average of aligned frames

    # use program-specific average if exists
    if not os.path.exists(name + ".avg"):

        # average unbinned stacks and save: name.avg
        apply_alignments_and_average(input_name, name, parameters)

        # average aligned tilt-series
        #         command="""
        # %s/bin/avgstack << EOF
        # %s.ali
        # %s.avg
        # /
        # EOF
        # """  % ( get_imod_path(), name, name )
        # input_fname = f"{name}.ali"
        # output_fname = f"{name}.avg"
        # start_end_section = "/"

        # output, error = avgstack(input_fname, output_fname, start_end_section)

    else:
        logger.warning("Using program generated frame average\n")

    # average aligned stack and save
    aligned_average = mrc.read(name + ".avg")

    return aligned_average

# sum all frames acording to parameter values without aligning
def sum_gain_correct_frames(movie, average, parameters):

    if parameters["gain_remove_hot_pixels"]:
        if Path(movie).suffix == '.mrc':
            preprocess.remove_xrays_from_file(Path(movie).stem,parameters['slurm_verbose'])
        else:
            logger.warning(f"Skipping hot pixel removal on images of format {Path(movie).suffix}")

    # get image dimensions
    x, y, z = get_image_dimensions(movie)

    # figure out range of frames to average
    first_frame = parameters["movie_first"] if "movie_first" in parameters else 0
    last_frame = parameters["movie_last"] if "movie_last" in parameters and parameters["movie_last"] != -1 else z

    # average frames in the specified range
    output, error = avgstack(
        movie, average, f"{first_frame},{last_frame}"
    )

    # are we using a gain reference?
    if "gain_reference" in parameters.keys() and parameters["gain_reference"] and os.path.exists(
        project_params.resolve_path(parameters["gain_reference"])
        ):
        gain_reference_file = project_params.resolve_path(parameters["gain_reference"])
        gain_file = os.path.basename(gain_reference_file)
        gain = f"../{gain_file}"
        if os.path.exists(gain):
            gain_reference_file = gain
        else:
            gain_reference, gain_reference_file = get_gain_reference(
                parameters, x, y
            )
    else:
        gain_reference_file = None

    # if using eer format, figure out the reduce factor
    if movie.endswith(".eer"):
        binning = 1
        if 'movie_eer_reduce' in parameters:
            binning = int(4/parameters['movie_eer_reduce'])
        elif gain_reference_file != None:
            gain_x, gain_y, gain_z = get_image_dimensions(gain_reference_file)
            binning = int(x / gain_x)
        if binning > 1:
            com = f"{get_imod_path()}/bin/newstack {average} {average} -bin {binning}"
            run_shell_command(com)

    # apply gain reference if we are using one
    if gain_reference_file != None:
        com = f'{get_imod_path()}/bin/clip multiply "{average}" "{gain_reference_file}" "{average}"; rm -f {average}~'
        output, error = run_shell_command(com)

        if "error" in output.lower():
            logger.error(output)
            if "sizes must be equal" in output.lower():
                logger.error("Did you apply the correct transformation to the gain reference?")
                x, y, z = get_image_dimensions(average)
                logger.info(f"{average} dimensions are {x} x {y}")
                x, y, z = get_image_dimensions(gain_reference_file)
                logger.info(f"{gain_reference_file} dimensions are {x} x {y}")
            raise Exception("Failed to apply gain reference")

def align_movie_super(parameters, name, suffix, isfirst = False):

    tmp_directory = name
    os.mkdir(tmp_directory)
    os.chdir(tmp_directory)

    movie_file = name + suffix
    aligned_average = name + ".avg"

    pixel = float(parameters["scope_pixel"])
    binning = float(parameters["data_bin"])
    voltage = float(parameters["scope_voltage"])
    init_dose = float(parameters["scope_init_dose"])
    dose_rate = float(parameters["scope_dose_rate"])
    mag_major = float(parameters["scope_mag_major"])
    mag_minor = float(parameters["scope_mag_minor"])
    distort_angle = float(parameters["scope_distort_ang"])
    actual_pixel = (
        pixel
        * float(parameters["data_bin"])
    )

    if 'motioncor' in parameters["movie_ali"]:

        # patch tracking
        patches_x = parameters["movie_motioncor_patch_x"] if "movie_motioncor_patch_x" in parameters else 1
        patches_y = parameters["movie_motioncor_patch_y"] if "movie_motioncor_patch_y" in parameters else 1
        if patches_x + patches_y > 2:
            patches = f" -Patch {parameters['movie_motioncor_patch_x']} {parameters['movie_motioncor_patch_y']}"
            if parameters.get("movie_motioncor_patch_overlap"):
                patches += f" {parameters['movie_motioncor_patch_overlap']}"
        else:
            patches = ""

        """
        -InMrc
        -InTiff
        -InEer
        -InSuffix
        -OutMrc
        -FmIntFile
        -ArcDir
        -FullSum
        -Gain
        -Dark
        -DefectFile
        -DefectMap
        -InAln
        -OutAln
        -TmpFile
        -LogDir
        -FmIntFile
        -Serial          0
        -EerSampling     1
        -Patch           0  0  0
        -Iter            15
        -Tol             0.10
        -Bft             500.00 100.00
        -PhaseOnly       0
        -FtBin           1.00
        -InitDose        0.00
        -FmDose          0.00
        -PixSize         0.00
        -kV              300
        -Cs              2.70
        -AmpCont         0.07
        -ExtPhase        0.00
        -Throw           0
        -Trunc           0
        -SumRange        3.00  25.00
        -SplitSum        0
        -Group           1  4
        -FmRef           -1
        -OutStack        0  1
        -RotGain         0
        -FlipGain        0
        -InvGain         0
        -Align           1
        -Tilt            0.00  0.00
        -Mag             1.00  1.00  0.00
        -InFmMotion      0
        -Crop            0  0
        -Gpu             0
        -UseGpus         1
        -GpuMemUsage     0.75
        -OutStar         0
        -TiffOrder       1
        -CorrInterp      0
        """

        if 'mrc' in suffix:
            input = f"-InMrc ../{movie_file}"
        elif 'tif' in suffix:
            input = f"-InTiff ../{movie_file}"
        elif 'eer' in suffix:
            input = f"-InEer ../{movie_file}"

            eer_frames_perimage = int(parameters["movie_eer_frames"])
            eer_superres_factor = int(parameters["movie_eer_reduce"])
            input = f"{input} -EerSampling {eer_superres_factor} -Group {eer_frames_perimage}"

        if "gain_reference" in parameters.keys() and os.path.exists(
            project_params.resolve_path(parameters["gain_reference"])
            ):
            gain_reference_file = project_params.resolve_path(parameters["gain_reference"])
            gain_file = os.path.basename(gain_reference_file)
            gain = f" -Gain ../{gain_file}"

            # If both -RotGain and -FlipGain are enabled, the gain reference will be rotated first and flipped next.
            if "gain_flipv" in parameters.keys() and parameters["gain_flipv"]:
                gain += f" -FlipGain 1"
            elif "gain_fliph" in parameters.keys() and parameters["gain_fliph"]:
                gain += f" -FlipGain 2"
            if "gain_rotation" in parameters.keys() and abs(int(parameters["gain_rotation"])) >= 0:
                gain += f" -RotGain {parameters['gain_rotation']}"
        else:
            gain = ""

        frame_options = ""
        x, y, total_frames = get_image_dimensions(f"../{movie_file}")
        if parameters["movie_first"] > 0:
            frame_options += f" -Throw {parameters['movie_first']}"
        if parameters["movie_last"] != -1:
            frame_options += f" -Trunc {total_frames - parameters['movie_last']}"
        if parameters["movie_group"] > 1 and "EerSampling" not in input:
            frame_options += f" -Group {parameters['movie_group']}"
        frame_options += f" -Bft {parameters['movie_motioncor_bfactor_global']} {parameters['movie_motioncor_bfactor_local']}"
        frame_options += f" -Tol {parameters['movie_motioncor_tol']} -Iter {parameters['movie_motioncor_iter']}"
        frame_options += f" -SumRange {parameters['movie_motioncor_sumrange_min']} {parameters['movie_motioncor_sumrange_max']}"
        if parameters.get("movie_motioncor_phase_only"):
            frame_options += " -PhaseOnly"
        if parameters.get("movie_motioncor_corr_interp"):
            frame_options += " -CorrInterp"
        if parameters.get("movie_motioncor_in_frame_motion"):
            frame_options += " -InFmMotion"

        if parameters["movie_motioncor_frameref"] > 0:
            frame_ref = parameters['movie_motioncor_frameref'] if parameters['movie_motioncor_frameref'] <= total_frames else total_frames
            frame_options += f" -FmRef {frame_ref}"

        dose_weighting_options = ""
        if parameters["movie_weights"]:
            dose_weighting_options += f" -InitDose {init_dose} -FmDose {dose_rate} -PixSize {pixel} -kV {voltage}" 
            dose_weighting_options += " -Cs 0" # NOT do CTF estimation

        mag_correction_options = ""
        if parameters["movie_magcorr"]:
            mag_correction_options += f" -Mag {mag_major} {mag_minor} {distort_angle}"

        """
        Usage: MotionCor3 Tags

        -InMrc
        1. Input MRC file that stores dose fractionated stacks.
        2. It can be a MRC file containing a single stack collected
            in Leginon or multiple stacks collected in UcsfTomo.
        3. It can also be the path of a folder containing multiple
            MRC files when -Serial option is turned on.

        -InTiff
        1. Input TIFF file that stores a dose fractionated stack.
        -InEer
        1. Input EER file that stores a dose fractionated stack.
        -OutMrc
        1. Output MRC file that stores the frame sum.
        2. It can be either a MRC file name or the prefix of a series
            MRC files when -Serial option is turned on.

        -ArcDir
        1. Path of the archive folder that holds the archived raw
            stacks with each pixel packed into 4 bits.
        2. The archived stacks are saved in MRC file with the gain
            reference saved in the extended header.
        3. The rotated and/or flipped gain reference will be saved
            if -RotGain and or -FlipGain are enabled.

        -FullSum
        1. MRC file for global-motion corrected, unweighted sum.
        2. This file is generated as soon as the global motion
            correction is completed while the program continues
            lengthy local motion correction. This file allows users
            to perform CTF estimate to gain quick feedback on the
            image quality.
        3. This file is temporary, when the next stack is processed,
            its content will be overwritten.

        -DefectFile
        1. Defect file stores entries of defects on camera.
        2. Each entry corresponds to a rectangular region in image.
        The pixels in such a region are replaced by neighboring
        good pixel values.
        3. Each entry contains 4 integers x, y, w, h representing
        the x, y coordinates, width, and heights, respectively.

        -InAln
        1. Specify the path to the directory where the alignment file
            will be loaded.
        2. The alignment file is a text file that stores the program
            setting and measured global and local motion. This file
            is created with -OutAln option.
        3. Once the alignment file is loaded, the alignment procedure
            will be bypassed with the loaded alignment data applied
            to generate motion-corrected images.

        -OutAln
        1. Specify the path to the directory where the alignment file
            will be saved.
        2. The alignment file is a text file that stores the program
            setting and measured global and local motion. This file can
            be reloaded next time into MotionCor2 that will bypass
            the alignment process.

        -DefectMap
        1. Defect map is a binary (0 or 1) map where defective pixels
        are assigned value of 1 and good pixels have value of 0.
        2. The defective pixels are corrected with a random pick of
        good pixels in its neighborhood.
        3. This is map must have the same dimension and orientation
        as the input movie frame.
        4. This map can be provided as either MRC or TIFF file that has
        MRC mode of 0 or 5 (unsigned 8 bit).

        -Serial
        1. Serial-processing all MRC files in a given folder whose
            name should be specified following -InMrc.
        2. The output MRC file name emplate should be provided
            folllowing -OutMrc
        3. 1 - serial processing, 0 - single processing, default.
        4. This option is only for single-particle stack files.

        -Gain
        MRC file that stores the gain reference. If not
        specified, MRC extended header will be visited
        to look for gain reference.

        -Dark
        1. MRC file that stores the dark reference. If not
            specified, dark subtraction will be skipped.
        2. If -RotGain and/or -FlipGain is specified, the
            dark reference will also be rotated and/or flipped.

        -TmpFile
        Temporary image file for debugging.

        -LogDir
        1. Log directory storing log files. Log files have the
            same file names as the output MRC files but with mrc
            replaced with log.

        -Patch
        1. It follows by  number of patches in x and y dimensions.
        2. The default values are 1 1, meaning only full-frame
            based alignment is performed.

        -Iter
        Maximum iterations for iterative alignment,
        default 5 iterations.

        -Tol
        Tolerance for iterative alignment,
        default 0.5 pixel.

        -Bft
        B-Factor for alignment, default 100.

        -PhaseOnly
        Only phase is used in cross correlation.
        default is 0, i.e., false.

        -FtBin
        Binning performed in Fourier space, default 1.0.

        -InitDose
        Initial dose received before stack is acquired

        -FmDose
        Frame dose in e/A^2. If not specified, dose
        weighting will be skipped.

        -PixSize
        Pixel size in A of input stack in angstrom. If not
        specified, dose weighting will be skipped.

        -kV
        High tension in kV needed for dose weighting.
        Default is 300.

        -Cs               1. Spherical aberration in mm. The default is set to
            zero, meaning NO CTF estimation.

        -AmpCont          1. Amplitude contrast. The default is 0.07.

        -ExtPhase         1. Extra phase shift in degree. The default is 0 degree,
            meaning NO estimation of extra phase shift.
        2. If a positive value is given, extra phase shift will
            estimated in a range centered at the given value. The
            range is limited within [0, 180] degrees.

        -Align
        Generate aligned sum (1) or simple sum (0)

        -Throw
        Throw initial number of frames, default is 0

        -Trunc
        Truncate last number of frames, default is 0

        -SumRange
        1. Sum frames whose accumulated doses fall in the
            specified range. The first number is the minimum
            dose and the second is the maximum dose.
        2. The default range is [3, 25] electrons per square
            angstrom.

        -Group
        1. Group every specified number of frames by adding
            them together. The alignment is then performed
            on the group sums. The so measured motion is
            interpolated to each raw frame.
        2. The 1st integer is for gobal alignment and the
            2nd is for patch alignment.

        -Crop
        1. Crop the loaded frames to the given size.
        2. By default the original size is loaded.

        -FmRef
        Specify a frame in the input movie stack to be the
        reference to which all other frames are aligned. The
        reference is 1-based index in the input movie stack
        regardless how many frames will be thrown. By default
        the reference is set to be the central frame.

        -OutStack
        1. It is followed by two integers used to specify if
            the aligned stack will be generated.
        2. When the 1st integer is set to 1, the aligned stack
            will be created.
        3. The 2nd integer specifies the z binning, i.e, the
            number of aligned frames to be summed in each
            output frame in the aligned stack.

        -RotGain
        Rotate gain reference counter-clockwise.
        0 - no rotation, default,
        1 - rotate 90 degree,
        2 - rotate 180 degree,
        3 - rotate 270 degree.

        -FlipGain
        Flip gain reference after gain rotation.
        0 - no flipping, default,
        1 - flip upside down,
        2 - flip left right.

        -InvGain
        Inverse gain value at each pixel (1/f). If a orginal
        value is zero, the inversed value is set zero.
        This option can be used together with flip and
        rotate gain reference.

        -Mag
        1. Correct anisotropic magnification by stretching
            image along the major axis, the axis where the
            lower magificantion is detected.
        2. Three inputs are needed including magnifications
            along major and minor axes and the angle of the
            major axis relative to the image x-axis in degree.
        3. By default no correction is performed.

        -InFmMotion
        1. 1 - Account for in-frame motion.
            0 - Do not account for in-frame motion.

        -Gpu
        GPU IDs. Default 0.
        For multiple GPUs, separate IDs by space.
        For example, -Gpu 0 1 2 3 specifies 4 GPUs.

        -GpuMemUsage
        1. GPU memory usage, default 0.5, meaning 50% of GPU
            memory will be used to buffer movie frames.
        2. The value should be between 0 and 0.5. When 0 is given,
            all movie frames are buffered on CPU memory.

        -UseGpus
        1. Specify number of GPUs out of free GPUs to use in the
            current process. By default, all free GPUs are used
        2. If less free GPUs are found than the specified number,
            the process will continue. If zero is found, the
            the process will quit.

        -SplitSum
        1. Generate odd and even sums using odd and even frames
            respectively when this option is enabled.

        -OutStar
        1. Generate the star file for Relion 4 polishing. By
            Default, it is diaabled. Set 1 to enable.
        """

        command = f"{get_motioncor3_path()} \
{input} \
-OutMrc {name}.mrc \
-FtBin {parameters.get('movie_motioncor_bin')} \
{gain} \
-OutAln {os.getcwd()} \
{frame_options} \
{patches} \
{dose_weighting_options} \
{mag_correction_options} \
-Gpu {get_gpu_id()} \
-UseGpus 1"
        [ output, error ] = run_shell_command(command, verbose=parameters["slurm_verbose"])

        if "Segmentation fault" in error or "Killed" in error:
            raise Exception(error)

        if "no CUDA-capable device is detected" in output or "All GPUs are in use" in output:
            if not parameters['slurm_verbose']:
                logger.error(output)
            logger.error('A GPU must be available for MotionCor3 to run')
            raise Exception(output)

        # rename frame average
        if parameters["movie_weights"]:
            if parameters['movie_motioncor_sumrange_min'] == 0 and parameters['movie_motioncor_sumrange_max'] == 0:
                shutil.move( name + "_DW.mrc", f"../{aligned_average}")
            else:
                shutil.move( name + "_DWS.mrc", f"../{aligned_average}")
        else:
            shutil.move( name + ".mrc", f"../{aligned_average}")

        # read shifts and save in txt format
        newf = open(f"{name}_clean.aln", "w")

        # output file has header and footer (if patch is used)
        NUM_LINES_HEADER = 8
        with open(f"{name}.aln", "r") as f:
            for idx, line in enumerate(f.readlines()):
                if NUM_LINES_HEADER < idx + 1 and idx < NUM_LINES_HEADER + total_frames:
                    newf.write(line)
        newf.close()
        shifts = np.loadtxt(f"{name}_clean.aln",ndmin=2)
        # only keep shift values for integrated frames if using eer movies
        if "EerSampling" in input:
            shifts = shifts[::eer_frames_perimage,:]
        if parameters.get("movie_force_integer"):
            shifts = shifts.round()
        np.savetxt(f"../{name}_shifts.txt",shifts[:,1:],fmt="%.4f")

    elif 'unblur' in parameters["movie_ali"]:

        if "gain_reference" in parameters.keys() and os.path.exists(
            project_params.resolve_path(parameters["gain_reference"])
            ):
            gain_reference_file = project_params.resolve_path(parameters["gain_reference"])
            gain_file = os.path.basename(gain_reference_file)

            gain_corrected = "no"

            if ("gain_rotation" in parameters.keys()
            and abs(int(parameters["gain_rotation"])) >= 0
            ):
                gain_rotation = abs(int(parameters["gain_rotation"]))

            if "gain_fliph" in parameters.keys():
                if parameters["gain_fliph"]:
                    gain_fliph = "yes"
                else:
                    gain_fliph = "no"

            if "gain_flipv" in parameters.keys():
                if parameters["gain_flipv"]:
                    gain_flipv = "yes"
                else:
                    gain_flipv = "no"

            gain_operate = "\n../%s\n%s\n%s\n%d" % (gain_file, gain_flipv, gain_fliph, gain_rotation)
        else:
            gain_corrected = "yes"
            gain_operate = ""

        if "movie_weights" in parameters.keys() and parameters["movie_weights"]:
            weighted = "yes\n%s\n%s\n%s" % (
                voltage,
                dose_rate,
                init_dose,
            )
            restore_Noise_power = "\nYES"
        else:
            weighted = "NO"
            restore_Noise_power = ""

        save_aligned_frames = False

        if save_aligned_frames:
            save_frames = "yes\n%s_aligned_frames.mrc" % name
        else:
            save_frames = "no"

        if "eer" in suffix:
            eer_frames_perimage = int(parameters["movie_eer_frames"])
            eer_superres_factor = int(parameters["movie_eer_reduce"])
            eer = "\n%d\n%d" % (eer_frames_perimage, eer_superres_factor)
            actual_pixel /= eer_superres_factor
        else:
            eer = ""

        if parameters["movie_force_integer"]:
            forceinteger = "yes"
        else:
            forceinteger = "no"

        if "movie_magcorr" in parameters.keys() and parameters["movie_magcorr"]:
            mag_corrections = "yes\n%s\n%s\n%s" % (
                distort_angle,
                mag_major,
                mag_minor,
            )
        else:
            mag_corrections = "no"

        bfactor = float(parameters["movie_bfactor"])
        first_frame = int(parameters["movie_first"]) + 1 # pyp from 0, unblur starts from 1
        last_frame = int(parameters["movie_last"]) + 1 if int(parameters["movie_last"]) != -1 else 0 # pyp's end is -1, unblur's end is 0
        running_average = parameters["movie_group"]
        maximum_shifts_in_A = 40.0
        minimum_shifts_in_A = 0.0
        threads = min(6,parameters["slurm_tasks"]) if "spr" in parameters["data_mode"].lower() else 1

        """
                **   Welcome to Unblur   **

                    Version : 2.00
                Compiled : Jun 30 2022
            Library Version : 2.0.0-alpha--1--dirty
                From Branch : main
                    Mode : Interactive

        Input stack filename [my_movie.mrc]                :
        Output aligned sum [my_aligned_sum.mrc]            :
        Output shift text file [my_shifts.txt]             :
        Pixel size of images (A) [1.0]                     :
        Output binning factor [1]                          :
        Apply Exposure filter? [yes]                       : no
        Set Expert Options? [no]                           : yes
        Minimum shift for initial search (A) [2.0]         :
        Outer radius shift limit (A) [80.0]                :
        B-factor to apply to images (A^2) [1500]           :
        Half-width of vertical Fourier mask [1]            :
        Half-width of horizontal Fourier mask [1]          :
        Termination shift threshold (A) [1]                :
        Maximum number of iterations [20]                  :
        Input stack is dark-subtracted? [yes]              :
        Input stack is gain-corrected? [yes]               :
        First frame to use for sum [1]                     :
        Last frame to use for sum (0 for last frame) [0]   :
        Number of frames for running average [1]           :
        Save Aligned Frames? [no]                          :
        Correct Magnification Distortion? [no]             :
        Max. threads to use for calculation [1]            :
        """

        # unblur cisTEM 2.0
        unblur_path = get_unblur2_path()
        command = f"""
{unblur_path}/unblur_gain << EOF
../{movie_file}
../{aligned_average}
../{name}_shifts.txt
{actual_pixel}
{binning}
{weighted}
yes
{minimum_shifts_in_A}
{maximum_shifts_in_A}
{bfactor}
1
1
1
20{restore_Noise_power}
yes
{gain_corrected}{gain_operate}
{first_frame}
{last_frame}
{running_average}
{save_frames}{eer}
{forceinteger}
{mag_corrections}
{threads}
EOF
"""
        command = (
            "export OMP_NUM_THREADS={0}; export NCPUS={0}; ".format(threads)
            + command
        )
        if parameters['data_mode'] == 'tomo' and parameters["slurm_verbose"] and not isfirst:
            [output, error] = run_shell_command(command, verbose=False)
        else:
            [output, error] = run_shell_command(command, verbose=parameters["slurm_verbose"])

        if "Segmentation fault" in error or "Killed" in error:
            logger.error("Try increasing the Memory per task in the Resources tab (or --slurm_memory parameter in the CLI)")
            raise Exception(error)

    elif 'skip' in parameters["movie_ali"]:

        # write identity matrix for null shifts
        x, y, total_frames = get_image_dimensions(f"../{movie_file}")
        shifts = np.zeros([total_frames,2])
        np.savetxt(f"../{name}_shifts.txt",shifts,fmt="%.4f")
        sum_gain_correct_frames(f"../{movie_file}", f"../{aligned_average}", parameters)

    # go back to parent directory and cleanup
    os.chdir("..")
    shutil.rmtree(tmp_directory)
    os.remove(f"{movie_file}")
    # convert shifts to .xf file
    shifts = np.loadtxt(name + "_shifts.txt", comments="#", ndmin=2)

    xfshifts = np.zeros((shifts.shape[0], 6))
    xfshifts[:, 0] = 1
    xfshifts[:, 3] = 1
    xfshifts[:, 4] = shifts[:, 0] / actual_pixel
    xfshifts[:, 5] = shifts[:, 1] / actual_pixel
    np.savetxt(name + ".xf", xfshifts, fmt="%13.7f")

    # maximum displacement
    error = np.hypot(xfshifts[:, -2], xfshifts[:, -1]).max()

    # save .xf file without binning
    binning = int(parameters["data_bin"])
    if binning > 1:
        t = np.loadtxt("%s.xf" % name, ndmin=2)
        t[:, -2:] *= binning
        np.savetxt("%s.xf" % name, t, fmt="%13.7f")

    # average aligned stack and save
    aligned_average = mrc.read(name + ".avg")

    return aligned_average

def generate_thumbnail(aligned_average, name, parameters):

    # save bin8 image png
    if aligned_average.shape[0] > 6096:
        binning = 4
    elif aligned_average.shape[0] < 2048:
        binning = 1
    else:
        binning = 2

    small = (
        aligned_average.reshape(
            aligned_average.shape[0] // binning,
            binning,
            aligned_average.shape[1] // binning,
            binning,
        )
        .mean(3)
        .mean(1)
    )
    writepng(small, "{}_small.png".format(name))
    contrast_stretch(name + "_small.png", name + ".jpg", 50)

    img2webp(f"{name}.jpg",f"{name}.webp")

    # bin image
    if False and binning > 1:
        size = (
            min(aligned_average.shape[0], aligned_average.shape[1])
            * parameters["data_bin"]
        )
        binning = math.floor(size / 512)
        if binning % 2 > 0:
            binning += 1
        new_size = binning * 512 // parameters["data_bin"]
        view = aligned_average[0 : int(new_size), 0 : int(new_size)]
        small = view.reshape(512, view.shape[0] // 512, 512, -1).mean(3).mean(1)

    if parameters["data_mode"] == "spr":
        new_name = "image.png"
    else:
        new_name = name + "_image.png"

    writepng(small, new_name)
    contrast_stretch(new_name)
    return binning


@Timer("align", text="Alignment took: {}", logger=logger.info)
def align_tilt_series(name, parameters, rotation=0):
    """
        Tilt series alignment.

        1. Data is binned by 2 if not already binned
        # 2. Hot-pixel removal with ccderaser
        3. Pre-alignment using tiltxcorr
        4. Alignment using RAPTOR or patches

        Alignment transformation is saved unbinned.
        """

    dim = int(mrc.readHeaderFromFile(name + ".st")["nx"])
    if dim > 8192:
        binning = 10
    elif dim >= 6144:
        binning = 8
    elif dim >= 4096:
        binning = 4
    elif dim >= 2048:
        binning = 2
    elif dim >= 1024:
        binning = 1
    else:
        binning = 1

    if not parameters["tomo_ali_auto_bin"] and parameters["tomo_ali_binning"] > 0:
        binning = parameters["tomo_ali_binning"]

    if binning > 1:
        command = "{0}/bin/newstack {1}.st {1}_bin.st -bin {2}".format(
            get_imod_path(), name, binning
        )
        run_shell_command(command,verbose=parameters["slurm_verbose"])
    else:
        shutil.copy2( f"{name}.st", f"{name}_bin.st" )

    tilt_series_size_x, tilt_series_size_y, tilt_series_size_z = get_image_dimensions(
        name + "_bin.st"
    )

    # always redo coarse alignment
    if not 'aretomo' in parameters["tomo_ali_method"].lower():
        logger.info("Doing pre-alignment using IMODs tiltxcorr")

        if parameters["tomo_ali_square"]:
            tapper_size = int(
                min(int(512 / binning), min(tilt_series_size_x, tilt_series_size_y) / 4)
            )
            border_tapper = f"-border {tapper_size},{tapper_size} -taper {tapper_size},{tapper_size} "
            tapper_edge = "-taper 1,1"
        else:
            tapper_size = 0
            border_tapper = ""
            tapper_edge = ""

        tiltxcorr_options = "-tiltfile {0}.rawtlt -binning {1} -rotation {2} -radius1 0.050000 -sigma1 0.030000 -radius2 0.100000 -sigma2 0.030000 -iterate 5 {3}".format(
            name, int(parameters["movie_bin"]), rotation, border_tapper,
        )

        command = "{0}/bin/tiltxcorr -input {1}_bin.st -output {1}_first.prexf {2}".format(
            get_imod_path(), name, tiltxcorr_options
        )
        run_shell_command(command,verbose=parameters["slurm_verbose"])

        # convert to global shifts with respect to middle frame
        command = "{0}/bin/xftoxg -nfit 0 -input {1}_first.prexf -goutput {1}_first.prexg".format(
            get_imod_path(), name
        )
        run_shell_command(command,verbose=parameters["slurm_verbose"])

        # generate aligned stack
        command = "{0}/bin/newstack -linear -xform {1}_first.prexg {1}_bin.st {1}_bin.preali -mode 1 -multadd 1,0 {2}".format(
            get_imod_path(), name, tapper_edge,
        )
        run_shell_command(command,verbose=parameters["slurm_verbose"])

        error = 1
        iteration = 0
        while error > 1e-3 and iteration < int(parameters["movie_iters"]):

            # re-align stack to cumulative frame average
            run_shell_command(
                "{0}/bin/tiltxcorr -input {1}_bin.preali -output {1}_cumulative.prexf {2}".format(
                    get_imod_path(), name, tiltxcorr_options
                ),
                verbose=parameters["slurm_verbose"],
            )

            # convert to global shifts with respect to middle frame
            run_shell_command(
                "{0}/bin/xftoxg -nfit 0 -input {1}_cumulative.prexf -goutput {1}_cumulative.prexg".format(
                    get_imod_path(), name
                ), verbose=parameters["slurm_verbose"]
            )

            # concatenate with latest transform
            run_shell_command(
                "{0}/bin/xfproduct {1}_first.prexg {1}_cumulative.prexg {1}.prexg".format(
                    get_imod_path(), name
                ), verbose=parameters["slurm_verbose"]
            )

            # generate aligned stack with latest alignment parameters
            run_shell_command(
                "{0}/bin/newstack -linear -xform {1}.prexg {1}_bin.st {1}_bin.preali -scale 0,32767 -mode 1 {2}".format(
                    get_imod_path(), name, tapper_edge,
                ),
                verbose=parameters["slurm_verbose"],
            )

            # update current transform
            run_shell_command("mv {0}.prexg {0}_first.prexg".format(name),verbose=parameters["slurm_verbose"])

            newerror = abs(
                np.loadtxt("{0}_cumulative.prexf".format(name))[:, 4:5]
            ).max()
            if iteration > 0 and newerror >= error:
                logger.info(
                    "Error did not decrease, this will be the last iteration"
                )
                error = 0
            else:
                error = newerror

            iteration += 1
            logger.info(
                "Max detected shift change at iteration {0} is {1} ".format(
                    iteration, newerror
                )
            )

        # update current transform
        run_shell_command("mv {0}_first.prexg {0}.prexg".format(name),verbose=parameters["slurm_verbose"])

    actual_pixel = (
        float(parameters["scope_pixel"]) * float(parameters["data_bin"]) * binning
    )

    """
        # Fine alignment using fiducials or patches
        if os.path.isfile( '{0}_tiltalignScript.txt'.format(name) ):

            # use existing alignments
            try:
                os.mkdir('IMOD')
            except:
                pass
            shutil.copy( name + '.rawtlt', 'IMOD/{0}_bin.rawtlt'.format( name ) )
            shutil.copy( name + '.fid.txt', 'IMOD/' + name + '_bin.fid.txt' )

            # re-run tiltalign to reflect changes in fiducial model
            com = '{0}/bin/tiltalign -param {1}_tiltalignScript.txt'.format( os.environ("IMOD_DIR"), name )
            commands.getoutput( com )
        """

    # check if fiducial/patch tracking coordinates exist
    if 'aretomo' in parameters["tomo_ali_method"]:

            logger.info("Align tilt-series using AreTomo2")
 
            binning_tomo = parameters["tomo_rec_binning"]
            thickness = parameters["tomo_rec_thickness"] + parameters['tomo_rec_thickness'] % 2

            specimen_thickness = parameters["tomo_ali_aretomo_zheight"]
            assert (specimen_thickness < thickness), f"Height of specimen ({specimen_thickness}) needs to be smaller than tomogram thickness ({thickness})"

            if "aretomo" not in parameters["tomo_rec_method"]:
                # skip reconstruction if using IMOD
                thickness = 0

            # default using SART for reconstruction
            reconstruct_option = f"-Sart {parameters['tomo_rec_aretomo_sart_iter']} {parameters['tomo_rec_aretomo_sart_num_projs']}"
            if not parameters["tomo_rec_aretomo_sart"]:
                reconstruct_option = "-Wbp 1"

            # correct the tilt offset
            tilt_offset_option = "1" if parameters['tomo_ali_aretomo_measure_tiltoff'] else f"1 {parameters['tomo_ali_aretomo_tiltoff']}"

            # local motion by giving the number of patches
            # patch tracking
            patches_x = parameters["tomo_ali_patches_x"] if "tomo_ali_patches_x" in parameters else 1
            patches_y = parameters["tomo_ali_patches_y"] if "tomo_ali_patches_y" in parameters else 1
            if patches_x + patches_y > 2:
                patches = f" -Patch {patches_x} {patches_y}"
            else:
                patches = ""

            """ Usage: AreTomo2 Tags

            -InMrc
            1. Input MRC file that stores tomo tilt series.

            -OutMrc
            1. Output MRC file that stores the aligned tilt series.

            -AlnFile
            1. Alignment file to be loaded.
            2. It will be applied to the loaded tilt series.

            -AngFile
            1. A single- or multi-column Text file that contains tilt
                angles in the first column.
            2. Both the number and the order of tilt angles must match
                the number and order of projection images in the input
                MRC file.

            -TmpFile
            1. Temporary image file for debugging.

            -LogFile
            1. Log file storing alignment data.

            -TiltRange
            Min and max tilts. By default the header values are used.

            -TiltAxis
            Tilt axis, default header value.

            -AlignZ
            Volume height for alignment, default 256

            -VolZ
            1. Volume z height for reconstrunction. It must be
                greater than 0 to reconstruct a volume.
            2. Default is 0, only aligned tilt series will
                generated.

            -OutBin
            Binning for aligned output tilt series, default 1

            -Gpu
            GPU IDs. Default 0.

            -TiltCor
            1. Correct the offset of tilt angle.
            2. This argument can be followed by two values. The
                first value can be -1, 0, or 1. and the  default is 0,
                indicating the tilt offset is measured for alignment
                only  When the value is 1, the offset is applied to
                reconstion too. When a negative value is given, tilt
                is not measured not applied.
            3. The second value is user provided tilt offset. When it
                is given, the measurement is disabled.

            -ReconRange
            1. It specifies the min and max tilt angles from which
                a 3D volume will be reconstructed. Any tilt image
                whose tilt ange is outside this range is exclueded
                in the reconstruction.

            -PixSize                                                                                                                                                                                                                                                [29/1972]
            1. Pixel size in Angstrom of the input tilt series. It
                is only required for dose weighting. If missing, dose
                weighting will be disabled.

            -Kv
            1. High tension in kV
            2. Required for dose weighting and CTF estimation

            -ImgDose
            1. Dose on sample in each image exposure in e/A2. Note
                this is not accumulated dose. If missing, dose weighting
                will be disabled.

            -Cs
            1. Spherical aberration in mm
            2. Requred only for CTF correction

            $-10s
            1. Amplitude contrast, default 0.07

            -10s
            1. Guess of phase shift and search range in degree.
            2. Only required for CTF estimation and with
            3. Phase plate installed.

            -FlipVol
            1. By giving a non-zero value, the reconstructed
                volume is saved in xyz fashion. The default is
                xzy.
            -FlipInt
            1. Flip the intensity of the volume to make structure white.
                Default 0 means no flipping. Non-zero value flips.
            -Sart
            1. Specify number of SART iterations and number
                of projections per update. The default values
                are 15 and 5, respectively

            -Wbp
            1. By specifying 1, weighted back projection is enabled
                to reconstruct volume.

            -DarkTol
            1. Set tolerance for removing dark images. The range is
                in (0, 1). The default value is 0.7. The higher value is
                more restrictive.

            -TiltScheme
            1. This option is used to determine the sequence each
                tilt image is acquired. This sequence is needed for the
                determination of accumulated dose on sample. If this
                option is missing, dose weighting will be disabled.
            2. Three parameters are needed. This first one is the
                starting angle. The second, tilt step, positive or
                negative,  indicates tilting direction direction
                after the starting angle. The third is 1, 2, or 3,
                corresponding
            to single-branch, two-branch, or Hagen
                scheme of data collection, respectively.

            -OutXF
            1. When set by giving no-zero value, IMOD compatible
                XF file will be generated.

            -OutImod
            1. It generates the Imod files needed by Relion4 or Warp
                for subtomogram averaging. These files are saved in the
                subfolder named after the output MRC file name.
            2. 0: default, do not generate any IMod files.
            3. 1: generate IMod files needed for Relion 4.
            4. 2: generate IMod files needed for WARP.
            5. 3: generate IMod files when the aligned tilt series
                    is used as the input for Relion 4 or WARP.

            -Align
            1. Skip alignment when followed by 0. This option is
                used when the input MRC file is an aligned tilt series.
                The default value is 1.

            -CropVol
            1. Crop the reconstructed volume to the specified sizes
                in x and y directions.
            2. Size x is the length perpendicular to tilt axis and size
                y is the length along the tilt axis.
            3. This option is only enabled when -RoiFile is enabled.

            -Bft
            1. B-factors for low-pass filter used in the cross
                correlation. The first value is used for global
                measurement. The second for the local measurement.

            -IntpCor
            1. When enabled, the correction for information loss due
                to linear interpolation will be perform. The default
                setting value 1 enables the correction.

            """

            command = f"{get_aretomo_path()} \
-InMrc {name}.mrc \
-OutMrc {name}_aretomo.rec \
-AngFile {name}.rawtlt \
-VolZ {thickness} \
-OutBin {binning_tomo} \
-TiltAxis {rotation} \
-DarkTol {parameters['tomo_ali_aretomo_dark_tol']} \
-AlignZ {specimen_thickness} \
{reconstruct_option} \
-TiltCor {tilt_offset_option} \
-OutImod 2 {patches} \
-Gpu {get_gpu_id()}"
            [ output, error ] = run_shell_command(command, verbose=parameters["slurm_verbose"])

            # save output
            try:
                shutil.copy2(f"{name}_Imod/{name}_st.xf", f"{name}.xf")
                shutil.copy2(f"{name}_Imod/{name}_st.tlt", f"{name}.tlt")
                os.symlink(f"{name}_aretomo.rec", f"{name}.rec")
            except:
                if 'Error: GPU' in output:
                    if not parameters['slurm_verbose']:
                        logger.error(output)
                    logger.error('A GPU must be available for AreTomo2 to run')
                raise Exception("AreTomo2 failed to run")
            return
    else:

        # alignment using gold fiducials
        if "tomo_ali_fiducial" in parameters and parameters["tomo_ali_fiducial"] > 0 and "tomo_ali_method" in parameters and parameters["tomo_ali_method"] == "imod_gold":

            # Alignment with RAPTOR
            logger.info("Align tilt-series using gold fiducials (IMOD/RAPTOR)")

            gold_diameter = int(round(parameters["tomo_ali_fiducial"] / actual_pixel))

            shutil.copy2("%s.rawtlt" % name, "%s_bin.rawtlt" % name)

            # fiducial based alignment with RAPTOR ( -minNeigh 10 -maxDist 200 )
            load_imod_cmd = imod_load_command()
            # command = "{0}; {1}/RAPTOR -seed 96 -execPath {1} -path . -input {2}_bin.preali -output . -diameter {3} -markers -1 -verb 1".format(
            #     load_imod_cmd,
            #     os.environ["PYP_DIR"] + "/TOMO/RAPTOR3.0/bin",
            #     name,
            #     gold_diameter,
            # )

            markers = parameters["tomo_ali_fiducial_number"]
            if markers > 0:
                fid_markers = f"-markers {markers} "
            else:
                fid_markers = ""

            command = "{0} export PATH=$PATH:{1}; {1}/RAPTOR -seed 96 -execPath {1} -path . -input {2}_bin.preali -output . -diameter {3} {4}-verb 1".format(
                load_imod_cmd, get_imod_path() + "/bin", name, gold_diameter, fid_markers
            )
            run_shell_command(command,verbose=parameters["slurm_verbose"])

            # try to recover from failure by re-running RAPTOR using fixed number of fiducials
            if not os.path.exists("IMOD/{0}_bin.xf".format(name)):
                # command = "{0}; {1}/RAPTOR -seed 96 -execPath {1} -path . -input {2}_bin.preali -output . -diameter {3} -markers 20 -verb 1".format(
                #     load_imod_cmd,
                #     os.environ["PYP_DIR"] + "/TOMO/RAPTOR3.0/bin",
                #     name,
                #     gold_diameter,
                # )
                command = "{0} export PATH=$PATH:{1}; {1}/RAPTOR -seed 96 -execPath {1} -path . -input {2}_bin.preali -output . -diameter {3} -markers 30 -verb 1".format(
                    load_imod_cmd, get_imod_path() + "/bin", name, gold_diameter
                )
                run_shell_command(command,verbose=parameters["slurm_verbose"])

            # if second try also failed, switch back to patch tracking
            if not os.path.exists("IMOD/{0}_bin_tiltalignScript.txt".format(name)):

                # try patch tracking instead
                parameters["tomo_ali_fiducial"] = 0

            else:

                rotation_angle = [
                    line
                    for line in open("IMOD/{0}_bin_tiltalignScript.txt".format(name))
                    if "RotationAngle" in line
                ][0].split()[1]
                include_list = [
                    line
                    for line in open("IMOD/{0}_bin_tiltalignScript.txt".format(name))
                    if "IncludeList" in line
                ][0].split()[1]

                # re-run tiltalign
                command = """
%s/bin/tiltalign -StandardInput << EOF
ModelFile       ./IMOD/%s_bin.fid.txt
ImagesAreBinned 1
OutputModelFile %s.3dmod
OutputResidualFile      ./IMOD/%s_bin.resid
OutputFidXYZFile        ./IMOD/%s_bin.fid.xyz
OutputTiltFile  ./IMOD/%s_bin.tlt
OutputXAxisTiltFile     ./IMOD/%s_bin.xtilt
OutputTransformFile     ./IMOD/%s_bin.xf
OutputZFactorFile       ./IMOD/%s.zfac
RotationAngle   %s
IncludeList     %s
TiltFile        ./IMOD/%s_bin.rawtlt
AngleOffset     0.0
RotOption       1
RotDefaultGrouping      5
TiltOption      5
TiltDefaultGrouping     5
MagReferenceView        1
MagOption       0
MagDefaultGrouping      4
XStretchOption  0
SkewOption      0
XStretchDefaultGrouping 7
SkewDefaultGrouping     11
BeamTiltOption  0
ResidualReportCriterion 3.0
SurfacesToAnalyze       0
MetroFactor     0.25
MaximumCycles   1000
KFactorScaling  1.0
AxisZShift      0.0
ShiftZFromOriginal      1
LocalAlignments 0
OutputLocalFile %slocal.xf
TargetPatchSizeXandY    700,700
MinSizeOrOverlapXandY   0.5,0.5
MinFidsTotalAndEachSurface      8,3
FixXYZCoordinates       0
LocalOutputOptions      1,0,1
LocalRotOption  3
LocalRotDefaultGrouping 6
LocalTiltOption 5
LocalTiltDefaultGrouping        6
LocalMagReferenceView   1
LocalMagOption  3
LocalMagDefaultGrouping 7
LocalXStretchOption     0
LocalXStretchDefaultGrouping    7
LocalSkewOption 0
LocalSkewDefaultGrouping        11
RobustFitting
EOF
""" % (
                    get_imod_path(),
                    name,
                    name,
                    name,
                    name,
                    name,
                    name,
                    name,
                    name,
                    rotation_angle,
                    include_list,
                    name,
                    name,
                )
                # [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
                # print output

                shutil.copy2("align/%s_bin_RAPTOR.log" % name, "%s_RAPTOR.log" % name)
                shutil.copy2(
                    "IMOD/%s_bin_tiltalignScript.txt" % name,
                    "%s_tiltalignScript.txt" % name,
                )
                shutil.copy2("IMOD/%s_bin.fid.txt" % name, "%s.fid.txt" % name)

                """
                    # create aligned stack
                    com='{0}/bin/newstack -linear -xform IMOD/{1}_bin.xf {1}_bin.preali {1}_bin.ali -taper 1,1'.format(get_imod_path(),name)
                    print com
                    print commands.getoutput(com)
                    """

        # if not using fiducials, or if RAPTOR failed
        if parameters["tomo_ali_fiducial"] == 0 or parameters["tomo_ali_method"] == "imod_patch":
            # Fiducial-less alignment

            logger.info(
                "Align tilt-series using patch tracking (IMOD)"
            )

            max_size_x = parameters.get("tomo_ali_patches_size_x")
            max_size_y = parameters.get("tomo_ali_patches_size_y")
            if max_size_x == None or max_size_x == 0:
                max_size_x = min(
                    tilt_series_size_x - 2 * tapper_size,1280
                )
            if max_size_y == None or max_size_y == 0:
                max_size_y = min(
                    tilt_series_size_y - 2 * tapper_size, 1280
                )

            # patch tracking
            command = "{0}/bin/tiltxcorr -input {1}_bin.preali -output {1}_patches.fid {2} -size {3},{4} -number {5},{6}".format(
                get_imod_path(),
                name,
                tiltxcorr_options,
                max_size_x,
                max_size_y,
                parameters["tomo_ali_patches_x"],
                parameters["tomo_ali_patches_y"],
            )
            run_shell_command(command, verbose=parameters["slurm_verbose"])

            # Chop up contours
            command = "{0}/bin/imodchopconts -input {1}_patches.fid -output {1}.fid -overlap 4 -surfaces 1".format(
                get_imod_path(), name
            )
            run_shell_command(command)

            shutil.copy2("%s.fid" % name, "%s.fid.txt" % name)

    # make alignment output between different variants uniform
    if parameters["tomo_ali_method"] == "imod_gold":

        # alignment using gold fiducials

        # move files around if using existing fiducial model
        if not os.path.exists("IMOD"):
            os.mkdir("IMOD")
            shutil.copy("%s.rawtlt" % name, "IMOD/%s_bin.rawtlt" % name)
            shutil.copy(name + ".fid.txt", "IMOD/{0}_bin.fid.txt".format(name))

        # re-run tiltalign
        com = "{0}/bin/tiltalign -param {1}_tiltalignScript.txt".format(
            get_imod_path(), name
        )
        run_shell_command(com,verbose=parameters["slurm_verbose"])

        # combine RAPTOR output with prealign step and unbinning
        if os.path.isfile("IMOD/%s_bin.xf" % name):

            # rename files for proper saving
            shutil.copy2("IMOD/%s_bin.xf" % name, "%s_bin.xf" % name)
            shutil.copy2("IMOD/%s_bin.tlt" % name, "%s.tlt" % name)

        else:

            # RAPTOR failed, make reported tilt-axis coincide with the Y-axis

            logger.info(
                "ERROR - RAPTOR failed to run on {0}. Switching back to pre-alignment results.\n".format(
                    name
                )
            )

            rot = vtk.rotation_matrix(np.radians(rotation), [0, 0, 1])
            rot2D = np.array(
                [rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1], 0.0, 0.0], ndmin=2
            )
            np.savetxt("{0}_bin.xf".format(name), rot2D, fmt="%13.7f")
            shutil.copy2("%s.rawtlt" % name, "%s.tlt" % name)

    elif parameters["tomo_ali_method"] == "imod_patch":

        # patch-based alignment

        # run tiltalign
        command = """
%s/bin/tiltalign -StandardInput << EOF
ModelFile       %s.fid
ImageFile       %s_bin.preali
ImagesAreBinned 1
OutputModelFile %s.3dmod
OutputResidualFile      %s.resid
OutputFidXYZFile        %sfid.xyz
OutputTiltFile  %s.tlt
OutputXAxisTiltFile     %s.xtilt
OutputTransformFile     %s_bin.xf
RotationAngle   %f
TiltFile        %s.rawtlt
AngleOffset     0.0
RotOption       1
RotDefaultGrouping      5
TiltOption      0
TiltDefaultGrouping     5
MagReferenceView        1
MagOption       0
MagDefaultGrouping      4
XStretchOption  0
SkewOption      0
XStretchDefaultGrouping 7
SkewDefaultGrouping     11
BeamTiltOption  0
ResidualReportCriterion 3.0
SurfacesToAnalyze       1
MetroFactor     0.25
MaximumCycles   1000
KFactorScaling  1.0
AxisZShift      0.0
ShiftZFromOriginal      1
LocalAlignments 0
OutputLocalFile %slocal.xf
TargetPatchSizeXandY    700,700
MinSizeOrOverlapXandY   0.5,0.5
MinFidsTotalAndEachSurface      8,3
FixXYZCoordinates       0
LocalOutputOptions      1,0,1
LocalRotOption  3
LocalRotDefaultGrouping 6
LocalTiltOption 5
LocalTiltDefaultGrouping        6
LocalMagReferenceView   1
LocalMagOption  3
LocalMagDefaultGrouping 7
LocalXStretchOption     0
LocalXStretchDefaultGrouping    7
LocalSkewOption 0
LocalSkewDefaultGrouping        11
RobustFitting
WeightWholeTracks
EOF
""" % (
            get_imod_path(),
            name,
            name,
            name,
            name,
            name,
            name,
            name,
            name,
            rotation,
            name,
            name,
        )
        run_shell_command(command,verbose=parameters["slurm_verbose"])

        shutil.copy2("%s.fid" % name, "%s.fid.txt" % name)

    # compose pre alignment with fiducial/patch based alignments
    command = "{0}/bin/xfproduct {1}.prexg {1}_bin.xf {1}.xf -scale {2},{2}".format(
        get_imod_path(), name, binning
    )
    run_shell_command(command,verbose=parameters["slurm_verbose"])

    # create aligned fiducial model
    command = "{0}/bin/imodtrans -2 {1}_bin.xf {1}.fid.txt {1}_aligned.fid".format(
        get_imod_path(), name
    )
    run_shell_command(command,verbose=parameters["slurm_verbose"])


def check_parfile_match_allboxes(par_file: str, allboxes_file: str):
    """check_parfile_match_allboxes 

        Check existence of parfile (alignment metadata) and allboxes (coordinates) and also check if they match

    Parameters
    ----------
    par_file : str
        Input parfile 
    allboxes_file : str
        Input coordinate file
    """
    assert (Path(par_file).exists()), f"{par_file} does not exist"
    assert (Path(allboxes_file).exists()), f"{allboxes_file} does not exist" 

    pardata = Parameters.from_file(par_file).data
    allboxes = np.loadtxt(allboxes_file, ndmin=2)
    # add more info about how to avoid this error
    assert (pardata.shape[0] == allboxes.shape[0]), f"Number of particles in parfile and metadata do not match: {pardata.shape[0]} != {allboxes.shape[0]}. You may have a different set of particles than that used during pre-processing or you may have cleaned (modified) your particles after refinement while still using old particle coordinates."


