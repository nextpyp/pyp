import math
import multiprocessing
import os
import shutil
import sys

import numpy as np
import pandas as pd
import scipy
from pathlib import Path

from pyp import merge
from pyp.analysis import statistics, plot, geometry
from pyp.inout.image import mrc
from pyp.inout.metadata import frealign_parfile, pyp_metadata 
from pyp.refine.frealign import frealign
from pyp.system import project_params, user_comm
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path, timer, symlink_relative
from pyp.inout.utils.pyp_edit_box_files import read_boxx_file_async, write_boxx_file_async

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def per_frame_scoring(
    parameters, name, current_path, allboxes, allparxs, particle_filenames
):
    """In particle frame refinement step, score the Gaussian averaged particle frames."""
    # follow settings in align.align_stack_super
    if os.path.exists(
        os.path.split(parameters["refine_parfile"])[0] + "/../frealign.config"
    ):
        fparameters = project_params.load_fyp_parameters(
            os.path.split(parameters["class_par"])[0] + "/../"
        )
        maxiter = int(fparameters["maxiter"])
        low_res = float(project_params.param(fparameters["rlref"], maxiter))
        high_res_refi = high_res_eval = float(
            project_params.param(fparameters["rhref"], maxiter)
        )
        # logger.info("high_res_refine", float( project_params.param( fparameters['rhref'], maxiter ) ))
        metric = (
            project_params.param(fparameters["metric"], maxiter)
            + " -fboost "
            + project_params.param(fparameters["fboost"], maxiter)
            + " -maskth "
            + project_params.param(fparameters["maskth"], maxiter)
        )
        # metric_weights = metric
        # print 'Retrieving FREALIGN compatible FREALIGN parameters: rhlref = %.2f, rhref = %.2f, metric = %s' % ( low_res, high_res_refi, metric )
    else:
        logger.warning("Could not find FREALIGN parameters to insure consistency")

    # temporarily disable PYP_SCRATCH
    pyp_scratch_bk = os.environ["PYP_SCRATCH"]
    os.environ["PYP_SCRATCH"] = ""

    if not os.path.exists("../log"):
        os.makedirs("../log")

    particle_stacks = [
        os.path.join("frealign_" + particle_filename, particle_filename + "_stack.mrc")
        for particle_filename in particle_filenames
    ]
    particle_parfiles = [
        particle_filename + "_r01_02.par" for particle_filename in particle_filenames
    ]

    # merge the frame stacks
    mrc.merge(particle_stacks, name + "_frames_stack.mrc")
    # merge the par files
    frealign_parfile.Parameters.merge_parameters(
        particle_parfiles,
        name + "_frames_r01_02.par",
        metric=parameters["refine_metric"],
        parx=True,
    )

    # save the averaged as backup
    shutil.copy2(name + "_frames_stack.mrc", name + "_frames_stack.mrc.real")

    # write out the stack file and par file into a txt for later processing
    with open("../stacks.txt", "a") as f:
        f.write(os.path.join(name, name + "_frames_stack.mrc\n"))
    with open("../pars.txt", "a") as f:
        f.write(os.path.join(name, name + "_frames_r01_02.par\n"))
    # if the project directory file is not written
    with open("../project_dir.txt", "w") as f:
        f.write(str(current_path))

    # find film number for this micrograph to figure out particle alignments
    # TODO: write out to func
    try:
        with open(os.path.join(current_path, parameters["data_set"] + ".films")) as x:
            series = [
                num
                for num, line in enumerate(x, 1)
                if "{}".format(name.replace("_r01", "")) == line.strip()
            ][0] - 1
        # write the overall order of the files
        with open("../ordering.txt", "a") as f:
            f.write("{}\n".format(series))
    except:
        sys.stderr.write("ERROR - Cannot find film number for " + name)
        sys.exit()

    # score the frames
    # call FREALIGN directly to improve performance
    mp = parameters.copy()
    # fp = fparameters.copy()
    mp["refine_mode"] = "1"
    mp["refine_mask"] = "0,0,0,0,0"
    # mp["refine_rlref"] = "{}".format(low_res)
    # mp["refine_rhref"] = "{}".format(high_res_refi)
    mp["refine_dataset"] = name + "_frames"

    # frames = len(np.unique(film_arr[:, scanor_col]))
    # TODO: write out to func
    frames = len(
        [
            boxes
            for boxes, line in zip(allboxes, allparxs[0])
            if float(line.split()[15]) == 0
        ]
    )

    frame_weights_width = int(
        math.floor(frames * 0.4)
    )  # width of gaussian used for frame weighting
    if frame_weights_width % 2 == 0:
        frame_weights_width += 1
    frame_weights_step = False  # use step-like weights for frame weighting

    # build weights for frame averaging
    all_weights = np.zeros([frames, frames])
    for i in range(frames):
        weights = np.exp(-pow((np.arange(frames) - float(i)), 2) / frame_weights_width)
        # apply hard threshold if using simple running averages
        if frame_weights_step:
            weights = np.where(weights > 0.5, 1, 0)
        all_weights[i, :] = weights / weights.mean() / frames

    # weight each particle stack
    blurred_stack = []
    for particle_stack in particle_stacks:
        merge.weight_stack(particle_stack, particle_stack + ".blur", all_weights)
        blurred_stack.append(particle_stack + ".blur")
    mrc.merge(blurred_stack, name + "_frames_stack.mrc")

    # necessary parfiles for scoring/recon
    os.symlink(name + "_frames_r01_02.par", name + "_frames_r01_02_used.par")
    os.symlink(name + "_frames_r01_02.par", name + "_r01_02.par")
    os.symlink(name + "_r01_02.par", name + "_r01_02_used.par")
    shutil.copy2(
        os.path.join(os.getcwd(), name + "_frames_stack.mrc"),
        "../" + name + "_frames_stack.mrc",
    )

    # score the gaussian blurred frames
    header = mrc.readHeaderFromFile(name + "_frames_stack.mrc")
    total_frames = header["nz"]
    score_metric = "cc3m -fboost T"
    command = frealign.mrefine_version(
        mp, 1, total_frames, 2, 1, name + "_r01_02", "", "log.txt", "", score_metric,
    )
    run_shell_command(command)

    # move back the original stack
    shutil.copy(name + "_frames_stack.mrc.real", name + "_frames_stack.mrc")

    # copy back the scored parfile
    # use the   LOGP      SIGMA   SCORE  CHANGE from score par -- cols 12 - 15
    original_par = frealign_parfile.Parameters.from_file(
        name + "_frames_r01_02.par"
    ).data
    # scored_par = frealign_parfile.Parameters.from_file(name + "_r01_02.par_").data
    scored_par = frealign_parfile.Parameters.from_file(name + "_r01_02.par").data
    original_par[:, 12:16] = scored_par[:, 12:16]
    frealign_parfile.Parameters.write_parameter_file(
        name + "_frames_r01_02.par", original_par, parx=True
    )
    # os.rename(name + '_r01_02.par_', name + '_frames_r01_02.par')

    # save the parameters for array job
    mp = parameters.copy()
    # fp = fparameters.copy()
    # mp["refine_rlref"] = "{}".format(low_res)
    # mp["refine_rhref"] = "{}".format(high_res_refi)
    mp["refine_dataset"] = name + "_frames"

    # save fp and mp into the main slurm job folder
    logger.info("saving mp and fp parameter files")
    project_params.save_pyp_parameters(mp, "..")
    # project_params.save_fyp_parameters(fp, "..")

    os.remove("../" + name + "_frames_stack.mrc")

    # set PYP_SCRATCH back to regular
    os.environ["PYP_SCRATCH"] = pyp_scratch_bk


def assign_angular_defocus_groups(
    parfile: str, angles: int, defocuses: int, frealignx: bool = False
):
    """Divide particles in parameter file into a discrete number of angular and defocus groups.

    Parameters
    ----------
    parfile : str
        Frealign .par file.
    angles : int
        Number of groups used to partition the data
    defocuses : int
        Number of groups used to partition the data
    frealignx : bool, optional
        Specify if .par parameter file is in frealignx format, by default False

    Returns
    -------
    par_obj : Parameters
        Object representing par file data
    angular groups : numpy.ndarray
        Result of angular clustering
    defocus groups : numpy.ndarray
        Result defous clustering.
    """


    
    # load .parx file
    if os.path.isfile(parfile):

        par_obj = frealign_parfile.Parameters.from_file(parfile)
        input = par_obj.data
        
        if "_used.par" in parfile:
            if frealignx:
                input = input[input[:, 12] > 0]
            else:
                input = input[input[:, 11] > 0]
            par_obj.data = input

    else:
        logger.error("{} not found.".format(parfile))
        sys.exit(0)

    angular_group = np.floor(np.mod(input[:, 2], 180) * angles / 180)
    if input.shape[0] > 0:
        mind, maxd = (
            int(math.floor(input[:, 8].min())),
            int(math.ceil(input[:, 8].max())),
        )
    else:
        mind = maxd = 0
    if maxd == mind:
        defocus_group = np.zeros(angular_group.shape)
    else:
        defocus_group = np.round((input[:, 8] - mind) / (maxd - mind) * (defocuses - 1))

    # return input, angular_group, defocus_group
    return par_obj, angular_group, defocus_group


def generate_cluster_stacks(inputstack, parfile, angles=25, defocuses=25):
    par_obj, angular_group, defocus_group = assign_angular_defocus_groups(
        parfile, angles, defocuses
    )
    input = par_obj.data

    # create new stacks
    pool = multiprocessing.Pool()
    for g in range(angles):
        for f in range(defocuses):
            cluster = input[np.logical_and(angular_group == g, defocus_group == f)]
            if cluster.shape[0] > 0:
                cluster = cluster[cluster[:, 11].argsort()]
                indexes = (cluster[:, 0] - 1).astype("int").tolist()
                outputstack = "{0}_{1}_{2}_stack.mrc".format(
                    os.path.splitext(inputstack)[0], g, f
                )
                pool.apply_async(
                    mrc.extract_slices, args=(inputstack, indexes, outputstack)
                )
    pool.close()

    # Wait for all processes to complete
    pool.join()


def shape_phase_residuals(
    inputparfile,
    angles,
    defocuses,
    threshold,
    mindefocus,
    maxdefocus,
    firstframe,
    lastframe,
    mintilt,
    maxtilt,
    minazh,
    maxazh,
    minscore,
    maxscore,
    binning,
    reverse,
    consistency,
    scores,
    frealignx,
    odd,
    even,
    outputparfile,
    png_file
):


    if scores and not frealignx:
        field = 14
        occ = 11
    elif frealignx:
        field = 15
        occ = 12
    else:
        field = 11

    # read interesting part of input file
    par_obj, angular_group, defocus_group = assign_angular_defocus_groups(
        inputparfile, angles, defocuses, frealignx
    )
    input = par_obj.data

    # figure out tomo or spr by check tilt angles
    tltangle = field + 3
    ptlindex = field + 2
    
    if np.any(input[:, tltangle] !=0 ):
        is_tomo = True
    else:
        is_tomo = False

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # additional sorting if matches available
    name = inputparfile[:-4]
    fmatch_stack = "../maps/{0}_match_unsorted.mrc".format(name)
    metric_weights = np.ones(input.shape[0])
    if os.path.exists(fmatch_stack):
        msize = int(mrc.readHeaderFromFile(fmatch_stack)["nx"]) / 2
        y, x = np.ogrid[-msize:msize, -msize:msize]
        mask = np.where(x ** 2 + y ** 2 <= msize ** 2, 1, 0)
    else:
        msize = 0
        mask = np.array([])

    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    results = manager.Queue()

    # determine per-cluster threshold
    thresholds = np.empty([angles, defocuses])
    thresholds[:] = np.nan
    min_scores = np.empty([angles, defocuses])
    min_scores[:] = np.nan
    max_scores = np.empty([angles, defocuses])
    max_scores[:] = np.nan
    for g in range(angles):
        for f in range(defocuses):
            # get all images in present cluster
            cluster = np.logical_and(angular_group == g, defocus_group == f)
            size = 1

            # make sure we have enough points for computing statistics
            while (
                np.extract(cluster == 1, input[:, field]).size < 100
                and input.shape[0] > 100
            ):

                cluster = np.logical_and(
                    np.logical_and(
                        angular_group >= g - size, angular_group <= g + size
                    ),
                    np.logical_and(
                        defocus_group >= f - size, defocus_group <= f + size
                    ),
                )
                size += 1

            if cluster.size > 0:
                # find cluster threshold using either percentage of size or absolute number of images

                if threshold == 0:
                    prs = np.extract(cluster == 1, input[:, field])
                    optimal_threshold = 1.075 * statistics.optimal_threshold(samples=prs, criteria="optimal")
                    if prs.size > 1:
                        mythreshold = optimal_threshold
                        cutoff = (
                            1.0 - 1.0 * np.argmin(np.fabs(prs - mythreshold)) / prs.size
                        )
                        # logger.info('Bi-modal distributions detected with means: {0}, {1}'.format( gmix.means_[0][0], gmix.means_[1][0] ))
                        logger.info(f'Using optimal threshold from bimodal distribution = {mythreshold:.2f}')
                        if prs.size > 20:
                            thresholds[g, f] = mythreshold
                        else:
                            logger.warning(
                                "Not enough points for estimating statistics %d %d %d",
                                g,
                                f,
                                prs.size,
                            )

                elif threshold <= 1:
                    # cluster = input[ np.logical_and( angular_group == g, defocus_group == f ) ]
                    if scores or frealignx:
                        # thresholds[g,f] = cluster[ cluster[:,field].argsort() ][ int( (cluster.shape[0]-1) * (1-threshold) ), field ]
                        if is_tomo:
                            bool_array = np.full(input.shape[0], False, dtype=bool)
                            bool_array[cluster] = True
                            take_values = np.logical_and(bool_array, np.abs(input[:, tltangle]) < 10)
                            used_array = input[take_values]
                            scores_used = used_array[:, [field, ptlindex]]
                            take_mean = []
                            for i in np.unique(scores_used[:, 1]):
                                take_mean.append(np.mean(scores_used[:, 0], where=scores_used[:,1]==i))

                            meanscore = np.array(take_mean)
                            thresholds[g, f] = np.sort(meanscore)[
                                int((meanscore.shape[0] - 1) * (1 - threshold))
                            ]
                            logger.info(f"Minimum score used for reconstruction = {thresholds[g, f]:.2f}")
                        else:
                            thresholds[g, f] = np.sort(input[cluster, field])[
                                int((cluster.shape[0] - 1) * (1 - threshold))
                            ]
                    else:
                        # thresholds[g,f] = cluster[ cluster[:,field].argsort() ][ int( (cluster.shape[0]-1) * threshold ), field ]
                        thresholds[g, f] = np.sort(input[cluster, field])[
                            int((cluster.shape[0] - 1) * threshold)
                        ]
                else:
                    # cluster = input[ np.logical_and( angular_group == g, defocus_group == f ) ]
                    if scores or frealignx:
                        thresholds[g, f] = cluster[cluster[:, field].argsort()][
                            min(cluster.shape[0] - threshold, cluster.shape[0]) - 1,
                            field,
                        ]
                    else:
                        thresholds[g, f] = cluster[cluster[:, field].argsort()][
                            min(threshold, cluster.shape[0]) - 1, field
                        ]

                prs = np.extract(cluster == 1, input[:, field])
                if minscore < 1:
                    min_scores[g, f] = prs.min() + minscore * (prs.max() - prs.min())
                else:
                    min_scores[g, f] = minscore

                if maxscore <= 1:
                    max_scores[g, f] = prs.max() - (1 - maxscore) * (
                        prs.max() - prs.min()
                    )
                else:
                    max_scores[g, f] = maxscore

                # print thresholds[g,f], min_scores[g,f], max_scores[g,f]

            else:
                logger.warning("No points in angular/defocus group.")

    pool.close()
    pool.join()

    # Collate periodogram averages
    bad_particles = 0
    total_particles = 0
    while results.empty() == False:
        current = results.get()
        thresholds[current[0], current[1]] = current[2]
        metric_weights[current[3]] = current[4]
        if current[0] + current[1] == 0:
            logger.info(
                "Processing group (%d,%d) containing %d particles and eliminating %d",
                current[0],
                current[1],
                len(current[3]),
                np.where(current[4] == 0, 1, 0).sum(),
            )
        bad_particles += np.where(current[4] == 0, 1, 0).sum()
        total_particles += len(current[3])

    from scipy.ndimage.filters import gaussian_filter

    thresholds = gaussian_filter(thresholds, sigma=1)

    if angles + defocuses > 2:
        plt.clf()
        cax = plt.imshow(thresholds, interpolation="nearest", cmap=cm.jet)
        plt.title("Thresholds per orientation\n and defocus group")
        plt.xlabel("Defocus Group")
        plt.ylabel("Orientation Group")
        plt.colorbar(cax, ticks=[np.nanmin(thresholds), np.nanmax(thresholds)])
        plt.savefig("../maps/%s_thresholds.png" % os.path.splitext(inputparfile)[0])

    # apply new PR threshold
    for g in range(angles):
        for f in range(defocuses):
            # if threshold != 1:
            if scores or frealignx:
                # input[:,field] = np.where( np.logical_and( np.logical_and( angular_group == g, defocus_group == f ), input[:,field] < thresholds[g,f] ), np.nan, input[:,field] )
                # input[:,occ] = np.where( np.logical_and( np.logical_and( angular_group == g, defocus_group == f ), input[:,field] < thresholds[g,f] ), 0, input[:,occ] )
                if is_tomo and thresholds[g, f] > 0:
                    input_group = input[np.logical_and(angular_group == g, defocus_group == f)]
                    ptl_index = np.unique(input_group[:, ptlindex])

                    for i in ptl_index:
                        ptl_field_array = input_group[input_group[:, ptlindex] == i, field]
                        tltangle_array = input_group[input_group[:, ptlindex] == i, tltangle]
                        meanfrom = ptl_field_array[np.abs(tltangle_array) < 10]

                        input[input[:, ptlindex] == i, occ] = np.where(
                        np.array( [ 0 if meanfrom.size == 0 else np.mean(meanfrom) ] * ptl_field_array.shape[0] ) < thresholds[g, f],
                        0,
                        input[input[:, ptlindex] == i, occ],
                        )
                else:
                    input[:, occ] = np.where(
                        np.logical_and(
                            np.logical_and(angular_group == g, defocus_group == f),
                            np.logical_or(
                                np.logical_or(
                                    input[:, field] < thresholds[g, f],
                                    input[:, field] < min_scores[g, f],
                                ),
                                input[:, field] > max_scores[g, f],
                            ),
                        ),
                        0,
                        input[:, occ],
                    )
                number = input[input[:, occ]==0].shape[0]
                logger.info(f"Number of particles with zero occupancy = {number:,} out of {input.shape[0]:,} ({number/input.shape[0]*100:.2f}%)")

    if os.path.exists(fmatch_stack):
        logger.info(
            "Removing %d bad particles by distance sorting (%d)",
            bad_particles,
            np.where(metric_weights == 0, 1, 0).sum(),
        )
        logger.info("Total particles = %d", total_particles)
        input[:, occ] = np.where(metric_weights == 0, 0, input[:, occ])
        fmatch_stack_removed = "../maps/%s_match_removed.mrc" % (
            os.path.splitext(inputparfile)[0]
        )
        mrc.extract_slices(
            fmatch_stack,
            np.nonzero(input[:, occ] == 0)[0].tolist(),
            fmatch_stack_removed,
        )

    # ignore if defocus outside permissible range
    if scores or frealignx:
        input[:, occ] = np.where(
            np.logical_or(input[:, 8] < mindefocus, input[:, 8] > maxdefocus),
            0,
            input[:, occ],
        )
    """
    else:
        input[:, field] = np.where(
            np.logical_or(input[:, 8] < mindefocus, input[:, 8] > maxdefocus),
            np.nan,
            input[:, field],
        )
    """
    # shape accorging to assigned top/side view orientations using mintilt and maxtilt values
    if maxazh < 180 or minazh > 0:
        if scores or frealignx:
            input[:, occ] = np.where(
                np.logical_or(
                    np.mod(input[:, 2], 180) < minazh, np.mod(input[:, 2], 180) > maxazh
                ),
                0,
                input[:, occ],
            )
        """
        else:
            input[:, field] = np.where(
                np.logical_or(
                    np.mod(input[:, 2], 180) < minazh, np.mod(input[:, 2], 180) > maxazh
                ),
                np.nan,
                input[:, field],
            )
        """
    # if extended .parx format

    if (scores and input.shape[1] > 16) or (frealignx and input.shape[1] > 17):

        # shape based on exposure sequence
        if lastframe > -1:
            if scores and not frealignx:
                input[:, occ] = np.where(
                    np.logical_or(input[:, 19] < firstframe, input[:, 19] > lastframe),
                    0,
                    input[:, occ],
                )
            elif frealignx:
                input[:, occ] = np.where(
                    np.logical_or(input[:, 20] < firstframe, input[:, 20] > lastframe),
                    0,
                    input[:, occ],
                )
            """
            else:
                input[:, field] = np.where(
                    np.logical_or(input[:, 19] < firstframe, input[:, 19] > lastframe),
                    np.nan,
                    input[:, field],
                )
            """
        else:
            if scores and not frealignx:
                input[:, occ] = np.where(input[:, 19] < firstframe, 0, input[:, occ])
            elif frealignx:
                input[:, occ] = np.where(input[:, 20] < firstframe, 0, input[:, occ])
            """
            else:
                input[:, field] = np.where(
                    input[:, 19] < firstframe, np.nan, input[:, field]
                )
            """
        # shape based on tilt-angle
        if scores and not frealignx:
            input[:, occ] = np.where(
                np.logical_or(input[:, 17] < mintilt, input[:, 17] > maxtilt),
                0,
                input[:, occ],
            )
        elif frealignx:
            input[:, occ] = np.where(
                np.logical_or(input[:, 18] < mintilt, input[:, 18] > maxtilt),
                0,
                input[:, occ],
            )
        """
       else:
            input[:, field] = np.where(
                np.logical_or(input[:, 17] < mintilt, input[:, 17] > maxtilt),
                np.nan,
                input[:, field],
            )
        """
    # revert phase residual polarity so that lowest PR become highest and viceversa
    if reverse:
        min_pr = np.extract(np.isfinite(input[:, field]), input[:, field]).min()
        max_pr = np.extract(np.isfinite(input[:, field]), input[:, field]).max()
        input[:, field] = np.where(
            np.isfinite(input[:, field]),
            max_pr - input[:, field] + min_pr,
            input[:, field],
        )

    # apply binning to image shifts (FREALIGN 9 measures shifts in Angstroms so we don't need this)
    if not scores:
        input[:, 4:6] *= binning

    ## particle selection based on consistency of angles/shifts determination
    if consistency:

        prevparfile = "%s%02d.par" % (
            inputparfile.split(".")[0][:-2],
            int(inputparfile.split(".")[0][-2:]) - 1,
        )

        if os.path.isfile(prevparfile):

            # read parameters from previous iteration
            previous = np.array(
                [
                    line.split()
                    for line in open(prevparfile)
                    if not line.startswith("C")
                ],
                dtype=float,
            )

            # detect euler angle jump
            anglejumps = np.mod(abs(input[:, 2] - previous[:, 2]), 360)
            anglejumps_sorted = anglejumps[anglejumps.argsort()]
            maxanglejump = anglejumps_sorted[
                min(int((anglejumps.shape[0] - 1) * threshold), anglejumps.shape[0] - 1)
            ]

            # detect differential shift
            shiftjumps = abs(
                np.hypot(input[:, 4] - previous[:, 4], input[:, 5] - previous[:, 5])
            )
            shiftjumps_sorted = shiftjumps[shiftjumps.argsort()]
            maxshiftjump = shiftjumps_sorted[
                min(int((shiftjumps.shape[0] - 1) * threshold), shiftjumps.shape[0] - 1)
            ]

            # keep only particles with jumps below thresholds
            input[:, field] = np.where(
                np.logical_or(anglejumps > maxanglejump, shiftjumps > maxshiftjump),
                np.nan,
                input[:, field],
            )

    if odd:
        if scores or frealignx:
            input[::2, occ] = 0
        else:
            input[::2, field] = np.nan

    if even:
        if scores:
            input[1::2, occ] = 0
        else:
            input[1::2, field] = np.nan

    # write output parameter file
    # new_par_obj = frealign_parfile.Parameters(version, extended=extended, data=input, prologue=prologue, epilogue=epilogue)
    # new_par_obj.write_file(outputparfile)

    par_obj.write_file(outputparfile)
    return par_obj
    # f = open(outputparfile, "w")
    # for line in open(inputparfile):
    #     if line.startswith("C"):
    #         f.write(line)
    # for i in range(input.shape[0]):
    #     if scores:
    #         f.write(frealign_parfile.NEW_PAR_STRING_TEMPLATE % tuple(input[i, :16]))
    #     elif frealignx:
    #         f.write(frealign_parfile.FREALIGNX_PAR_STRING_TEMPLATE % tuple(input[i, :17]))
    #     else:
    #         f.write(frealign_parfile.CCLIN_PAR_STRING_TEMPLATE % tuple(input[i, :13]))
    #     f.write("\n")
    # f.close()
    #
    # if scores:
    #     columns = 16
    #     width = 137
    # elif frealignx:
    #     columns = 17
    #     width = 145
    # else:
    #     columns = 13
    #     width = 103

    # if input.shape[1] > columns:
    #     # compose extended .parx file
    #     long_file = [line for line in open(inputparfile) if not line.startswith("C")]
    #     short_file = [line for line in open(outputparfile) if not line.startswith("C")]
    #     if (
    #         len(long_file[0].split()) > columns
    #         and not len(short_file[0].split()) > columns
    #         and not "star" in outputparfile
    #     ):

    #         logger.info("merging", inputparfile, "with", outputparfile, "into", outputparfile)

    #         f = open(outputparfile, "w")
    #         [f.write(line) for line in open(inputparfile) if line.startswith("C")]
    #         for i, j in zip(short_file, long_file):
    #             f.write(i[:-1] + j[width - 1 :])
    #         f.close()

@timer.Timer(
    "call_shape_phase_residuals", text="Shaping scores took: {}", logger=logger.info
)
def call_shape_phase_residuals(
    input_par_file, output_par_file, png_file, fp, iteration
):

    mindefocus = float(project_params.param(fp["reconstruct_mindef"], iteration))
    maxdefocus = float(project_params.param(fp["reconstruct_maxdef"], iteration))
    firstframe = int(project_params.param(fp["reconstruct_firstframe"], iteration))
    lastframe = int(project_params.param(fp["reconstruct_lastframe"], iteration))
    mintilt = float(project_params.param(fp["reconstruct_mintilt"], iteration))
    maxtilt = float(project_params.param(fp["reconstruct_maxtilt"], iteration))
    minazh = float(project_params.param(fp["reconstruct_minazh"], iteration))
    maxazh = float(project_params.param(fp["reconstruct_maxazh"], iteration))
    minscore = float(project_params.param(fp["reconstruct_minscore"], iteration))
    maxscore = float(project_params.param(fp["reconstruct_maxscore"], iteration))

    shapr = project_params.param(fp["reconstruct_shapr"], iteration)
    reverse = False
    consistency = False
    if "reverse" in shapr.lower():
        reverse = True
    if "consistency" in shapr.lower():
        consistency = True

    # use NO cutoff if we are using multiple references
    if int(project_params.param(fp["class_num"], iteration)) > 1:
        cutoff = 1
    else:
        cutoff = project_params.param(fp["reconstruct_cutoff"], iteration)

    angle_groups = int(project_params.param(fp["reconstruct_agroups"], iteration))
    defocus_groups = int(project_params.param(fp["reconstruct_dgroups"], iteration))
    cutoff = float(cutoff)
    binning = 1.0
    odd = False
    even = False
    scores = True
    is_frealignx = True or ("frealignx" in project_params.param(fp["refine_metric"], iteration)
    or project_params.param(fp["dose_weighting_enable"], iteration) 
    or "tomo" in project_params.param(fp["data_mode"], iteration)
    or "local" in project_params.param(fp["extract_fmt"], iteration)
    )

    par_obj = shape_phase_residuals(
        input_par_file,
        angle_groups,
        defocus_groups,
        cutoff,
        mindefocus,
        maxdefocus,
        firstframe,
        lastframe,
        mintilt,
        maxtilt,
        minazh,
        maxazh,
        minscore,
        maxscore,
        binning,
        reverse,
        consistency,
        scores,  # since we are always using scores (not phase residuals)
        is_frealignx,
        odd,
        even,
        output_par_file,
        png_file
    )
    
    return par_obj

def eval_phase_residual(
    defocus, mparameters, fparameters, input, name, film, scanor, tolerance
):

    if math.fabs(defocus) > tolerance:
        logger.info("Evaluating %f = %d", defocus, np.nan)
        return np.nan

    particles = input.shape[0]

    eval = np.copy(input)
    eval[:, 8:10] += defocus
    # input[:,10] += 100000 * defocus

    # print input[:,8].mean(), input[:,9].mean()

    # substitue new defocus value and write new parfile
    frealign_parameter_file = "scratch/" + name + "_r01_02.par"

    if "frealignx" in fparameters["metric"]:
        version = "frealignx"
    else:
        version = "new"

    par_obj = frealign_parfile.Parameters(version, data=input)
    par_obj.write_file(frealign_parameter_file)
    # with open(frealign_parameter_file, "w") as f:
    #     # write out header (including resolution table)
    #     # [ f.write(line) for line in open( parfile ) if line.startswith('C') ]
    #     for i in range(particles):
    #         if "frealignx" in fparameters["metric"]:
    #             f.write(
    #                 frealign_parfile.FREALIGNX_PAR_STRING_TEMPLATE % (tuple(eval[i, :16]))
    #             )
    #         else:
    #             f.write(frealign_parfile.NEW_PAR_STRING_TEMPLATE % (tuple(eval[i, :16])))
    #         f.write("\n")

    stack = "data/%s_stack.mrc" % (name).replace("_short", "")
    local_stack = "%s_stack.mrc" % (name)
    if not os.path.exists(local_stack):
        try:
            symlink_relative(os.path.join(os.getcwd(),stack), local_stack)
        except:
            logger.info("symlink failed %s %s", local_stack, stack)
            pass

    # call FREALIGN directly to improve performance
    mp = mparameters.copy()
    fp = fparameters.copy()
    fp["mode"] = "1"
    fp["mask"] = "0,0,0,0,0"
    fp["dataset"] = name

    local_model = os.getcwd() + "/scratch/%s_r01_01.mrc" % (name)
    if not os.path.exists(local_model):
        symlink_relative(mparameters["class_ref"], local_model)

    os.chdir("scratch")

    command = frealign.mrefine_version(
        mp, fp, 1, particles, 2, 1, name + "_r01_02", "", "/dev/null", "", fp["metric"]
    )
    run_shell_command(command)

    output_parfile = name + "_r01_02.par_"

    # open output .par file and average scores
    if "frealignx" in fp["metric"]:
        scores = np.array(
            [
                float(line[129:136])
                for line in open(output_parfile)
                if not line.startswith("C")
            ],
            dtype=float,
        )
    else:
        scores = np.array(
            [
                float(line[121:128])
                for line in open(output_parfile)
                if not line.startswith("C")
            ],
            dtype=float,
        )

    logger.info("Evaluating %f = %f", defocus, scores.mean())

    os.chdir("..")

    return -scores.mean()


def score_particles_fromparx(par_data, mintilt: float, maxtilt: float, min_num_projections: int, pixel_size: float, metric="new"):
    """ Compute scores of sub-volume from their corresponding projections in the parfile
        Scores will be updated in box3d files, which will be later used for CSP

    Parameters:
    ----------
        par_data (numpy array): 
            information from parx file
        metric (str): 
            The format of the input parfile (currently only support extended metric new)
    """
    scores = {}
    shifts_3d = {}
    weights = []

    if metric == "new":
        film_col = 7
        occ_col = 11
        score_col = 14
        ptlidx_col = 16
        tiltan_col = 17
        scanord_col = 19
        normx_col = 24 - 1
        normy_col = 25 - 1
        normz_col = 26 - 1
        matrix0_col = 27 - 1
        matrix15_col = 42 - 1
    else:
        logger.error("Currently not support other metrics except metric new")
        sys.exit()

    # First, compute the weights for scoring
    max_scanord = max(np.unique(par_data[:, scanord_col].astype("int")))
    # prepare weights data structure
    # while len(weights) < max_scanord + 1:
    #     weights.append([])

    # # # compute average score per tilt
    # # for scanord, average in enumerate(weights):

    # #     scores = par_data[par_data[:, scanord_col] == scanord]

    # #     if len(scores) > 0 and scores.ndim != 1:
    # #         scores = scores[:, score_col]
    # #         weights[scanord] = sum(scores) / len(scores)
    # #     else:
    # #         weights[scanord] = 0.0

    # iterate through particles (ptlidx) in different tilt-series (film)
    films = np.unique(par_data[:, film_col].astype("int"))

    for film in films:

        tiltseries = par_data[par_data[:, film_col] == film]
        particles = np.unique(tiltseries[:, ptlidx_col].astype("int"))
        scores[film] = [-1.0 for _ in range(max(particles)+1)]
        shifts_3d[film] = [[0,0,0] for _ in range(max(particles)+1)]
        
        for ptl in particles:

            particle = tiltseries[tiltseries[:, ptlidx_col] == ptl]
            
            sum_score = 0.0
            if particle.ndim != 1:

                # take care of particle that does not have complete enough tilt coverage
                tilt_angles = particle[:, tiltan_col]
                valid_tilt_angles = [
                    angle for angle in tilt_angles if mintilt <= angle and angle <= maxtilt 
                ]
                if len(valid_tilt_angles) >= min_num_projections:
                    sum_weight = 0.0
                    for proj in particle:
                        weight = 1.0 
                        if proj[tiltan_col] <= maxtilt and proj[tiltan_col] >= mintilt:
                            sum_score += weight * proj[score_col]
                            sum_weight += weight
                    
                    if sum_weight > 0 and particle[0, occ_col] > 0:
                        sum_score /= sum_weight
                    else: 
                        sum_score = -1
                    
                    scores[film][ptl] = sum_score
                
                matrix = particle[0,matrix0_col: matrix15_col+1]
                matrix[12: 16] = np.array([0,0,0,1])
                dx, dy, dz = geometry.getShiftsForRecenter(particle[0,normx_col:normz_col+1], matrix, 0)
                dx, dy, dz = dx/pixel_size, dy/pixel_size, dz/pixel_size
                shifts_3d[film][ptl] = [dx, dy, dz]
            else:
                scores[film][ptl] = -1.0

    return scores, shifts_3d


def clean_particles_tomo(box3dfile, dist: float, threshold: float, shifts_3d_film: list):
    """Clean particles/sub-volume based on their distances/scores by setting the KEEP field in box3d files

    Parameters
    ----------
        box3dfile : list[str]
            Tomo particle metadata 
        dist int :
            Distance cutoff in 3D

    Returns
    ----------
        list: return box3d lines that stores ptlidx, 3d coord, scores and keep
    """
    if dist < 0.0:
        logger.error(
            "Distance cutoff has to be greater than 0. Dist of 0 to keep ALL particles,"
        )
        sys.exit()

    # remove header
    box3dfile = box3dfile[1:]
    # correct type
    for data in box3dfile:
        data[0], data[1], data[2], data[3], data[4], data[5] = (
            int(data[0]),
            float(data[1]),
            float(data[2]),
            float(data[3]),
            float(data[4]),
            str(data[5]),
        )

    # sort based on scores
    box3dfile = sorted(box3dfile, key=lambda x: x[4], reverse=True)

    # save valid points in 3D
    valid_points = np.array(box3dfile[0][1:4], ndmin=2)

    for idx, data in enumerate(box3dfile):
        
        if idx == 0:
            if box3dfile[0][4] >= threshold:
                data[5] = "Yes"
            else:
                data[5] = "No"
            continue

        ptlind = data[0]
        pos_x, pos_y, pos_z = data[1], data[2], data[3]
        try:
            # some ptlind in the parfiles may be removed
            pos_x += shifts_3d_film[ptlind][0] 
            pos_y += shifts_3d_film[ptlind][1] 
            pos_z += shifts_3d_film[ptlind][2]
        except:
            pass
        # check if the point is close to previous evaluated points
        dmin = scipy.spatial.distance.cdist(
            np.array([pos_x, pos_y, pos_z], ndmin=2), valid_points
        ).min()
        if data[4] < threshold or dmin <= dist:
            data[5] = "No"
        else:
            data[5] = "Yes"
            valid_points = np.vstack((valid_points, np.array([pos_x, pos_y, pos_z])))

    # sort the box based on particle index before return
    box3dfile = sorted(box3dfile, key=lambda x: x[0])

    return box3dfile


def particle_cleaning(parameters: dict):
    """ Particle cleaning (called from pyp_main) 

    Parameters
    ----------

    Returns
    ----------
        list: return box3d lines that stores ptlidx, 3d coord, scores and keep
    """
    
    try:
        filmlist_file = "{}.films".format(parameters["data_set"])
        films = np.loadtxt(filmlist_file, dtype='str', ndmin=1)
        # films = [film.strip() for film in f.readlines()]
    except:
        raise Exception(
            "{} does not exists".format("{}.films".format(parameters["data_set"]))
        )

    parfile = project_params.resolve_path(parameters["clean_parfile"])

    # first check class selection, and generate one parfile with occ=0 to mark the discarded particles
    if parameters["clean_class_selection"] and not parameters["clean_discard"]:
        sel = parameters["clean_class_selection"]
        selist = sel.split(",")
        selection = [int(x) for x in selist]
        merge_align = parameters["clean_class_merge_alignment"]
        
        output_parfile = pyp_metadata.merge_par_selection(parfile, selection, parameters, merge_align=merge_align)
    
        parfile = output_parfile

        parameters["refine_parfile"] = parfile
 
    if os.path.exists(parfile) and parfile.endswith(".bz2"):
        parfile = frealign_parfile.Parameters.decompress_parameter_file(parfile, parameters["slurm_tasks"])

    # single class cleaning regard to box files
    if "spr" in parameters["data_mode"]:

        pardata = frealign_parfile.Parameters.from_file(parfile).data

        if parameters["clean_spr_auto"]:
            # figure out optimal score threshold

            samples = np.array(frealign.get_phase_residuals(pardata,parfile,parameters,2))
            # samples = pardata[:, 14].astype("f")

            threshold = 1.075 * statistics.optimal_threshold(
                samples=samples, criteria="optimal"
            )
            logger.info(f"Using {threshold} as optimal threshold")
        else:
            threshold = parameters["clean_threshold"]

        parameters, new_pardata = clean_particle_sprbox(pardata, threshold, parameters, metapath="./pkl")
        
        if parameters["clean_discard"]:
            
            current_dir = os.getcwd()
            # clean_parfile = os.path.join(current_dir, "frealign", "maps", os.path.basename(parfile).replace(".par", "_clean.par").replace(".bz2", ""))
            clean_parfile = os.path.basename(parfile).replace(".par", "_clean.par").replace(".bz2", "")
            os.chdir("./frealign/maps/")
            frealign_parfile.Parameters.write_parameter_file(clean_parfile, new_pardata, parx=True, frealignx=False)
            
            frealign_parfile.Parameters.compress_parameter_file(clean_parfile, clean_parfile.replace(".par", ".par.bz2"), parameters["slurm_tasks"])
            # link the parfile and reference for workflows
            conventional_auto = clean_parfile.replace("_clean", "").replace(".par", ".par.bz2")
            if not Path(conventional_auto).exists():
                shutil.copy2(clean_parfile, clean_parfile.replace("_clean", ""))
                frealign_parfile.Parameters.compress_parameter_file(clean_parfile.replace("_clean", ""), conventional_auto)
                os.remove(clean_parfile.replace("_clean", ""))

            os.remove(clean_parfile)
            os.chdir(current_dir)
            # update extract selection after cleaning
            parameters["extract_cls"] += 1

    # we don't have box3d in spr, so this only works for tomo so far
    elif "tomo" in parameters["data_mode"]:
        
        if not parameters["clean_discard"]:

            # update mean scores if parfile is provided
            if os.path.exists(parfile):

                par_data = frealign_parfile.Parameters.from_file(parfile).data
                scores, shifts_3d = score_particles_fromparx(par_data, parameters["clean_mintilt"], 
                                                            parameters["clean_maxtilt"], 
                                                            parameters["clean_min_num_projections"],
                                                            parameters["scope_pixel"] 
                                                            )
                # update mean scores in boxes3d
                update_scores(films, scores)

            # remove bad particles with threshold and distance, and plot histograms
            thresholding_and_plot(films, shifts_3d, parameters)

        else:
            # completely remove particle projections from parfile and allboxes coordinate
            newfilms = deep_clean_parfile(parfile,films,parameters["data_set"])
            
            # update film file
            if newfilms.shape[0] < films.shape[0]:
                os.rename(filmlist_file, filmlist_file.replace(".films", ".films_original"))
                np.savetxt(filmlist_file, newfilms, fmt="%s")
                shutil.copy2(filmlist_file, filmlist_file.replace(".films", ".micrographs"))

    return parameters 


def thresholding_and_plot(films, shifts_3d: list, parameters: dict):
    films_used_particles = [[film, 0] for film in films]
    film_count = 0
    particle_used_count = 0
    particle_all_count = 0

    for film, tiltseries in enumerate(films):
        box3dfile = "csp/{}_boxes3d.txt".format(tiltseries)
        newbox3dfile = "csp/{}_boxes3d_clean.txt".format(tiltseries)
        particle_used_film = 0

        if not os.path.exists(box3dfile):
            logger.warning("{} does not exist".format(box3dfile))
        else:
            newf = open(newbox3dfile, "w")
            newf.write(
                "%8s\t%8s\t%8s\t%8s\t%8s\t%8s\n"
                % ("PTLIDX", "X", "Y", "Z", "Score", "Keep_CSP")
            )
            with open(box3dfile, "r") as f:
                box3d = [line.split() for line in f.readlines()]
                newbox3d = clean_particles_tomo(box3d, parameters["clean_dist"], parameters["clean_threshold"], shifts_3d[film])

                for line in newbox3d:
                    newf.write(
                        "%8d\t%8.1f\t%8.1f\t%8.1f\t%8.2f\t%8s\n" % tuple(line)
                    )
                    particle_all_count += 1
                    if "Yes" in line[5]:
                        particle_used_count += 1
                        particle_used_film += 1

            newf.close()
            os.remove(box3dfile)
            os.rename(newbox3dfile, box3dfile)

            mean_scores = [_[4] for _ in newbox3d]
            plot.histogram_particle_tomo(mean_scores, parameters["clean_threshold"], tiltseries, "csp")


        films_used_particles[film][1] = particle_used_film
        film_count += 1

    logger.warning(
        "{:,} particles ({:.1f}%) from {} tilt-series will be used".format(
            particle_used_count,
            (particle_used_count / particle_all_count * 100),
            film_count,
        )
    )


def update_scores(films, scores: list): 
    
    # parse new scores into 3d boxes
    for film, tiltseries in enumerate(films):
        box3dfile = "csp/{}_boxes3d.txt".format(tiltseries)
        newbox3dfile = "csp/{}_boxes3d_clean.txt".format(tiltseries)
        if not os.path.exists(box3dfile):
            logger.warning("{} does not exist".format(box3dfile))
        else:
            newf = open(newbox3dfile, "w")
            with open(box3dfile, "r") as f:
                for line in f.readlines():
                    if line.strip().startswith("PTLIDX"):
                        newf.write(line)
                    else:
                        data = line.split()
                        # PTLIDX, X, Y, Z, Score, Keep
                        (
                            data[0],
                            data[1],
                            data[2],
                            data[3],
                            data[4],
                            data[5],
                        ) = (
                            int(data[0]),
                            float(data[1]),
                            float(data[2]),
                            float(data[3]),
                            float(data[4]),
                            str(data[5]),
                        )

                        try:
                            data[4] = scores[film][data[0]]
                        except:
                            data[4] = -1.0 # happend at the end and might be discarded in parfile

                        newf.write(
                            "%8d\t%8.1f\t%8.1f\t%8.1f\t%8.2f\t%8s\n"
                            % tuple(data)
                        )
            newf.close()
            shutil.copy2(newbox3dfile, box3dfile)
            os.remove(newbox3dfile)

def deep_clean_parfile(parfile: str, films, dataset: str):
    assert (os.path.exists(parfile)), f"{parfile} is required to remove particle projections from parfile and allboxes"

    FILM_COL = 8 - 1
    OCC_COL = 12 - 1
    ORI_TAG = "_original"
    parx_name, parx_format = os.path.splitext(parfile)
    par = frealign_parfile.Parameters.from_file(parfile)
    par_data = par.data

    par_data_clean = par_data[par_data[:, OCC_COL] != 0.0]
    allboxes_line_count = 0
    empty_films = []

    for idx, film in enumerate(films):
        # check which allboxes we should use: frame format (*local.allboxes) is the first priority
        if os.path.exists(os.path.join("csp", film + "_local.allboxes")):
            allboxes_file = os.path.join("csp", film + "_local.allboxes")
            allboxes = np.loadtxt(
                os.path.join("csp", film + "_local.allboxes"), ndmin=2
            )
        elif os.path.exists(os.path.join("csp", film + ".allboxes")):
            allboxes_file = os.path.join("csp", film + ".allboxes")
            allboxes = np.loadtxt(
                os.path.join("csp", film + ".allboxes"), ndmin=2
            )
        else:
            raise Exception(f"{film} allboxes not found")

        allboxes_name, allboxes_format = os.path.splitext(allboxes_file)

        par_data_film = par_data[par_data[:, FILM_COL] == idx]

        if par_data_film.shape[0] != allboxes.shape[0]:
            raise Exception(
                f"The number of lines in {parfile} and {allboxes_file} does not match - {par_data_film.shape[0]} v.s. {allboxes.shape[0]}\n\
                        You probably deep-clean particles already. Try to look for *_clean.par.bz2 from a downstream block"
            )

        allboxes = np.delete(
            allboxes, np.argwhere(par_data_film[:, OCC_COL] == 0), axis=0
        )

        if allboxes.shape[0] == 0:
            empty_films.append(film)
            os.remove(allboxes_file)
        else:
            allboxes_line_count += allboxes.shape[0]
        
        if allboxes.shape[0] > 0:
            os.rename(allboxes_file, allboxes_name + ORI_TAG + allboxes_format)
            np.savetxt(allboxes_file, allboxes.astype(int), fmt="%i")

    if par_data_clean.shape[0] != allboxes_line_count:
        [
            os.rename(f, f.replace(ORI_TAG + allboxes_format, allboxes_format))
            for f in [
                os.path.join("csp", file)
                for file in os.listdir("csp")
                if file.endswith(ORI_TAG + allboxes_format)
            ]
        ]
        raise Exception(
            f"After cleaning the total number of lines in {parfile} and allboxes does not match - {par_data_clean.shape[0]} v.s. {allboxes_line_count}"
        )

    par_data_clean[:, 0] = np.array(
        [(_i + 1) % 10000000 for _i in range(par_data_clean.shape[0])]
    )

    # re-number films start from 0
    new_film_ids = par_data_clean[:, 7]
    uniquefilm = np.unique(new_film_ids)
    for i, old_id in enumerate(uniquefilm):
        film_mask = par_data_clean[:, 7] == old_id
        par_data_clean[film_mask, 7] = i

    par.data = par_data_clean
    current_dir = os.getcwd()
    # clean_parfile = os.path.join(os.getcwd() , "frealign", "maps" + os.path.basename(parfile).replace(".par", "_clean.par").replace(".bz2", ""))
    
    clean_parfile = f"{dataset}_r01_02_clean.par"
    parfile = clean_parfile.replace("_clean", "")
    # clean_parfile = os.path.basename(parfile).replace(".par", "_clean.par").replace(".bz2", "")
    # os.rename(parfile, parfile.replace(parx_format, ORI_TAG + parx_format))
    
    os.chdir("./frealign/maps/")
    par.write_file(clean_parfile)

    compressed_clean_parfile = clean_parfile.replace(".par", ".par.bz2")
    compressed_parfile = compressed_clean_parfile.replace("_clean", "")
    reference = compressed_parfile.replace(".par.bz2", ".mrc")
    prev_reference = reference.replace("_02.mrc", "_01.mrc")

    frealign_parfile.Parameters.compress_parameter_file(clean_parfile, compressed_clean_parfile)

    # link the parfile and reference for workflows
    if not Path(compressed_parfile).exists():
        shutil.copy2(clean_parfile, parfile)
        frealign_parfile.Parameters.compress_parameter_file(parfile, compressed_parfile)
        os.remove(parfile)
    if not Path(reference).exists() and Path(prev_reference).exists():
        shutil.copy2(prev_reference, reference)

    os.remove(clean_parfile)
    os.chdir(current_dir)
    
    logger.info("Successfully remove particles from the parfile and allboxes!")

    # remove empty film from original film list
    if len(empty_films) > 0:
        indices = np.where(np.isin(films, empty_films))
        newfilms = np.delete(films, indices)
        return newfilms
    else:
        return films
    # return clean_parfile.replace(".par", ".par.bz2") 

def clean_particle_sprbox(pardata, thresh, parameters, isfrealignx=False, metapath="./pkl"):

     # select particles based on Frealign score distribution 
     # modify pkl files to reflect the coordinates change

    if not isfrealignx:
        field = 14
        occ_field = 11
    else:
        field = 15
        occ_field = 12
    classification_pass = parameters["extract_cls"] + 1 
    occ_thresh = parameters["reconstruct_min_occ"]

    # filter particles that are too close to their neighbors
    pardata = remove_duplicates(pardata, field, occ_field, parameters)

    # discard_mask = np.logical_or(pardata[:, field] < thresh, pardata[:, 11] < occ_thresh)
    discard_mask = np.ravel(np.logical_or(pardata[:, field] < thresh, pardata[:, 11] < occ_thresh))
    logger.info(f"Score range [{min(pardata[:, field])},{max(pardata[:, field])}], threshold = {thresh:.2f}")
    logger.info(f"Occupancy range [{min(pardata[:, 11])},{max(pardata[:, 11])}], threshold {occ_thresh}")
    discard = pardata[discard_mask]
    newinput_keep = pardata[np.logical_not(discard_mask)]
    global_indexes_to_remove = (discard[:, 0] - 1).astype("int").tolist()
    global_indexes_to_keep = (newinput_keep[:, 0] - 1).astype("int").tolist()

    logger.info("Particles to remove = %i" % len(global_indexes_to_remove))
    logger.info("Particles to keep = %i" % len(global_indexes_to_keep))

    thresh_ratio = len(global_indexes_to_keep) / pardata.shape[0]

    if not parameters["clean_discard"]: 
        if parameters["clean_class_selection"]:
            # class selection using exiting occ
            parameters["reconstruct_cutoff"] = "1"
            return parameters, newinput_keep
        else:
            if thresh_ratio >= 0.0001:
                parameters["reconstruct_cutoff"] = "%.4f" % thresh_ratio
                logger.info(f"Reconstruction cutoff is changed to {thresh_ratio:.4f}")
                return parameters, newinput_keep
            else:
                Exception(f"Only {thresh_ratio * 100} percent of the particles will be used, which is too low, Abort.")
            
    else:
        # indexes are in base-0
        discard_filmid = np.unique(discard[:,7].astype(int))
        filmlist_file = "{}.films".format(parameters["data_set"])
        film_list = np.loadtxt(filmlist_file, dtype='str')
        discard_filmlist = film_list[discard_filmid]
        unchanged_filmlist = np.setdiff1d(film_list, discard_filmlist)
        
        emptyfilm = [] # record films with no particles left 

        # updating metadata 
        is_spr=True
        current_dir = os.getcwd()
        os.chdir(metapath)
        box_header = ["x", "y", "Xsize", "Ysize", "inside", "selection"]

        # in case some images have no particle removed, update selection column only
        if len(unchanged_filmlist) > 0:
            for film in unchanged_filmlist:
                metadata = pyp_metadata.LocalMetadata(film + ".pkl", is_spr=is_spr)
                boxx = metadata.data["box"].to_numpy()
                boxx[:, 5] = classification_pass
                df = pd.DataFrame(boxx, columns=box_header)
                metadata.updateData({'box':df})    
                metadata.write()

        for id, film in enumerate(discard_filmlist):
            filmid = discard_filmid[id]
            
            metadata = pyp_metadata.LocalMetadata(film + ".pkl", is_spr=is_spr)
            if "box" in metadata.data.keys():
            
                boxx = metadata.data["box"].to_numpy()
                # current valid particles
                boxx_valid = np.argwhere(np.logical_and(boxx[:, 4] >= 1, boxx[:, 5] >= classification_pass - 1))
                boxx_valid = boxx_valid.ravel()
                
                ptls_infilm = pardata[pardata[:, 7] == filmid] 
                assert len(boxx_valid) == ptls_infilm.shape[0], f"Valid particles in box {len(boxx_valid)} not equal to particles in parfile {ptls_infilm.shape[0]}"
                
                # the index want to keep from the current valid particles.
                ptl_to_keep = np.argwhere(np.logical_and(ptls_infilm[:, field] >= thresh, ptls_infilm[:, 11] >= occ_thresh)).ravel()
                
                if ptl_to_keep.shape[0] > 0:
                    boxx_keep_index = boxx_valid[ptl_to_keep]
                    # set clean particles to pass id
                    boxx[boxx_keep_index, 5] = classification_pass
                    # set other particles to lower level
                    all_indices = np.arange(boxx.shape[0])
                    complementary_indices = np.setdiff1d(all_indices, boxx_keep_index)
                    boxx[complementary_indices, 5] = classification_pass - 1 

                    # passed = boxx[np.logical_and(boxx[:, 4] >=1, boxx[:, 5]>= classification_pass)]
                    # logger.info(f"Particles before clean is {len(boxx_valid)}, particles after clean is {len(passed)}")
                else:
                    emptyfilm.append(film)
                
                df = pd.DataFrame(boxx, columns=box_header)
            
                metadata.updateData({'box':df})    
                metadata.write()

                # update allboxes here
                allboxfile = os.path.join(current_dir, "csp", film + ".allboxes")
                allboxes = np.loadtxt(allboxfile, ndmin=2).astype(int).tolist()
                previous_valid = boxx[boxx_valid]
                indexes = np.argwhere(
                    np.logical_and(
                        previous_valid[:, 4] == 1, previous_valid[:, 5] >= classification_pass
                    )
                )

                if len(allboxes) > len(indexes):
                    allboxes = [allboxes[i[0]] for i in indexes]
                    if Path(allboxfile).is_symlink():
                        os.remove(allboxfile)
                    np.savetxt(allboxfile, np.array(allboxes), fmt="%i")

            else:
                Exception("No box info from Pickle file.")

        os.chdir(current_dir)

        # remove empty film from original film list
        if len(emptyfilm) > 0:
            indices = np.where(np.isin(film_list, emptyfilm))
            newfilms = np.delete(film_list, indices)
            
            os.rename(filmlist_file, filmlist_file.replace(".films", ".films_original"))
            np.savetxt(filmlist_file, newfilms, fmt="%s")
            shutil.copy2(filmlist_file, filmlist_file.replace(".films", ".micrographs"))

        # produce corresponding .par file
        # reorder index
        newinput_keep[:, 0] = list(range(newinput_keep.shape[0]))
        newinput_keep[:, 0] += 1

        if newinput_keep.shape[0] != len(global_indexes_to_keep):
            logger.error(
                "Number of clean particles does not match number of particles to keep: {0} != {1}".format(
                    newinput_keep.shape[0], len(global_indexes_to_keep)
                )
            )
            sys.exit()

        # re-number films start from 0
        new_film_ids = newinput_keep[:, 7]
        uniquefilm = np.unique(new_film_ids)
        for i, old_id in enumerate(uniquefilm):
            film_mask = newinput_keep[:, 7] == old_id
            newinput_keep[film_mask, 7] = i
        
        """
        current_film = newinput_keep[0, 7]
        current_film = 0
        new_film_number = 0
        for i in range(newinput_keep.shape[0]):
            if newinput_keep[i, 7] != current_film:
                current_film = newinput_keep[i, 7]
                new_film_number += 1
            newinput_keep[i, 7] = new_film_number
        """
        # set occupancy to 100
        newinput_keep[:, 11] = 100.0

        # return clean_parfile.replace(".par", ".par.bz2")
        return parameters, newinput_keep


def update_boxx_files(global_indexes_to_remove, parameters, classification_pass, shifts=[]):

    pixel = float(parameters["scope_pixel"]) * float(parameters["data_bin"])

    micrographs = "{}.micrographs".format(parameters["data_set"])
    inputlist = [line.strip() for line in open(micrographs, "r")]

    current_global_counter = previous_global_counter = micrograph_counter = 0

    boxx = np.array([1])

    global_indexes_to_remove.sort()

    threads = 12

    # read all boxx files in parallel
    pool = multiprocessing.Pool(threads)
    manager = multiprocessing.Manager()
    results = manager.Queue()
    logger.info("Reading box files using %i threads", threads)
    for micrograph in inputlist:
        pool.apply_async(read_boxx_file_async, args=(micrograph, results))
    pool.close()

    # Wait for all processes to complete
    pool.join()

    boxx_dbase = dict()
    box_dbase = dict()
    while results.empty() == False:
        current = results.get()
        boxx_dbase[current[0]] = current[1]
        box_dbase[current[0]] = current[2]

    local_counter = 0

    # add infinity to force processing of micrograph list to the end
    import sys

    # global_indexes_to_remove.append( sys.maxint )

    logger.info("Updating particle database")

    # set last field in .boxx files to 0 for all bad particles
    for i in global_indexes_to_remove:

        while current_global_counter <= i:

            try:
                name = inputlist[micrograph_counter]
                boxxfile = "box/{}.boxx".format(name)
                boxfile = "box/{}.box".format(name)
            except:
                logger.exception(
                    "%d outside bounds %d", micrograph_counter, len(inputlist)
                )
                logger.exception("%d %d", current_global_counter, i)
                sys.exit()

            if os.path.exists(boxxfile):
                # boxx = numpy.loadtxt( boxxfile, ndmin=2 )
                boxx = boxx_dbase[name]
                box = box_dbase[name]

                if boxx.size > 0:

                    # only count particles that survived previous pass
                    # valid_particles = boxx[:,4].sum()
                    valid_particles = np.where(
                        np.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        1,
                        0,
                    ).sum()

                    previous_global_counter = current_global_counter
                    current_global_counter = (
                        current_global_counter + valid_particles
                    )  # 5th column contains actually extracted particles

                    # increment class membership pass for all particles of previous classification pass
                    boxxx = np.where(
                        np.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        classification_pass,
                        boxx[:, 5],
                    )

                    # alignments
                    counter = previous_global_counter
                    for j in range(boxx.shape[0]):
                        if boxx[j, 4] > 0 and boxx[j, 5] >= classification_pass - 1:
                            if len(shifts) > 0:
                                # boxx_dbase[name][j,0:2] = box[j,0:2] - boxx[j,2:4] / 2 - numpy.round( input[ counter, 4:6 ] / pixel )
                                boxx_dbase[name][j, 0:2] = (
                                    box[j, 0:2]
                                    - boxx[j, 2:4] / 2
                                    + np.round(shifts[local_counter, :] / pixel)
                                )
                                local_counter += 1
                            else:
                                boxx_dbase[name][j, 0:2] = (
                                    box[j, 0:2] - boxx[j, 2:4] / 2
                                )

                            counter += 1

                    boxx_dbase[name][:, 5] = boxxx
            else:
                logger.info("%s does not exist", boxxfile)

            if micrograph_counter < len(inputlist) - 1:
                micrograph_counter += 1
            # if micrograph_counter == len(inputlist):
            else:
                logger.info("Reached the end")
                break

        if micrograph_counter < len(inputlist):

            index_in_micrograph = 0
            global_counter = -1

            # print i, previous_global_counter
            while global_counter < i - previous_global_counter:
                if (
                    boxx[index_in_micrograph, 4] == 1
                    and boxx[index_in_micrograph, 5] >= classification_pass - 1
                ):
                    global_counter += 1
                index_in_micrograph += 1
            # boxx[index-1,5] = 0
            boxxx[index_in_micrograph - 1] = classification_pass - 1
            boxx_dbase[name][:, 5] = boxxx

    logger.info("Current global count %d", current_global_counter)
    # save all boxx files in parallel
    pool = multiprocessing.Pool(threads)
    manager = multiprocessing.Manager()
    results = manager.Queue()
    logger.info("Saving box files using %i threads", threads)
    for micrograph in inputlist:
        pool.apply_async(
            write_boxx_file_async, args=(micrograph, boxx_dbase[micrograph])
        )
        # write_boxx_file_async( micrograph, boxx_dbase[micrograph] )
    pool.close()

    # Wait for all processes to complete
    pool.join()
    return boxx_dbase




def remove_duplicates(pardata: np.ndarray, field: int, occ_field: int, parameters: dict) -> np.ndarray:
    """remove_duplicates Remove particles in SPA that are too close to their neighbors after alignment by setting their score to -1

    Parameters
    ----------
    pardata : np.ndarray
        Refined parfile
    field : int
        index where the score column is
    """

    FILM_COL = 8 - 1 
    pixel_size = parameters["scope_pixel"]

    filmlist_file = "{}.films".format(parameters["data_set"])
    film_list = np.loadtxt(filmlist_file, dtype='str')

    films = np.unique(pardata[:, FILM_COL].astype("int"))

    for film in films:
        micrograph = pardata[pardata[:, FILM_COL] == film]
        metadata = pyp_metadata.LocalMetadata("pkl/" + film_list[film] + ".pkl", is_spr=True)
        box = metadata.data["box"].to_numpy()

        micrograph[:, -2:] = box[:, :2]
        sort_pardata = micrograph[np.argsort(micrograph[:, field])][::-1]

        valid_points = np.array(
            [sort_pardata[0][-2] + (sort_pardata[0][4]/pixel_size), sort_pardata[0][-1]] + (sort_pardata[0][5]/pixel_size), 
            ndmin=2
            )

        for idx, line in enumerate(sort_pardata):
            if idx == 0:
                continue

            coordinate = np.array([line[-2] + (line[4]/pixel_size), line[-1] + (line[5]/pixel_size)], ndmin=2)
            dmin = scipy.spatial.distance.cdist(coordinate, valid_points).min()
            if dmin <= parameters["clean_dist"]:
                pardata[int(line[0]-1)][occ_field] = 0.0
            else:
                valid_points = np.vstack((valid_points, coordinate))
    
    return pardata
