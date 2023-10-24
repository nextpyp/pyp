import itertools
import math
import multiprocessing
import os
import subprocess
from collections import deque

import numpy as np

from pyp import analysis
from pyp.system import mpi
from pyp.inout.image import img2webp
from pyp.inout.metadata import frealign_parfile, pyp_metadata, isfrealignx, tomo_load_frame_xf
from pyp.inout.utils import pyp_edit_box_files as imod
from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.analysis.geometry.pyp_convert_coord import read_3dbox

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def fit_poly_trajectory(shifts, degree=0):

    frames = shifts.shape[0]
    x = shifts[:, -2]
    xs = np.linspace(1, frames, frames)
    y = shifts[:, -1]
    ys = np.linspace(1, frames, frames)

    polyx = np.polyfit(xs, x, degree)
    polyy = np.polyfit(ys, y, degree)

    newshifts = np.copy(shifts)
    newshifts[:, -2] = np.polyval(polyx, xs)
    newshifts[:, -1] = np.polyval(polyy, ys)
    return newshifts


def fit_spline_trajectory_1D_new(x, sigma=5):

    from scipy.ndimage import filters
    from scipy.signal import gaussian

    b = gaussian(20, sigma)
    average = filters.convolve1d(x, b / b.sum())

    return average


def fit_angular_trajectory_1D(x, sigma=5):

    # truncate values bigger than 360 and convert to unit sphere
    # x = np.where( x > 360, x % 360, x )
    y = np.sin(np.radians(x))

    from scipy import ndimage

    z = ndimage.median_filter(y, size=5)

    # filter values
    from scipy.ndimage import filters
    from scipy.signal import gaussian

    b = gaussian(20, sigma)
    yr = filters.convolve1d(z, b / b.sum(), mode="nearest")
    radians = np.where(y - yr > 1, 1, y - yr)
    output = x - np.degrees(np.arcsin(radians))
    # output = x - np.degrees(np.arcsin(y)) + np.degrees(np.arcsin(yr))
    return output


def fit_spatial_trajectory_1D_new(x, sigma=5):

    from scipy import ndimage

    xm = ndimage.median_filter(x, size=3)

    # filter values
    from scipy.ndimage import filters

    window_len = sigma
    window = "bartlett"
    b = eval("np." + window + "(window_len)")

    xr = filters.convolve1d(xm, b / b.sum(), mode="nearest")
    return xr


def fit_angular_trajectory_1D_new(x, sigma=5, normalize=False):

    # truncate values bigger than 360 and convert to unit sphere
    # if normalize:
    # x = np.where( x > 360, x % 360, x )
    s = np.sin(np.radians(x))
    c = np.cos(np.radians(x))

    from scipy import ndimage

    sm = ndimage.median_filter(s, size=3)
    cm = ndimage.median_filter(c, size=3)

    # filter values
    from scipy.ndimage import filters
    from scipy.signal import gaussian

    # b = gaussian( 21, sigma )
    # make sure window lenght is odd
    if sigma % 2 == 0:
        window_len = sigma + 1
    else:
        window_len = sigma

    window = "bartlett"
    # b = gaussian( window_len, sigma )
    b = eval("np." + window + "(window_len)")

    sr = filters.convolve1d(s, b / b.sum(), mode="nearest")
    cr = filters.convolve1d(c, b / b.sum(), mode="nearest")
    sr = np.where(sr > 1, 1, sr)
    sr = np.where(sr < -1, -1, sr)
    cr = np.where(cr > 1, 1, cr)
    cr = np.where(cr < -1, -1, cr)
    # output = x - np.degrees(np.arcsin(y)) + np.degrees(np.arcsin(yr))
    output = np.degrees(np.arctan2(sr, cr))
    if normalize:
        output = np.where(output < 0, output + 360, output)
    return output


def fit_spline_trajectory_1D(x, iters=1, k=3, factor=3.0):

    import scipy
    from scipy.interpolate import UnivariateSpline

    frames = len(x)
    xs = np.linspace(1, frames, frames)

    newshifts = np.copy(x)

    if np.fabs(x).sum() > 0 and frames > 3:

        """
        How to fix scipy's interpolating spline default behavior:
        http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/
        """

        from scipy.ndimage import filters
        from scipy.signal import gaussian

        sigma = 3
        b = gaussian(3, sigma)
        averagex = filters.convolve1d(x, b / b.sum())
        varx = filters.convolve1d(np.power(x - averagex, 2), b / b.sum())

        if np.fabs(varx).min() > 1e6 * np.finfo(float).eps:
            wx = 1 / np.sqrt(varx)
        else:
            return newshifts

        if k < 3:
            splinex = UnivariateSpline(xs, x, k=k)
        else:
            for iters in range(iters):
                splinex = UnivariateSpline(xs, x, w=wx * factor)
                x = splinex
            # splinex = UnivariateSpline( xs, x, w=wx*factor )
        newshifts = splinex(xs)

    return newshifts


def fit_spline_trajectory(shifts, k=3, factor=3.0):

    import scipy
    from scipy.interpolate import UnivariateSpline

    frames = shifts.shape[0]
    x = shifts[:, -2]
    xs = np.linspace(1, frames, frames)
    y = shifts[:, -1]
    ys = np.linspace(1, frames, frames)

    newshifts = np.copy(shifts)

    if np.fabs(shifts[:, -2:]).sum() > 0:

        """
        How to fix scipy's interpolating spline default behavior:
        http://www.nehalemlabs.net/prototype/blog/2014/04/12/how-to-fix-scipys-interpolating-spline-default-behavior/
        """

        from scipy.ndimage import filters
        from scipy.signal import gaussian

        sigma = 3
        b = gaussian(3, sigma)
        averagex = filters.convolve1d(x, b / b.sum())
        varx = filters.convolve1d(np.power(x - averagex, 2), b / b.sum())

        if np.fabs(varx).min() > 1e6 * np.finfo(float).eps:
            wx = 1 / np.sqrt(varx)
        else:
            return newshifts

        averagey = filters.convolve1d(y, b / b.sum())
        vary = filters.convolve1d(np.power(y - averagey, 2), b / b.sum())

        if np.fabs(vary).min() > 1e6 * np.finfo(float).eps:
            wy = 1 / np.sqrt(vary)
        else:
            return newshifts

        if k < 3:
            splinex = UnivariateSpline(xs, x, k=k)
            spliney = UnivariateSpline(ys, y, k=k)
        else:
            splinex = UnivariateSpline(xs, x, w=wx * factor)
            spliney = UnivariateSpline(ys, y, w=wy * factor)
        newshifts[:, -2] = splinex(xs)
        newshifts[:, -1] = spliney(ys)

    return newshifts


def trans_spl(frames, shx, shy):
    x_spl = np.poly1d((np.polyfit(frames, shx, 1)))
    y_spl = np.poly1d((np.polyfit(frames, shy, 1)))
    x_splined = x_spl(frames)
    y_splined = y_spl(frames)
    return x_splined, y_splined


def rot_spl(frames, psi, theta, phi):
    psi_spl = np.poly1d((np.polyfit(frames, psi, 1)))
    theta_spl = np.poly1d((np.polyfit(frames, theta, 1)))
    phi_spl = np.poly1d((np.polyfit(frames, phi, 1)))
    psi_splined = psi_spl(frames)
    theta_splined = theta_spl(frames)
    phi_splined = phi_spl(frames)
    return psi_splined, theta_splined, phi_splined


def regularize_film(
    filename,
    parfile,
    parameters,
    tilt_col,
    ptlind_col,
    scanor_col,
    actual_pixel,
    prev_arr,
    input_arr,
    tilt_count,
    xf_frames=np.array([])
):

    logger.info("Now processing tilt %d", tilt_count)
    tilt_idx = np.where(input_arr[:, tilt_col].astype(int) == tilt_count)[0]
    
    if tilt_idx.shape[0] == 0:
        logger.warning(f"Tilt {tilt_count} does not contain particles. Skipping... ")
        return

    tilt_arr = input_arr[tilt_idx, :]
    prev_tilt_arr = prev_arr[tilt_idx]

    # get the index of tilt to obtain global frame alignment from the right file
    tilt_list_indexes = np.unique(input_arr[:, tilt_col], return_index=True)[1]
    tilt_list = np.array([input_arr[:, tilt_col][index] for index in sorted(tilt_list_indexes)])
    tilt_index_xf_files = np.where(tilt_list == tilt_count)[0][0]

    # get total number of particles
    ptlind_list = np.sort(np.unique(tilt_arr[:, ptlind_col]).astype(int))
    particle_num = len(ptlind_list)

    frame_num = len(np.unique(tilt_arr[:, scanor_col]))

    total_shift_dimensions = 5

    noisy_shifts = np.zeros((particle_num, frame_num, total_shift_dimensions))
    noisy_regularized_shifts = np.copy(noisy_shifts)
    clean_shifts = np.copy(noisy_shifts)
    clean_regularized_shifts = np.copy(noisy_shifts)
    clean_regularized_poses = np.copy(noisy_shifts)


    time_sigma = parameters["csp_time_sigma"]
    spatial_sigma = parameters["csp_spatial_sigma"]
    name_png = filename

    # traverse particles
    for particle_count, ptlind in enumerate(ptlind_list):
        save_plots = tilt_count == 0 and particle_count == 0
        particle_indices = np.where(
            tilt_arr[:, ptlind_col].astype(int) == ptlind
        )[0]

        # XD: SCANOR starts from 0
        """
        XD note 2020 10 31: Rethink regularization for tomography. Removed valid_indexes to make it easier for SPR.
        """
        # valid_indexes = ( film_arr[particle_indices,scanor_col] - 1 ).astype('i')

        # frame_order = np.argsort( film_arr[particle_indices,scanor_col] )

        # valid_indexes = np.sort( film_arr[particle_indices,scanor_col] ).astype('i')
        # import pdb; pdb.set_trace()
        # XD: Possible issues
        particle_arr = tilt_arr[particle_indices]
        prev_particle_arr = prev_tilt_arr[particle_indices]

        psi, theta, phi, shx, shy = [particle_arr[:, i] for i in range(1, 6)]


        # subtract the baseline shifts
        d_psi, d_theta, d_phi, d_shx, d_shy = list(
            map(
                lambda x, y: x - y,
                [psi, theta, phi, shx, shy],
                [prev_particle_arr[:, i] for i in range(1, 6)],
            )
        )

        # xf_d_shx, xf_d_shy = -d_shx / actual_pixel, -d_shy / actual_pixel
        xf_d_shx, xf_d_shy = - particle_arr[:, -5] / actual_pixel, - particle_arr[:, -4] / actual_pixel

        if parameters["csp_rotreg"]:
            # two_rotations = np.concatenate((d_phi.reshape((-1,1)), d_theta.reshape((-1,1))), axis=1)
            # noisy_shifts[particle_count,valid_indexes,:2] = two_rotations

            three_rotations = np.concatenate(
                (
                    d_psi.reshape((-1, 1)),
                    d_theta.reshape((-1, 1)),
                    d_phi.reshape((-1, 1)),
                ),
                axis=1,
            )
            noisy_shifts[particle_count, :, :3] = three_rotations

            # using old AB's method
            # d_psi_splined = np.degrees( np.arctanh( fit_spline_trajectory_1D( np.tanh(np.radians(d_psi)),1,3,1) ))
            # d_theta_splined = np.degrees( np.arctanh( fit_spline_trajectory_1D(np.tanh(np.radians(d_theta)),1,3,1) ))
            # d_phi_splined = np.degrees( np.arctanh( fit_spline_trajectory_1D(np.tanh(np.radians(d_phi)),1,3,1) ))

            # d_psi_splined = fit_angular_trajectory_1D_new( d_psi, time_sigma )
            # d_theta_splined = fit_angular_trajectory_1D_new( d_theta, time_sigma )
            # d_phi_splined = fit_angular_trajectory_1D_new( d_phi, time_sigma )

            d_rot1, d_rot2, d_rot3 = d_psi, d_theta, d_phi

            if time_sigma > 0:
                # using XD method
                if "xd" in parameters["csp_rotreg_method"].lower():
                    d_psi_splined = fit_spline_trajectory_1D(d_rot1, k=5, factor=0.6)
                    d_theta_splined = fit_spline_trajectory_1D(d_rot2, k=5, factor=0.6)
                    d_phi_splined = fit_spline_trajectory_1D(d_rot3, k=5, factor=0.6)

                # using AB1 method
                elif "ab1" in parameters["csp_rotreg_method"].lower():
                    d_psi_splined = np.degrees(
                        np.arctanh(
                            fit_spline_trajectory_1D(
                                np.tanh(np.radians(d_rot1)), 1, 3, 1
                            )
                        )
                    )
                    d_theta_splined = np.degrees(
                        np.arctanh(
                            fit_spline_trajectory_1D(
                                np.tanh(np.radians(d_rot2)), 1, 3, 1
                            )
                        )
                    )
                    d_phi_splined = np.degrees(
                        np.arctanh(
                            fit_spline_trajectory_1D(
                                np.tanh(np.radians(d_rot3)), 1, 3, 1
                            )
                        )
                    )

                # using AB2 method
                else:
                    d_psi_splined = fit_angular_trajectory_1D_new(d_rot1, time_sigma)
                    d_theta_splined = fit_angular_trajectory_1D_new(d_rot2, time_sigma)
                    d_phi_splined = fit_angular_trajectory_1D_new(d_rot3, time_sigma)
            else:
                d_psi_splined = d_psi
                d_theta_splined = d_theta
                d_phi_splined = d_phi

            # d_rot1 = d_psi
            # d_rot2 = d_theta
            # d_rot3 = d_phi
            # d_rot2_splined = d_theta_splined
            # d_rot1_splined = d_phi_splined

            # if save_plots and particle_count % 100 == 0 and len(psi) > 3:
            #     angles_before = np.array( ( d_rot1, d_rot2 ) ).transpose()
            #     angles_after = np.array( ( d_rot1_splined, d_rot2_splined ) ).transpose()
            #     try:
            #         analysis.plot.plot_angular_trajectory( angles_after, '../scratch/' + parfile.replace('.par','') + '_rot_F%04d_P%04d.png' % ( film_count, particle_count ), angles_before )
            #     except:
            #         pass

            noisy_regularized_shifts[particle_count, :, :3] = np.stack(
                (d_psi_splined, d_theta_splined, d_phi_splined)
            ).T

        if parameters["csp_transreg"]:
            translations = np.concatenate(
                (xf_d_shx.reshape((-1, 1)), xf_d_shy.reshape((-1, 1))), axis=1
            )

            noisy_shifts[particle_count, :, -2:] = translations
            
            if time_sigma > 0:
                # using XD method
                if "xd" in parameters["csp_transreg_method"].lower():
                    new_trans_shifts = fit_spline_trajectory(
                        translations, k=5, factor=0.6
                    )
                    # d_x_splined = new_trans_shifts[:, 0]
                    # d_y_splined = new_trans_shifts[:, 1]
                    d_x_splined = new_trans_shifts[:, -2]
                    d_y_splined = new_trans_shifts[:, -1]

                # using particle frame alignment method
                elif "spline" in parameters["csp_transreg_method"].lower():
                    splined_shifts = fit_spline_trajectory(
                        noisy_shifts[particle_count, :, :]
                    )
                    # d_x_splined = splined_shifts[:, 0]
                    # d_y_splined = splined_shifts[:, 1]
                    d_x_splined = splined_shifts[:, -2]
                    d_y_splined = splined_shifts[:, -1]

                # using AB method
                else:
                    d_x_splined = fit_spatial_trajectory_1D_new(xf_d_shx, time_sigma)
                    d_y_splined = fit_spatial_trajectory_1D_new(xf_d_shy, time_sigma)
            else:
                d_x_splined = xf_d_shx
                d_y_splined = xf_d_shy

            # noisy_regularized_shifts[particle_count,:,-2:] = np.stack((d_x_splined, d_y_splined)).T
            noisy_regularized_shifts[particle_count, :, -2:] = np.stack(
                (d_x_splined, d_y_splined)
            ).T

            if save_plots:
                # translations_before = np.array( ( noisy_shifts[particle_count,:,-2], noisy_shifts[particle_count,:,-1] ) ).transpose()
                translations_before = np.array((xf_d_shx, xf_d_shy)).transpose()
                translations_after = np.array((d_x_splined, d_y_splined)).transpose()
                
                try:
                    analysis.plot.plot_trajectory_raw(
                        translations_after,
                        "./"
                        + name_png
                        + "_trans_T%04d_P%04d_noisy.png" % (tilt_count, particle_count),
                        translations_before,
                    )
                    
                except:
                    pass

    project_dir = os.getcwd()
    movie_name = filename + ".mrc"
    ali_path = os.path.join(project_dir, movie_name)
    box_path = os.path.join(project_dir, movie_name)
    name = box_path.strip(".mrc")

    if "tomo" in parameters["data_mode"].lower():
        # retrieve 3D particle positions
        # box = imod.coordinates_from_mod_file(
        #     project_dir + "/mod/%s.spk" % movie_name.strip(".mrc")
        # )

        # # HF Liu: correct the binning
        # ctf = np.loadtxt(project_dir + "/ctf/%s.ctf" % movie_name.strip(".mrc"))
        # micrographsize_x, micrographsize_y = ctf[6], ctf[7]
        # squarex = int(math.pow(2, math.ceil(math.log(micrographsize_x, 2))))
        # squarey = int(math.pow(2, math.ceil(math.log(micrographsize_y, 2))))
        # square = max(micrographsize_x, micrographsize_y)
        # binning = int(math.ceil(square / 512))

        micrograph_drift = np.round(xf_frames[tilt_index_xf_files])

        # box *= binning
        boxes3d_file = f"{name}_boxes3d.txt"
        box = np.array(read_3dbox(boxes3d_file))

        particles_indexes_in_tilt_series = np.unique(tilt_arr[:, ptlind_col])

        box = box[particles_indexes_in_tilt_series.astype("int"), 1:4]
        box = box.astype(float)

        assert (
            box.shape[0] == particle_num
        ), f"boxx file selected particles and par file number of particles don't match - {box.shape[0]} v.s. {particle_num}"

    else:

        micrograph_drift = np.round(np.loadtxt(ali_path.replace(".mrc", ".xf"), ndmin=2))

        boxx = pyp_metadata.LocalMetadata(name + ".pkl", is_spr=True).data["box"].to_numpy()

        # logger.info("boxx path:", name)
        # logger.info("boxx shape:", boxx.shape)
        # logger.info(np.logical_and( boxx[:,4] == 1, boxx[:,5] >= int( mparameters['extract_cls']) ))
        # logger.info(np.__version__)
        # logger.info("particle cls: ", mparameters['extract_cls'])

        # XD: added this to account for possible error in updating box files
        box = boxx[
            np.logical_and(
                boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"])
            )
        ]

        if box.shape[0] != particle_num:
            box = boxx[boxx[:, 5] >= int(parameters["extract_cls"])]

        assert (
            box.shape[0] == particle_num
        ), f"boxx file selected particles and par file number of particles don't match - {box.shape[0]} v.s. {particle_num}"


    import scipy.spatial.distance

    # set zero weights for missing particle projections
    particle_indices = set(np.unique(tilt_arr[:, ptlind_col]).astype("i"))
    # mask = np.array([(i in particle_indices) for i in range(box.shape[0])])
    mask = np.array([True for i in range(box.shape[0])])

    # XD: decide on the iter we will perform this for
    for particle_count, ptlind in enumerate(ptlind_list):
        save_plots = tilt_count == 0 and particle_count == 0
        if "tomo" in parameters["data_mode"].lower():
            distances = scipy.spatial.distance.cdist(
                box, box[particle_count].reshape(1, 3)
            )
        else:
            distances = scipy.spatial.distance.cdist(
                box[:, :2], np.array(box[particle_count][:2], ndmin=2)
            )


        if spatial_sigma > 0:
            # At a minimum, average this number of particles
            minimum_particles_to_average = 5

            particle_spatial_sigma = spatial_sigma
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

            # set zero weights for missing particle projections
            weights[~mask] = 0

            weights /= weights.sum()

            # update the locally weighted trajectories
            if parameters["csp_rotreg"]:
                clean_shifts[particle_count, :, 0] = np.multiply(
                    noisy_shifts[:, :, 0], weights
                ).sum(axis=0)
                clean_shifts[particle_count, :, 1] = np.multiply(
                    noisy_shifts[:, :, 1], weights
                ).sum(axis=0)
                clean_shifts[particle_count, :, 2] = np.multiply(
                    noisy_shifts[:, :, 2], weights
                ).sum(axis=0)

            if parameters["csp_transreg"]:
                clean_shifts[particle_count, :, -1] = np.multiply(
                    noisy_shifts[:, :, -1], weights
                ).sum(axis=0)
                clean_shifts[particle_count, :, -2] = np.multiply(
                    noisy_shifts[:, :, -2], weights
                ).sum(axis=0)

                # add global translations
                # logger.info("clean shifts before global")
                # logger.info(clean_shifts[ particle_count, :, -2: ])
                
                if distances.size > minimum_particles_to_average:
                    clean_shifts[particle_count, :, -2:] += micrograph_drift[:, -2:]
                else:
                    # revert to global drift values
                    clean_shifts[particle_count, :, -2:] = micrograph_drift[:, -2:]
                
                
                # logger.info("global movements")
                # logger.info(micrograph_drift[ :, -2: ])

                # logger.info("clean shifts after global")
                # logger.info(clean_shifts[ particle_count, :, -2: ])

                if "tomo" in parameters["data_mode"].lower():
                    global_unrounded = xf_frames[tilt_index_xf_files]
                else:
                    global_unrounded = np.loadtxt(ali_path.replace(".mrc", ".xf"), ndmin=2)

                
                if save_plots:
                    # translations_before = np.array( ( noisy_shifts[particle_count,:,-2], noisy_shifts[particle_count,:,-1] ) ).transpose()
                    translations_before = global_unrounded[:, -2:]
                    translations_after = global_unrounded[:, -2:]
                    try:
                        analysis.plot.plot_trajectory_raw(
                            translations_after,
                            "./"
                            + name_png
                            + "_trans_T%04d_P%04d_global.png"
                            % (tilt_count, particle_count),
                            translations_before,
                        )
                    except:
                        pass

        else:
            clean_shifts[particle_count, :, 0] = noisy_shifts[particle_count, :, 0]
            clean_shifts[particle_count, :, 1] = noisy_shifts[particle_count, :, 1]
            clean_shifts[particle_count, :, 2] = noisy_shifts[particle_count, :, 2]
            clean_shifts[particle_count, :, -1] = noisy_shifts[particle_count, :, -1]
            clean_shifts[particle_count, :, -2] = noisy_shifts[particle_count, :, -2]

        particle_indices = np.where(
            tilt_arr[:, ptlind_col].astype(int) == ptlind
        )[0]

        # XD: SCANOR starts from 0

        # valid_indexes = ( film_arr[particle_indices,scanor_col] - 1 ).astype('i')
        """
        XD 2020 10 31: refer to comment above on the use of valid_indexes
        """
        # valid_indexes = np.sort( film_arr[particle_indices,scanor_col] ).astype('i')
        # logger.info("valid indexes", valid_indexes)

        # XD: potential issues here
        particle_arr = tilt_arr[particle_indices]
        prev_particle_arr = prev_tilt_arr[particle_indices]

        # import pdb; pdb.set_trace()

        psi, theta, phi, shx, shy = [particle_arr[:, i] for i in range(1, 6)]

        if parameters["csp_rotreg"] and len(psi) > 3:
            # d_rot1, d_rot2 = clean_shifts[particle_count,valid_indexes,0], clean_shifts[particle_count,valid_indexes,1]
            # two_rotations = np.concatenate((d_rot1.reshape((-1,1)), d_rot2.reshape((-1,1))), axis=1)

            d_rot1, d_rot2, d_rot3 = (
                clean_shifts[particle_count, :, 0],
                clean_shifts[particle_count, :, 1],
                clean_shifts[particle_count, :, 2],
            )

            # new_rot_shifts = fit_spline_trajectory(two_rotations, k=5, factor=0.6)
            # d_rot1_splined = new_rot_shifts[:,0]
            # d_rot2_splined = new_rot_shifts[:,1]

            if time_sigma > 0:
                # using XD method
                if "xd" in parameters["csp_rotreg_method"].lower():
                    d_psi_splined = fit_spline_trajectory_1D(d_rot1, k=5, factor=0.6)
                    d_theta_splined = fit_spline_trajectory_1D(d_rot2, k=5, factor=0.6)
                    d_phi_splined = fit_spline_trajectory_1D(d_rot3, k=5, factor=0.6)
                # using AB1 method
                elif "ab1" in parameters["csp_rotreg_method"].lower():
                    d_psi_splined = np.degrees(
                        np.arctanh(
                            fit_spline_trajectory_1D(
                                np.tanh(np.radians(d_rot1)), 1, 3, 1
                            )
                        )
                    )
                    d_theta_splined = np.degrees(
                        np.arctanh(
                            fit_spline_trajectory_1D(
                                np.tanh(np.radians(d_rot2)), 1, 3, 1
                            )
                        )
                    )
                    d_phi_splined = np.degrees(
                        np.arctanh(
                            fit_spline_trajectory_1D(
                                np.tanh(np.radians(d_rot3)), 1, 3, 1
                            )
                        )
                    )
                # using AB2 method
                else:
                    d_psi_splined = fit_angular_trajectory_1D_new(d_rot1, time_sigma)
                    d_theta_splined = fit_angular_trajectory_1D_new(d_rot2, time_sigma)
                    d_phi_splined = fit_angular_trajectory_1D_new(d_rot3, time_sigma)
            else:
                d_psi_splined = d_rot1
                d_theta_splined = d_rot2
                d_phi_splined = d_rot3

            # d_psi_splined, d_theta_splined, d_phi_splined = d_rot1, d_rot2, d_rot3

            # d_rot2 = d_theta
            # d_rot1 = d_phi
            # d_rot2_splined = d_theta_splined
            # d_rot1_splined = d_phi_splined

            # print "Doing rotational regularization"
            # psi_splined, theta_splined, phi_splined = rot_spl(frames, psi, theta, phi)
            # psi_splined = np.degrees( np.arctanh( fit_spline_trajectory_1D( np.tanh(np.radians(psi)),1,3,1) ))
            # theta_splined = np.degrees( np.arctanh( fit_spline_trajectory_1D(np.tanh(np.radians(theta)),1,3,1) ))
            # phi_splined = np.degrees( np.arctanh( fit_spline_trajectory_1D(np.tanh(np.radians(phi)),1,3,1) ))

            # phi_splined = d_rot1_splined + prev_particle_arr[:,3]
            # theta_splined = d_rot2_splined + prev_particle_arr[:,2]
            # psi_splined = psi

            mask = parameters["refine_mask"].split(":")[-1].split(",")
            if int(mask[0]) > 0:
                psi_splined = d_psi_splined
            else:
                psi_splined = d_rot1
            if int(mask[1]) > 0:
                theta_splined = d_theta_splined
            else:
                theta_splined = d_rot2
            if int(mask[2]) > 0:
                phi_splined = d_phi_splined
            else:
                phi_splined = d_rot3

            clean_regularized_shifts[particle_count, :, :3] = np.stack(
                (psi_splined, theta_splined, phi_splined)
            ).T

            # add back differences to the baseline
            psi_splined += prev_particle_arr[:, 1]
            theta_splined += prev_particle_arr[:, 2]
            phi_splined += prev_particle_arr[:, 3]

            # normalize final angular values
            psi_splined = np.where(psi_splined > 360, psi_splined % 360, psi_splined)
            psi_splined = np.where(psi_splined < 0, psi_splined + 360, psi_splined)
            theta_splined = np.where(
                theta_splined > 360, theta_splined % 360, theta_splined
            )
            theta_splined = np.where(
                theta_splined < 0, theta_splined + 360, theta_splined
            )
            phi_splined = np.where(phi_splined > 360, phi_splined % 360, phi_splined)
            phi_splined = np.where(phi_splined < 0, phi_splined + 360, phi_splined)

            # if particle_count == 10:
            #    print 'psi_splined, theta_splined, phi_splined (after normalization)', psi_splined, theta_splined, phi_splined

            clean_regularized_poses[particle_count, :, :3] = np.stack(
                (psi_splined, theta_splined, phi_splined)
            ).T

            if (
                save_plots
                and len(psi) > 3
            ):
                angles_before = np.array((theta, phi)).transpose()
                angles_after = np.array((d_theta_splined, d_phi_splined)).transpose()

                try:
                    analysis.plot.plot_angular_trajectory(
                        angles_after,
                        "./"
                        + name_png
                        + "_rot_T%04d_P%04d.png" % (tilt_count, particle_count),
                        angles_before,
                    )
                except:
                    pass

        else:
            # print "Not rotational regularization"
            psi_splined, theta_splined, phi_splined = psi, theta, phi

        if parameters["csp_transreg"]:
            # import pdb; pdb.set_trace()
            xf_d_shx, xf_d_shy = (
                clean_shifts[particle_count, :, -2],
                clean_shifts[particle_count, :, -1],
            )

            if time_sigma > 0:
                # using XD method
                # import pdb; pdb.set_trace()
                if "xd" in parameters["csp_transreg_method"].lower():
                    translations = np.concatenate(
                        (xf_d_shx.reshape((-1, 1)), xf_d_shy.reshape((-1, 1))), axis=1
                    )
                    new_trans_shifts = fit_spline_trajectory(
                        translations, k=5, factor=0.6
                    )
                    d_x_splined = new_trans_shifts[:, -2]
                    d_y_splined = new_trans_shifts[:, -1]

                # using particle frame alignment method
                elif "spline" in parameters["csp_transreg_method"].lower():
                    splined_shifts = fit_spline_trajectory(
                        clean_shifts[particle_count, :, :]
                    )
                    d_x_splined = splined_shifts[:, -2]
                    d_y_splined = splined_shifts[:, -1]

                # using AB method
                else:
                    d_x_splined = fit_spatial_trajectory_1D_new(xf_d_shx, time_sigma)
                    d_y_splined = fit_spatial_trajectory_1D_new(xf_d_shy, time_sigma)
                # import pdb; pdb.set_trace()
                # logger.info("d x y splined stacked", np.stack((d_x_splined, d_y_splined)).T)
            else:
                d_x_splined = xf_d_shx
                d_y_splined = xf_d_shy
            # import pdb; pdb.set_trace()
            if save_plots:
                # translations_before = np.array( ( noisy_shifts[particle_count,valid_indexes,-2], noisy_shifts[particle_count,valid_indexes,-1] ) ).transpose()
                translations_before = np.array((xf_d_shx, xf_d_shy)).transpose()
                translations_after = np.array((d_x_splined, d_y_splined)).transpose()
                try:
                    analysis.plot.plot_trajectory_raw(
                        translations_after,
                        "./"
                        + name_png
                        + "_trans_T%04d_P%04d_total.png" % (tilt_count, particle_count),
                        translations_before,
                    )
                except:
                    pass
            
            clean_regularized_shifts[particle_count, :, -2:] = np.stack(
                (d_x_splined, d_y_splined)
            ).T
            
            if spatial_sigma > 0:
                d_x_splined = d_x_splined - micrograph_drift[:, -2]
                d_y_splined = d_y_splined - micrograph_drift[:, -1]
                xf_d_shx -= micrograph_drift[:, -2]
                xf_d_shy -= micrograph_drift[:, -1]
            
            # import pdb; pdb.set_trace()

            # clean_regularized_shifts[particle_count, :, -2:] = np.stack(
            #     (d_x_splined, d_y_splined)
            # ).T

            if save_plots:
                translations_before = np.array(
                    (
                        noisy_shifts[particle_count, :, -2],
                        noisy_shifts[particle_count, :, -1],
                    )
                ).transpose()
                # translations_before = np.array( ( xf_d_shx , xf_d_shy ) ).transpose()
                translations_after = np.array((d_x_splined, d_y_splined)).transpose()
                try:
                    analysis.plot.plot_trajectory_raw(
                        translations_after,
                        "./"
                        + name_png
                        + "_trans_T%04d_P%04d_local.png" % (tilt_count, particle_count),
                        translations_before,
                    )
                except:
                    pass

                # combine global, local, total translation plots into one
                xf_name_template = (
                    "./"
                    + name_png
                    + "_trans_T%04d_P%04d" % (tilt_count, particle_count)
                )
                global_xf_name = xf_name_template + "_global.png"
                local_xf_name = xf_name_template + "_local.png"
                total_xf_name = xf_name_template + "_total.png"
                combined_xf_name = xf_name_template + "_combined.png"
                com = "/usr/bin/montage {0} {1} {2} -geometry +0+0 -tile 3x1 -geometry +0+0 {3}".format(
                    global_xf_name, local_xf_name, total_xf_name, combined_xf_name
                )
                subprocess.getstatusoutput(com)
                
                img2webp(combined_xf_name, combined_xf_name.replace(".png", ".webp"))

            # convert from stack size pixels to FREALIGN shifts
            x_splined = ( -d_x_splined * actual_pixel - prev_particle_arr[:,-5] ) + prev_particle_arr[:, 4]
            y_splined = ( -d_y_splined * actual_pixel - prev_particle_arr[:,-4] ) + prev_particle_arr[:, 5]
            # import pdb; pdb.set_trace()

        else:
            x_splined, y_splined = shx, shy

        clean_regularized_poses[particle_count, :, -2:] = np.stack(
            (x_splined, y_splined)
        ).T

        clean_regularized_dx = x_splined - prev_particle_arr[:, 4] + prev_particle_arr[:, -5] 
        clean_regularized_dy = y_splined - prev_particle_arr[:, 5] + prev_particle_arr[:, -4]

        # import pdb; pdb.set_trace()
        new_poses = np.stack(
            (psi_splined, theta_splined, phi_splined, x_splined, y_splined)
        )
        new_dshisfts = np.stack(
                (clean_regularized_dx, clean_regularized_dy)
        )

        particle_arr[:, 1:6] = np.around(new_poses, decimals=2).T

        particle_arr[:, -5:-3] = new_dshisfts.T

        tilt_arr[particle_indices, :] = particle_arr

    

    # show sample trajectories
    if tilt_count == 0:

        if "tomo" in parameters["data_mode"].lower():
            # use positions of the middle frame to plot trajectories
            allboxes = np.loadtxt(f"{filename}_local.allboxes", dtype=int)
            frame_indexes = np.unique(allboxes[:, -1])
            middle_frame_index = int(np.median(frame_indexes))
            box_2d = allboxes[tilt_idx, :]
            box_2d = box_2d[box_2d[:, -1] == middle_frame_index]
        else: 
            box_2d = box 

        # logger.info("regularized shifts", clean_regularized_shifts[-1])
        if parameters["csp_rotreg"]:
            analysis.plot.plot_trajectories(
                name_png,
                ali_path,
                box_2d, 
                clean_regularized_shifts,
                parameters,
                rotation=True,
                savefig=True,
            )
        else:
            analysis.plot.plot_trajectories(
                name_png,
                ali_path,
                box_2d,
                clean_regularized_shifts,
                parameters,
                rotation=False,
                savefig=True,
            )

    input_arr[tilt_idx, :] = tilt_arr

    
    


def regularize(
    filename,
    parfile,
    prev_parfile,
    output,
    parameters,
):

    if parameters["csp_rotreg"] or parameters["csp_transreg"]:

        tilt_col = 20 - 1
        ptlind_col = 17 - 1
            
        # NOTE : lets use CNFDNC as film col
        scanor_col = 21 - 1

        actual_pixel = float(parameters["scope_pixel"]) * float(
            parameters["data_bin"]
        )

        # prev_arr = np.array( [line.split() for line in open( prev_parfile ) if not line.startswith('C') ], dtype=float)
        prev_arr = frealign_parfile.Parameters.from_file(prev_parfile).data
        # input_arr = np.array( [line.split() for line in open( parfile ) if not line.startswith('C') ], dtype=float )
        input_arr = frealign_parfile.Parameters.from_file(parfile).data
        
        if "tomo" in parameters["data_mode"].lower():
            try:
                xf_frames, xf_files = tomo_load_frame_xf(parameters, filename, xf_path="./")
            except:
                metadata = pyp_metadata.LocalMetadata(f"{filename}.pkl").data
                xf_frames = [metadata["drift"][tilt_idx].to_numpy() for tilt_idx in sorted(metadata["drift"].keys())]
        else: xf_frames = np.array([])
        
        if input_arr.shape[1] > 20:

            # traverse micrographs/tilt-series
            # use multiprocessing
            
            arguments = []
            for tilt_count in range(len(np.unique(input_arr[:, tilt_col]))):
                # pool.apply_async( regularize_film, args=(parfile, rotational_refinement, translational_refinement,rotational_method, translational_method, save_plots, spatial_sigma, time_sigma, \
                # mparameters, fparameters, film_col, ptlind_col, scanor_col, actual_pixel, prev_arr, input_arr, film_count) )
                 
                if not (tilt_count >= parameters["csp_UseImagesForRefinementMin"] and ( parameters["csp_UseImagesForRefinementMax"] == -1 or tilt_count <= parameters["csp_UseImagesForRefinementMax"] )):
                    continue

                regularize_film(
                    filename,
                    parfile,
                    parameters,
                    tilt_col,
                    ptlind_col,
                    scanor_col,
                    actual_pixel,
                    prev_arr,
                    input_arr,
                    tilt_count,
                    xf_frames,
                )
                """
                arguments.append(
                    (
                    filename,
                    parfile,
                    parameters,
                    tilt_col,
                    ptlind_col,
                    scanor_col,
                    actual_pixel,
                    prev_arr,
                    input_arr,
                    tilt_count,
                    xf_frames,
                    )
                )

            mpi.submit_function_to_workers(regularize_film, arguments)
            """

            allparxs = deque()

            # frealignx = isfrealignx(parfile)
            (
                fieldwidths,
                fieldstring,
                _,
                _,
            ) = frealign_parfile.Parameters.format_from_parfile(parfile)

            for row in input_arr:

                allparxs.append(fieldstring[:-1] % tuple(row.tolist()))

            f = open(output, "w")

            for line in open(parfile):
                if line.startswith("C"):
                    f.write(line)
                else:
                    break

            # write header information
            # f.write("""C FREALIGN parameter file for CSP\n""")
            # f.write("""C     1       2       3       4         5         6       7     8        9       10      11      12        13         14      15      16       17        18        19        20        21        22        23        24        25        26        27        28        29        30        31        32        33        34        35        36        37        38        39        40        41        42        43        44        45\n""")
            # f.write("""C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LOGP      SIGMA   SCORE  CHANGE   PTLIND    TILTAN    DOSEXX    SCANOR    CNFDNC    PTLCCX      AXIS     NORM0     NORM1     NORM2  MATRIX00  MATRIX01  MATRIX02  MATRIX03  MATRIX04  MATRIX05  MATRIX06  MATRIX07  MATRIX08  MATRIX09  MATRIX10  MATRIX11  MATRIX12  MATRIX13  MATRIX14  MATRIX15      PPSI    PTHETA      PPHI\n""")

            f.writelines("%s\n" % item for item in allparxs)
            f.close()

