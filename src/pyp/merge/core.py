import math
import os
import shutil
import subprocess
import datetime
import time

from pathlib import Path

import numpy as np

from pyp import utils
from pyp.analysis import plot
from pyp.inout.image import mrc
from pyp.inout.image.core import get_image_dimensions
from pyp.inout.utils import pyp_edit_box_files as imod
from pyp.merge import weights as pyp_weights
from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path, get_aretomo_path, get_topaz_path, get_gpu_ids
from pyp.utils import get_relative_path
from pyp.utils.timer import Timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def tomo_is_done(name, directory):
    extensions = ["rec"] 
    return utils.has_files(os.path.join(directory, name), extensions)


def apply_weights(
    filename,
    shifts,
    binFactor=2,
    delta=250,
    deltaF=0.0000001,
    radiK=0.5,
    fweights=np.empty([0]),
    blend=np.empty([0]),
    scores=np.empty([0]),
    output_stack="",
    diagnostics=False,
):

    # middle = np.log(0.0025)/np.log(0.85)

    # apply combined weights
    boxsize = int(mrc.readHeaderFromFile(filename)["nx"])
    frames = int(mrc.readHeaderFromFile(filename)["nz"])
    fourier_average = np.zeros([boxsize, boxsize // 2 + 1], dtype=complex)
    weights = pyp_weights.combined_weights_movie(
        shifts, radiK, delta, deltaF, boxsize, boxsize, binFactor, scores
    )

    # blend with measured weights
    if len(blend) > 0:
        for f in range(frames):
            weights[f, :, :] = (1 - blend) * fweights[f, :, :] + blend * weights[
                f, :, :
            ]
        wsum = np.sum(weights, axis=0)
        for f in range(frames):
            weights[f, :, :] /= wsum

    # graphical output
    if diagnostics:
        squares = 38  # 19
        M = plot.contact_sheet(weights[:, : boxsize // 2 + 1 : 1, ::1], squares, False)
        N = plot.contact_sheet(weights[:, : boxsize // 2 + 1 : 1, ::1], squares, True)
        # writepng( np.vstack([M,N]), '{0}_weights.png'.format( filename[:-4] ) )
        import matplotlib.pyplot as plt

        plt.clf()
        plt.figure(figsize=(17, 2.5))
        plt.imshow(np.vstack([M, N]), cmap="coolwarm")
        a = plt.gca()
        a.set_frame_on(False)
        a.set_xticks([])
        a.set_yticks([])
        plt.axis("off")
        # plt.savefig( '{0}_weights.png'.format( filename[:-4] ), bbox_inches='tight', pad_inches=0, dpi=302 )
        plt.savefig(
            "{0}_weights.png".format(filename[:-4]), bbox_inches="tight", pad_inches=0
        )
        plt.close()
        plot.guinier_plot(
            weights,
            "{0}_weights_new.png".format(filename[:-4]),
            binFactor * float(project_params.load_pyp_parameters(".")["scope_pixel"]),
        )
        # plot.guinier_plot( weights, '{0}_weights_new.pdf'.format( filename[:-4] ), binFactor * float( project_params.load_pyp_parameters('.')['scope_pixel'] ) )

    for f in range(frames):
        # weights_sum += weights[f,:,:]
        if len(output_stack) > 0:
            weighted_frame = np.fft.irfft2(
                weights[f, :, :] * np.fft.rfft2(mrc.readframe(filename, f))
            )
            if f > 0:
                mrc.append(weighted_frame, output_stack)
            else:
                mrc.write(weighted_frame, output_stack)
        else:
            fourier_average += weights[f, :, :] * np.fft.rfft2(
                mrc.readframe(filename, f)
            )

    # aligned_average = np.fft.irfft2( fourier_average / weights_sum )
    if len(output_stack) > 0:
        return
    else:
        return np.fft.irfft2(fourier_average)


def do_exclude_virions(name):
    """Identify ignored virions (exclude_virions) from _exclude_virions.mod."""
    # HF vir: add function obtaining the threshold on all kinds of size of png
    ############################################################################
    exclude_model = "%s_exclude_virions.mod" % name
    if os.path.isfile(exclude_model):
        exclude_virions = imod.coordinates_from_mod_file(exclude_model)
        x_dimension_exclude_virions = imod.dimension_from_mod_file(exclude_model)[0]
        # use the x-coordinate to figure out which threshold to use for segmentation
        if exclude_virions.size == 0:
            exclude_virions = np.array([[8, 0, 0]])
        else:
            exclude_virions[:, 0] = exclude_virions[:, 0].astype(int) / (
                x_dimension_exclude_virions / 9
            )
            exclude_virions = exclude_virions.astype(int)
    else:
        exclude_virions = np.empty(0)
    ###########################################################################
    return exclude_virions


def do_exclude_views(name, tilt_angles):
    """Identify ignored views (exclude_views) from s_exclude_views.mod and _RAPTOR.log"""
    exclude_model = "%s_exclude_views.mod" % name
    if os.path.isfile(exclude_model):
        ignore = imod.indexes_from_mod_file(exclude_model)
        if not ignore == []:
            exclude_views = "-EXCLUDELIST2 " + ",".join(
                str(e) for e in np.array(sorted(ignore), dtype="int") + 1
            )
        else:
            logger.warning("Exclusion view file {0} is empty".format(exclude_model))
            exclude_views = ""
    else:
        exclude_views = ""
    # exclude_views = "-EXCLUDELIST2 1,2,3,4,5,6,7,8,34,35,36,37,38,39,40,41"

    # Update excluded views per RAPTOR alignment
    # checks for existing tilt series alignment
    if os.path.exists("{0}_RAPTOR.log".format(name)):
        excluded = []
        # supposedly produces clean tilt series by excluding unalignable tilts
        # but apparently not doing this rn
        if excluded:
            logger.warning(f"Failed to align views {excluded}")
            if exclude_views == "":
                exclude_views = "-EXCLUDELIST2 " + ",".join(excluded)
            else:
                # combine excluded views into unique sorted list
                excluded.extend(
                    exclude_views.replace(" ", "").split("-EXCLUDELIST2")[1].split(",")
                )
                exclude_views = "-EXCLUDELIST2 " + ",".join(
                    [str(i) for i in sorted([int(j) for j in list(set(excluded))])]
                )

            # create clean aligned tilt series and corresponding tilt-angle file
            # Look at this
            newstack_exclude_views = "-fromone -exclude " + ",".join(
                [str(i) for i in sorted([int(j) for j in list(set(excluded))])]
            )
            command = "{0}/bin/newstack -input {1}.ali -output {1}_clean.ali {2}".format(
                get_imod_path(), name, newstack_exclude_views
            )
            logger.info(command)
            logger.info(
                subprocess.check_output(
                    command, stderr=subprocess.STDOUT, shell=True, text=True
                )
            )

            tilt_angles_clean = []
            for i in range(tilt_angles.size):
                if str(i + 1) not in excluded:
                    tilt_angles_clean.append(str(tilt_angles[i]))

            f = open("{0}_clean.tlt".format(name), "w")
            f.write("\n".join(str(elem) for elem in tilt_angles_clean))
            f.close()

    return exclude_views


def weight_stack(input_stack, output_stack, weights):

    # figure out image dimensions
    header = mrc.readHeaderFromFile(input_stack)
    x, y, frames = header["nx"], header["ny"], header["nz"]

    # calculate frame weighted averages
    for i in range(frames):
        weighted_frame = np.zeros([x, y])
        for j in range(frames):
            weighted_frame += weights[i, j] * mrc.readframe(input_stack, j)
        if i > 0:
            mrc.append(weighted_frame, output_stack)
        else:
            mrc.write(weighted_frame, output_stack)


def weight_stack_array(input_stack, weights):

    # figure out image dimensions
    frames, x, y = input_stack.shape

    output_stack = np.zeros(input_stack.shape)

    # calculate frame weighted averages
    for i in range(frames):
        output_stack[i, :, :] = np.zeros([x, y])
        for j in range(frames):
            output_stack[i, :, :] += weights[i, j] * input_stack[j, :, :]

    return output_stack


def weight_average(input_stack, output_stack, weights):

    # figure out image dimensions
    frames = mrc.readHeaderFromFile(input_stack)["nz"]

    # calculate frame weighted averages
    for i in range(frames):
        weighted_frame = weights[i] * mrc.readframe(input_stack, i)
        if i > 0:
            mrc.append(weighted_frame, output_stack)
        else:
            mrc.write(weighted_frame, output_stack)

def get_tilt_options(parameters,exclude_views):
    # Reconstruction options 
    # -RADIAL 0.125,0.15, -RADIAL 0.25,0.15 (autoem2), 0.35,0.05 (less stringent)
    tilt_options = "-MODE 2 -OFFSET 0.00 -PERPENDICULAR -SCALE 0.0,0.002 -SUBSETSTART 0,0 -XAXISTILT 0.0 -FlatFilterFraction 0.0 {0}".format(exclude_views)
    
    if parameters.get("tomo_rec_filter_form") == "fakesirt":
        tilt_options += f" -FakeSIRTiterations {parameters.get('tomo_rec_fake_sirt_iterations')}"
    if parameters.get("tomo_rec_filtering_method") == "gaussian":
        tilt_options += f" -RADIAL {parameters['tomo_rec_lpradial_cutoff']},{parameters['tomo_rec_lpradial_falloff']}"
    elif parameters.get("tomo_rec_filtering_method") == "hamming":
        tilt_options += f" -HammingLikeFilter {parameters.get('tomo_rec_hamming')}"

    return tilt_options

def reconstruct_tomo(parameters, name, x, y, binning, zfact, tilt_options, force=False):
    """Perform 3D reconstruction for tomoswarm."""

    if parameters.get("tomo_rec_2d_filtering_method") != "none":
        if parameters.get("tomo_rec_2d_filtering_method") == "doseweighting" and os.path.exists("%s.order" % name):

            dose_file = open("%s.dose" % name, "w")
            with open("%s.order" % name, "r") as f:
                for line in f.readlines():
                    prev_accumulative_dose = float(line.strip()) * float(
                        parameters["scope_dose_rate"]
                    )
                    dose_per_tilt = float(parameters["scope_dose_rate"])
                    dose_file.write(
                        "%f\t%f\n" % (prev_accumulative_dose, dose_per_tilt)
                    )
            dose_file.close()

            command = "{0}/bin/mtffilter -dtype 2 -dfile {1}.dose -volt {2} -verbose 1 -input {1}.ali -output {1}.mtf.ali".format(
                get_imod_path(), name, int(float(parameters["scope_voltage"]))
            )
            # suppress long log
            if ["slurm_verbose"]:
                logger.info(command)
            run_shell_command(command, verbose=False)
        elif parameters.get("tomo_rec_2d_filtering_method") == "lowpass":
            command = "{0}/bin/mtffilter {1}.ali {1}.mtf.ali -lowpass {2},{3}".format(
                get_imod_path(), name, parameters["tomo_rec_mtfilter_cutoff"], parameters["tomo_rec_mtfilter_falloff"]
            )
            run_shell_command(command, verbose=parameters["slurm_verbose"])

        shutil.move("{0}.mtf.ali".format(name), "{0}.ali".format(name))

    # create binned raw stack
    command = "{0}/bin/newstack -input {1}.st -output {1}_bin.mrc -bin {2}".format(
        get_imod_path(), name, binning
    )
    run_shell_command(command,verbose=parameters["slurm_verbose"])

    # create binned aligned stack
    command = "{0}/bin/newstack -input {1}.ali -output {1}_bin.ali -mode 2 -origin -linear -bin {2}".format(
        get_imod_path(), name, binning
    )
    run_shell_command(command,verbose=parameters["slurm_verbose"])

    # create binned reconstruction
    # only reconstruct tomograms if we're not using aretomo2
    thickness = parameters["tomo_rec_thickness"] + parameters['tomo_rec_thickness'] % 2

    if 'imod' in parameters["tomo_rec_method"].lower():

        if False and parameters["tomo_ali_square"]:
            command = "{0}/bin/tilt -input {1}_bin.ali -output {1}.rec -TILTFILE {1}.tlt -SHIFT 0.0,0.0 -SLICE 0,{2} -THICKNESS {3} -WIDTH {4} -IMAGEBINNED {5} -FULLIMAGE {6},{4} {7} {8}".format(
                get_imod_path(), name, x - 1, thickness, y, binning, x, tilt_options, zfact,
            )
        else:
            command = "{0}/bin/tilt -input {1}_bin.ali -output {1}.rec -TILTFILE {1}.tlt -SHIFT 0.0,0.0 -THICKNESS {2} -IMAGEBINNED {3} -FULLIMAGE {4},{5} {6} {7}".format(
                get_imod_path(), name, thickness, binning,  x, y, tilt_options, zfact,
            )
        run_shell_command(command,verbose=parameters["slurm_verbose"])

    elif "aretomo" in parameters["tomo_rec_method"].lower() and ( "aretomo" not in parameters["tomo_ali_method"].lower() or force):

        if Path(f"{name}_aretomo.rec").exists():
            os.rename(f"{name}_aretomo.rec", f"{name}.rec")
        else:
            reconstruct_option = f"-Sart {parameters['tomo_rec_aretomo_sart_iter']} {parameters['tomo_rec_aretomo_sart_num_projs']}"
            if not parameters["tomo_rec_aretomo_sart"]:
                reconstruct_option = "-Wbp 1"

            # the new version of AreTomo2 apparently need the alignments
            if not os.path.exists(f"{name}.aln"):
                tilt_angles = np.loadtxt(f"{name}.tlt",ndmin=2)
                tilt_order = np.loadtxt(f"{name}.order",ndmin=2)
                x, y, z = get_image_dimensions(f"{name}.ali")
                with open(f"{name}.aln",'w') as f:
                    f.write("# AreTomo Alignment / Priims bprmMn)\n")
                    f.write(f"# RawSize = {x} {y} {z}\n")
                    f.write(f"# NumPatches = 0\n")
                    f.write(f"# SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT\n")
                    for tilt, order in zip(tilt_angles, tilt_order):
                        #    0    00.0000    1.00000   1493.353   -740.186     1.00     1.00     1.00     0.00    -60.00
                        f.write("%5d%11.4f%11.5f%11.3f%11.3f%9.2f%9.2f%9.2f%9.2f%9.2f\n" % (order,0,1,0,0,1,1,1,0,tilt))

            command = f"{get_aretomo_path()} \
-InMrc {name}.ali \
-OutMrc {name}.rec \
-AngFile {name}.tlt \
-AlnFile {name}.aln \
-VolZ {int(1.0 * thickness)} \
-OutBin {binning} \
-DarkTol {parameters['tomo_ali_aretomo_dark_tol']} \
{reconstruct_option} \
-Align 0 \
-Gpu {get_gpu_ids(parameters,separator=' ')}"
            run_shell_command(command, verbose=parameters["slurm_verbose"])
