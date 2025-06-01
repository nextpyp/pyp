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
from pyp.system.utils import get_imod_path, get_aretomo_path, get_aretomo3_path, get_gpu_ids
from pyp.utils import get_relative_path

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


def do_exclude_views(name):
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
    thickness = parameters["tomo_rec_thickness"]

    if 'imod' in parameters["tomo_rec_method"].lower():

        # get tomogram dimensions directly from aligned tilt-series
        x, y, _ = get_image_dimensions(f"{name}_bin.ali")
        x *= binning
        y *= binning

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

            if "aretomo" == parameters["tomo_rec_method"].lower():

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
                
            elif "aretomo3" == parameters["tomo_rec_method"].lower():

                # rename all the files because aretomo3 expects an mrc extension for the input
                os.symlink( f"{name}.ali", f"{name}_aligned.mrc")
                os.symlink( f"{name}.rawtlt", f"{name}_aligned.rawtlt")
                os.symlink( f"{name}.aln", f"{name}_aligned.aln")
                
                """ Usage: AreTomo3 Tags

     ******  Common Parameters  *****
-InPrefix
  1. Prefix of input file name(s), ogether with Insuffix
     and InSkips, is used to form either a single or subset
     for file name(s), which are processed by AreTomo3.
  2. If the suffix is mdoc, any mdoc file that starts with
     the prefix string will be selected.
  3. If the suffix is mrc, any mrc file that starts with
     the prefix string will be selected and processed.
  4. The prefix can also be the path of a folder containing
     the movie files (tiff or eer) or tilt series (mrc).
  5. Note that movie files must be in the same directory
     as the mdoc files.


  1. If MDOC files have .mdoc file name extension, then
     .mdoc should be given after (null). If another extension
     is used, it should be used instead.

  1. If a MDOC file contains any string given behind %s,
     those MDOC files will not be processed.

-OutDir
  1. Path to output folder to store generated tilt series, tomograms,
     and alignment files.

-PixSize
  1. Pixel size in A of input stack in angstrom.

-kV
  1. High tension in kV needed for dose weighting.
  2. Default is 300.

-Cs
  1. Spherical aberration in mm for CTF estimation.

-FmDose
  1. Per frame dose in e/A2.

-SplitSum
   1. Generate odd and even sums using odd and even frames.
   2. The default value is 1, which enables the split, 0
      disables this function.
   3. When enabled, 3 tilt series and 3 tomograms will be
      generated. Tilt series and tomograms generated from
      odd and even sums are appended _ODD and _EVN in
      their file names, respectively.
   4. When disabled, odd and even tilt series and tomograms
      will not be generated. Tilt series and tomogram from
      full sums are generated only.
 
 -Cmd
  1. Default 0 starts processing from motion correction.
  2. -Cmd 1 starts processing from tilt series alignment
     including CTF estimation, correction, tomographic
     alignment and reconstruction.
  3. -Cmd 2 starts processing from CTF correction and
     then tomographic reconstruction.
  4. -Cmd 1 and -Cmd 2 ignore -Resume.

-Resume
  1. Default 0 processes all the data.
  2. -Resume 1 starts from what are left by skipping all the mdoc
     files in MdocDone.txt file in the output folder.

-Gpu
   GPU IDs. Default 0.
   For multiple GPUs, separate IDs by space.
   For example, -Gpu 0 1 2 3 specifies 4 GPUs.

-DefectFile
1. Defect file stores entries of defects on camera.
2. Each entry corresponds to a rectangular region in image.
   The pixels in such a region are replaced by neighboring
   good pixel values.
3. Each entry contains 4 integers x, y, w, h representing
   the x, y coordinates, width, and heights, respectively.

-Gain
1. MRC or TIFF file that stores the gain reference.
2. Falcon camera produced .gain file can also be used
   since it is a TIFF file.

-Dark
  1. MRC file that stores the dark reference. If not
     specified, dark subtraction will be skipped.
  2. If -RotGain and/or -FlipGain is specified, the
     dark reference will also be rotated and/or flipped.

-McPatch
  1. It is followed by numbers of patches in x and y dimensions.
  2. The default values are 1 1, meaning only full-frame
     based alignment is performed.

-McIter
   Maximum iterations for iterative alignment,
   default 7 iterations.

-McTol
   Tolerance for iterative alignment,
   default 0.5 pixel.

-McBin
   Binning performed in Fourier space, default 1.0.

-Group
   1. Group every specified number of frames by adding
      them together. The alignment is then performed
      on the group sums. The so measured motion is
      interpolated to each raw frame.
   2. The 1st integer is for gobal alignment and the
      2nd is for patch alignment.

-FmRef
   Specify a frame in the input movie stack to be the
   reference to which all other frames are aligned. The
   reference is 1-based index in the input movie stack
   regardless how many frames will be thrown. By default
   the reference is set to be the central frame.

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

-TiltAxis
   1. User provided angle of tilt axis in degree. If users
      do not provide one, AreTomo3 will search in full range.
   2. If users provide one and do not want AreTomo3 to refine
      it, add -1 after the provided tilt axis.
   3. Otherwise, AreTomo3 regines the provided value in a
      in a smaller range.

-AlignZ
   Volume height for alignment, default 256

-VolZ
   1. Volume z height for reconstrunction. It must be
      greater than 0 to reconstruct a volume.
   2. Default is 0, only aligned tilt series will
      generated.

-ExtZ
   1. Extra volume z height for reconstrunction. This is
      the  space added to the estimated sample thickness for
      the final reconstruction of tomograms.
   2. This setting is relevant only when -VolZ -1 is set,
      which means users want to use the estimated sample
      thickness.

      greater than 0 to reconstruct a volume.
   2. Default is 0, only aligned tilt series will
      generated.

-AtBin
   1. Binnings for tomograms with respect to motion
      corrected tilt series. Users can specify two floats.
       2. The first number is required and 1.0 is default.
       3. The second number is optional. By default, the
      second resolution is disabled. It is actived only
      when users provide a different number.

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

-AmpContrast
   1. Amplitude contrast, default 0.07

-ExtPhase
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

-IntpCor
   1. When enabled, the correction for information loss due
      to linear interpolation will be perform. The default
      setting value 1 enables the correction.

-CorrCTF
   1. When enabled, local CTF correction is performed on
      raw tilt series. By default this function is enabled.
   2. Passing 0 disables this function.

            """
                command = f"{get_aretomo3_path()} \
-InPrefix {name}_aligned.mrc \
-OutDir ./ \
-Cmd 2 \
-AtBin {binning} \
-FlipVol 0 \
{reconstruct_option} \
-DarkTol {parameters['tomo_ali_aretomo_dark_tol']} \
-VolZ {int(1.0 * thickness)} \
-Gpu {get_gpu_ids(parameters,separator=' ')} \
-TmpDir {os.environ['PYP_SCRATCH']}"
                run_shell_command(command, verbose=parameters["slurm_verbose"])
                
                assert os.path.exists(f"{name}_aligned_Vol.mrc"), "AreTomo3 reconstruction failed, no output file found"

                # rename output from {name}_aligned_Vol.mrc to {name}.rec
                # rename output from {name}_Vol.mrc to {name}.rec
                rec_name = f"{name}.rec"
                if os.path.exists(rec_name):
                    os.remove(rec_name)
                os.symlink( f"{name}_aligned_Vol.mrc", rec_name )
                
                # cleanup
                os.remove(f"{name}_aligned.mrc")
                os.remove(f"{name}_aligned.rawtlt")
                os.remove(f"{name}_aligned.aln")
