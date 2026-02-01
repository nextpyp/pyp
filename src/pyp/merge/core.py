import glob
import logging
import math
import os
import shutil

from pathlib import Path

import numpy as np

from pyp import utils
from pyp import preprocess
from pyp.utils import symlink_force
from pyp.inout.image import mrc
from pyp.inout.image.core import get_image_dimensions, generate_aligned_tiltseries, get_tilt_axis_angle
import pyp.inout.image as imageio
from pyp.inout.utils import pyp_edit_box_files as imod
from pyp.merge import weights as pyp_weights
from pyp.system import project_params, mpi
from pyp.system.local_run import run_shell_command, stream_shell_command
from pyp.system.utils import get_imod_path, get_aretomo_path, get_aretomo3_path, get_gpu_ids
from pyp.analysis import plot

from pyp.system.logging import logger

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

def reconstruct_tomo(parameters, name, x, y, binning, zfact, tilt_options, force=False, erase_fiducials=False):
    """Perform 3D reconstruction for tomoswarm."""

    if not erase_fiducials:
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
                run_shell_command(command, log_level=logging.TRACE)
            elif parameters.get("tomo_rec_2d_filtering_method") == "lowpass":
                command = "{0}/bin/mtffilter {1}.ali {1}.mtf.ali -lowpass {2},{3}".format(
                    get_imod_path(), name, parameters["tomo_rec_mtfilter_cutoff"], parameters["tomo_rec_mtfilter_falloff"]
                )
                run_shell_command(command)

            # match the stats of the original file (in-place), because it apparently matters during fiducial erasure
            command = "{0}/bin/densmatch -scaled {1}.mtf.ali -reference {1}.ali -all && mv {1}.mtf.ali {1}.ali".format(
                get_imod_path(), name
            )
            run_shell_command(command)

    else:

        gold_mod = f"{name}_gold.mod"

        if not os.path.exists(gold_mod) and parameters["tomo_rec_force"]:
            # create binned aligned stack, if needed
            if not os.path.exists(f'{name}_bin.ali'):
                imod_binning_option = f"-shrink {binning}" if binning > 1 else ""
                size_x = round(x / binning)
                size_x -= size_x % 2
                size_y = round(y / binning)
                size_y -= size_y % 2
                command = "{0}/bin/newstack -quiet -input {1}.ali -output {1}_bin.ali -mode 2 -origin -linear {2} -size {3},{4}".format(
                    get_imod_path(), name, imod_binning_option, size_x, size_y
                )
                run_shell_command(command)
                
        if parameters["tomo_rec_erase_fiducials"]:
    
            if not os.path.exists(gold_mod):
                from pyp.detect import detect_gold_beads
                detect_gold_beads(parameters, name, binning, zfact, tilt_options)
            
            if not os.path.exists(gold_mod):
                logger.warning(f"Skipping gold erasure because no fiducials were detected")
            else:

                # save projected gold coordinates as txt file
                com = f"{get_imod_path()}/bin/model2point {gold_mod} {name}_gold_ccderaser.txt"
                run_shell_command(com)
                
                # convert to unbinned tilt-series coordinates, if needed
                if os.path.exists(f"{name}_gold_ccderaser.txt"):
                    try:
                        with open(f"{name}_gold_ccderaser.txt") as f:
                            gold_coordinates = np.array([line.split() for line in f.readlines() if '*' not in line and not "0.00        0.00        0.00" in line], dtype='f', ndmin=2)

                        gold_coordinates[:,:2] *= binning
                        np.savetxt(name + "_gold_ccderaser.txt",gold_coordinates)

                        # convert back to imod model using one point per contour
                        com = f"{get_imod_path()}/bin/point2model {name}_gold_ccderaser.txt {name}_gold_ccderaser.mod -scat -number 1"
                        run_shell_command(com)

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

                        com = f"{get_imod_path()}/bin/ccderaser -input {name}.ali -output {name}.ali~ -model {name}_gold_ccderaser.mod -expand {erase_iterations} -order {erase_order} -merge -exclude -circle 1 -better {parameters['tomo_ali_fiducial'] * erase_factor / parameters['scope_pixel']} -verbose && mv {name}.ali~ {name}.ali"
                        [ output, _ ] = run_shell_command(com)
                        if "The largest circle radius is too big for the arrays" in output:
                            raise Exception("ccderaser error: The largest circle radius is too big for the arrays. Try reducing the Fiducial radius factor.")

                        try:
                            os.remove(name + "_gold_ccderaser.txt")
                            os.remove(name + "_gold_ccderaser.mod")
                        except:
                            pass
                    except:
                        logger.warning(f"Failed to erase gold from tilt-series!")

    size_x = round(x / binning)
    size_x -= size_x % 2
    size_y = round(y / binning)
    size_y -= size_y % 2

    if binning > 1 and ( not os.path.exists(f"{name}_bin.mrc") or parameters["tomo_rec_erase_fiducials"] ):
        # create binned raw stack
        command = "{0}/bin/newstack -quiet -input {1}.st -output {1}_bin.mrc -shrink {2} -size {3},{4}".format(
            get_imod_path(), name, binning, size_x, size_y
        )
        run_shell_command(command)
    else:
        shutil.copy2(name+'.st',name+'_bin.mrc')

    if parameters["tomo_ali_force"] or not os.path.exists(f"{name}_bin.ali") or parameters["tomo_rec_erase_fiducials"]:
        imod_binning_option = f"-shrink {binning}" if binning > 1 else ""
        # create binned aligned stack
        command = "{0}/bin/newstack -quiet -input {1}.ali -output {1}_bin.ali -mode 2 -origin -linear {2} -size {3},{4}".format(
            get_imod_path(), name, imod_binning_option, size_x, size_y
        )
        run_shell_command(command)

    # create binned reconstruction
    # only reconstruct tomograms if we're not using aretomo2
    thickness = parameters["tomo_rec_thickness"]

    if 'imod' in parameters["tomo_rec_method"].lower():

        thickness = round(thickness/binning)
        thickness -= thickness % 2

        command = "{0}/bin/tilt -input {1}_bin.ali -output {1}.rec -TILTFILE {1}.tlt -SHIFT 0.0,0.0 -THICKNESS {2} {3} {4}".format(
            get_imod_path(), name, thickness, tilt_options, zfact
        )
        run_shell_command(command)

    elif "aretomo" in parameters["tomo_rec_method"].lower() and ( parameters["tomo_ali_method"].lower() != parameters["tomo_rec_method"].lower() or force):

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
                run_shell_command(command)
                
            elif "aretomo3" == parameters["tomo_rec_method"].lower():

                # rename all the files as required by aretomo3

                symlink_force( f"{name}.rawtlt", f"{name}_aligned.rawtlt")
                symlink_force( f"{name}.aln", f"{name}_aligned.aln")
                
                # aligned tilt series must have mrc extension
                # we also add a small constant to each image to trick AreTomo3's mass normalization because it fails when there are blank images
                command = f"{get_imod_path()}/bin/newstack -quiet {name}.ali {name}_aligned.mrc -multadd 1,1"
                run_shell_command(command)
                
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
                run_shell_command(command)
                
                assert os.path.exists(f"{name}_aligned_Vol.mrc"), "AreTomo3 reconstruction failed"

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

    else:
        logger.warning(f"Skipping reconstruction because {parameters["tomo_rec_method"].lower()} was already used for alignment")
        
        
def reconstruct_tomo_halves( name, parameters, project_path):
    """
        Generate half tomograms for N2N training
    """
    
    if os.path.exists("frame_list.txt") and parameters.get("tomo_rec_generate_halves_use_frames"):
        logger.debug(f"Calculating half-tomograms from odd/even frames")
        reconstruct_tomo_halves_from_frames( name, parameters, project_path )
    else:
        logger.debug(f"Calculating half-tomograms from odd/even tilts since no frame data was found")
        reconstruct_tomo_halves_from_odd_even_tilts( name, parameters )

def reconstruct_tomo_halves_from_frames( name, parameters, project_path):
    """
        Generate half tomograms for N2N training
    """
        
    with open("frame_list.txt", "r") as f:
        frame_list = f.read().split("\n")

    raw_image = frame_list

    # retrieve, pre-process, and split the raw frames into even/odd stacks, if needed
    if not os.path.exists(Path(frame_list[0]).stem + "_half1.avg"):
        arguments = []
        for f in frame_list:
            arguments.append((str(project_path) + "/raw/" + f, f))
        mpi.submit_function_to_workers(shutil.copy2, arguments)

        # convert eer files to mrc using movie_eer_reduce and movie_eer_frames parameters (flipping in x is required to match unblur/motioncorr convention)
        if frame_list[0].lower().endswith(".eer"):
            full_frame = get_image_dimensions(frame_list[0])[2]
            valid_averages = np.floor(full_frame / parameters['movie_eer_frames'])
            eer = True
            arguments = []
            for f in frame_list:
                # average eer frames
                command = f"{get_imod_path()}/bin/clip flipx -es {parameters['movie_eer_reduce']-1} -ez {parameters['movie_eer_frames']} {f} {f.replace('.eer','.mrc')}; rm -f {f}"
                arguments.append(command)
            mpi.submit_jobs_to_workers(arguments)

            raw_image = [ Path(i).stem + '.mrc' for i in frame_list ]
        else:
            raw_image = frame_list
            eer = False

        # convert tif movies to mrc files
        if ".tif" in raw_image[0].lower():
            commands = [] 
            for f in raw_image:
                com = "{0}/bin/newstack -quiet -mode 2 {1} {2}; rm -f {1}".format(
                    get_imod_path(), f, Path(f).stem + ".mrc"
                )
                commands.append(com)
            mpi.submit_jobs_to_workers(commands)
            
            raw_image = [Path(f).stem + ".mrc" for f in frame_list]
    
        # get dimensions
        dims = get_image_dimensions(raw_image[0])
        if eer:
            z_slices = int(valid_averages) - 1
        else:
            z_slices = dims[2] - 1

        # create half stacks
        arguments = []
        for _, f in enumerate(raw_image):
            with open(Path(f).stem+'.xf') as frame_alignments_file:
                frame_alignments = frame_alignments_file.read().split('\n')
            for half in [1, 2]:
                subset = np.arange(half-1, z_slices + 1, 2)
                if not os.path.exists(f"{Path(f).stem}_half{half}.mrc"):
                    command = f"{get_imod_path()}/bin/newstack -quiet -input {f} -secs {','.join(map(str, subset))} -output {Path(f).stem}_half{half}.mrc"
                    arguments.append(command)
                    with open(Path(f).stem+f"_half{half}.xf",'w') as output_half_xf:
                        for index in subset:
                            output_half_xf.write(f"{frame_alignments[index]}\n")
                    
        if len(arguments) > 0:
            mpi.submit_jobs_to_workers(arguments)
    
    elif frame_list[0].lower().endswith(".eer") or ".tif" in raw_image[0].lower():
        raw_image = [Path(f).stem + ".mrc" for f in frame_list]

    # generate half tomograms from half stacks
    dims = get_image_dimensions(name+'.mrc')
    for i in [1, 2]:
        newname = name + f"_half{i}"
        new_filelist = [Path(file).stem + f"_half{i}" + Path(file).suffix for file in raw_image]

        # copy the tilt alignment and angle files
        shutil.copy2(f"{name}.xf", newname + ".xf")
        shutil.copy2(f"{name}.tlt", newname + ".tlt")
        shutil.copy2(f"{name}.order", newname + ".order")

        # generate averages using existing xf
        preprocess.regenerate_average_quick(
            newname,
            parameters,
            dims,
            new_filelist,
        )

        os.symlink(newname + ".mrc", newname + ".st")

        # actual stack sizes
        headers = mrc.readHeaderFromFile(newname + ".mrc")
        x = int(headers["nx"])
        y = int(headers["ny"])

        # Resize aligned tilt-series depending on tilt-axis orientation
        tilt_axis_angle = get_tilt_axis_angle(name)
        if tilt_axis_angle % 180 > 45 and tilt_axis_angle % 180 < 135 and not parameters.get("tomo_ali_square") and x != y:
            x, y = y, x
            logger.info(f"Resizing aligned tilt-series to {x} x {y} to accomodate tilt-axis orientation")

        # binned reconstruction
        binning = parameters["tomo_rec_binning"]
        zfact = ""

        # regenerate aligned tilt-series
        generate_aligned_tiltseries(newname, parameters, x, y)

        exclude_views = do_exclude_views(newname)

        # Reconstruction options
        tilt_options = get_tilt_options(parameters,exclude_views)

        # produce binned tomograms
        if parameters["tomo_ali_method"] == "imod_gold" and parameters["tomo_rec_erase_fiducials"]:
            # erase fiducials if needed
            preprocess.erase_gold_beads(newname, parameters, tilt_options, binning, zfact, x, y)
        else:
            reconstruct_tomo(parameters, newname, x, y, binning, zfact, tilt_options, force=True)

def reconstruct_tomo_halves_from_odd_even_tilts( name, parameters):
    """
        Generate half tomograms for denoising training
    """
    
    # split tilt-series into even/odd stacks

    # get dimensions of raw tilt-series
    raw_tilt_series = name + ".mrc"
    dims = get_image_dimensions(raw_tilt_series)

    with open(name+'.tlt') as tilt_angles_file:
        tilts = tilt_angles_file.read().split('\n')
    with open(name+'.xf') as alignments_file:
        alignments = alignments_file.read().split('\n')
    with open(name+'.order') as order_file:
        orders = order_file.read().split('\n')

    # create half tilt-series and half tilt-angle, alignment, and order files
    arguments = []
    for half in [1, 2]:
        subset = np.arange(half-1, dims[2], 2)
        command = f"{get_imod_path()}/bin/newstack -quiet -input {raw_tilt_series} -secs {','.join(map(str, subset))} -output {name}_half{half}.mrc"
        arguments.append(command)
        with open(name+f"_half{half}.tlt",'w') as output_half_tlt:
            for index in subset:
                output_half_tlt.write(f"{tilts[index]}\n")
        with open(name+f"_half{half}.xf",'w') as output_half_xf:
            for index in subset:
                output_half_xf.write(f"{alignments[index]}\n")
        with open(name+f"_half{half}.order",'w') as output_half_order:
            for index in subset:
                output_half_order.write(f"{orders[index]}\n")
            
    if len(arguments) > 0:
        mpi.submit_jobs_to_workers(arguments)
    
    # generate half tomograms from half stacks
    for i in [1, 2]:
        newname = name + f"_half{i}"

        os.symlink(newname + ".mrc", newname + ".st")
        if os.path.exists(name + "_gold3d.mod"):
            # if we have a previsouly saved gold3d model, we need to flip Y-Z
            if parameters.get("tomo_rec_force"):
                rotate_option = ""
            else:
                rotate_option = "-Y"
            command = f"{get_imod_path()}/bin/imodtrans {rotate_option} {name}_gold3d.mod {newname}_gold3d.mod"
            run_shell_command(command)

        # actual stack sizes
        headers = mrc.readHeaderFromFile(newname + ".mrc")
        x = int(headers["nx"])
        y = int(headers["ny"])

        # Resize aligned tilt-series depending on tilt-axis orientation
        tilt_axis_angle = get_tilt_axis_angle(name)
        if tilt_axis_angle % 180 > 45 and tilt_axis_angle % 180 < 135 and not parameters.get("tomo_ali_square") and x != y:
            x, y = y, x
            logger.info(f"Resizing aligned tilt-series to {x} x {y} to accomodate tilt-axis orientation")

        # binned reconstruction
        binning = parameters["tomo_rec_binning"]
        zfact = ""

        # convert to square
        size_x, size_y, size_z = get_image_dimensions(newname + ".mrc")
        if parameters["tomo_ali_format"]:
            squarex = math.ceil(size_x / 512.0) * 512
            squarey = math.ceil(size_y / 512.0) * 512
        else:
            squarex = size_x
            squarey = size_y
        square = max(squarex, squarey)
        
        aligned_tilts = [ Path(f).stem for f in glob.glob(f"{newname}_????.ali") ]
        imageio.tiltseries_to_squares(newname, parameters, aligned_tilts, size_z, square, int(parameters["data_bin"]))

        # regenerate aligned tilt-series
        generate_aligned_tiltseries(newname, parameters, x, y)

        # get list of excluded views
        exclude_views = do_exclude_views(name)
        exclude_views_half = ""

        subset = np.arange(i-1, dims[2], 2)

        if len(exclude_views) > 0:
            excluded_indexes = exclude_views.split(" ")
            if len(excluded_indexes) > 0:
                exclude_views_half = "-EXCLUDELIST2 " + ",".join([ str(int(f)//2 + i % 2) for f in excluded_indexes[-1].split(",") if int(f) % 2 == i % 2 ])
                if len(exclude_views_half.split(" ")[-1]) == 0:
                    exclude_views_half = ""

        # Reconstruction options
        tilt_options = get_tilt_options(parameters,exclude_views_half)

        # produce binned tomograms, erase fiducials if needed
        if parameters["tomo_ali_method"] == "imod_gold" and parameters["tomo_rec_erase_fiducials"] and os.path.exists(newname+"_gold3d.mod") and len(imod.coordinates_from_mod_file(f"{newname}_gold3d.mod")) > 0:

            # first, project 3D gold coordiantes to respective aligned tilt-series
            thickness = parameters["tomo_rec_thickness"]
            thickness = round(thickness/binning)
            thickness -= thickness % 2

            if not os.path.exists(newname + "_bin.ali"):
                if binning > 1:
                    command = "{0}/bin/newstack -quiet -input {1}.ali -output {1}_bin.ali -shrink {2}".format(
                        get_imod_path(), newname, binning
                    )
                    run_shell_command(command)
                else:
                    shutil.copy2(newname+".ali",newname+"_bin.ali")

            command = "{0}/bin/tilt -input {1}_bin.ali -output {1}_gold.mod -TILTFILE {1}.tlt -SHIFT 0.0,0.0 -THICKNESS {2} {3} -ProjectModel {1}_gold3d.mod".format(
                get_imod_path(), newname, thickness, tilt_options
            )
            run_shell_command(command)

            # delete gold and reconstruct
            preprocess.erase_gold_beads(newname, parameters, tilt_options, binning, zfact, x, y)
        else:
            reconstruct_tomo(parameters, newname, x, y, binning, zfact, tilt_options, force=True)
