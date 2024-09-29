import glob
import math
import random
import os
import shutil

import numpy as np
from skimage import measure

from pyp import detect
from pyp.detect import joint
from pyp.analysis.geometry import calcSpikeNormXYZ, get_vir_binning_boxsize
from pyp.analysis.geometry import transformations as vtk
from pyp.analysis.image import normalize_volume
from pyp.inout.image import mrc, img2webp
from pyp.inout.image.core import get_image_dimensions
from pyp.inout.utils import pyp_edit_box_files as imod
from pyp.utils import timer
from pyp.system import local_run, mpi, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import (
    get_tomo_path,
    get_imod_path,
    check_env,
)
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_virion_segmentation_thresholds(seg_thresh):

    return np.array(
        [" 0.1 0.01 0.005 0.0025 0.001 0.0005 0.00025 0.0001 -0.000144325".split()]
    ).squeeze()[int(seg_thresh)]


def process_virion_multiprocessing(
    name,
    virion_name,
    virion,
    seg_thresh,
    x,
    y,
    binning,
    z_thickness,
    virion_binning,
    virion_size,
    band_width,
    tilt_angles,
    spike_size,
    tilt_options,
    parameters,
):
    # workflow will be first reconstructing virion volume, then searching virus membrane, finally template-matching particles on the membrane
    # build virion volume first
    build_virion_unbinned(
        virion,
        binning,
        virion_binning,
        virion_size,
        x,
        y,
        z_thickness,
        tilt_options,
        name,
        virion_name,
    )

    # The binning here controls the size of binned_nad.mrc, binned_nad_seg.mrc, which will be used for template matching
    virion_binned_size = 200
    spk_pick_binning = math.ceil(1.0 * virion_size / (virion_binning * virion_binned_size))

    command = "{0}/bin/binvol -bin {1} {2}.rec {2}_binned.mrc".format(
        get_imod_path(), spk_pick_binning, virion_name,
    )
    local_run.run_shell_command(command,parameters["slurm_verbose"])

    if not os.path.exists(f"{virion_name}_binned_nad_seg.mrc"):

        # increase the contrast of virus volume
        command = "{0}/bin/nad_eed_3d -n 10 {1}_binned.mrc {1}_binned_nad.mrc".format(
            get_imod_path(), virion_name,
        )
        local_run.run_shell_command(command,verbose=False)

        # produce virus membrane (surface only)
        correction = binning / float(spk_pick_binning) / virion_binning * parameters["tomo_vir_binn"]

        # get value of radius from virion detection operation
        radius = float(virion[-1]) * correction

        # read in tolerance value (making sure it is expressed in percentage)
        tolerance = min(100,abs(parameters["tomo_vir_seg_tol"])) / 100.

        # calculate admissible radius range
        min_radius = radius * ( 1 - tolerance )
        max_radius = radius + ( 1 + tolerance )

        # Run segmentation
        # USAGE: virus_segment_membrane input.mrc iradius oradius weight iterations variances output.mrc
        check_env()

        weight = parameters["tomo_vir_seg_smoothness"]
        iterations = parameters["tomo_vir_seg_iterations"]
        variances = 10
        command = f"{get_tomo_path()}/virus_segment_membrane {virion_name}_binned_nad.mrc {min_radius:.2f} {max_radius:.2f} {weight} {iterations} {variances} {virion_name}_binned_nad_seg.mrc"
        local_run.run_shell_command(command,verbose=parameters['slurm_verbose'])

        if os.path.exists("%s_binned_nad.mrc" % (virion_name)):
            # produce visualization for selecting thresholds of iso-surfaces
            command = "{0}/virus_segment_membrane_select_threshold {1}_binned_nad".format(
                get_tomo_path(), virion_name
            )
            local_run.run_shell_command(command,parameters["slurm_verbose"])
            command = "convert -resize 1080x360 {1}_binned_nad.png {1}_binned_nad.png".format(
                get_imod_path(), virion_name
            )
            local_run.run_shell_command(command,parameters["slurm_verbose"])

            img2webp(f"{virion_name}_binned_nad.png", f"{virion_name}_binned_nad.webp")

    # flipy if raw data is dm4
    if os.path.exists("isdm4"):
        command = "{0}/bin/clip flipx {1}.rec {1}.rec".format(
            get_imod_path(), virion_name
        )
        local_run.run_shell_command(command,parameters["slurm_verbose"])
        command = "{0}/bin/clip flipx {1}_binned_nad_seg.mrc {1}_binned_nad_seg.mrc".format(
            get_imod_path(), virion_name
        )
        local_run.run_shell_command(command,parameters["slurm_verbose"])

    threshold = get_virion_segmentation_thresholds(int(seg_thresh))

    autopick_template = virion_name + "_autopick_template.mrc"

    # run autopick
    if parameters["tomo_vir_detect_method"] != "none" and parameters.get("micromon_block") != "tomo-segmentation-closed":

        fresh_template_match = False

        if "tomo_ext_bin" in parameters and parameters["tomo_ext_bin"] > 1:
            factor = 4.2 / parameters["scope_pixel"] / virion_binning
        else:
            factor = 1.0


        if (
            not os.path.exists(virion_name + "_cut.txt")
        ):
            if "template" in parameters["tomo_vir_detect_method"]:
                # pick and extract particles
                lower_slice = int(
                    virion_size / 2 / virion_binning - band_width / virion_binning
                )
                upper_slice = int(
                    virion_size / 2 / virion_binning + band_width / virion_binning
                )

                # if using standard template, rescale template volume to match data pixel size
                parameters["tomo_vir_detect_ref"] = project_params.resolve_path(parameters["tomo_vir_detect_ref"]) if "tomo_vir_detect_ref" in parameters else ""
                if (
                    "hiv1bal-ang90_global_average_symmetrized.mrc.filtered.mrc"
                    in parameters["tomo_vir_detect_ref"]
                ):
                    # size = int( 36.0 * factor )
                    size = int(64.0 * factor)
                    if size % 2 > 0:
                        size += 1

                    resample_and_resize(
                        input=parameters["tomo_vir_detect_ref"],
                        output=autopick_template,
                        scale=factor,
                        size=size,
                    )

                # elif parameters["tomo_vir_detect_method"] == "mesh":
                #     size_x = size_y = size_z = 8
                if os.path.exists(parameters["tomo_vir_detect_ref"]):
                    shutil.copy2(parameters["tomo_vir_detect_ref"], autopick_template)
                    size_x = size_z = 0
                    size_y = autopick_template
                else: 
                    raise Exception("Please provide a valid reference for template-search")

                # Automatic spike picking by template matching on viral surface.
                #
                # Usage: external/TOMO/Correlation3DNew
                #  1. virus tomogram volume (excluding .rec extension)
                #  2. virus membrane volume, possibly binned (mrc file)
                #  3. level to extract surface from virus membrane volume
                #  4. surface binning factor wrt to virus volume (1,2,3,etc)
                #  5. lower tilt range,
                #     e.g. to ignore bottom and top of virus: 0(X), 1(Y) or 2(Z)
                #  6. upper tilt range,
                #     e.g. to ignore bottom and top of virus: 0(X), 1(Y) or 2(Z)
                #  7. dimension in which to restrict the template search,
                #     e.g. to ignore bottom and top of virus: 0(X), 1(Y) or 2(Z)
                #  8. lowest slice to search (in dimension specified above)
                #  9. highest slice to search (in dimension specified above)
                # 10. template size in X dimension
                # 11. template size in Y dimension
                # 12. template size in Z dimension
                # 13. minimum spacing between adjacent correlation peaks
                # 14. correlation threshold (peaks with lower correlation
                #     values than threshold are discarded) autoem2=0.15
                # 15. file with extracted spikes (txt file)
                # 16. correlation surface for visualization (xml file)
                # 17. spike locations for visualization (xml file)
                command = "{0}/Correlation3DNew {1} {1}_binned_nad_seg.mrc {2} {3} {4} {5} 2 {6} {7} 0 {10} 0 {8} {9} {1}_cut.txt {1}_ccc.xml {1}_spikes.xml".format(
                    get_tomo_path(),
                    virion_name,  # 1. virus tomogram volume (mrc file) Note: only the BASENAME
                    threshold,
                    # 3. level to extract surface from virus [ 0.1 0.01 0.005 0.0025 0.001 0.0005 0.00025 0.0001 -0.000144325 ]
                    spk_pick_binning,  # 4. surface binning factor (*_binned_nad.mrc) wrt to virus volume (*.mrc)
                    min(tilt_angles),
                    max(tilt_angles),
                    lower_slice,
                    upper_slice,
                    str(parameters["tomo_vir_detect_dist"]),
                    str(parameters["tomo_vir_detect_thre"]),
                    autopick_template,
                )
                t = timer.Timer(text="Spike detection using Correlation3DNew took: {}", logger=logger.info)
                t.start()
                command = f"{get_tomo_path()}/Correlation3DNew {virion_name} {virion_name}_binned_nad_seg.mrc {threshold} {spk_pick_binning} {min(tilt_angles)} {max(tilt_angles)} 2 {lower_slice} {upper_slice} {size_x} {size_y} {size_z} {parameters['tomo_vir_detect_dist']} {parameters['tomo_vir_detect_thre']} {virion_name}_cut.txt {virion_name}_ccc.xml {virion_name}_spikes.xml"
                local_run.run_shell_command(command,verbose=parameters['slurm_verbose'])
                t.stop()
                fresh_template_match = True

            # Using uniform coordinates from virion surface
            elif  "tomo_vir_detect_method" in parameters and parameters["tomo_vir_detect_method"] == "mesh":
                fresh_template_match = True
                bandwidth = band_width / virion_binning
                distance = parameters["tomo_vir_detect_dist"]
                # scale_factor = virion_binning * spk_pick_binning
                mesh_coordinate_generator(virion_name, threshold, distance, bandwidth)

            # flipx cmm coordinates
            if os.path.exists("isdm4"):
                f = open("{0}_auto_new.cmm".format(virion_name), "w")
                with open("{0}_auto.cmm".format(virion_name)) as cmm:
                    for line in cmm:
                        if "marker id" in line:
                            y = float(
                                line.split()[3].replace("y=", "").replace('"', "")
                            )
                            f.write(line.replace(line.split()[3], 'y="%f"' % (120 - y)))
                        else:
                            f.write(line)
                f.close()
                shutil.copy2(
                    "{0}_auto_new.cmm".format(virion_name),
                    "{0}_auto.cmm".format(virion_name),
                )

        else:
            logger.info("Using existing autopick results")

        # HF Liu - if haven't yet extracted spikes before AND wanted to correct the heights (close or away from membranes)
        if fresh_template_match:

            # extract positions from cmm file (markers should sit at the base of the spike at the membrane level)
            if os.path.exists("{0}_auto.cmm".format(virion_name)):
                f = open("{0}.pos".format(virion_name), "w")

                with open("{0}_auto.cmm".format(virion_name)) as cmm:

                    for line in cmm:
                        if "marker id" in line:
                            x = float(
                                line.split()[2].replace("x=", "").replace('"', "")
                            )
                            y = float(
                                line.split()[3].replace("y=", "").replace('"', "")
                            )
                            z = float(
                                line.split()[4].replace("z=", "").replace('"', "")
                            )
                            f.write("%3.4f\t%3.4f\t%3.4f\n" % (x, y, z))
                f.close()

                # spike height default = 228A
                spike_height = (
                    float(parameters["tomo_vir_detect_offset"])
                    / parameters["scope_pixel"]
                    / virion_binning
                )

                # negative distance from membrane (where marker sits) to center of spike (compatible with Correlation3DNew)
                # offset = - ( spike_height + 4 ) / 2
                offset = -spike_height

                if offset != 0 or "mesh" in parameters["tomo_vir_detect_method"]:
                    com = "{0}/LoopCreateVolumeList {1} {2} {3} {4}.rec {4}.pos {5} {4}_binned_nad_seg.mrc {6} {4}_cut.txt".format(
                        get_tomo_path(),
                        spk_pick_binning,
                        min(tilt_angles),
                        max(tilt_angles),
                        virion_name,
                        offset,
                        threshold,
                    )
                    local_run.run_shell_command(com)

        # revert flipy to ensure compatibility
        if os.path.exists("isdm4"):
            command = "{0}/bin/clip flipx {1}_binned_nad_seg.mrc {1}_binned_nad_seg.mrc".format(
                get_imod_path(), virion_name
            )
            local_run.run_shell_command(command)

        EXTRACT_USING_PYTHON = True
        spike_binning = parameters["tomo_ext_bin"] if 'tomo_ext_bin' in parameters else 1

        # USAGE: external/TOMO/CutVolumes3DFromPositions
        # CutVolumes3DFromPositions volumes cutsize prefix noeulers start
        # volumes - txt file obtained from LoopCreateVolumeList.
        # cutsize - crop size of output volumes.
        # prefix - string to preceed all volumes filename.
        # noeulers - USE 1 (cropping done without using euler angles). 0 if cropping done using euler angles.
        # start - index within the volumes list to start processing. Set to -1 to produce the output txt file alone (without generating cropped volumes).
        command = "unset LD_LIBRARY_PATH; {0}/CutVolumes3DFromPositions {1}_cut.txt {2} {1} 1 0".format(
            get_tomo_path(), virion_name, spike_size
        )
        # print commands.getoutput(command)

        # Cut unbinned volumes making sure the output size is padded if outside of virion
        #
        # CutVolumes3DFromPositions cannot be used due to the 32-bit 2GB file size restriction.
        # Instead, we extract the volumes directly in python:
        if EXTRACT_USING_PYTHON:

            newfile = virion_name + ".txt"
            coordinate_file = virion_name + "_cut.txt"
            tmp_coordinate_file = virion_name + "_cut.txt.tmp"
            """
            if os.path.exists('isdm4'):
                command = '{0}/bin/clip flipx {1}_unbinned.rec {1}_unbinned.rec'.format(get_imod_path(),virion_name)
                print command
                print commands.getoutput(command)
            """
            # resample
            if spike_binning != 1:
                command = "{0}/bin/binvol -binning {1} {2}.rec {2}.rec".format(
                    get_imod_path(), spike_binning, virion_name
                )
                local_run.run_shell_command(command)

            A = mrc.read(virion_name + ".rec")
            # normalize densities (as it is done in CutVolumes3DFromPositions)
            A = (A - A.mean()) / A.std()
            # mimic rotx operation for dm4 files
            if os.path.exists("isdm4"):
                A = A[:, ::-1, :]
            if ("tomo_ext_fmt" in parameters
                and "eman" in parameters["tomo_ext_fmt"].lower()
                and not parameters["data_invert"]
            ):
                logger.info(
                    "Invert the virion volume for EMAN format sub-volume extraction when not using data_invert"
                )
                A = -A

            index = 0

            spikesize = parameters["tomo_ext_size"] if "tomo_ext_size" in parameters  else 0

            if spike_binning != 1:
                spikesize //= int(spike_binning)
                logger.info(
                    "Bin spike with binning factor {0} after virion bin {1}".format(
                        spike_binning, virion_binning
                    )
                )

            with open(newfile, "w") as newf:

                with open(tmp_coordinate_file, "w") as tmpf:

                    tmpf.write(
                        "number\tlwedge\tuwedge\tposX\tposY\tposZ\tgeomVirusX\tgeomVirusY\tgeomVirusZ\tnormalX\tnormalY\tnormalZ\tmatrix[0]\tmatrix[1]\tmatrix[2]\tmatrix[3]\tmatrix[4]\tmatrix[5]\tmatrix[6]\tmatrix[7]\tmatrix[8]\tmatrix[9]\tmatrix[10]\tmatrix[11]\tmatrix[12]\tmatrix[13]\tmatrix[14]\tmatrix[15]\tmagnification[0]\tmagnification[1]\tmagnification[2]\tcutOffset\tfilename\n"
                    )

                    with open(coordinate_file) as f:
                        for line in f.readlines():

                            data = line.split("\t")

                            if not "number" in data[0]:

                                name = virion_name + "_spk%04d.mrc" % index
                                # check if spikes were already extracted using retrieved _cut.txt
                                # if not, only process the lines which are height corrected

                                if (
                                        fresh_template_match and float(parameters["tomo_vir_detect_offset"])
                                ) > 0 and data[6] != "0":
                                    continue

                                pos = (np.array(data[3:6], dtype=float)).astype("int")
                                if spike_binning != 1:
                                    pos //= int(spike_binning)

                                subA = np.ones([spikesize, spikesize, spikesize]) * A.mean()

                                # x, y, z coordinates in .txt = pos[0], pos[1], pos[2]
                                # when converted to model (eg. spk):
                                # x = pos[0]
                                # y = boxsize - pos[2]
                                # z = pos[1]
                                poszl = pos[2] - spikesize // 2
                                poszh = pos[2] + spikesize // 2
                                posyl = (A.shape[1] - pos[1]) - spikesize // 2
                                posyh = (A.shape[1] - pos[1]) + spikesize // 2
                                posxl = pos[0] - spikesize // 2
                                posxh = pos[0] + spikesize // 2

                                # highx = highy = highz = binning * spikesize
                                lowx = lowy = lowz = 0
                                highx = highy = highz = spikesize
                                if posxl < 0:
                                    lowx = -posxl
                                    posxl = 0
                                if posxh >= A.shape[2]:
                                    highx = subA.shape[2] - (posxh - A.shape[2])
                                    posxh = A.shape[2]
                                if posyl < 0:
                                    lowy = -posyl
                                    posyl = 0
                                if posyh >= A.shape[1]:
                                    highy = subA.shape[1] - (posyh - A.shape[1])
                                    posyh = A.shape[1]
                                if poszl < 0:
                                    lowz = -poszl
                                    poszl = 0
                                if poszh >= A.shape[0]:
                                    highz = subA.shape[0] - (poszh - A.shape[0])
                                    poszh = A.shape[0]

                                if spikesize > 0 and parameters["tomo_ext_fmt"] != "none":
                                    try:
                                        subA[lowz:highz, lowy:highy, lowx:highx] = A[
                                            poszl:poszh, posyl:posyh, posxl:posxh
                                        ]
                                        mrc.write(subA, name)
                                    except:
                                        logger.exception(
                                            "This spike volume cannot be extracted correctly due to being completely out of bound."
                                        )
                                        continue

                                    # normalize volumes
                                    normalize_volume(name)

                                # update the metadata of sub-tomogram in _cut.txt (coordinates saved in sva/) and .txt (gathered into 3DAVG/*_original_volumes.txt)
                                data[32] = name
                                data[0] = str(index + 1)
                                # write the dimension of virus
                                data[6:9] = [str(A.shape[2]), str(A.shape[1]), str(A.shape[0])]

                                # rescale back for true virus size
                                if spike_binning != 1:
                                    data[6:9] = [
                                        str(A.shape[2] * spike_binning),
                                        str(A.shape[1] * spike_binning),
                                        str(A.shape[0] * spike_binning),
                                    ]

                                tmpf.write("\t".join(data) + "\n")

                                # replace the dimension of virus with the dimension of sub-tomogram
                                data[6:9] = [str(spike_size / spike_binning)] * 3
                                # replace the position of spike in virus with the center of sub-tomogram
                                data[3:6] = [str(spike_size / spike_binning // 2)] * 3

                                newf.write("\t".join(data) + "\n")

                                index += 1

            # only keep the virions line in _cut.txt if they're correctly extracted
            os.remove(coordinate_file)
            shutil.copy(tmp_coordinate_file, coordinate_file)

    # cleanup
    if (
        os.path.exists(virion_name + "_unbinned.rec")
        and not parameters.get("tomo_vir_seg_debug")
        and not virion_name.endswith("_vir0000")
    ):
        os.remove(virion_name + "_unbinned.rec")

def detect_virions(parameters, virion_size, binning, name):

    # HF vir: change the pixel size in the header of .rec binned tomogram
    # HoughMaxRadius is still set to 200 (Angstrom)
    # Assume tomo_vir_size is 3/2 times bigger than the diameter of actual virus
    ##############################################################################################################
    hough_diameter = 200 * 2
    hough_box = hough_diameter * (3 / 2)
    hough_pixel = hough_box / (virion_size / float(parameters["data_bin"]) / binning)
    hough_diameter = virion_size * 2

    # Since segmentation can work with original size of volume, we don't rescale the tomo.rec pixel size anymore
    # hough_radius = float(parameters["tomo_vir_size"])
    ################################################################################################################

    # use uniform pixel size in mrc header because avl relies in this value for specifying virion dimensions to search for
    command = "{0}/bin/alterheader -o 0,0,0 -del {1},{1},{1} {2}.rec".format(get_imod_path(), "%.2f" % hough_pixel, name)
    [ output, error ] = local_run.run_shell_command(command,parameters["slurm_verbose"])

    # find virions
    if not os.path.isfile("{0}.vir".format(name)) and ( parameters["tomo_vir_method"] == "auto" or parameters["tomo_spk_method"] == "virions" ):

        extension = "rec"
        if parameters['tomo_vir_binn'] > 1:
            # additional binning to speed up processing
            extension = "bin"
            command = f"{get_imod_path()}/bin/binvol -bin {parameters['tomo_vir_binn']} {name}.rec {name}.bin"
            [ output, error ] = local_run.run_shell_command(command,parameters["slurm_verbose"])

        # use uniform pixel size in mrc header because avl relies in this value for specifying virion dimensions to search for
        command = "{0}/bin/alterheader -o 0,0,0 -del {1},{1},{1} {2}.{3}".format(get_imod_path(), "%.2f" % (hough_pixel*parameters['tomo_vir_binn']), name, extension)
        [ output, error ] = local_run.run_shell_command(command,parameters["slurm_verbose"])

        """
        USAGE:

        VirusLocation  [--cuvatureThreholding] [--planarity]
                        [--cannyLowerThreshold <difference of pixels>]
                        [--cannyUpperThreshold <difference of pixels>] [-o
                        <filename>] [--diffusionNumberOfIterations <natural
                        number>] [--houghNumberOfVirus <natural number>]
                        [--houghMaximumRadius <single value in native physical
                        units>] [--houghMinimumRadius <single value in native
                        physical units>] [--rejectionCriteria2 <natural number>]
                        [--rejectionCriteria1 <single value in native physical
                        units>] [-V] ...  [--] [--version] [-h] <source_image>


        Where:

        --cuvatureThreholding
            use curvature thresholding to further reduce the feature pixel
            selected for the hough transfor ( default true )

        --planarity
            use a planarity instead of canny edge for feature detection into the
            hough transform ( default false )

        --cannyLowerThreshold <difference of pixels>
            smallest gradient magnitude of edge to trace (default = .75)

        --cannyUpperThreshold <difference of pixels>
            minimum gradient magnitude to seed edge trace (default = 1.0)

        -o <filename>,  --outputIMOD <filename>
            output IMOD model file to write virus as scattered open points in
            contour

        --diffusionNumberOfIterations <natural number>
            number if iterations of gradient diffusion to run (default = 10)

        --houghNumberOfVirus <natural number>
            number of virus to locate (default = 30)

        --houghMaximumRadius <single value in native physical units>
            maximum radius of virus in physical space  (default = 200)

        --houghMinimumRadius <single value in native physical units>
            minimum radius of virus in physical space (default = 90)

        --rejectionCriteria2 <natural number>
            Automatically dertermines threshold for the number of hough votes
            based on clustering,  the argument is the number of virus to perform
            cluster analysis on.

        --rejectionCriteria1 <single value in native physical units>
            Rejects viruses if the entire sphere plus padding is not with in the
            image boundary, the argument is padding applied to radius. The padding
            is in physical space

        -V,  --verbosity  (accepted multiple times)
            increases the information outputed

        --,  --ignore_rest
            Ignores the rest of the labeled arguments following this flag.

        --version
            Displays version information and exits.

        -h,  --help
            Displays usage information and exits.

        <source_image>
            (required)  source image file name to be read


        Automatic virus locator for electron tomography.

        There are four phases to this program: denosing, feature detection,
        sphere detection, and virus selection.

        For denoising, anisotropic diffusion is used followed by curvature flow.
        The number of iterations can be controlled.

        There are two methods for feature detection. The first utilized the
        Canny edge detector, which takes an upper and lower threshold of the
        gradient magnitude. Alternatively, a planarity detector can be used.
        This filter calculate the Hessian matrix and it's eigenvalues, then a
        planarity value is computed based on this geometry. In additional to
        either of these, the feature point can be further reduced by bounding
        the curvature of the iso-contours at the point.

        Next a spherical Hough transform utilizes the feature points to vote on
        the center of spheres with in a minimum and maximum radius.

        Lastly, the viruses are selected. The maximum number of virus canidates
        are choosen by a command line parameter. Then there are two optional
        rejection criteria which can be applied. The first rejects if the virus'
        radius plus paramaterize radius is not contained with in the image. The
        second, utilized a clustering algorithm to split a paramaterized number
        of virus into two groups.
        """

        # bound the size of virions by the given tolerance
        tolerance = min(100,abs(parameters["tomo_vir_det_tol"])) / 100.
        max_radius = parameters["tomo_vir_rad"] * ( 1 + tolerance ) / 2.0
        min_radius = parameters["tomo_vir_rad"] * ( 1 - tolerance ) / 2.0

        command = f"{get_tomo_path()}/itkCLT-next VirusLocation --cannyLowerThreshold {parameters['tomo_vir_canny_low']} --cannyUpperThreshold {parameters['tomo_vir_canny_high']} --diffusionNumberOfIterations {parameters['tomo_vir_iterations']} --houghNumberOfVirus {parameters['tomo_vir_number']} --houghMinimumRadius {min_radius} --houghMaximumRadius {max_radius} --rejectionCriteria1 10 --rejectionCriteria2 100 {name}.{extension} -o {name}.vir"
        [output, error] = local_run.run_shell_command(command,parameters["slurm_verbose"])

        # cleanup
        if parameters['tomo_vir_binn'] > 1:
            os.remove(name + ".bin")
            
        rec_x, rec_z, rec_y = get_image_dimensions(f"{name}.rec")

        command = f"{get_imod_path()}/bin/imodtrans -Y -n {rec_x},{rec_y},{rec_z} -sx {parameters['tomo_vir_binn']} -sy {parameters['tomo_vir_binn']} -sz {parameters['tomo_vir_binn']} {name}.vir {name}.vir"
        [output, error] = local_run.run_shell_command(command,False)

def process_virions(
    name, x, y, binning, tilt_angles, tilt_options, exclude_virions, parameters,
):
    """Performs virion detection/extraction in tomogram then spike detection/extraction in virion.

    Input files
    -----------
    name.vir : file, optional
        Virion coordinates in tomo volume

    Output files
    ------------
    name.vir
        Virion coordinates in tomo volume
    virion_name.txt
        Spike coordinates in extracted virion volume
    virion_name.rec
        Extracted virion volume
    """

    spike_size = parameters["tomo_ext_size"] if "tomo_ext_size" in parameters  else 0

    virion_binning, virion_size = get_vir_binning_boxsize(parameters["tomo_vir_rad"], parameters["scope_pixel"])
    _, rec_z, _ = get_image_dimensions(f"{name}.rec")

    if virion_size <= 0:
        logger.warning("Virion size not set, skipping virion processing")
        return

    # detect virions
    if not os.path.exists(f"{name}.vir"):
        detect_virions(parameters, virion_size, binning, name)

    if os.path.isfile("{0}.vir".format(name)):
        # load virion coordinates (and apply binning)
        virions = imod.coordinates_from_mod_file("%s.vir" % name)

        logger.info(f"Found {len(virions)} virions")

        # process virions in parallel
        arguments = []

        for virion in range(virions.shape[0]):
            """
            seg_thresh = 1

            if exclude_virions.shape[0] > 0:
                current_virion = exclude_virions[ exclude_virions[:,2] == virion ]
                if current_virion.shape[0] > 0:
                    # seg_thresh = current_virion.squeeze()[0]
                    seg_thresh = current_virion[0,0]
            else:
                print 'WARNING: Segmentation threshold not specified for virion {0}. Using second column in png as default.'.format( virion )
            """
            # HF vir: ignore virions which the thresholds are not selected
            ########################################################################################################
            virion_name = name + "_vir%04d" % virion

            if exclude_virions.size == 0:
                # using thresholds from website?
                seg_thresh = 0
                virion_thresholds = "virion_thresholds.next"
                if os.path.exists(virion_thresholds) and os.stat("virion_thresholds.next").st_size > 0 and parameters.get("micromon_block") != "tomo-segmentation-closed":
                    metadata = np.loadtxt( virion_thresholds, ndmin=2, dtype="str" )
                    virions_in_tilt_series = metadata[ metadata[:,0] == name ]
                    threshold = virions_in_tilt_series[ virions_in_tilt_series[:,1] == str(virion) ]
                    if threshold.size > 0:
                        seg_thresh = int(threshold.squeeze()[-1]) - 1
                    else:
                        # ignore virion
                        seg_thresh = 8
            else:
                # logger.info("Already screen over the segmentation thresholds in png")
                # logger.info(f"selected threshold coordinate for virion # {virion}:")
                current_virion = exclude_virions[exclude_virions[:, 2] == virion]
                seg_thresh = 8
                if current_virion.shape[0] > 0:
                    # if multiple thresholds are selected for this virion, we choose the last selection
                    seg_thresh = current_virion[-1, 0]

            if seg_thresh < 8:

                band_width = parameters["tomo_vir_detect_band"] / parameters["scope_pixel"]

                arguments.append(
                    (
                        name,
                        virion_name,
                        virions[virion],
                        seg_thresh,
                        x,
                        y,
                        binning,
                        rec_z,
                        virion_binning,
                        virion_size,
                        band_width,
                        tilt_angles,
                        spike_size,
                        tilt_options,
                        parameters,
                    )
                )

                if virion_name.endswith("_vir0000"):
                    if parameters["tomo_vir_detect_method"] != "none" and parameters["micromon_block"] != "tomo-segmentation-closed":
                        if not os.path.exists(virion_name + "_cut.txt"):
                            if "template" in parameters["tomo_vir_detect_method"]:
                                logger.info("Detecting spikes using template search")
                            elif  "tomo_vir_detect_method" in parameters and parameters["tomo_vir_detect_method"] == "mesh":
                                logger.info("Detecting spikes using mesh vertices")
                            else:
                                logger.warning("No spike detection method provided")

                logger.info(
                    "Using {0} as segmentation threshold (column {1}) for virion {2}".format(
                    get_virion_segmentation_thresholds(seg_thresh),
                    seg_thresh + 1,
                    virion_name.split("_vir")[-1] )
                )

            else:
                logger.warning(f"Ignoring virion {virion} with no segmentation threshold (column {seg_thresh})")

            #######################################################################################################

        if len(arguments) > 0:

            if virion_binning > 1:
                # down-sample aligned tilt-series first before reconstructing virions
                command = "%s/bin/newstack -ftreduce %d %s.ali %s_bin_vir.ali" % (get_imod_path(), virion_binning, name, name)
                [output, error] = local_run.run_shell_command(command,parameters["slurm_verbose"])
            else:
                shutil.copy2( f'{name}.ali', f'{name}_bin_vir.ali' )

            mpi.submit_function_to_workers(process_virion_multiprocessing, arguments, verbose=parameters["slurm_verbose"])

            # save all coordinates as .spk file
            if parameters["tomo_vir_detect_method"] != "none":
                global_spike_coordiantes = get_global_spike_coordiantes( name, parameters, x, y )
                if len(global_spike_coordiantes) > 0:
                    np.savetxt(f"{name}_all_spikes.txt", np.asarray(global_spike_coordiantes,dtype='f'))
                    command = (
                        f"{get_imod_path()}/bin/point2model -scat -sphere 10 -color 0,0,255 -input {name}_all_spikes.txt -output {name}.spk"
                    )
                    local_run.run_shell_command(command,verbose=parameters['slurm_verbose'])
                    os.remove(f"{name}_all_spikes.txt")

    else:
        logger.warning(f"No virions found")

def get_global_spike_coordiantes( name, parameters, micrographsize_x, micrographsize_y ):

    virion_bin, virion_boxsize = get_vir_binning_boxsize(parameters["tomo_vir_rad"], parameters["scope_pixel"])

    # set virion box size
    if virion_boxsize > 0:
        virion_boxsize /= virion_bin
    else:
        virion_bin = 1
        virion_boxsize = 0

    # tomogram binning factor with respect to raw micrographs
    squred_image = parameters["tomo_ali_square"]
    binning = parameters["tomo_rec_binning"]

    recZ = parameters["tomo_rec_thickness"] + parameters['tomo_rec_thickness'] % 2
    logger.info(f"Tomogram Z dimension is {recZ} (unbinned voxels)")
    # get size of full unbinned reconstruction from .rec file
    # recZ = rec_z * binning

    if os.path.exists("%s.vir" % name):
        virion_coordinates = imod.coordinates_from_mod_file("%s.vir" % name)
    else:
        virion_coordinates = np.empty( shape=(0, 0) )

    global_spike_coordinates = []

    # traverse all virions in tilt series
    for vir in range(virion_coordinates.shape[0]):

        vir_x, vir_y, vir_z = [
            binning * virion_coordinates[vir, 0],
            binning * virion_coordinates[vir, 1],
            recZ - binning * virion_coordinates[vir, 2],
        ]

        # check if we have picked spikes for this virion
        virion_file = "%s_vir%04d_cut.txt" % (name, vir)
        if os.path.isfile(virion_file):
            spikes_in_virion = np.loadtxt(
                virion_file, comments="number", usecols=(list(range(32))), ndmin=2
            )
            if spikes_in_virion.shape[0] == 0:
                logger.warning(
                    "File {0} contains no spikes. Skipping".format(virion_file)
                )
                continue
        else:
            logger.warning(f"File {virion_file} not found. Skipping.")
            continue

        # for all spikes in current virion
        for spike in range(spikes_in_virion.shape[0]):

            # extract local spike coordinates [0-479]
            spike_x, spike_y, spike_z = spikes_in_virion[spike, 3:6]

            # virion boxsize is supposed to be included in coordinates.txt
            virion_boxsize = spikes_in_virion[spike, 6]

            spike_x, spike_y, spike_z = spike_x, (virion_boxsize - spike_y), spike_z

            # compute global spike coordinates from virus box size
            spike_X = vir_x + (spike_x - virion_boxsize // 2) * virion_bin
            spike_Y = vir_y + (spike_y - virion_boxsize // 2) * virion_bin
            spike_Z = vir_z + (spike_z - virion_boxsize // 2) * virion_bin
        
            global_spike_coordinates.append( [spike_X / binning, spike_Y / binning, spike_Z / binning] )
    
    return global_spike_coordinates

def build_virion(virion, binning, virion_size, x, y, tilt_options, name, virion_name):
    # The first slice is the centroid minus half of the boxsize
    fslice = float(virion[1]) * binning - (virion_size / 2)

    # The last slice is fslice + boxsize - 1
    lslice = fslice + (virion_size - 1)

    # Check that slices do not extend beyond the edge of the tomogram on y
    # i.e. that there are no smaller than 0 or larger than ysize
    ypad_up = ypad_dn = 0

    # Restrict upper Y and calculate padding (Y is actually Z in the reconstructed virion)
    if lslice > x - 1:
        ypad_up = lslice - x
        lslice = x - 1

    # Restrict lower Y and calculate padding
    if fslice < 0:
        ypad_dn = fslice
        fslice = 0

    shiftx = y / 2 - float(virion[0]) * binning
    rec_z = 256
    shiftz = (
        rec_z / 2 - float(virion[2])
    ) * binning  # shifty = y / binning - float(virion[1]) * binning

    # reconstruct virion
    command = "{0}/bin/tilt -input {1}_virbin.ali -output {2}.rec -TILTFILE {1}.tlt -SHIFT {3},{4} -SLICE {5},{6} -THICKNESS {7} -WIDTH {7} -IMAGEBINNED 1 -FULLIMAGE {8},{9} {10}".format(
        get_imod_path(),
        name,
        virion_name,
        shiftx,
        shiftz,
        int(fslice),
        int(lslice),
        virion_size,
        x,
        y,
        tilt_options,
    )
    local_run.run_shell_command(command,verbose=False)

    # pad volume to have uniform dimensions
    if math.fabs(ypad_dn) > 0 or math.fabs(ypad_up) > 0:
        command = "{0}/bin/newstack -secs {1}-{2} -input {3}.rec -output {3}.rec -blank".format(
            get_imod_path(), int(ypad_dn), int(virion_size - 1 + ypad_dn), virion_name,
        )
        local_run.run_shell_command(command)

    # rotate volume to align with Z-axis
    command = "{0}/bin/clip rotx {1}.rec {1}.rec".format(get_imod_path(), virion_name)
    local_run.run_shell_command(command)


def build_virion_unbinned(
    virion, binning, virion_binning, virion_size, x, y, z_thickness, tilt_options, name, virion_name
):
    # virion_size *= virion_binning
    # x *= virion_binning
    # y *= virion_binning
    # binning *= virion_binning

    # The first slice is the centroid minus half of the boxsize
    # fslice = float(virion[1]) * binning - ( virion_size / 2 )
    fslice = float(virion[1]) * binning - (virion_size / 2) - 1

    # The last slice is fslice + boxsize - 1
    lslice = fslice + (virion_size - 1)

    # Check that slices do not extend beyond the edge of the tomogram on y
    # i.e. that there are no smaller than 0 or larger than ysize
    ypad_up = ypad_dn = 0

    # Restrict upper Y and calculate padding (Y is actually Z in the reconstructed virion)
    if lslice > x - 1:
        ypad_up = lslice - x
        lslice = x - 1

    # Restrict lower Y and calculate padding
    if fslice < 0:
        ypad_dn = fslice
        fslice = 0

    shiftx = y / 2 - float(virion[0]) * binning # images rotated
    rec_z = z_thickness

    shiftz = (rec_z / 2 - float(virion[2])) * binning  
    # shifty = y / binning - float(virion[1]) * binning

    # reconstruct virion
    command = "{0}/bin/tilt -input {1}_bin_vir.ali -output {2}_unbinned.rec -TILTFILE {1}.tlt -SHIFT {3},{4} -SLICE {5},{6} -THICKNESS {7} -WIDTH {7} -FULLIMAGE {8},{9} {10} -IMAGEBINNED {11}".format(
        get_imod_path(),
        name,
        virion_name,
        shiftx,
        shiftz,
        int(fslice),
        int(lslice),
        virion_size,
        x,
        y,
        tilt_options,
        virion_binning
    )
    local_run.run_shell_command(command,verbose=False)

    # pad volume to have uniform dimensions
    if math.fabs(ypad_dn) > 0 or math.fabs(ypad_up) > 0:
        command = "{0}/bin/newstack -secs {1}-{2} -input {3}_unbinned.rec -output {3}_unbinned.rec -blank".format(
            get_imod_path(), int(ypad_dn)/virion_binning, int(virion_size - 1 + ypad_dn)/virion_binning, virion_name,
        )
        local_run.run_shell_command(command)

    # rotate volume to align with Z-axis
    command = "{0}/bin/clip rotx {1}_unbinned.rec {1}_unbinned.rec".format(
        get_imod_path(), virion_name
    )
    local_run.run_shell_command(command)

    # bin virion volume
    # command = "{0}/bin/binvol --binning {1} {2}_unbinned.rec {2}_unbinned.rec".format(
    #     get_imod_path(), virion_binning, virion_name
    # )
    # local_run.run_shell_command(command)

    os.rename("%s_unbinned.rec" % (virion_name), "%s.rec" % (virion_name))


def extract_regions(parameters, name, x, y, binning, zfact, tilt_options):
    """Extracts specified regions from tomo volume.

    From [1]_:
        ROIs: this generates reconstructions for each selected ROI (optionally phase-flipped if you are doing CTF correction). 
        The option -extract_bin controls the binning factor applied to the reconstructed regions. 
        The default is not to bin the data (-extract_bin 1)

    Input files
    -----------
    name_regions.mod
        Specifies coordinates in tomo volume

    Output files
    ------------
    name_region_number.rec
        Extracted volume

    References
    ----------
    .. [1] Bartesaghi Lab Wiki
    """
    coordinates = imod.extract_irregular_regions("%s_regions.mod" % name)

    region = 0
    for coord in coordinates:
        logger.info(coord)
        fslice = (coord[2] - coord[5] / 2) * binning
        lslice = (fslice + coord[5] * binning) - 1
        shiftx = (512 / 2 - coord[0]) * binning
        shifty = (256 / 2 - coord[1]) * binning
        thickness = coord[4] * binning
        width = coord[3] * binning

        command = "{0}/bin/tilt -input {1}.ali -output {1}_region_{2}.rec -TILTFILE {1}.tlt -SHIFT {3},{4} -SLICE {5},{6} -THICKNESS {7} -WIDTH {8} -FULLIMAGE {9},{10} {11} {12}".format(
            get_imod_path(),
            name,
            region,
            shiftx,
            shifty,
            int(fslice),
            int(lslice),
            thickness,
            width,
            x,
            y,
            tilt_options,
            zfact,
        )
        local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])

        # apply binning factor
        if "extract_bin" in parameters and parameters["extract_bin"] > 1:
            command = "{0}/bin/binvol -input {1}_region_{2}.rec -output {1}_region_{2}.rec -binning {3}".format(
                get_imod_path(), name, region, parameters["extract_bin"]
            )
            local_run.run_shell_command(command)

        # rotate volume to align with Z-axis
        command = "{0}/bin/clip rotx {1}_region_{2}.rec {1}_region_{2}.rec".format(
            get_imod_path(), name, region
        )
        local_run.run_shell_command(command)

        # flipy if dm4 format
        if os.path.exists("isdm4"):
            command = "{0}/bin/clip flipx {1}_region_{2}.rec {1}_region_{2}.rec".format(
                get_imod_path(), name, region
            )
            local_run.run_shell_command(command)

        region += 1


def spk_extract_and_process(
    name,
    spike_name,
    shiftx,
    shiftz,
    fslice,
    lslice,
    spike_size,
    x,
    y,
    tilt_options,
    zfact,
    ypad_dn,
    ypad_up,
    pad_factor,
    parameters,
):

    # reconstruct virion
    command = "{0}/bin/tilt -input {1}.ali -output {2}.rec -TILTFILE {1}.tlt -SHIFT {3},{4} -SLICE {5},{6} -THICKNESS {7} -WIDTH {7} -IMAGEBINNED 1 -FULLIMAGE {8},{9} {10} {11}".format(
        get_imod_path(),
        name,
        spike_name,
        shiftx,
        shiftz,
        int(fslice),
        int(lslice),
        spike_size,
        x,
        y,
        tilt_options,
        zfact,
    )
    local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

    # pad volume to have uniform dimensions
    if math.fabs(ypad_dn) > 0 or math.fabs(ypad_up) > 0:
        command = "{0}/bin/newstack -secs {1}-{2} -input {3}.rec -output {3}.rec -blank && rm {3}.rec~".format(
            get_imod_path(), int(ypad_dn), int(spike_size - 1 + ypad_dn), spike_name,
        )
        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

    # rotate volume to align with Z-axis
    command = "{0}/bin/clip rotx {1}.rec {1}.rec && rm {1}.rec~".format(get_imod_path(), spike_name)
    local_run.run_shell_command(command)

    # padding volume
    if pad_factor > 1:
        command = "{0}/bin/clip resize -ox {2} -oy {2} -oz {2} {1}.rec {1}.rec && rm {1}.rec~".format(
            get_imod_path(), spike_name, spike_size / pad_factor
        )
        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

    # TODO: remove eman2 dependency
    # HF Liu: normalize the spikes
    """
    command = "{0}; e2proc3d.py --clip {3} --process=normalize.edgemean --apix {2} {1}.rec {1}.rec".format(
        eman_load_command(),
        spike_name,
        parameters["scope_pixel"],
        parameters["extract_box"],
    )
    local_run.run_shell_command(command)
    """

    # apply spike binning factor
    if "tomo_ext_binn" in parameters and  parameters["tomo_ext_binn"] > 1:
        command = "{0}/bin/binvol -input {1}.rec -output {1}.rec -binning {2} && rm {1}.rec~".format(
            get_imod_path(), spike_name, str(parameters["tomo_ext_binn"])
        )
        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

    # flipy if dm4 format
    if os.path.exists("isdm4"):
        command = "{0}/bin/clip flipx {1}.rec {1}.rec".format(
            get_imod_path(), spike_name
        )
        local_run.run_shell_command(command)

    # make projection for cleaning particles by 2D classification
    command = "{0}/bin/xyzproj -input {1}.rec -axis Y -angles 0,0,0 -output {1}.proj".format(
        get_imod_path(), spike_name
    )
    # local_run.run_shell_command(command)

def detect_and_extract_particles( name, parameters, current_path, binning, x, y, zfact, tilt_angles, tilt_options, exclude_virions ):

    # are we detecting virions?
    virion_mode = ( 
                   parameters.get("tomo_vir_method") != "none" and parameters.get("tomo_vir_rad") > 0
                   or parameters["micromon_block"] == "tomo-segmentation-closed"
                   or parameters["micromon_block"] == "tomo-picking-closed"
                   )

    spike_mode = ( parameters.get("tomo_spk_rad") > 0 and not virion_mode 
                  and
                  ( 
                   parameters.get("tomo_spk_method") != "none"
                   or parameters.get("tomo_pick_method") != "none" 
                   or parameters.get("tomo_srf_detect_method") != "none"
                   )
                   or virion_mode and parameters.get("tomo_vir_detect_method") != "none"
                   or parameters.get("micromon_block") == "tomo-particles-eval"
    )

    # initialize coordinate variables
    coordinates = virion_coordinates = spike_coordinates = np.array([])

    # use this radius when no estimation is available
    unbinned_virion_radius = parameters["tomo_vir_rad"] / parameters["scope_pixel"] / parameters['data_bin'] * parameters["tomo_vir_binn"]
    binned_virion_radius = unbinned_virion_radius / binning
    unbinned_spike_radius = parameters["tomo_spk_rad"] / parameters["scope_pixel"] / parameters['data_bin']
    binned_spike_radius = unbinned_spike_radius / binning

    # TODO: 1. pyp-eval
    if ( 
        parameters.get("micromon_block") == "tomo-preprocessing"
        and ( 
             virion_mode and parameters.get("tomo_vir_method") == "pyp-eval"
             or not virion_mode and parameters.get("tomo_spk_method") == "pyp-eval"
        )
        or parameters.get("micromon_block") == "tomo-particles-eval"
    ):

        logger.info("Doing NN-based picking")

        if not os.path.exists( project_params.resolve_path(parameters["detect_nn3d_ref"]) ):
            raise Exception(f"Trained model not found: {project_params.resolve_path(parameters['detect_nn3d_ref'])}")
        else:
            coordinates = joint.tomoeval(parameters,name)[:,[0,2,1]]
            # calculate unbinned coordinates
            coordinates *= binning
            # add radius (unbinned)
            if virion_mode:
                radius = unbinned_virion_radius
            else:
                radius = unbinned_spike_radius
            coordinates = np.hstack( ( coordinates.copy(), radius * np.ones((coordinates.shape[0],1)) ) )

    # 2. virions (new-only)
    elif parameters.get("tomo_pick_method") == "virions" and parameters.get("micromon_block") == "tomo-picking":

        logger.info("Doing automatic virion picking")

        # figure out virion parameters
        _, virion_size = get_vir_binning_boxsize(parameters["tomo_vir_rad"], parameters["scope_pixel"])
        
        # use new parameters for figuring out virion size
        parameters["tomo_vir_rad"] = parameters["tomo_spk_vir_rad"]

        # virion detection
        detect_virions(parameters, virion_size, parameters["tomo_rec_binning"], name)
        
        # read output and convert to unbinned coordinates
        coordinates = imod.coordinates_from_mod_file(f"{name}.vir")
        coordinates *= binning
        coordinates[:,-1] *= parameters['tomo_vir_binn']

    # 3. size-based
    elif parameters.get("tomo_spk_method") == "auto" or parameters.get("tomo_pick_method") == "auto":

        logger.info("Doing size-based picking")

        from pyp.detect.tomo import picker
        picker.pick( 
                    name,
                    radius = parameters["tomo_spk_rad"],
                    pixelsize = parameters["scope_pixel"],
                    auto_binning = binning,
                    contract_times = parameters["tomo_spk_contract_times_3d"],
                    gaussian = parameters["tomo_spk_gaussian_3d"],
                    sigma = parameters["tomo_spk_sigma_3d"],
                    stdtimes_cont = parameters["tomo_spk_stdtimes_cont_3d"],
                    min_size = parameters["tomo_spk_min_size_3d"],
                    dilation = parameters["tomo_spk_dilation_3d"],
                    radius_times = parameters["tomo_spk_radiustimes_3d"],
                    inhibit = parameters["tomo_spk_inhibit_3d"],
                    detection_width = parameters["tomo_spk_detection_width_3d"],
                    stdtimes_filt = parameters["tomo_spk_stdtimes_filt_3d"],
                    remove_edge = parameters["tomo_spk_remove_edge_3d"],
                    show = False
                    )

        # read and convert output to unbinned coordinates
        coordinates = imod.coordinates_from_mod_file(f"{name}.spk")
        coordinates *= binning
        coordinates = np.hstack( ( coordinates.copy(), unbinned_spike_radius * np.ones((coordinates.shape[0],1)) ) )

    # 4. manual
    elif ( 
          ( parameters.get("tomo_spk_method") == "manual" and not virion_mode ) 
          or parameters.get("tomo_vir_method") == "manual" 
          or parameters.get("tomo_pick_method") == "manual" ):
        
        logger.info("Using manual picking")

        if not os.path.exists( name + ".next" ):
            logger.warning("No coordinates file found for this tilt-series")
        else:
            # read unbinned coordinates from website
            coordinates = np.loadtxt(f"{name}.next",ndmin=2)
            
            if virion_mode:
                coordinates[:,-1] /= parameters["tomo_vir_binn"]

            # clean up
            remote_next_file = os.path.join( current_path, 'next', name + '.next' )
            if os.path.exists(remote_next_file):
                os.remove(remote_next_file)
            if os.path.exists(name + ".next"):
                os.remove( name + ".next")

    if virion_mode:
        
        # convert virion (unbinned) coordinates to pyp's .vir format, if needed
        if len(coordinates) > 0 and not os.path.exists(f"{name}.vir"):
            pyp_coordinates = coordinates[:,[0,2,1,3]] / binning
            imod.coordinates_to_model_file( pyp_coordinates, f"{name}.vir", radius=binned_virion_radius )
        # Performs virion detection and/or spike detection
        process_virions(
            name, x, y, binning, tilt_angles, tilt_options, exclude_virions, parameters,
        )

        # read virion coordinates and convert to unbinned, if needed
        if len(coordinates) == 0 and os.path.exists(f"{name}.vir"):
            coordinates = imod.coordinates_from_mod_file(f"{name}.vir")
            coordinates *= binning

        if len(coordinates) > 0:
            coordinates[:,-1] *= parameters["tomo_vir_binn"]
            virion_coordinates = coordinates

        # read spike coordinates and convert to unbinned, if needed
        if (
            os.path.exists(f"{name}.spk")
            and (
                parameters.get("tomo_srf_detect_method") != "none"
                or parameters.get("tomo_vir_detect_method") != "none"
            )
        ):
            coordinates = imod.coordinates_from_mod_file("%s.spk" % name)
            if len(coordinates) > 0:
                _, rec_z, _ = get_image_dimensions(f"{name}.rec")
                coordinates[:,2] = rec_z - coordinates[:,2]
                coordinates *= binning
                coordinates = np.hstack( ( coordinates.copy(), unbinned_spike_radius * np.ones((coordinates.shape[0],1)) ) )
            
    if spike_mode and len(coordinates) > 0:
        spike_coordinates = coordinates
        pyp_coordinates = coordinates[:,[0,2,1,3]] / binning
        pyp_coordinates[:,-1] *= parameters.get("tomo_vir_binn")
        imod.coordinates_to_model_file( pyp_coordinates, f"{name}.spk", radius=binned_spike_radius)

    # generate *_volumes.txt file and extract particles, if needed
    if (
        os.path.isfile("%s.spk" % name)
        or os.path.isfile("%s.txt" % name)
        or os.path.isfile("%s.openmod" % name)
    ) and parameters.get("tomo_vir_method") == "none" and parameters.get("micromon_block") != "tomo-picking-closed":
        t = timer.Timer(text="Sub-volume extraction took: {}", logger=logger.info)
        t.start()
        extract_spk_direct(
            parameters, name, x, y, binning, zfact, tilt_angles, tilt_options
        )
        t.stop()

    return virion_coordinates, spike_coordinates

def extract_spk_direct(
    parameters, name, x, y, binning, zfact, tilt_angles, tilt_options
):
    """Performs spike detection/extraction directly from tomo volume.

    Input files
    -----------
    name.{spk,txt} : file, optional
        Spike coordinates in tomo volume

    Output files
    ------------
    name_vir0000.txt
        Spike coordinates in tomo volume
    spike_name.rec
        Extracted spk volume
    """
    pad_factor = parameters["tomo_ext_padd"] if "tomo_ext_padd" in parameters else 1

    spike_size = pad_factor * ( parameters["tomo_ext_size"] if "tomo_ext_size" in parameters  else 0 ) * parameters["tomo_ext_binn"]

    # load spike coordinates
    if os.path.isfile("%s.spk" % name):
        spikes = imod.coordinates_from_mod_file("%s.spk" % name)
    # if there's txt file from Wendy and Ye (particles picked in cryolo)
    elif os.path.isfile("%s.txt" % name):
        spikes = np.loadtxt(
            "%s.txt" % name, comments="particle_x", usecols=(list(range(10))), ndmin=2,
        )
    # from manual picking of filament
    elif os.path.isfile("%s.openmod" % name):
        # spacing is 1/4 of box size, ~1/2 real particle size
        spikes = imod.regular_points_from_line(
            "%s.openmod" % name, spike_size / binning / 4
        )
    else:
        logger.info("No sub-volumes to process")
        return

    # load INVERSE affine transformations and tilt angle file
    inversexf = "%s_inverse.xf" % name

    command = """
%s/bin/xfinverse << EOF
%s.xf
%s
1
EOF
    """ % (
        get_imod_path(),
        name,
        inversexf,
    )
    [output, error] = local_run.run_shell_command(command,verbose=False)

    inversexf_file = np.loadtxt(inversexf, ndmin=2)
    tilts = np.loadtxt("%s.tlt" % name)
    micrograph_x, micrograph_y = x, y
    rec_X, z_thickness, rec_Y = get_image_dimensions(name +".rec")

    with open("%s_vir0000.txt" % name, "w") as f:

        # invert volume contrast for eman particles
        if not parameters["data_invert"] and parameters["tomo_ext_fmt"].lower() == "eman":
            command = "{0}/bin/newstack {1}.ali {1}.ali -multadd -1,0".format(
                get_imod_path(), name
            )
            local_run.run_shell_command(command)

        arguments = []

        for spk in range(spikes.shape[0]):

            spike = spikes[spk]
            if os.path.isfile("%s.txt" % name):
                # convert txt coordinate (MSCS data) to IMOD model convention
                # the binning factor of binned tomogram (512,512,256) with respect to tomogram
                # where particles were picked
                BINNING_FOR_PICKING = 2
                # The z height of tomograms where you pick particles
                Z_FOR_PICKING = z_thickness

                logger.info(("Information read from txt = %s", spikes[spk][0:6]))
                spike_x, spike_y, spike_z, virion_x, virion_y, virion_z = list(
                    [x / BINNING_FOR_PICKING for x in spikes[spk][0:6]]
                )

                # Particles were picked in tomogram in size of 1024 and height of 256
                # convert their coordinates in tomogram in size of 512 and height of 256
                spike_y = (
                    spike_y
                    - (Z_FOR_PICKING // (2 * (BINNING_FOR_PICKING)))
                    + (z_thickness // 2)
                )
                virion_y = (
                    virion_y
                    - (Z_FOR_PICKING // (2 * (BINNING_FOR_PICKING)))
                    + (z_thickness // 2)
                )

                # This is the TRUE coordinate observed when opening model and tomogram in IMOD
                spike_y, spike_z, virion_y, virion_z = (
                    spike_z,
                    z_thickness - spike_y,
                    virion_z,
                    z_thickness - virion_y,
                )

                logger.info(
                    "Spike position = [ %f, %f, %f ] ", spike_x, spike_y, spike_z
                )
                logger.info(
                    "Virion position = [ %f, %f, %f ] ", virion_x, virion_y, virion_z
                )

                # To abide by the IMOD model convention for subsequent extraction
                ### z = ( Height of tomogram - z )
                spike[0:3] = [spike_x, spike_y, z_thickness - spike_z]

            spike_name = "%s_spk%04d" % (name, spk)

            # logger.info("( %s ) = %s", spike_name, spike[0:3])

            # Only extract particles that are in full coverage of tilts
            # to bypass particles in the corner that lose one or more tilted projecitons

            [spike_X, spike_Y, spike_Z] = [spike[0], spike[1], z_thickness - spike[2]]
            [spike_X, spike_Y, spike_Z] = list(
                [x * binning for x in [spike_X, spike_Y, spike_Z]]
            )
            [center_X, center_Y, center_Z] = list(
                [x * binning // 2 for x in [rec_X, rec_Y, z_thickness]]
            )

            KEEP_ONLY_VOL_FULL = False
            lose_tilt = False

            for idx, tilt in enumerate(tilts):
                angle = math.radians(tilt)
                tilt_x = (spike_X - center_X) * math.cos(angle) + (
                    spike_Z - center_Z
                ) * math.sin(angle)
                tilt_y = spike_Y - center_Y
                T = inversexf_file[idx, :6]
                tilt_X = T[0] * tilt_x + T[1] * tilt_y + T[4] + center_X
                tilt_Y = T[2] * tilt_x + T[3] * tilt_y + T[5] + center_Y

                half_spk_size = spike_size / 2

                if (
                    tilt_X - half_spk_size < 0
                    or tilt_X + half_spk_size > micrograph_x
                    or tilt_Y - half_spk_size < 0
                    or tilt_Y + half_spk_size > micrograph_y
                ):
                    lose_tilt = True
                    break

            if lose_tilt and KEEP_ONLY_VOL_FULL:
                continue

            # figure out IMOD's tilt bound for reconstructing virion

            # The first slice is the centroid minus half of the boxsize
            fslice = float(spike[1]) * binning - (spike_size / 2)

            # The last slice is fslice + boxsize - 1
            lslice = fslice + (spike_size - 1)

            # Check that slices do not extend beyond the edge of the tomogram on y
            # i.e. that there are no smaller than 0 or larger than ysize
            ypad_up = ypad_dn = 0

            # Restrict upper Y and calculate padding (Y is actually Z in the reconstructed virion)
            if lslice > x - 1:
                ypad_up = lslice - x
                lslice = x - 1

            # Restrict lower Y and calculate padding
            if fslice < 0:
                ypad_dn = fslice
                fslice = 0

            # skip if reversed slice
            if fslice > lslice:
                continue

            # shiftx = y / 2 - float(spike[0]) * binning
            shiftx = x / 2 - float(spike[0]) * binning

            shiftz = (
                z_thickness / 2 - float(spike[2])
            ) * binning  # shifty = y / binning - float(virion[1]) * binning

            # compile arguments for parallel processing
            arguments.append(
                (
                    name,
                    spike_name,
                    shiftx,
                    shiftz,
                    fslice,
                    lslice,
                    spike_size,
                    x,
                    y,
                    tilt_options,
                    zfact,
                    ypad_dn,
                    ypad_up,
                    pad_factor,
                    parameters,
                )
            )

            # three Euler angles transforming spike norms vertically
            normX = normY = normZ = 0

            if os.path.isfile("%s.txt" % name):
                # norm comes directly from txt
                # [ normX, normY, normZ ] = list(map(float, spikes[spk][7:10]))
                logger.info(
                    "Information read from txt before norm calc = %s %s %s %s %s %s",
                    spike_x,
                    spike_y,
                    spike_z,
                    virion_x,
                    virion_y,
                    virion_z,
                )
                normX, normY, normZ = calcSpikeNormXYZ(
                    spike_x, spike_y, spike_z, virion_x, virion_y, virion_z
                )
                logger.info("NormX, NormY, NormZ = [ %f, %f, %f ]", normX, normY, normZ)

                CHECK_NORM = True
                if CHECK_NORM:
                    vector = np.array(
                        [spike_x - virion_x, spike_y - virion_y, spike_z - virion_z, 0]
                    )
                    norm = np.linalg.norm(vector)
                    vector = np.array(
                        [vector[0] / norm, vector[1] / norm, vector[2] / norm, 0]
                    )
                    normZ_m = vtk.rotation_matrix(np.radians(-normZ), [0, 0, 1])
                    normX_m = vtk.rotation_matrix(np.radians(-normX), [1, 0, 0])
                    result = np.dot(normX_m, np.dot(normZ_m, vector))
                    # result should be ( 0,0,1 )
                    logger.info("Vector after normZ & normX rotation is ", result)
            elif parameters['tomo_spk_rand']:
                # random normx normz, normy will be changed during merge
                normX = 360 * (random.random() - 0.5)
                normZ = 360 * (random.random() - 0.5)

            # Write txt for 3DAVG
            # number  lwedge  uwedge  posX    posY    posZ    geomX   geomY   geomZ   normalX normalY normalZ matrix[0]       matrix[1]       matrix[2]        matrix[3]       matrix[4]       matrix[5]       matrix[6]       matrix[7]       matrix[8]       matrix[9]       matrix[10]       matrix[11]      matrix[12]      matrix[13]      matrix[14]      matrix[15]      magnification[0]       magnification[1]      magnification[2]        cutOffset       filename
            f.write(
                """%d\t%.2f\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n"""
                % (
                    spk + 1,
                    tilt_angles.min(),
                    tilt_angles.max(),
                    spike_size / (2 * parameters["tomo_ext_binn"]),
                    spike_size / (2 * parameters["tomo_ext_binn"]),
                    spike_size / (2 * parameters["tomo_ext_binn"]),
                    spike_size / parameters["tomo_ext_binn"],
                    spike_size / parameters["tomo_ext_binn"],
                    spike_size / parameters["tomo_ext_binn"],
                    normX,
                    normY,
                    normZ,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    spike_name + ".rec",
                )
            )
        if detect.tomo_subvolume_extract_is_required(parameters):
            mpi.submit_function_to_workers(spk_extract_and_process, arguments, verbose=parameters["slurm_verbose"])


def mesh_coordinate_generator(virion_name, threshold, distance, bandwidth):

    # using smoothened surface virion_binned_nad_seg.mrc as template for grid coordinates
    if os.path.isfile("{0}_binned_nad_seg.mrc".format(virion_name)):

        virion_volume = "{0}_binned_nad_seg.mrc".format(virion_name)

        if mrc.readHeaderFromFile(virion_volume)['mode'] != 2:
            command = "{0}/bin/newstack {1} {1} -mode 2".format(
                get_imod_path(), virion_volume
            )
            local_run.run_shell_command(command)

        volume = mrc.read(virion_volume)
        contour = float(threshold)

        verts, faces, _, _ = measure.marching_cubes(
            volume,
            level=contour,
            spacing=(1, 1, 1),
            step_size=1,
            allow_degenerate=False,
            method="lewiner",
            mask=None,
        )

        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)

        normals = compute_normals(verts, faces)

        clean_vts, clean_norms = clean_verts(verts, normals, distance)

        z = volume.shape[2]

        if bandwidth > 0 and bandwidth < z / 2:
            inner_band = z/2 - bandwidth
            outer_band = z/2 + bandwidth
            keep_mask = np.logical_and(clean_vts[:, 0] >= inner_band, clean_vts[:, 0] <= outer_band)
            clean_vts = clean_vts[keep_mask]
            clean_norms = clean_norms[keep_mask]

        # need to swap x, z the same as the vertices coordinates
        clean_norms_swapxz = clean_norms.copy()
        clean_norms_swapxz[:, [0, 2]] = clean_norms[:, [2, 0]]

        write_mesh_cmm(clean_vts, virion_name)

    else:
        logger.info("Cannot find {0}_binned_nad_seg.mrc".format(virion_name))

def compute_normals(vertices, faces):
    # Compute the normals for each triangle face
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]])

    # Normalize face normals
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    # Compute vertex normals by averaging the face normals of the adjacent faces
    vertex_normals = np.zeros(vertices.shape)
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]
    vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)

    return vertex_normals


def clean_verts(verts, normals, distance):

    from scipy.spatial import KDTree

    kdtree = KDTree(verts)
    visited = set()
    clean_verts = []
    clean_norm = []

    for i, vert in enumerate(verts):
        if i not in visited:
            neighbors = kdtree.query_ball_point(vert, distance)
            clean_vert = np.mean(verts[neighbors], axis=0)
            clean_verts.append(clean_vert)
            # use the first normal of the neighbor
            clean_norm.append(normals[neighbors[0]])
            visited.update(neighbors)

    return np.array(clean_verts), np.array(clean_norm)

def write_mesh_cmm(verts, name):

    with open("{}_auto.cmm".format(name), "w") as cmm:
        cmm.write('<marker_sets>\n<marker_set name="grid_picking">\n')
        id = 1
        for xyz in verts:
            x = float(xyz[2])
            y = float(xyz[1])
            z = float(xyz[0])
            cmm.write(
                """<marker id="{0}" x="{1}" y="{2}" z="{3}" r="1" g="0.8" b="0.2" radius="1"/>\n""".format(  
                str(id), str(x), str(y), str(z),
                )
            )
            id += 1
        cmm.write("</marker_set>\n</marker_sets>")
