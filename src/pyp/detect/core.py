import os
import shutil
import socket

import numpy as np

from pyp import utils
from pyp.analysis import plot
from pyp.detect import joint, topaz
from pyp.inout.image import mrc, writepng
from pyp.inout.metadata import frealign_parfile
from pyp.streampyp.web import Web
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path, get_parameter_files_path
from pyp.utils import get_relative_path
from pyp.utils.timer import Timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def tomo_spk_is_required(parameters):
    """Whether to detect and extract spikes."""
    return "tomo_spk_rad" in parameters and parameters["tomo_spk_rad"] > 0 or parameters.get("tomo_vir_detect_method") != "none"

def tomo_subvolume_extract_is_required(parameters):
    return "tomo_ext_size" in parameters and parameters["tomo_ext_size"] > 0 and parameters["tomo_ext_fmt"] != "none"

def tomo_vir_is_required(parameters):
    """Whether to detect and extract virions."""
    return ( "tomo_vir_method" in parameters and parameters["tomo_vir_method"] != "none" and parameters["tomo_vir_method"] != "pyp-train" and "tomo_vir_rad" in parameters and parameters["tomo_vir_rad"] > 0
            or parameters["micromon_block"] == "tomo-picking-closed" 
            or parameters["micromon_block"] == "tomo-segmentation-closed" )


def is_required(parameters,name):
    """Whether to detect and extract particles."""
    extensions = ["box"]
    return "detect_rad" in parameters and parameters["detect_rad"] > 0 and not utils.has_files(name, extensions)


def is_done(name):
    extensions = ["box"]
    return utils.has_files(name, extensions)


def gold_beads_is_done(name):
    suffixes = ["_gold3d.mod"]
    return utils.has_files(name, suffixes, suffix=True)


@Timer(
    "detect_gold_beads",
    text="Detecting gold beads took: {}",
    logger=logger.info,
)
def detect_gold_beads(parameters, name, x, y, binning, zfact, tilt_options):
    """Detect and project gold beads in 3D reconstruction."""

    # find gold beads in 3D reconstruction
    size_of_gold = parameters["tomo_ali_fiducial"] / parameters["scope_pixel"] / binning
    command = "{0}/bin/findbeads3d -size {1} {2}.rec {2}_gold3d.mod -max 500 -threshold 0.1 -store 0.5".format(
        get_imod_path(), size_of_gold, name
    )
    local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])

    thickness = parameters["tomo_rec_thickness"]
    # project gold beads into raw tilt-series
    command = "{0}/bin/tilt -input {1}_bin.ali -output {1}_gold.mod -TILTFILE {1}.tlt -SHIFT 0.0,0.0  -THICKNESS {2} -IMAGEBINNED {3} -FULLIMAGE {4},{5} {6} {7} -ProjectModel {1}_gold3d.mod".format(
        get_imod_path(), name, thickness, binning, x, y, tilt_options, zfact,
    )
    local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])


@Timer("detect", text="Particle picking took: {}", logger=logger.info)
def pick_particles(
    name, image, mparameters, pixelsize, setdefocus=10000,
):
    radiuses = mparameters["detect_rad"]
    data_bin = int(mparameters["data_bin"])
    if "extract_box" in mparameters.keys():
        boxsize = mparameters["extract_box"]
        boundsize = int(mparameters["extract_bnd"])
        binning = mparameters["extract_bin"]
        symmetry = mparameters["particle_sym"]
        classification_pass = mparameters["extract_cls"]
    else:
        boxsize = 0
        boundsize = 0
        binning = 1
        symmetry = "C1"
        classification_pass = 0

    # radius = np.array(radiuses.split(","), dtype=float).max()
    radius = radiuses

    if True or not os.path.isfile("{}.box".format(name)):

        if os.path.exists("{}.boxx".format(name)):
            os.remove("{}.boxx".format(name))

        logger.info(
            "Pick particles using strategy: {}".format(mparameters["detect_method"])
        )

        if "grabber" in mparameters["detect_method"].lower():
            # GRABBER: prepare parameter file using template
            f = open("{0}/grabber.xml".format(get_parameter_files_path()))
            parameters = f.read()
            f.close()
            parameters = parameters.replace("OUTPATH", os.getcwd() + "/")
            parameters = parameters.replace("PATH", os.getcwd() + "/grabber/")
            parameters = parameters.replace("MICROGRAPH", name)
            parameters = parameters.replace("RADIUS", str(radius))

            f = open("{}_grabber.xml".format(name), "w")
            f.write(parameters)
            f.close()

            os.mkdir("grabber")

            # use larger binning if in super-resolution mode to prevent jasper's memory explosion
            if image.shape[0] > 6096:
                auto_binning = 8
            else:
                auto_binning = 4
            com = "{0}/bin/newstack {1}.mrc grabber/{1}.mrc -bin {2}".format(
                get_imod_path(), name, binning
            )
            local_run.run_shell_command(com)

            shutil.copy("{}.mrc".format(name), "grabber")
            if os.path.exists("{0}.xml".format(name)):
                shutil.copy("{0}.xml".format(name), "grabber/{0}.xml".format(name))
            else:
                # shutil.copy('{0}/micrograph.xml'.format(get_parameter_files_path(), 'grabber/{0}.xml'.format(name))
                f = open("{0}/micrograph.xml".format(get_parameter_files_path()))
                parameters = f.read()
                f.close()
                parameters = parameters.replace(
                    "SETPIXELSIZE", str(pixelsize * auto_binning)
                )
                parameters = parameters.replace("SETDEFOCUS", str(setdefocus))
                f = open("grabber/{0}.xml".format(name), "w")
                f.write(parameters)
                f.close()

            # run grabber
            command = "{0}/scripts/grabberfixed {1}_grabber.xml".format(
                os.environ["GRABBERDIR"], name
            )
            local_run.run_shell_command(command)
            shutil.rmtree("grabber")

        elif "jasper" in mparameters["detect_method"].lower():

            if "," in radiuses:
                radx, rady, radz = radiuses.split(",")
            else:
                radx = rady = radz = radius

            # use larger binning if in super-resolution mode to prevent jasper's memory explosion
            if image.shape[0] > 6096:
                auto_binning = 8
            else:
                auto_binning = 4

            # JASPER: prepare parameter file using template
            if os.path.isfile("jasper.xml"):
                f = open("jasper.xml")
            else:
                f = open("{0}/jasper.xml".format(get_parameter_files_path()))
            parameters = f.read()
            f.close()
            parameters = parameters.replace("OUTPATH", os.getcwd() + "/")
            parameters = parameters.replace("PATH", os.getcwd() + "/jasper/")
            if "RADIUSX" in parameters:
                parameters = parameters.replace(
                    "RADIUSX", str(2.0 * float(radx) / auto_binning)
                )
                parameters = parameters.replace(
                    "RADIUSY", str(2.0 * float(rady) / auto_binning)
                )
                parameters = parameters.replace(
                    "RADIUSZ", str(2.0 * float(radz) / auto_binning)
                )
            else:
                # OLD STYLE XML FILE
                parameters = parameters.replace(
                    "RADIUS", str(2.0 * float(radius) / auto_binning)
                )
            parameters = parameters.replace("SYMMETRY", symmetry)
            parameters = parameters.replace("BINNING", "1")

            f = open("{}_jasper.xml".format(name), "w")
            f.write(parameters)
            f.close()

            os.mkdir("jasper")

            # shutil.copy('{}.mrc'.format(name), 'jasper')

            com = "{0}/bin/newstack {1}.avg jasper/{1}.mrc -bin {2}".format(
                get_imod_path(), name, auto_binning
            )
            local_run.run_shell_command(com)

            if os.path.exists("{0}.xml".format(name)):
                shutil.copy("{0}.xml".format(name), "jasper/{0}.xml".format(name))
            else:
                f = open("{0}/micrograph.xml".format(get_parameter_files_path()))
                parameters = f.read()
                f.close()
                parameters = parameters.replace("SETPIXELSIZE", str(pixelsize))
                parameters = parameters.replace("SETDEFOCUS", str(setdefocus))
                f = open("jasper/{0}.xml".format(name), "w")
                f.write(parameters)
                f.close()

            logger.info(f"Particle radius used for picking: {radius}")

            # run jasper
            command = "unset DISPLAY; {0}/jasper pickSingle {1}_jasper.xml {1}".format(
                os.environ["JASPERDIR"], name, name
            )
            logger.info("\nUsing configuration file {0}_jasper.xml:\n".format(name))
            local_run.run_shell_command("cat {0}_jasper.xml".format(name))
            local_run.run_shell_command(command)
            shutil.rmtree("jasper")

            # save picked particles coordinate in unbinned format
            if os.path.exists("{}.box".format(name)):
                box = np.loadtxt("{}.box".format(name), ndmin=2)
                np.savetxt(
                    "{}.box".format(name), box * data_bin * auto_binning, fmt="%i\t"
                )
            else:
                logger.warning("No particles found")

        elif "shape" in mparameters["detect_method"].lower() and "," in radiuses:

            parameters = project_params.load_pyp_parameters()
            radx, rady, radz = radiuses.split(",")

            try:
                os.mkdir("pick")
            except:
                pass
            shutil.copy("{}.avg".format(name), "pick/{0}.mrc".format(name))

            command = "module load Anaconda/2.0.1; unset DISPLAY; export EMAN2DIR=/usr/local/apps/EMAN2/2.1pre_mpi1.6.5_eth; export LD_LIBRARY_PATH=$EMAN2DIR/lib:$EMAN2DIR/extlib/lib; export PYTHONPATH=/usr/local/apps/Anaconda/2.0.1/lib/python2.7/site-packages/:$EMAN2DIR/lib:$EMAN2DIR/bin:$EMAN2DIR/extlib/site-packages; python {0}/pick/pick.py -m {1} -i {2} -o {2} --size_x_A={3} --size_y_A={4} --size_z_A={5} --symtype={6} -p {7} --bin_sz={8}".format(
                os.environ["PYTHONDIR"],
                name,
                os.getcwd() + "/pick",
                radx,
                rady,
                radz,
                parameters["particle_sym"].lower(),
                pixelsize,
                1024,
            )

            local_run.run_shell_command(command)

            # fix coordinates to match pipeline's convention and save new box file
            if os.path.exists("pick/{0}.box".format(name)):
                box = np.loadtxt("pick/{0}.box".format(name), dtype=float)
                zeros = np.zeros([box.shape[0], 2])
                box = np.hstack((box[:, :2][:, ::-1], zeros))
                np.savetxt("{0}.box".format(name), box, fmt="%i\t")

        elif mparameters["detect_method"].endswith("-train") or mparameters["detect_method"].endswith("-eval"):

            if "topaz" in mparameters["detect_method"].lower():
                boxs = topaz.spreval(mparameters,name)
            else:
                boxs = joint.spreval(mparameters,name)
            if boxs is not None:
                if len(boxs) > 0:
                    boxes = np.zeros( [boxs.shape[0], 4] )
                    boxes[:,0] = boxs[:,0]
                    boxes[:,1] = boxs[:,1]
                    np.savetxt(
                        "{}.box".format(name), boxes * data_bin, fmt="%i\t"
                    )
        elif mparameters["detect_method"] == "manual":
            # convert manually picked coordinates to spk file
            radius_in_pixels = int(mparameters["detect_rad"] / mparameters["scope_pixel"] / binning)
            # convert coordinates from next to pyp format
            try:

                coordinate_file = np.loadtxt( "{0}.next".format(name), dtype='str', ndmin=2)
                next_coordinates = coordinate_file.astype("float")

                # clean up
                with open("project_folder.txt") as f:
                    project_folder = f.read()

                remote_next_file = os.path.join( project_folder, 'next', name + '.next' )
                if os.path.exists(remote_next_file):
                    os.remove(remote_next_file)
                os.remove( name + ".next")

                pyp_coordinates = np.zeros( [ next_coordinates.shape[0], next_coordinates.shape[1] + 2 ] )
                pyp_coordinates[:,:2] = next_coordinates[:,:2]

                # save positions to box file
                if len(pyp_coordinates) > 0:
                    np.savetxt(
                        "{}.box".format(name), pyp_coordinates * data_bin, fmt="%i\t"
                    )
                logger.info(f"{pyp_coordinates.shape[0]} particles found")
            except:
                logger.warning("No particles picked for this micrograph")
        elif (
            "all" in mparameters["detect_method"].lower()
            or "auto" in mparameters["detect_method"].lower()
            or ("detect_ref" in mparameters and os.path.exists(mparameters["detect_ref"]))
        ):

            # size based particle picking

            # set up enviroment for FREALIGN refinement
            # ctffile = name + ".ctf"
            # ctf = np.loadtxt(ctffile)
            # try:
            #    os.mkdir("ctf")
            # except:
            #    pass
            # shutil.copy2(ctffile, "ctf")

            try:
                os.mkdir("frealign")
            except:
                pass

            # micrograph binning for performing search
            auto_binning = 6
            # if ctf[6] > 6096:
            #     auto_binning *= 2
            com = "{0}/bin/newstack {1}.avg frealign/{1}.mrc -bin {2}".format(
                get_imod_path(), name, auto_binning
            )
            local_run.run_shell_command(com)
            micrograph = mrc.read("frealign/%s.mrc" % name)

            # box size to extract particles (3x particle radius)
            tilesize = int(3 * radius / pixelsize / auto_binning)
            if tilesize % 2 > 0:
                tilesize += 1

            """ tile complete image
            overlap = tilesize / 8
            for x in range(0, micrograph.shape[0]-tilesize+1, overlap):
                for y in range(0, micrograph.shape[1]-tilesize+1, overlap):
                    boxs.append( [y,x] )
            """

            # detect contamination
            import scipy
            from skimage.morphology import disk, remove_small_objects

            G = micrograph - scipy.ndimage.gaussian_filter(micrograph, sigma=25)  # 150
            mask = G < G.mean()
            cmask = scipy.ndimage.morphology.binary_opening(mask, disk(1))
            clean = remove_small_objects(cmask, min_size=150)
            segmentation = scipy.ndimage.morphology.binary_closing(clean, disk(2))
            # dilate 56A around contamination
            contamination_dilation = int(56.0 / auto_binning / pixelsize)
            # contamination_dilation = 15
            area = scipy.ndimage.morphology.binary_dilation(
                segmentation, disk(contamination_dilation)
            )
            if mparameters["detect_ignore_contamination"]:
                area = np.full(area.shape,False)
            # np.save( '{0}_mask.npy'.format(name), area )

            # detect density peaks in particle's size range
            radius1 = radius2 = 0.5 * pixelsize * auto_binning / radius
            sigma1 = 0
            sigma2 = 0.01
            radius1 = 0.001
            sigma2 = 0.001  # (used for apoferritin together with change below)
            com = "{0}/bin/mtffilter -radius1 {2} -hi {3} -l {4},{5} frealign/{1}.mrc frealign/{1}_bp.mrc".format(
                get_imod_path(), name, radius1, sigma1, radius2, sigma2
            )
            local_run.run_shell_command(com)

            lowres = mrc.read("frealign/{0}_bp.mrc".format(name))
            locality = int(radius / pixelsize / auto_binning)
            locality = int(mparameters["detect_dist"])
            # locality = max( 5, int( radius / pixelsize / auto_binning / 4 ) )
            # locality = max( 5, int( radius / pixelsize / auto_binning ) ) # used for apoferritin because contrast is higher in center of particle (not lower)
            logger.info(f"Using locality value of {locality} for particle detection")
            if not "noise" in mparameters["detect_method"].lower():
                minimas = (
                    lowres == scipy.ndimage.minimum_filter(lowres, locality)
                ).nonzero()
            else:
                minimas = (
                    lowres == scipy.ndimage.maximum_filter(lowres, locality)
                ).nonzero()
            boxs = []
            for i in range(minimas[0].shape[0]):
                x = minimas[0][i] - tilesize / 2
                y = minimas[1][i] - tilesize / 2
                clean = not area[minimas[0][i], minimas[1][i]]
                inside = (
                    x >= 0
                    and x < micrograph.shape[0] - tilesize + 1
                    and y >= 0
                    and y < micrograph.shape[1] - tilesize + 1
                )
                if clean and inside:
                    boxs.append([y, x])

            # pre-filtering of positions done based on image statistics of background vs. foreground
            boxes = boxs[:]
            boxes_not_normalized = boxs[:]

            if len(boxs) > 0:
                
                from pyp import extract

                particles = extract.extract_particles_old(
                    micrograph, boxs, radius, tilesize, 1, pixelsize * auto_binning
                )

                mrc.write(particles, "frealign/" + name + "_stack.mrc")

                # compute image indicators
                particles = extract.extract_particles_old(
                    lowres,
                    boxes_not_normalized,
                    radius,
                    tilesize,
                    1,
                    pixelsize * auto_binning,
                    False,
                )

                x, y = np.mgrid[0:tilesize, 0:tilesize] - tilesize // 2
                if radius / pixelsize / auto_binning > tilesize / 2:
                    logger.warning(
                        f"Particle radius falls outside box {radius} > {tilesize // 2 * pixelsize}"
                    )
                    iradius = tilesize * pixelsize * auto_binning / 2
                else:
                    iradius = radius
                condition = np.hypot(x, y) > iradius / pixelsize / auto_binning
            else:
                particles = np.array([])

            boxs = []
            boxs_colors = []
            boxs_discarded = []
            boxs_discarded_colors = []
            indicators = np.zeros([particles.shape[0], 5])
            for p in range(particles.shape[0]):
                raw = particles[p, :, :]
                background = np.extract(condition, raw)
                foreground = np.extract(np.logical_not(condition), raw)
                indicators[p, 0] = (
                    foreground.mean() < 0.015 and foreground.std() > 1.025
                )
                indicators[p, 1] = foreground.mean()
                indicators[p, 2] = foreground.std()
                indicators[p, 3] = background.mean()
                indicators[p, 4] = background.std()
                # if foreground.mean() < 0 and foreground.std() > 1.000:

                # switch between different picking strategies
                if "auto" in mparameters["detect_method"].lower():
                    # conditione = foreground.std() > background.std()
                    # conditione = ( foreground.std() > 1.25 ) and ( foreground.mean() < -.5 )
                    # conditione = ( foreground.mean() < .965 * background.mean() ) and ( foreground.std() > background.std() )

                    # conditione = ( foreground.mean() < .25 * background.mean() ) and ( foreground.std() > 1.5 * background.std() )

                    # conditione = ( foreground.mean() < .98 * background.mean() ) and ( foreground.mean() < .97 * particles.mean() )
                    # conditione = ( foreground.mean() < .98 * background.mean() ) and ( raw.std() > 1.05 * background.std() )
                    conditione = foreground.std() > background.std()

                elif "all" in mparameters["detect_method"].lower():
                    conditione = True
                # elif 'noise' in mparameters['detect_method'].lower():
                #    conditione = foreground.mean() > background.mean() and foreground.std() < background.std()
                else:
                    conditione = foreground.mean() < lowres.mean()

                # if 'all' in mparameters['detect_method'].lower() or foreground.mean() < lowres.mean():
                # if foreground.std() > 1:

                if conditione:
                    boxs.append([boxes[p][0], boxes[p][1]])
                    boxs_colors.append(np.median(foreground))
                else:
                    boxs_discarded.append([boxes[p][0], boxes[p][1]])
                    boxs_discarded_colors.append(np.median(foreground))

                """ plot histogram for each particle
                import matplotlib.pyplot as plt
                from matplotlib import cm
                fig, ax = plt.subplots( figsize=(7.5,2) )
                n, bins, patches = ax.hist( foreground.ravel(), 100, density=1, facecolor='green', alpha=0.75 )
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                col = bin_centers - min(bin_centers)
                col /= max(col)
                for co, pa in zip(col, patches):
                    plt.setp(pa, 'facecolor', cm.rainbow(co))
                plt.savefig('{0}_hist_{1}.png'.format(name,'%03d'%p))
                """

            """
            # plot histogram for each particle
            import matplotlib.pyplot as plt
            from matplotlib import cm
            fig, ax = plt.subplots( figsize=(7.5,2) )
            
            n, bins, patches = ax.hist( np.array( boxs_colors + boxs_discarded_colors ).ravel(), 25, facecolor='green', alpha=0.75 )
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - min(bin_centers)
            col /= max(col)
            for co, pa in zip(col, patches):
                plt.setp(pa, 'facecolor', cm.hot(co))
            plt.savefig('{0}_energy.png'.format(name))
            """

            # np.save( '{0}_indicators.npy'.format(name), indicators )

            logger.info(f"Keeping {len(boxs)} from {particles.shape[0]}")
            # output diagnositcs
            """
            colors = []
            for i in range(particles.shape[0]):
                strenght = -lowres[ int( energy[i,1] + tilesize / 2 ), int( energy[i,2] + tilesize / 2 ) ]
                if prs[i] >= threshold and strenght >= umbral:
                    dx = energy[i,3] / pixelsize / auto_binning
                    dy = energy[i,4] / pixelsize / auto_binning
                    boxs_score_colors.append( energy[i,0] )
            """

            if False:
                import matplotlib.pyplot as plt
                from matplotlib import cm

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                msize = int(radius / pixelsize / auto_binning)
                msize *= msize
                # msize /= 4
                msize *= 3
                from scipy import ndimage

                smicrograph = ndimage.gaussian_filter(micrograph, 1)
                smicrograph = lowres
                mmean = smicrograph.mean()
                mstd = 2.0 * smicrograph.std()
                ax.imshow(
                    smicrograph, cmap=cm.Greys_r, vmin=mmean - mstd, vmax=mmean + mstd
                )
                if area.max() > area.min():
                    ax.contour(area, 1, colors="#00ff00", linewidths=4)
                ax.axis("off")
                if len(boxs) > 0:
                    if len(boxs_colors) > 0:
                        ax.scatter(
                            np.array(boxs)[:, 0] + tilesize // 2,
                            np.array(boxs)[:, 1] + tilesize // 2,
                            s=msize / 50,
                            facecolor="none",
                            c=boxs_colors,
                            cmap=cm.hot,
                            vmin=np.array(boxs_colors).min(),
                            vmax=np.array(boxs_colors).max(),
                            marker="o",
                            lw=1,
                        )
                        # ax.scatter( np.array(boxs)[:,0] + tilesize//2, np.array(boxs)[:,1] + tilesize//2, s=msize/3, facecolor='none', edgecolor='blue', marker="o", lw=2 )
                    if len(boxs_discarded_colors) > 0:
                        ax.scatter(
                            np.array(boxs_discarded)[:, 0] + tilesize // 2,
                            np.array(boxs_discarded)[:, 1] + tilesize // 2,
                            s=msize / 50,
                            facecolor="none",
                            c=np.array(boxs_discarded_colors),
                            cmap=cm.hot,
                            vmin=np.array(boxs_discarded_colors).min(),
                            vmax=np.array(boxs_discarded_colors).max(),
                            marker="s",
                            lw=1,
                            alpha=1,
                        )
                        # ax.scatter( np.array(boxs_discarded)[:,0]+tilesize//2, np.array(boxs_discarded)[:,1]+tilesize//2, s=msize/3, facecolor='none', edgecolor='red', marker="s", lw=1 )
                plt.savefig("{0}_peaks.pdf".format(name), bbox_inches="tight")
                plt.close()

            # template based particle selection using FREALIGN
            # if os.path.exists( mparameters['detect_ref'] ) and not 'all' in mparameters['detect_method'] and not 'auto' in mparameters['detect_method']:
            if "detect_ref" in mparameters and os.path.isfile(
                mparameters["detect_ref"]
            ):

                stringency = mparameters["detect_method"]
                if (
                    not "loose" in stringency.lower()
                    and not "tight" in stringency.lower()
                ):
                    logger.warning(
                        "Picking option %s not valid. Using tight mode." % stringency
                    )
                    stringency = "tight"

                logger.info(f"Using particle picking stringency {stringency}")

                # re-extract positions that survived initial criteria
                boxes = boxs[:]

                from pyp import extract

                particles = extract.extract_particles_old(
                    micrograph, boxs, radius, tilesize, 1, pixelsize * auto_binning
                )
                mrc.write(particles, "frealign/" + name + "_stack.mrc")
                particles = len(boxes)

                # set correct pixel size in mrc header
                command = """
    %s/bin/alterheader << EOF
    frealign/%s_stack.mrc
    del
    %s,%s,%s
    done
    EOF
    """ % (
                    get_imod_path(),
                    name,
                    pixelsize * auto_binning,
                    pixelsize * auto_binning,
                    pixelsize * auto_binning,
                )
                local_run.run_shell_command(command)

                frealign_parfile.Parameters.generateFrealignParFile(
                    [name], name, name, mparameters["ctf_use_ast"]
                )

                oparameters = project_params.load_pyp_parameters()
                mparameters = project_params.load_pyp_parameters()
                if (
                    mparameters["scope_pixel"] == "0"
                    or "auto" in mparameters["scope_pixel"].lower()
                ):
                    mparameters["scope_pixel"] = str(ctf[9])
                if (
                    mparameters["scope_voltage"] == "0"
                    or "auto" in mparameters["scope_voltage"].lower()
                ):
                    mparameters["scope_voltage"] = str(ctf[10])
                if (
                    mparameters["scope_mag"] == "0"
                    or "auto" in mparameters["scope_mag"].lower()
                ):
                    mparameters["scope_mag"] = str(ctf[11])
                mparameters["extract_box"] = str(tilesize)
                mparameters["extract_bin"] = auto_binning
                project_params.save_pyp_parameters(mparameters)

                energy_file = "{0}_energy.npy".format(name)
                re_run = True
                if os.path.exists(energy_file):
                    re_run = np.load(energy_file).shape[0] != particles
                    if re_run:
                        logger.info(
                            "Energy dimensions do not match ({0} != {1}), recomputing\n".format(
                                np.load(energy_file).shape[0], particles
                            )
                        )
                    else:
                        logger.info("Using existing energy {0}\n".format(energy_file))

                if re_run:

                    logger.info(
                        "Evaluating FREALIGN score on %d locations (tilesize=%d)\n"
                        % (len(boxes), tilesize)
                    )

                    frealign_iters = 2

                    # launch FREALIGN refinement (skip reconstruction, full search)
                    use_rhref = 16

                    # for NTSR1
                    # use_rhref = 8

                    # mparameters['cpu_count'] = '48'

                    command = "cd frealign; export MYCORES={4}; echo {5} > `pwd`/mynode; export MYNODES=`pwd`/mynode; {0}/bin/fyp -dataset {1} -iter 2 -maxiter {2} -mode 4 -mask 1,1,1,1,1 -cutoff -1 -rlref 100 -rhref {6} -ipmax 50 -itmax 25 -dang 0 -fmatch T -model {3}".format(
                        os.environ["PYP_DIR"],
                        name,
                        frealign_iters,
                        mparameters["detect_ref"],
                        mparameters["slurm_tasks"],
                        socket.gethostname(),
                        use_rhref,
                    )
                    # command = 'cd frealign; export MYCORES={4}; echo {5} > `pwd`/mynode; export MYNODES=`pwd`/mynode; {0}/refine/frealign/frealign.py -dataset {1} -iter 2 -maxiter {2} -mode 1 -mask 0,0,0,1,1 -cutoff -1 -rlref 100 -rhref 20 -ipmax 1 -itmax 1 -dang 0 -fmatch F -model {3}'.format( os.environ['PYTHONDIR'], name, frealign_iters, mparameters['particle_ref'], mparameters['cpu_count'], socket.gethostname() )
                    local_run.run_shell_command(command)

                    # restore original parameters
                    project_params.save_pyp_parameters(oparameters)

                    # parse parameter file
                    parfile = "frealign/maps/%s_r01_%02d.par" % (name, frealign_iters)
                    input = np.array(
                        [
                            line.split()
                            for line in open(parfile)
                            if not line.startswith("C")
                        ],
                        dtype=float,
                    )

                    if input.shape[1] > 13:
                        field = 14
                    else:
                        field = 11

                    logger.info("saving mp and fp parameter files")
                    """
                    energy = np.empty( [ particles_x, particles_y, 6 ] )
                    newshape = [ particles_x, particles_y ]
                    energy[ :, :, 0 ] = np.reshape( input[:,field], newshape )                    # parse FREALIGN scores
                    energy[ :, :, 1 ] = np.reshape( np.array(boxes)[:,1], newshape )           # corresponding x coordinate
                    energy[ :, :, 2 ] = np.reshape( np.array(boxes)[:,0], newshape )           # corresponding y coordinates
                    energy[ :, :, 3 ] = np.reshape( -input[:,4], newshape )                       # parse FREALIGN x-shifts
                    energy[ :, :, 4 ] = np.reshape( -input[:,5], newshape )                       # parse FREALIGN y-shift
                    energy[ :, :, 5 ] = np.reshape( np.mod( input[:,2], 180 ), newshape )      # parse FREALIGN orientations
                    """

                    energy = np.empty([particles, 6])
                    energy[:, 0] = input[:, field]  # parse FREALIGN scores
                    energy[:, 1] = np.array(boxes)[:, 1]  # corresponding x coordinate
                    energy[:, 2] = np.array(boxes)[:, 0]  # corresponding y coordinates
                    energy[:, 3] = -input[:, 4]  # parse FREALIGN x-shifts
                    energy[:, 4] = -input[:, 5]  # parse FREALIGN y-shift
                    energy[:, 5] = np.mod(
                        input[:, 2], 180
                    )  # parse FREALIGN orientations

                    np.save("{0}_energy.npy".format(name), energy)

                    matches = "frealign/maps/%s_r01_%02d_match_unsorted.mrc" % (
                        name,
                        frealign_iters,
                    )
                    if os.path.exists(matches):

                        # compute distances between images and projections within mask
                        msize = int(mrc.readHeaderFromFile(matches)["nx"]) / 2.0
                        y, x = np.ogrid[-msize:msize, -msize:msize]
                        mask = np.where(x ** 2 + y ** 2 <= msize ** 2, 1, 0)
                        match = mrc.read(matches)
                        scores = np.empty([match.shape[0]])

                        for i in range(scores.size):
                            energy[i, 0] = -np.abs(
                                mask
                                * (match[i, msize * 2 :, :] - match[i, : msize * 2, :])
                            ).mean()

                        energy[:, 0] -= energy[:, 0].min()

                        # resort images based on new metric
                        indexes = np.argsort(energy[:, 0])[::-1]
                        mrc.write(match[indexes, :, :], matches)

                        logger.info(energy[:, 0])
                        logger.info(indexes)
                        logger.info(energy[indexes, 0])

                        A = mrc.read(matches)
                        M = plot.contact_sheet(A)
                        writepng(M, "{0}_matches.png".format(name))
                        command = "convert -resize 200% {0}_matches.png {0}_matches.png".format(
                            name
                        )
                        local_run.run_shell_command(command)

                else:
                    energy = np.load("{0}_energy.npy".format(name))

                # prs = energy[:,:,0]
                prs = energy[:, 0]

                # optimal FREALIGN score threshold based on bimodal distribution
                from sklearn import mixture

                gmix = mixture.GMM(n_components=2, covariance_type="full")
                # gmix.fit( prs[ prs > 0 ].ravel())
                gmix.fit(np.reshape(prs[prs > 0].ravel(), [prs[prs > 0].size, 1]))
                # threshold = ( gmix.means_ + .8 * gmix.covars_ ).squeeze()
                # threshold = np.array( gmix.means_ ).mean()
                threshold = np.array(gmix.means_).min()
                threshold = np.array(gmix.means_).max()
                if "tight" in stringency.lower():
                    threshold = prs[prs > 0].mean()
                logger.info(
                    f"Gaussian means = {gmix.means_} \nCovariances {gmix.covars_} \nThreshold = {threshold}"
                )

                """
                from sklearn import svm
                from scipy import stats
                outliers_fraction = 0.25
                gmix = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,kernel="rbf", gamma=0.1)
                from sklearn.covariance import EllipticEnvelope
                gmix = EllipticEnvelope(contamination=.1)
                new = np.reshape( prs, [1,prs.size] )
                gmix.fit(new)
                y_pred = gmix.decision_function(new).ravel()
                threshold = stats.scoreatpercentile(y_pred,100 * outliers_fraction)
                print 'SVM', threshold
                """

                # extract local maxima
                # neighborhood_size = 10
                # maximas = ( prs == scipy.ndimage.maximum_filter( prs, neighborhood_size ) ).nonzero()

                """
                for i in range( maximas[0].shape[0] ):
                    x = maximas[0][i]
                    y = maximas[1][i]
                    if prs[x,y] >= threshold:
                        # test = energy[x-1:x+2,y-1:y+2,0]
                        # test_ori = np.fabs( energy[x-1:x+2,y-1:y+2,5] - energy[x,y,5] )
                        # if test[ test >= .95 * threshold ].size > 1 and test_ori[ test_ori > 0 ].min() < 5:
                        dx = energy[x,y,3] / pixelsize / auto_binning
                        dy = energy[x,y,4] / pixelsize / auto_binning
                        if not area[ int( energy[x,y,1] + dx + tilesize / 2 ), int( energy[x,y,2] + dy + tilesize / 2 ) ]:
                            boxs.append( [ energy[x,y,2]  + dy + tilesize / 2, energy[x,y,1] + dx + tilesize / 2, tilesize, tilesize ] )
                            colors.append( energy[x,y,0] )
                            # colors.append( np.reshape( indicators[:,2], [ particles_x, particles_y ] )[x,y] )
                """

                # optimal density threshold for particle detection
                allboxs = []
                allpeaks = []
                for i in range(particles):
                    clean = not area[
                        int(energy[i, 1] + tilesize / 2),
                        int(energy[i, 2] + tilesize / 2),
                    ]
                    if clean:
                        allboxs.append(
                            [
                                energy[i, 2] + tilesize / 2,
                                energy[i, 1] + tilesize / 2,
                                tilesize,
                                tilesize,
                            ]
                        )
                        allpeaks.append(
                            -lowres[
                                int(energy[i, 1] + tilesize / 2),
                                int(energy[i, 2] + tilesize / 2),
                            ]
                        )
                umbral = (np.array(allpeaks).min() + np.array(allpeaks).mean()) / 2
                # umbral = np.array(allpeaks).min()
                if "tight" in stringency.lower():
                    peaks = np.array(allpeaks)
                    gmix = mixture.GMM(n_components=2, covariance_type="diag")
                    gmix.fit(peaks.ravel())
                    first_lobe_index = gmix.means_.argmin()
                    umbral = (
                        gmix.means_[first_lobe_index]
                        + 0.8 * gmix.covars_[first_lobe_index]
                    ).squeeze()
                    umbral = peaks.mean()

                logger.info(f"Using density threshold {umbral}")

                # eliminate outliers by robust covariance
                # gmix = mixture.GMM(n_components=1, covariance_type='full')
                # gmix.fit( allpeaks.ravel() )
                # umbral_outliers = np.array( gmix.means_ ).mean() + 2 * np.array( gmix.covars_ )
                # print 'Gaussian means = ', gmix.means_, '\nCovariances', gmix.covars_, '\nThreshold = ', threshold

                # find particles with scores and density above thresholds
                boxs = []
                boxs_score_colors = []
                boxs_density_colors = []
                boxs_indexes = []
                low_score = []
                low_score_colors = []
                low_score_indexes = []
                low_density = []
                low_density_colors = []
                low_density_indexes = []
                for i in range(particles):
                    strenght = -lowres[
                        int(energy[i, 1] + tilesize / 2),
                        int(energy[i, 2] + tilesize / 2),
                    ]
                    # if prs[i] >= threshold and strenght >= umbral:
                    if prs[i] >= threshold:
                        dx = energy[i, 3] / pixelsize / auto_binning
                        dy = energy[i, 4] / pixelsize / auto_binning
                        clean = not area[
                            int(energy[i, 1] - dy + tilesize / 2),
                            int(energy[i, 2] - dx + tilesize / 2),
                        ]
                        if clean:
                            boxs.append(
                                [
                                    energy[i, 2] - dx + tilesize / 2,
                                    energy[i, 1] - dy + tilesize / 2,
                                    tilesize,
                                    tilesize,
                                ]
                            )
                            boxs_score_colors.append(energy[i, 0])
                            boxs_density_colors.append(strenght)
                            boxs_indexes.append(i)
                    else:
                        clean = not area[
                            int(energy[i, 1] + tilesize / 2),
                            int(energy[i, 2] + tilesize / 2),
                        ]
                        if clean:
                            if prs[i] < threshold:
                                low_score.append(
                                    [
                                        energy[i, 2] + tilesize / 2,
                                        energy[i, 1] + tilesize / 2,
                                        tilesize,
                                        tilesize,
                                    ]
                                )
                                low_score_colors.append(energy[i, 0])
                                low_score_indexes.append(i)
                            if strenght < umbral:
                                low_density.append(
                                    [
                                        energy[i, 2] + tilesize / 2,
                                        energy[i, 1] + tilesize / 2,
                                        tilesize,
                                        tilesize,
                                    ]
                                )
                                low_density_colors.append(strenght)
                                low_density_indexes.append(i)

                msize = int(radius / pixelsize / auto_binning)
                msize *= msize

                # output diagnositcs
                import matplotlib.pyplot as plt
                from matplotlib import cm

                colormap = cm.rainbow
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                mmean = micrograph.mean()
                mstd = 1.5 * micrograph.std()
                ax[0].imshow(
                    micrograph, cmap=cm.Greys_r, vmin=mmean - mstd, vmax=mmean + mstd
                )
                if area.max() > area.min():
                    ax[0].contour(area, 1, colors="#00ff00", linewidths=4)
                ax[0].axis("off")
                if len(boxs) > 0:
                    # ax[0].scatter( np.array(boxs)[:,0], np.array(boxs)[:,1], s=msize, facecolor='none', c=np.array(boxs_score_colors), cmap=cm.jet, vmin=prs.min() , vmax=prs.max() , marker="o", lw=1.5 )
                    for i in range(len(boxs_indexes)):
                        pcolor = colormap(
                            (boxs_score_colors[i] - prs.min()) / (prs.max() - prs.min())
                        )
                        ax[0].annotate(
                            boxs_indexes[i] + 1,
                            boxs[i][:2],
                            xytext=(-10, 10),
                            textcoords="offset points",
                            ha="right",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.25", fc=pcolor, alpha=0.5),
                            fontsize=8,
                            fontweight="bold",
                        )
                    # arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0')
                    # ax[0].scatter( np.array(low_score)[:,0], np.array(low_score)[:,1], s=100, facecolor='none', c=np.array(low_score_colors), cmap=cm.jet, vmin=prs.min() , vmax=prs.max() , marker="x", lw=.5 )
                    # for i in range(len(low_score_indexes)):
                    #    pcolor = colormap( ( low_score_colors[i] - prs.min() ) / ( prs.max() - prs.min() ) )
                    #    ax[0].annotate( low_score_indexes[i] + 1, low_score[i][:2], xytext = (-10, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.1', fc = pcolor, alpha = 0.5), fontsize=8 )
                # mmean = lowres.mean()
                # mstd = 1.5 * lowres.std()
                cax = ax[1].imshow(
                    micrograph, cmap=cm.Greys_r, vmin=mmean - mstd, vmax=mmean + mstd
                )
                if area.max() > area.min():
                    ax[1].contour(area, 1, colors="#00ff00", linewidths=4)
                if len(boxs) > 0:
                    for i in range(len(low_score_indexes)):
                        pcolor = colormap(
                            (low_score_colors[i] - prs.min()) / (prs.max() - prs.min())
                        )
                        ax[1].annotate(
                            low_score_indexes[i] + 1,
                            low_score[i][:2],
                            xytext=(-10, 10),
                            textcoords="offset points",
                            ha="right",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.1", fc=pcolor, alpha=0.5),
                            fontsize=8,
                        )
                    ## ax[1].scatter( np.array(boxs)[:,0], np.array(boxs)[:,1], s=msize, facecolor='none', c=np.array(boxs_density_colors), cmap=cm.jet, vmin=np.array(allpeaks).min() , vmax=np.array(allpeaks).max() , marker="o", lw=1.5 )
                    # for i in range(len(boxs_indexes)):
                    #    pcolor = colormap( ( boxs_density_colors[i] - np.array(allpeaks).min() ) / ( np.array(allpeaks).max() - np.array(allpeaks).min() ) )
                    #    ax[1].annotate( boxs_indexes[i] + 1, boxs[i][:2], xytext = (-10, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.1', fc = pcolor, alpha = 0.5), fontsize=8, fontweight='bold' )
                    # if len(low_density) > 0:
                    #    # ax[1].scatter( np.array(low_density)[:,0], np.array(low_density)[:,1], s=100, facecolor='none', c=np.array(low_density_colors), cmap=cm.jet, vmin=np.array(allpeaks).min() , vmax=np.array(allpeaks).max() , marker="x", lw=.5 )
                    #    for i in range(len(low_density_indexes)):
                    #        pcolor = colormap( ( low_density_colors[i] - np.array(allpeaks).min() ) / ( np.array(allpeaks).max() - np.array(allpeaks).min() ) )
                    #        ax[1].annotate( low_density_indexes[i] + 1, low_density[i][:2], xytext = (-10, 10), textcoords = 'offset points', ha = 'right', va = 'bottom', bbox = dict(boxstyle = 'round,pad=0.1', fc = pcolor, alpha = 0.5), fontsize=8 )
                # caxd = fig.add_axes([0.95, 0.16, 0.01, 0.675])
                # fig.colorbar(cax,caxd)
                plt.axis("off")
                plt.savefig("{0}_images.png".format(name), bbox_inches="tight")
                plt.close()

                fig, ax = plt.subplots(1, 2, figsize=(13, 2))
                n, bins, patches = ax[1].hist(
                    np.array(allpeaks).ravel(),
                    25,
                    density=1,
                    facecolor="green",
                    alpha=0.75,
                )
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                col = bin_centers - min(bin_centers)
                col /= max(col)
                for c, p in zip(col, patches):
                    plt.setp(p, "facecolor", cm.rainbow(c))
                ax[1].plot(
                    (umbral, umbral),
                    (0, 1.05 * n.max()),
                    "k-",
                    label="Th=%.2f" % umbral,
                    lw=4,
                )
                n, bins, patches = ax[0].hist(
                    energy[:, 0].ravel(), 25, density=1, facecolor="green", alpha=0.75
                )
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                col = bin_centers - min(bin_centers)
                col /= max(col)
                for c, p in zip(col, patches):
                    plt.setp(p, "facecolor", cm.rainbow(c))
                ax[0].plot(
                    (threshold, threshold),
                    (0, 1.05 * n.max()),
                    "k-",
                    label="Th=%.2f" % threshold,
                    lw=4,
                )
                ax[0].legend(prop={"size": 10})
                plt.savefig("{0}_histog.png".format(name))
                plt.close()

                command = "montage -geometry +0+0 -tile x2 {0}_images.png {0}_histog.png {0}_energy.png".format(
                    name
                )
                local_run.run_shell_command(command, verbose=mparameters["slurm_verbose"])

            else:

                boxes = boxs[:]
                boxs = []
                for b in boxes:
                    boxs.append(
                        [b[0] + tilesize / 2, b[1] + tilesize / 2, tilesize, tilesize]
                    )

            logger.info(f"{len(boxs)} particles found")

            # save positions to box file
            boxes = np.array(boxs)
            if len(boxs) > 0:
                boxes[:, 2:] = 0

                np.savetxt(
                    "{}.box".format(name), boxes * data_bin * auto_binning, fmt="%i\t"
                )

        elif mparameters["detect_method"] == "import" and os.path.exists(project_params.resolve_path(mparameters["detect_files"])):
            box_file = os.path.join(project_params.resolve_path(mparameters["detect_files"]), f"{name}.box")
            if os.path.exists(box_file):
                boxes = np.loadtxt(box_file, ndmin=2)
                add_columns = np.zeros([boxes.shape[0], 2])
                boxes = np.hstack((boxes, add_columns))
                np.savetxt(
                        "{}.box".format(name), boxes, fmt="%f\t"
                        )
            else:
                logger.warning("Couldn't find any box file for this image")
        else:
            logger.error(
                f"Particle picking strategy not recognized: {mparameters['detect_method']}"
            )

    if not os.path.isfile("{}.box".format(name)):
        logger.info("Did not find any particles.")
    elif os.stat("{}.box".format(name)).st_size == 0:
        open("{}.boxx".format(name), "a").close()
        logger.info("Did not find any particles.")
    else:
        # read unbinned .box file (only first 4 columns)
        boxes = np.loadtxt("{}.box".format(name), ndmin=2)[:, :4]

        boxes[:, :-2] /= data_bin

        if boxes.size == 0:
            logger.info("Did not find any particles.")
            shutil.copy2("{0}.webp".format(name), "{0}_boxed.webp".format(name))
        else:
            # extended box file: add two columns to standard .box file, first to indicate particles inside micrograph, second for particles that were classified
            if os.path.isfile("{}.boxx".format(name)):
                boxxs = np.loadtxt("{}.boxx".format(name), ndmin=2)
            else:
                boxxs = np.zeros([boxes.shape[0], boxes.shape[1] + 2])
                boxxs[:, -2] = 1
                boxxs[:, :-2] = boxes
                boxxs[:, :2] *= data_bin

            boxxs[:, :-2] /= data_bin

            # decorate png's with picked particles
            options = ""
            newoptions = ""
            pick_bin = 1.0
            if image.shape[0] != image.shape[1]:
                if image.shape[0] > 7000:
                    display_bin = 14.0
                else:
                    display_bin = 7.0
            else:
                display_bin = 8.0
            display_bin = image.shape[0] / 512.0
            display_bin = 8
            # TODO: revisit this, why do we need to reset data_bin when > 1
            data_bin = 1
            for i in range(boxes.shape[0]):
                # centerX = ( boxes[i,0] + boxes[i,2] / 2 ) * pick_bin / display_bin * data_bin
                # centerY = ( boxes[i,1] + boxes[i,3] / 2 ) * pick_bin / display_bin * data_bin
                centerX = (
                    (boxxs[i, 0] + boxxs[i, 2] / 2) * pick_bin / display_bin * data_bin
                )
                centerY = (
                    (boxxs[i, 1] + boxxs[i, 3] / 2) * pick_bin / display_bin * data_bin
                )
                x1 = centerX + radius * data_bin / pixelsize / display_bin
                x2 = centerX + radius * data_bin / pixelsize / display_bin / 2
                x2 = centerX + radius * data_bin / pixelsize / display_bin / 10
                # if boxxs[i,4] + boxxs[i,5] > 1:
                if boxxs[i, 4] == 1 and boxxs[i, 5] >= int(classification_pass):
                    options = options + (
                        " -draw 'circle %.1f,%.1f %.1f,%.1f'"
                        % (centerX, centerY, x1, centerY)
                    )
                elif boxxs[i, 4] + boxxs[i, 5] > 0:
                    newoptions = newoptions + (
                        " -draw 'circle %.1f,%.1f %.1f,%.1f'"
                        % (centerX, centerY, x1, centerY)
                    )

            # only do this if we have the webp file
            if not Web.exists and os.path.exists(name+".webp"):
                command = "{0}/convert {1}.webp -flip -fill none -stroke green1 {2} {1}_tmp.webp".format(
                    os.environ["IMAGICDIR"], name, options
                )
                local_run.run_shell_command(command, verbose=False)

                command = "{0}/convert {1}_tmp.webp -flip -fill none -stroke red {2} {1}_boxed.webp".format(
                    os.environ["IMAGICDIR"], name, newoptions
                )
                local_run.run_shell_command(command, verbose=False)

            # pythonesque way
            if False:
                import matplotlib.pyplot as plt
                from matplotlib import cm

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                # use bin12 micrograph in frealign folder if available
                binned_micrograph = "frealign/%s.mrc" % name
                auto_binning = 12
                if not os.path.exists(binned_micrograph):
                    # micrograph binning for performing search
                    com = "{0}/bin/newstack {1}.avg {1}_binned.mrc -bin {2}".format(
                        get_imod_path(), name, auto_binning
                    )
                    local_run.run_shell_command(com)
                    binned_micrograph = f"{name}_binned.mrc"
                micrograph = mrc.read(binned_micrograph)
                mparameters = project_params.load_pyp_parameters()

                """
                ctffile = name + ".ctf"
                ctf = np.loadtxt(ctffile)
                if mparameters["scope_pixel"] == 0:
                    mparameters["scope_pixel"] = ctf[9]
                if mparameters["scope_voltage"] == 0:
                    mparameters["scope_voltage"] = ctf[10]
                if mparameters["scope_mag"] == 0:
                    mparameters["scope_mag"] = ctf[11]
                """

                msize = int(
                    float(mparameters["detect_rad"])
                    / float(mparameters["scope_pixel"])
                    / auto_binning
                )
                tilesize = int(3 * msize)
                msize *= msize
                msize *= 3
                from scipy import ndimage

                mmean = micrograph.mean()
                mstd = 2.0 * micrograph.std()
                ax.imshow(micrograph, cmap=cm.Greys_r, vmin=mmean - mstd, vmax=mmean + mstd)
                ax.axis("off")
                for i in range(boxes.shape[0]):
                    if boxxs[i, 4] == 1 and boxxs[i, 5] >= int(classification_pass):
                        ax.scatter(
                            (np.array(boxxs)[i, 0] + boxxs[i, 2] / 2) / auto_binning,
                            (np.array(boxxs)[i, 1] + boxxs[i, 3] / 2) / auto_binning,
                            s=msize / 3,
                            facecolor="none",
                            edgecolor="lime",
                            marker="o",
                            lw=2,
                        )
                    else:
                        ax.scatter(
                            (np.array(boxxs)[i, 0] + boxxs[i, 2] / 2) / auto_binning,
                            (np.array(boxxs)[i, 1] + boxxs[i, 3] / 2) / auto_binning,
                            s=msize / 3,
                            facecolor="none",
                            edgecolor="red",
                            marker="o",
                            lw=0.75,
                        )
                plt.savefig("{0}_boxs.pdf".format(name), bbox_inches="tight")
                plt.close()
                # done

            boxsize *= binning
            boundsize *= binning

            # extract particles
            if False and boxsize > 0:

                # use boxx coordinates if available
                if os.path.isfile("{}.boxx".format(name)):
                    boxxs[:, 0:2] = boxxs[:, 0:2] + boxxs[0, 2] / 2 - boxsize / 2
                    boxes = boxxs[:, :-2]
                else:
                    boxes[:, 0:2] = boxes[:, 0:2] + boxes[0, 2] / 2 - boxsize / 2
                    boxxs[:, :-2] = boxes
                boxxs[:, 2:4] = boxsize

                if boundsize > boxsize:
                    difference = (boundsize - boxsize) / 2
                else:
                    difference = 0

                # check if inside micrograph bounds
                for i in range(boxes.shape[0]):
                    # if boxes[i,0] >= 0 and boxes[i,0]+boxsize < image.shape[-1] and boxes[i,1] >= 0 and boxes[i,1]+boxsize < image.shape[-2]:
                    inside = (
                        boxes[i, 0] >= difference
                        and boxes[i, 0] + boxsize + difference < image.shape[-1]
                        and boxes[i, 1] >= difference
                        and boxes[i, 1] + boxsize + difference < image.shape[-2]
                    )
                    if inside:
                        boxxs[i, 4] = 1
                    else:
                        boxxs[i, 4] = 0
                boxs = boxxs[
                    np.logical_and(
                        boxxs[:, 4] == 1, boxxs[:, 5] >= int(classification_pass)
                    )
                ]

                if len(boxs) > 0:

                    from pyp import extract

                    particles = extract.extract_particles_old(
                        image,
                        boxs.tolist(),
                        radius * data_bin,
                        boxsize,
                        binning,
                        pixelsize * data_bin,
                    )
                    mrc.write(particles, "{}_stack.mrc".format(name))
                    mrc.write(-particles, "{}_particles.mrcs".format(name))

                    mparameters = project_params.load_pyp_parameters()
                    if "amp" in mparameters["extract_fmt"].lower():
                        amplitudes = abs(np.fft.fftshift(np.fft.fft2(particles)))
                        # need to normalize?
                        mrc.write(amplitudes, "{}_amps_stack.mrc".format(name))

            boxxs[:, :-2] *= data_bin
            np.savetxt("{}.boxx".format(name), boxxs, fmt="%2.2f", delimiter="\t")
