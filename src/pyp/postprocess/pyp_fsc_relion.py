#!/usr/bin/env -S python -B

import matplotlib

matplotlib.use("Agg")
import argparse
import datetime
import math
import multiprocessing
import os
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy
from matplotlib import cm

from pyp.inout.image import mrc
from pyp.system import utils
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def fsc_multiprocessing(
    apix,
    half1,
    half2,
    average,
    threshold,
    shell,
    gauss,
    phases,
    results,
    mask,
    keep=False,
):

    use_external_mask = len(mask) > 0

    if len(mask) > 0:
        name = mask.replace(".mrc", "")
    else:
        name = "%s_th_%0.8f_sh_%0.4f_gs_%0.4f" % (
            average.replace(".mrc", ""),
            threshold,
            shell,
            gauss,
        )
        mask = "mask_" + name + ".mrc"

    if math.fabs(threshold) > 0 or use_external_mask:

        # compute mask if it doesn't exist
        if not use_external_mask:
            # compute mask
            com = "{5}; e2proc3d.py {0} {1} --process=mask.auto3d:radius=100:threshold={2}:nshells={3}:nshellsgauss={4}:nmaxseed=0:return_mask=1".format(
                average, mask, threshold, shell, gauss, utils.eman_load_command()
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)

            # shutil.copy2( average, mask )

            # used lp=83 for p97 figures
            com = "{1}; e2proc3d.py {0} {0} --process=filter.lowpass.gauss:cutoff_pixels=36".format(
                mask, utils.eman_load_command()
            )
            com = "{1}; e2proc3d.py {0} {0} --process=filter.lowpass.gauss:cutoff_pixels=30.9".format(
                mask, utils.eman_load_command()
            )  # emd-2984 vs. pdb-5a1a
            com = "{1}; e2proc3d.py {0} {0} --process=filter.lowpass.gauss:cutoff_pixels=29".format(
                mask, utils.eman_load_command()
            )  # emd-2984 vs. half maps
            com = "{1}; e2proc3d.py {0} {0} --process=filter.lowpass.gauss:cutoff_pixels=20".format(
                mask, utils.eman_load_command()
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)

            current_mask = mrc.read(mask)
            mrc.write(
                (current_mask - current_mask.min())
                / (current_mask.max() - current_mask.min()),
                mask,
            )

        msk = mrc.read(mask)
        A = numpy.concatenate(
            (
                numpy.concatenate(
                    (msk[:, :, msk.shape[2] / 2], msk[:, msk.shape[1] / 2, :]), axis=1
                ),
                msk[msk.shape[0] / 2, :, :],
            ),
            axis=1,
        )
        B = numpy.concatenate(
            (
                numpy.concatenate(
                    (numpy.sum(msk, axis=0), numpy.sum(msk, axis=1)), axis=1
                ),
                numpy.sum(msk, axis=2),
            ),
            axis=1,
        )
        import matplotlib.pyplot as pltmask

        pltmask.clf()
        pltmask.imshow(A, interpolation="nearest", cmap=cm.Greys_r)
        pltmask.axis("off")
        pltmask.title("Th = %0.8f" % threshold)
        pltmask.savefig(mask.replace(".mrc", ".png"), bbox_inches="tight")

        # mask first half
        com = "{4}; e2proc3d.py {0} {1}_{2}.mrc --multfile={3}".format(
            half1, os.path.basename(half1)[:-4], name, mask, utils.eman_load_command()
        )
        if args.verbose:
            logger.info(com)
        subprocess.getoutput(com)

        # mask second half
        com = "{4}; e2proc3d.py {0} {1}_{2}.mrc --multfile={3}".format(
            half2, os.path.basename(half2)[:-4], name, mask, utils.eman_load_command()
        )
        if args.verbose:
            logger.info(com)
        subprocess.getoutput(com)

        # randomize phases
        nome = os.path.basename(half2)[:-4] + "_" + name
        if len(phases) > 0:
            com = "{3}; e2proc3d.py {0}.mrc {0}_phases.mrc --process=filter.lowpass.gauss:cutoff_freq={1}:sigma={2}".format(
                nome,
                phases.split(",")[0],
                phases.split(",")[-1],
                utils.eman_load_command(),
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)
        else:
            # print 'Bypassing phase filtering'
            shutil.copy2("%s.mrc" % nome, "%s_phases.mrc" % nome)

        # compure FSC
        com = "{4}; e2proc3d.py {0}_{2}.mrc fsc_{2}.txt --apix={3} --calcfsc={1}_{2}_phases.mrc".format(
            os.path.basename(half1)[:-4],
            os.path.basename(half2)[:-4],
            name,
            apix,
            utils.eman_load_command(),
        )
        if args.verbose:
            logger.info(com)
        subprocess.getoutput(com)

        # clanup
        if not keep:
            os.remove("{0}_{1}.mrc".format(os.path.basename(half1)[:-4], name))
            os.remove("{0}_{1}.mrc".format(os.path.basename(half2)[:-4], name))
            os.remove("{0}_{1}_phases.mrc".format(os.path.basename(half2)[:-4], name))
            if not mask:
                os.remove("{0}".format(mask))
    else:
        # compute unmaskewd FSC
        com = "{0}; e2proc3d.py {1} fsc_{3}.txt --apix={4} --calcfsc={2}".format(
            utils.eman_load_command(), half1, half2, name, apix
        )
        if args.verbose:
            logger.info(com)
        subprocess.getoutput(com)

    # parse result
    fsc = numpy.loadtxt("fsc_{0}.txt".format(name))
    if not keep:
        os.remove("fsc_{0}.txt".format(name))

    results.put([threshold, shell, gauss, fsc])

    return


def fsc_cutoff(x, cutoff=0.5):

    # detect first dip below cutoff
    for i in range(x.shape[0]):
        if x[i, 1] > float(cutoff):
            ia = i
        else:
            break

    xa = x[ia, 0]
    ya = x[ia, 1]
    xb = x[ia + 1, 0]
    yb = x[ia + 1, 1]

    a = (yb - ya) / (xb - xa)
    b = ya - (a * xa)
    value = a / (float(cutoff) - b)

    return value


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FSC plots")
    parser.add_argument(
        "-half1",
        "--half1",
        "-h1",
        "--h1",
        help="First half-map",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-half2",
        "--half2",
        "-h2",
        "--h2",
        help="Second half-map",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-thresholds",
        "--thresholds",
        "-t",
        "--t",
        help="Threshold used for shape masking (min:max:inc).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-shells",
        "--shells",
        "-s",
        "--s",
        help="EMAN2s nshell (1:2:1)",
        type=str,
        default="1:2:1",
    )
    parser.add_argument(
        "-gausses",
        "--gausses",
        "-g",
        "--g",
        help="EMAN2s nshellgauss (2:3:1)",
        type=str,
        default="1:2:1",
    )
    parser.add_argument(
        "-phases",
        "--phases",
        help="Sigma for main filter (.3,.05)",
        type=str,
        default=".3,.05",
    )
    parser.add_argument(
        "-apix", "--apix", "-a", "--a", help="Pixel size", type=float, required=True
    )
    parser.add_argument(
        "-keep",
        "--keep",
        help="Do not delete mask files used for FSC computations.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-align",
        "--align",
        help="Align second volume to average.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-lowpass",
        "--lowpass",
        help="Lowpass filter applied to map before thresholing operation.",
        type=float,
        default=20,
    )
    parser.add_argument(
        "-mask",
        "--mask",
        help="Provide external file to create mask.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-crop",
        "--crop",
        help="Crop half volumes to this size before FSC calculation.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-verbose",
        "--verbose",
        help="Output processing commands.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-relion",
        "--relion",
        help="Use relion_postprocessing.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-pdbres",
        "--pdbres",
        help="Resolution to filter pdb coordinates.",
        type=float,
        default=2,
    )

    args = parser.parse_args()

    with open(".pyplog", "a") as f:
        f.write(str(datetime.datetime.now()) + ": " + " ".join(sys.argv) + "\n\n")

    # compute average map

    dataset = (
        os.path.basename(args.half1)[:-4] + "_" + os.path.basename(args.half2)[:-4]
    )

    if args.mask:
        average = args.mask
    else:
        average = dataset + ".mrc"

    if args.half1[-3:] == "pdb" or args.half2[-3:] == "pdb":

        if args.half1[-3:] == "pdb":
            pdbfile = args.half1[:-4]
            reference = mrc.readHeaderFromFile(args.half2)
            second_half = args.half2
        else:
            pdbfile = args.half2[:-4]
            reference = mrc.readHeaderFromFile(args.half1)
            second_half = args.half1

        x = reference["nx"]
        y = reference["ny"]
        z = reference["nz"]

        first_half = pdbfile + ".mrc"
        if True or not os.path.exists(first_half):
            com = "{0}; e2pdb2mrc.py -A {1} -R {2} -B {3},{4},{5} {6}.pdb {6}.mrc".format(
                utils.eman_load_command(), args.apix, args.pdbres, x, y, z, pdbfile
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)
        else:
            logger.info("Using existing map for pdb coordinates.")

        if not os.path.exists(average):

            logger.info(
                "Generating low resolution map for selecting masking threshold: %f",
                average,
            )

            com = "{4}; e2proc3d.py {0} {1} --apix={2} --process=filter.lowpass.gauss:cutoff_pixels={3}".format(
                second_half, average, args.apix, args.lowpass, utils.eman_load_command()
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)
            mask = mrc.read(average)
            logger.info(
                "Density range of filtered volume is = [%f,%f]", mask.min(), mask.max()
            )
            sys.exit()
        else:
            logger.info("Using existing map to generate mask: %f", average)

    else:

        if len(args.crop) > 0:
            com = "{0}; e2proc3d.py {1} {2} --clip={3}".format(
                utils.eman_load_command(),
                args.half1,
                args.half1.replace(".mrc", "_crop.mrc"),
                args.crop,
            )
            logger.info(com)
            subprocess.getoutput(com)
            args.half1 = args.half1.replace(".mrc", "_crop.mrc")
            com = "{0}; e2proc3d.py {1} {2} --clip={3}".format(
                utils.eman_load_command(),
                args.half2,
                args.half2.replace(".mrc", "_crop.mrc"),
                args.crop,
            )
            logger.info(com)
            subprocess.getoutput(com)
            args.half2 = args.half2.replace(".mrc", "_crop.mrc")

        first_half = args.half1
        second_half = args.half2

        half1 = mrc.read(args.half1)
        half2 = mrc.read(args.half2)

        if not os.path.exists(average):

            mrc.write((half1 + half2) / 2.0, average)

            # lp=15 (used for p97 plots)
            com = "{3}; e2proc3d.py {0} {0} --apix={1} --process=filter.lowpass.gauss:cutoff_pixels={2}".format(
                average, args.apix, args.lowpass, utils.eman_load_command()
            )
            if args.verbose:
                logger.info(com)
            logger.info(subprocess.getoutput(com))

            logger.info(
                "Generated lowpass-filtered average of two half maps %f", average
            )

            densities = mrc.read(average)
            logger.info(
                "Density limits = [\t{0},\t{1}]".format(
                    densities.min(), densities.max()
                )
            )
            sys.exit()
        else:
            logger.info("Using existing average of two half maps %f", average)

    if args.align:

        # make sure files have same header
        h = mrc.readHeaderFromFile(average)
        headerbytes = mrc.makeHeaderData(h)
        data = mrc.read(first_half)
        f = open(first_half, "wb")
        f.write(headerbytes)
        mrc.appendArray(data, f)
        f.close()

        shutil.copy2(
            "{0}/python3/chimerafitnogui.py".format(os.environ["PYP_DIR"]), "."
        )
        command = 'module load chimera; chimera --nogui --script ""chimerafitnogui.py {0} {1}""'.format(
            first_half, average
        )
        logger.info(command)
        logger.info(subprocess.getoutput(command))
        logger.info(
            "Moving %s %s", first_half.replace(".mrc", "_rotated.mrc"), first_half
        )
        shutil.move(first_half.replace(".mrc", "_rotated.mrc"), first_half)

    if args.relion:

        # create sym links to files

        auxs = ["relion_half1_class001_unfil.mrc", "relion_half2_class001_unfil.mrc"]
        for aux in auxs:
            if os.path.exists(aux):
                os.remove(aux)
        os.symlink(first_half, "relion_half1_class001_unfil.mrc")
        os.symlink(second_half, "relion_half2_class001_unfil.mrc")

        name = first_half.replace(".mrc", "")

        if args.mask:
            mask_option = "--mask " + args.mask
        else:
            mask_option = "--auto_mask"
            mask_option = "--auto_mask --inimask_threshold {0} --extend_inimask {1} --width_mask_edge {2}".format(
                args.thresholds.split(":")[0],
                args.shells.split(":")[0],
                args.gausses.split(":")[0],
            )
        # mask_option = '--auto_mask --extend_inimask 0 --width_mask_edge 3'.format( args.thresholds.split(':')[0] )
        # mask_option += ' --randomize_at_fsc 0.94'
        com = "/dscrhome/ab690/code/relion/build/bin/relion_postprocess --i relion --angpix {0} {1} --auto_bfac".format(
            args.apix, mask_option
        )
        com = "/hpc/home/ab690/code/relion/build/bin/relion_postprocess --i relion --angpix {0} {1} --adhoc_bfac -175.4287 --low_pass 4.86956".format(
            args.apix, mask_option
        )
        com = "/hpc/home/ab690/code/relion-3.0_beta/build/bin/relion_postprocess --i {2} --i2 {3} --angpix {0} {1} --adhoc_bfac -175.4287 --low_pass 4.86956".format(
            args.apix, mask_option, first_half, second_half
        )
        com = "module load Relion/3.0-GPU; relion_postprocess --i {2} --i2 {3} --angpix {0} {1} --adhoc_bfac -175.4287 --low_pass 4.86956".format(
            args.apix, mask_option, first_half, second_half
        )

        logger.info(com)
        results = subprocess.getoutput(com)

        logger.info(results)

        star_file = "postprocess.star"

        # plot results
        post = open(star_file).read().split("data_")
        data = numpy.array(
            [
                line.split()
                for line in post[2]
                .split("_rlnFourierShellCorrelationUnmaskedMaps #4")[-1]
                .split("\n")
                if len(line) > 3
            ]
        )
        if len(data.shape) == 1:
            data = numpy.array(
                [
                    line.split()
                    for line in post[2]
                    .split(
                        "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps #7"
                    )[-1]
                    .split("\n")
                    if len(line) > 3
                ]
            )

        FinalResolution = [
            float(line.split()[1].replace("#", ""))
            for line in open(star_file)
            if "_rlnFinalResolution" in line
        ][0]
        BfactorUsedForSharpening = [
            float(line.split()[1].replace("#", ""))
            for line in open(star_file)
            if "_rlnBfactorUsedForSharpening" in line
        ][0]

        plt.plot(data[:, 1].astype("f"), data[:, 3].astype("f"), label="Corrected")
        if len(data.shape) > 1:
            plt.plot(
                data[:, 1].astype("f"), data[:, 4].astype("f"), label="Unmasked Maps"
            )
            plt.plot(
                data[:, 1].astype("f"), data[:, 5].astype("f"), label="Masked Maps"
            )
            plt.plot(
                data[:, 1].astype("f"),
                data[:, 6].astype("f"),
                label="Corrected PR Masked Maps",
            )
        # plt.ylim( [-.05,1.05] )
        plt.plot(data[:, 1].astype("f"), 0.143 * numpy.ones(data[:, 1].shape), "k:")
        plt.plot(data[:, 1].astype("f"), 0.5 * numpy.ones(data[:, 1].shape), "k:")
        plt.ylim((-0.1, 1.05))
        plt.xlim((data[:, 1].astype("f")[0], data[:, 1].astype("f")[-1]))
        plt.legend(prop={"size": 10})
        plt.title(
            "%s\nFinal Resolution = %.2f, Bfactor = %.2f"
            % (
                first_half.replace("_crop.mrc", ""),
                FinalResolution,
                BfactorUsedForSharpening,
            )
        )
        plt.savefig(first_half.replace(".mrc", "_relion.png"))

        # clean up
        shutil.move("postprocess.star", first_half.replace(".mrc", ".star"))
        shutil.move(
            "postprocess_masked.mrc", first_half.replace(".mrc", "_postprocess.mrc")
        )
        list = [
            "relion_half1_class001_unfil.mrc",
            "relion_half2_class001_unfil.mrc",
            "postprocess.mrc",
            "postprocess_automask.mrc",
            "postprocess_fsc.xml",
        ]
        if not args.keep:
            [os.remove(f) for f in list if os.path.exists(f)]

        sys.exit()

    # collate parameter variations
    ths = args.thresholds.split(":")
    thresholds = numpy.arange(float(ths[0]), float(ths[1]), float(ths[2]))
    # thresholds = numpy.array([0, .013, .016, .019])
    shs = args.shells.split(":")
    shells = list(range(int(shs[0]), int(shs[1]), int(shs[2])))
    gss = args.gausses.split(":")
    gausses = list(range(int(gss[0]), int(gss[1]), int(gss[2])))

    densities = mrc.read(average)
    min_threshold = densities.min()
    max_threshold = densities.max()

    # split FSC calculations
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    results = manager.Queue()

    for threshold in thresholds:
        if threshold < min_threshold or threshold > max_threshold:
            logger.info(
                "Threshold %0.8f falls outside density range [%0.8f,%0.8f]",
                threshold,
                min_threshold,
                max_threshold,
            )
        else:
            for shell in shells:
                for gauss in gausses:
                    logger.info(
                        "Submitting th = %0.8f, sh = %0.4f, gs = %0.4f",
                        threshold,
                        shell,
                        gauss,
                    )
                    # fsc_multiprocessing( args.apix, first_half, second_half, average, threshold, shell, gauss, args.phases, results, args.keep )
                    # sys.exit()
                    pool.apply_async(
                        fsc_multiprocessing,
                        args=(
                            args.apix,
                            first_half,
                            second_half,
                            average,
                            threshold,
                            shell,
                            gauss,
                            args.phases,
                            results,
                            args.mask,
                            args.keep,
                        ),
                    )
                    # fsc_multiprocessing( args.apix, first_half, second_half, average, threshold, shell, gauss, args.phases, results, args.keep )
    pool.close()

    # Wait for all processes to complete
    pool.join()

    # plot all curves
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(10,8))
    # fig, ax = plt.subplots()

    if results.empty() == True:
        logger.error("Could not get FSC curves.")
    else:

        # Collate periodogram averages
        sorted_results = []
        while results.empty() == False:
            sorted_results.append(results.get())
        sorted_results.sort()

        if args.half1[-3:] == "pdb" or args.half2[-3:] == "pdb":
            logger.info("Using .5 cutoff")
            cutoff_value = 0.5
        else:
            logger.info("Using .143 cutoff")
            cutoff_value = 0.143

        for result in sorted_results:
            fsc = result[-1]
            # ax.plot( fsc[:,0], fsc[:,1], label='Th=%02.4f, Sh=%02.4f, Gs=%02.4f' % ( result[0], result[1], result[2] ) )
            cutoff = fsc_cutoff(fsc, cutoff_value)
            logger.info("Resolution = %f", cutoff)
            # ax.plot( fsc[:,0], fsc[:,1], label='Mask %02d (.5 = %.2f %s, .143 = %.2f %s)' % ( count, cutoffs[0], u'\u00c5', cutoffs[1], u'\u00c5' ) )
            plt.plot(
                fsc[:, 0],
                fsc[:, 1],
                label="Th %.3f (%.3f = %.2f %s)"
                % (result[0], cutoff_value, cutoff, "\u00c5"),
            )

    if not args.relion:
        plt.plot(fsc[:, 0], 0.143 * numpy.ones(fsc[:, 1].shape), "k:")
        plt.plot(fsc[:, 0], 0.5 * numpy.ones(fsc[:, 1].shape), "k:")

        legend = plt.legend(loc="upper right", shadow=True)
        plt.ylim((-0.1, 1.05))
        plt.xlim((fsc[0, 0], 1 * fsc[-1, 0]))
        plt.title(
            "FSC for %s\nVs.\n%s"
            % (os.path.basename(args.half1)[:-4], os.path.basename(args.half2)[:-4])
        )
        plt.xlabel("Frequency (1/" + "\u00c5" + ")")
        plt.ylabel("FSC")
        # ltext  = legend.get_texts()
        # plt.setp(ltext, fontsize='xx-small')
    plt.legend(prop={"size": 10})
    plt.savefig("%s_fsc.png" % dataset)

    numpy.savetxt("%s_fsc.txt" % dataset, fsc, fmt="%10.5f")

    # cleanup
    if not args.keep:
        try:
            # os.remove( average )
            if len(args.crop) > 0:
                os.remove(args.half1)
                os.remove(args.half2)
        except:
            pass
