#! /usr/bin/env python

import matplotlib

matplotlib.use("Agg")
import argparse
import datetime
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
    apix, half1, half2, average, threshold, shell, gauss, phases, results, keep=False
):

    name = "%s_th_%0.8f_sh_%0.4f_gs_%0.4f" % (
        average.replace(".mrc", ""),
        threshold,
        shell,
        gauss,
    )
    mask = "mask_" + name + ".mrc"

    if threshold != 0:
        # compute mask
        com = "{5}; e2proc3d.py {0} {1} --process=mask.auto3d:radius=100:threshold={2}:nshells={3}:nshellsgauss={4}:nmaxseed=0:return_mask=1".format(
            average, mask, threshold, shell, gauss, utils.eman_load_command()
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

    else:
        mask = average

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
            numpy.concatenate((numpy.sum(msk, axis=0), numpy.sum(msk, axis=1)), axis=1),
            numpy.sum(msk, axis=2),
        ),
        axis=1,
    )
    plt.imshow(A, interpolation="nearest", cmap=cm.Greys_r)
    plt.axis("off")
    plt.title("Th = %0.8f" % threshold)
    plt.savefig(mask.replace(".mrc", ".png"), bbox_inches="tight")

    # mask first half
    if True or threshold != 0:
        # if threshold != 0:
        com = "{4}; e2proc3d.py {0} {1}_{2}.mrc --multfile={3}".format(
            half1, os.path.basename(half1)[:-4], name, mask, utils.eman_load_command(),
        )
        if args.verbose:
            logger.info(com)
        subprocess.getoutput(com)

        # mask second half
        com = "{4}; e2proc3d.py {0} {1}_{2}.mrc --multfile={3}".format(
            half2, os.path.basename(half2)[:-4], name, mask, utils.eman_load_command(),
        )
        if args.verbose:
            logger.info(com)
        subprocess.getoutput(com)

    else:
        shutil.copy2(half1, os.path.basename(half1)[:-4] + "_" + name + ".mrc")
        shutil.copy2(half2, os.path.basename(half2)[:-4] + "_" + name + ".mrc")

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

    # sys.exit()

    # clanup
    if not keep:
        os.remove("{0}_{1}.mrc".format(os.path.basename(half1)[:-4], name))
        os.remove("{0}_{1}.mrc".format(os.path.basename(half2)[:-4], name))
        os.remove("{0}_{1}_phases.mrc".format(os.path.basename(half2)[:-4], name))
        # os.remove( '{0}'.format( mask ) )
    else:
        # compure FSC
        com = "{4}; e2proc3d.py {0} fsc_{2}.txt --apix={3} --calcfsc={1}".format(
            half1, half2, name, apix, utils.eman_load_command()
        )
        # print com
        subprocess.getoutput(com)

    # parse result
    fsc = numpy.loadtxt("fsc_{0}.txt".format(name))
    if not keep:
        os.remove("fsc_{0}.txt".format(name))

    results.put([threshold, shell, gauss, fsc])

    return


def fsc_cutoff(x, cutoff=0.5):

    # detect first dip below cutoff
    """
    for i in range(x.shape[0]):
        if x[i,1] > float(cutoff) and x[i,0] > .1:
            ia = i
        else:
            break
    """

    zero_crossings = numpy.where(numpy.diff(numpy.sign(x[:, 1] - float(cutoff))))[0]
    ia = x.shape[-1] - 2
    for i in zero_crossings:
        if x[i, 0] > 0.1:
            ia = i
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
        "-model",
        "--model",
        help="Use .5-cutoff to report resolution.",
        action="store_true",
        default=False,
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
        if not os.path.exists(first_half):
            nyquist = float(args.pixel) * 2
            com = "{5}; e2pdb2mrc.py -A {0} -R {6} -B {1},{2},{3} --center {4}.pdb {4}.mrc".format(
                args.apix, x, y, z, pdbfile, utils.eman_load_command(), nyquist
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)
        else:
            logger.info("Using existing map for pdb coordinates.")

        if not os.path.exists(average):

            logger.info(
                "Generating low resolution map for selecting masking threshold:",
                average,
            )

            com = "module load EMAN1; proc3d {0} {1} apix={2} lp={3}".format(
                second_half, average, args.apix, args.lowpass
            )
            com = "{4}; e2proc3d.py {0} {1} --apix={2} --process=filter.lowpass.gauss:cutoff_pixels={3}".format(
                second_half,
                average,
                args.apix,
                args.lowpass,
                utils.eman_load_command(),
            )
            if args.verbose:
                logger.info(com)
            subprocess.getoutput(com)
            sys.exit()
        else:
            logger.info("Using existing map to generate mask: %f", average)

    else:

        if len(args.crop) > 0:
            com = "module load EMAN1; proc3d {0} {1} clip={2}".format(
                args.half1, args.half1.replace(".mrc", "_crop.mrc"), args.crop
            )
            com = "{3}; e2proc3d.py {0} {1} --clip={2}".format(
                args.half1,
                args.half1.replace(".mrc", "_crop.mrc"),
                args.crop,
                utils.eman_load_command(),
            )
            logger.info(com)
            subprocess.getoutput(com)
            args.half1 = args.half1.replace(".mrc", "_crop.mrc")
            com = "module load EMAN1; proc3d {0} {1} clip={2}".format(
                args.half2, args.half2.replace(".mrc", "_crop.mrc"), args.crop
            )
            com = "{3}; e2proc3d.py {0} {1} --clip={2}".format(
                args.half2,
                args.half2.replace(".mrc", "_crop.mrc"),
                args.crop,
                utils.eman_load_command(),
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
            com = "module load EMAN1; proc3d {0} {0} apix={1} lp={2}".format(
                average, args.apix, args.lowpass
            )
            com = "module load EMAN2; e2proc3d.py {0} {0} --apix={1} --process=filter.lowpass.gauss:cutoff_pixels={2} --process=mask.zeroedge3d:x0=2:y0=2:x1=2:y1=2:z0=2:z1=2".format(
                average, args.apix, args.lowpass
            )
            com = "{3}; e2proc3d.py {0} {0} --apix={1} --process=filter.lowpass.gauss:cutoff_pixels={2} --process=mask.zeroedge3d:x0=2:y0=2:x1=2:y1=2:z0=2:z1=2".format(
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

    # sys.exit()

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
                "Threshold %0.8f falls outside density range [%0.4f,%0.4f]",
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
                    # pool.apply_async(fsc_multiprocessing, args=( args.apix, first_half, second_half, average, threshold, shell, gauss, args.phases, results, args.keep ) )
                    fsc_multiprocessing(
                        args.apix,
                        first_half,
                        second_half,
                        average,
                        threshold,
                        shell,
                        gauss,
                        args.phases,
                        results,
                        args.keep,
                    )
    pool.close()

    # Wait for all processes to complete
    pool.join()

    # plot all curves
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    if results.empty() == True:
        logger.error("Could not get FSC curves.")
    else:
        # Collate periodogram averages
        sorted_results = []
    while results.empty() == False:
        sorted_results.append(results.get())
        sorted_results.sort()

        if args.half1[-3:] == "pdb" or args.half2[-3:] == "pdb" or args.model:
            logger.info("Using .5 cutoff")
            cutoff_value = 0.5
        else:
            logger.info("Using .143 cutoff")
            cutoff_value = 0.143
        # cutoff_value = .5

        for result in sorted_results:
            fsc = result[-1]
            # ax.plot( fsc[:,0], fsc[:,1], label='Th=%02.4f, Sh=%02.4f, Gs=%02.4f' % ( result[0], result[1], result[2] ) )
            cutoff = fsc_cutoff(fsc, cutoff_value)
            logger.info("Resolution = %f", cutoff)
        # ax.plot( fsc[:,0], fsc[:,1], label='Mask %02d (.5 = %.2f %s, .143 = %.2f %s)' % ( count, cutoffs[0], u'\u00c5', cutoffs[1], u'\u00c5' ) )
        ax.plot(
            fsc[:, 0],
            fsc[:, 1],
            label="Th %.3f (%.3f = %.2f %s"
            % (result[0], cutoff_value, cutoff, "\u00c5"),
        )

        ax.plot(fsc[:, 0], 0.143 * numpy.ones(fsc[:, 1].shape), "k:")
        ax.plot(fsc[:, 0], 0.5 * numpy.ones(fsc[:, 1].shape), "k:")
        ax.plot(fsc[:, 0], numpy.zeros(fsc[:, 1].shape), "k")

    legend = ax.legend(loc="lower left", shadow=True, fontsize=10)
    ax.set_ylim((-0.1, 1.05))
    ax.set_xlim((fsc[0, 0], 1 * fsc[-1, 0]))
    plt.title(
        "FSC for %s\nVs.\n%s"
        % (os.path.basename(args.half1)[:-4], os.path.basename(args.half2)[:-4])
    )
    plt.xlabel("Frequency (1/" + "\u00c5" + ")")
    plt.ylabel("FSC")
    # ltext  = legend.get_texts()
    # plt.setp(ltext, fontsize='xx-small')
    plt.savefig("%s_fsc.pdf" % dataset)

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
