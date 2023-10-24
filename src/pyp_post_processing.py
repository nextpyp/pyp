#!/usr/bin/env python

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"

import matplotlib

matplotlib.use("Agg")

import argparse
import math
import os
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy

from pyp.inout.image import mrc
from pyp.system import project_params, utils
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def plotsf(result, pixel_size):

    # produce structure factor
    com = "{2}; e2proc3d.py {0}.mrc .dummy.mrc --apix={1} --calcsf={0}.sf".format(
        result, pixel_size, utils.eman_load_command()
    )
    logger.info("\nComputing structure factor... %s\n", com)
    logger.info(subprocess.getoutput(com))
    cryo = numpy.loadtxt("%s.sf" % result)
    plt.plot(cryo[:, 0], numpy.log(cryo[:, 1]), label="%s" % result)
    # xray = numpy.loadtxt( '%s.sf' % pdbname )
    # plt.plot( xray[:,0], numpy.log(xray[:,1]), label='%s' % pdbname )
    plt.legend(prop={"size": 8})
    plt.savefig("%s.png" % result)
    os.remove(result + ".sf")


def optimize_bfactor():

    from scipy.stats import kurtosis

    kurtosis(image)


def histogram_equalization(image, nrbins=256):

    imhist, bins = numpy.histogram(image.flatten(), nrbins, normed=True)
    cdf = imhist.cumsum()
    cdf = nrbins * cdf / cdf[-1]
    output = numpy.interp(image.flatten(), bins[:-1], cdf)
    return output.reshape(image.shape)


def local_bfactor(input_map, bfactor, pixel_size, local):

    minbfactor, low_res, high_res = [float(i) for i in bfactor.split(",")]
    bfactor *= -1

    maxbfactor, lowrange, highrange, number_of_bins = [
        float(i) for i in local.split(",")
    ]

    # change low pass filter instead of bfactor level
    max_low_res, lowrange, highrange, number_of_bins = [
        float(i) for i in local.split(",")
    ]
    maxbfactor = minbfactor

    A = mrc.read(input_map)

    # generate all bfactor corrected maps

    # use image modulo as bfactor modulation
    import scipy.ndimage.filters

    # modulo = numpy.log( 1 + 1 * scipy.ndimage.filters.gaussian_filter( numpy.fabs(A), 1 ) )
    # modulo = scipy.ndimage.filters.gaussian_filter( A, 1 )
    modulo = A.copy()

    # mask density
    import scipy.ndimage

    mask = scipy.ndimage.filters.gaussian_filter(
        numpy.where(modulo > lowrange, 1.0, 0.0), 0.5
    )
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mrc.write(mask, "mask.mrc")

    # write masked map
    input_map_masked = input_map.replace(".mrc", "_masked.mrc")
    # mrc.write( mask * A, input_map_masked )
    # input_map = input_map_masked

    if True or number_of_bins < 0:
        mrc.write(modulo, "modulo.mrc")
        # sys.exit()

    # 2, 4
    hist, levels = numpy.histogram(
        modulo, bins=number_of_bins, range=[lowrange, highrange]
    )
    # levels = numpy.hstack( [ modulo.min(), levels ] )

    # minbfactor = 90
    # maxbfactor = bfactor
    # bfactors = numpy.hstack( [ 90, numpy.linspace( minbfactor, maxbfactor, number_of_bins + 1 ) ] )
    bfactors = numpy.linspace(minbfactor, maxbfactor, number_of_bins + 1)

    logger.info("Using local bfactors %f between density levels %f", bfactors, levels)

    low_res_values = numpy.linspace(low_res, max_low_res, number_of_bins + 1)

    logger.info(
        "Using local low pass %f between density levels %f", low_res_values, levels
    )

    # apply general bfactor
    corrected_map = input_map.replace(".mrc", "") + "_bfactor.mrc"
    com = "{4}/bin/bfilter -Bfactor {0} -sampling {1},{1},{1} {2} {3}".format(
        "%f,%f,%f" % (bfactors[0], low_res_values[0], high_res),
        pixel_size,
        input_map,
        corrected_map,
        utils.get_bsoft_path(),
    )
    logger.info(com)
    subprocess.getoutput(com)
    # command = 'module load EMAN2; e2proc3d.py {0} {0} --process=normalize.edgemean'.format( corrected_map )
    # commands.getoutput(command)

    # output = A.copy()
    output = mrc.read(corrected_map)
    previous_map = output.copy()
    lth = levels[0]
    for i in range(1, levels.size):

        # generate and load next bfactor level
        com = "{4}/bin/bfilter -Bfactor {0} -sampling {1},{1},{1} {2} {3}".format(
            "%f,%f,%f" % (bfactors[i], low_res_values[i], high_res),
            pixel_size,
            input_map,
            corrected_map,
            utils.get_bsoft_path(),
        )
        logger.info(com)
        subprocess.getoutput(com)
        next_map = mrc.read(corrected_map)
        hth = levels[i]

        logger.info(
            "Applying bfactor range [ %f, %f ], low pass range [ %f, %f ] within density range [ %f, %f ]",
            bfactors[i - 1],
            bfactors[i],
            low_res_values[i - 1],
            low_res_values[i],
            lth,
            hth,
        )
        interpolation = (modulo - lth) / (hth - lth)
        output = numpy.where(
            numpy.logical_and(modulo > lth, modulo <= hth),
            interpolation * previous_map + (1 - interpolation) * next_map,
            output,
        )
        # output = numpy.where( numpy.logical_and( modulo > lth, modulo < hth ), ( interpolation * previous_map + ( 1 - interpolation ) * next_map + output ) / 2., output )

        # prepare for next level
        previous_map = next_map.copy()
        lth = hth

    output = numpy.where(modulo > hth, previous_map, output)

    os.remove(corrected_map)
    os.remove(input_map)

    # output *= mask

    return output


if __name__ == "__main__":

    logger.info(sys.version)

    args = project_params.parse_arguments("postprocessing")

    pdb_path = project_params.resolve_path(args.pdb)
    if len(pdb_path) > 1:
        if os.path.splitext(os.path.split(args.pdb)[-1])[-1] == ".gz":
            pdbname = (
                os.path.split(args.pdb)[-1]
                .replace("pdb", "")
                .replace(".ent.gz", "")
                .upper()
            )
            subprocess.getoutput("gunzip %s -c > %s.pdb" % (args.pdb, pdbname))
        else:
            pdbname = args.pdb.replace(".pdb", "")

    if len(args.bfactor.split(",")) > 1:
        name = "U%d_B%s_L%s_FL%s_FH%s_I%03d" % (
            args.upsample,
            args.bfactor.split(",")[0],
            args.local.replace(",", "_"),
            args.bfactor.split(",")[1],
            args.bfactor.split(",")[2],
            args.iters,
        )
    else:
        name = "U%d_B%s_L%s_F%.1f_I%03d" % (
            args.upsample,
            args.bfactor,
            args.local.replace(",", "_"),
            args.blowpass,
            args.iters,
        )

    if os.path.splitext(args.input)[1] == ".mrc":
        input = os.path.splitext(args.input)[0]
    else:
        input = args.input

    # figure out pixel size
    if args.apix:
        pixel_size = args.apix
    else:
        command = (
            "module load IMOD; header %s.mrc | grep Angstroms | awk '{print $4}'"
            % input
        )
        pixel_size = float(subprocess.getoutput(command).split()[-1])
        logger.info("\nDetected pixel size = %s", pixel_size)

    pixel_size /= args.upsample

    # crop volume
    if args.radius > 0:
        size = 2 * int(float(args.radius) / float(pixel_size))
        clip = "--scale={0} --clip={1},{1},{1} --process=normalize.edgemean".format(
            args.upsample, size
        )
        # clip = '--scale={0} --clip={1},{1},{1}'.format(args.upsample,size)
        com = "{2}; e2proc3d.py {0}.mrc {0}_clip.mrc {1}".format(
            input, clip, utils.eman_load_command()
        )
        logger.info("\nCropping volume...\n", com)
        logger.info(subprocess.getoutput(com))
    else:
        shutil.copy2("%s.mrc" % input, "%s_clip.mrc" % input)
        size = mrc.readHeaderFromFile(input + ".mrc")["nz"]

    # shape mask from original volume before bfactor correction
    if (
        not os.path.exists("{0}_mask.mrc".format(input))
        and math.fabs(args.threshold) > 0
    ):
        if False:
            com = "module load EMAN2; e2proc3d.py {0}_clip.mrc {0}_mask.mrc --process=mask.auto3d:radius=100:threshold={1}:nshells=1:nshellsgauss=3:nmaxseed=0:return_mask=1 {2}".format(
                input, args.threshold, clip
            )
            logger.info("\nComputing shape mask...\n", com)
            logger.info(subprocess.getoutput(com))

        else:

            # create mask by thresholding map density
            import scipy.ndimage.filters

            # modulo = numpy.log( 1 + 1 * scipy.ndimage.filters.gaussian_filter( numpy.fabs(A), 1 ) )
            A = mrc.read(input + "_clip.mrc")
            G = scipy.ndimage.filters.gaussian_filter(A, 2.0)
            G = A

            mrc.write(G, "modulo.mrc")

            # mask apodization using Gaussian filter
            # mask = scipy.ndimage.filters.gaussian_filter( numpy.where( G > args.threshold, 1.0, 0.0 ), 0.5 )
            mask = scipy.ndimage.filters.gaussian_filter(
                numpy.where(G > args.threshold, 1.0, 0.0), 1
            )

            # mask normalization
            mask = (mask - mask.min()) / (mask.max() - mask.min())

            # save mask to file
            mrc.write(mask, input + "_mask.mrc")

            # sys.exit()
            # mrc.write( G, input + '_modulo.mrc' )

            # save masked volume
            # mrc.write( A * mask, input + '_clip.mrc' )

            # sys.exit()

    else:
        logger.info("\nMask already exists or is not requested.")

    if len(pdb_path) > 1:
        # create map from pdb file
        com = "{4}; e2pdb2mrc.py {0}.pdb {0}.mrc -A {1} -R {2} -B {3}".format(
            pdbname, pixel_size, args.blowpass, size, utils.eman_load_command()
        )
        logger.info("\nConverting pdb to mrc...\n %s", com)
        subprocess.getoutput(com)
        # compute structure factor
        com = "{2}; e2proc3d.py {0}.mrc .dummy.mrc --apix={1} --calcsf={0}.sf".format(
            pdbname, pixel_size, utils.eman_load_command()
        )
        logger.info("\nComputing structure factor...\n %s", com)
        logger.info(subprocess.getoutput(com))
        pdb = "--setsf={0}.sf".format(pdbname)
        newname = "_" + pdbname
        # com='module load EMAN2; e2proc3d.py {0}_clip.mrc {0}_clip.mrc {1}'.format( input, pdb )
        # print '\nCropping volume...\n', com
        # print commands.getoutput(com)
    else:
        pdb = ""
        newname = ""

    # bfactor sharpening original map
    if len(args.bfactor) > 0:

        for iter in range(args.iters):

            if numpy.fabs(float(args.local.split(",")[0])) > 0:

                # doing local bfactor correction
                output = local_bfactor(
                    "{0}_clip.mrc".format(input), args.bfactor, pixel_size, args.local
                )
                mrc.write(output, input + "_clip_%s.mrc" % name)

            elif len(args.bfactor.split(",")) == 1:
                shutil.move(
                    "{0}_clip.mrc".format(input),
                    "%s_clip_U%d.mrc" % (input, args.upsample),
                )
                com = "{0}/bfactor_v1.04.sh {1}_clip_{2}.mrc {3} 10 4 {4} 2 {5} 10".format(
                    utils.get_bfactor_path(),
                    input,
                    "U%d" % args.upsample,
                    pixel_size,
                    args.bfactor,
                    args.blowpass,
                )
                logger.info(com)
                logger.info(subprocess.getoutput(com))
                # _clip_U1_B-60_F0.0
                logger.info(
                    input + "_clip_U%d_B%d_F%.1f.mrc",
                    int(args.upsample),
                    int(args.bfactor),
                    float(args.blowpass),
                )
                shutil.move(
                    input
                    + "_clip_U%d_B%d_F%.1f.mrc"
                    % (int(args.upsample), int(args.bfactor), float(args.blowpass)),
                    input + "_clip_" + name + ".mrc",
                )
            else:

                if args.embfactor != 0:
                    com = "{0}/bin/embfactor -Bfactor {1} -maxresol {2},{3} -sampling {4},{4},{4} -noscale {5}_clip.mrc {5}_clip_{6}.mrc".format(
                        utils.get_embfactor_path,
                        args.bfactor.split(",")[0],
                        args.bfactor.split(",")[1],
                        args.bfactor.split(",")[2],
                        pixel_size,
                        input,
                        name,
                    )
                else:
                    com = "{0}/bin/bfilter -Bfactor {1} -sampling {2},{2},{2} {3}_clip.mrc {3}_clip_{4}.mrc".format(
                        utils.get_bsoft_path(), args.bfactor, pixel_size, input, name
                    )

                logger.info("\nApplying bfactor...\n %s", com)
                logger.info(subprocess.getoutput(com))

                continue

                if args.iters >= 1:

                    # save copy of original volume
                    if iter == 0:
                        shutil.copy2(input + "_clip.mrc", input + "_clip_original.mrc")
                        # command = 'module load EMAN2; e2proc3d.py {0}_clip.mrc {0}_clip_original.mrc --process=normalize.edgemean'.format( input, name )
                        # print command
                        # print commands.getoutput(command)

                    # normalize sharpened map
                    command = "{2}; e2proc3d.py {0}_clip_{1}.mrc {0}_clip_{1}.mrc --process=normalize.edgemean".format(
                        input, name, utils.eman_load_command()
                    )
                    logger.info(command)
                    logger.info(subprocess.getoutput(command))

                    if True:
                        # multiply by original image
                        command = "{2}; e2proc3d.py {0}_clip_original.mrc {0}_clip_next.mrc --multfile={0}_clip_{1}.mrc".format(
                            input, name, utils.eman_load_command()
                        )
                        logger.info(command)
                        logger.info(subprocess.getoutput(command))
                    else:
                        # add two volumes
                        mrc.write(
                            mrc.read(input + "_clip_original.mrc")
                            + mrc.read("%s_clip_%s.mrc" % (input, name)),
                            input + "_clip_next.mrc",
                        )

                    # save output if final iteration
                    if iter < args.iters - 1:
                        shutil.move(input + "_clip_next.mrc", input + "_clip.mrc")
                    else:
                        shutil.copy2(
                            input + "_clip_next.mrc", input + "_clip_%s.mrc" % name
                        )

                    # equalize histogram
                    if False:
                        A = mrc.read(input + "_clip_%s.mrc" % name)
                        mrc.write(
                            histogram_equalization(A, nrbins=256),
                            input + "_clip_%s.mrc" % name,
                        )

        if args.iters >= 1:
            try:
                os.remove(input + "_clip_original.mrc")
                os.remove(input + "_clip_next.mrc")
            except:
                pass

    else:
        shutil.copy2(
            "{0}_clip.mrc".format(input), "{0}_clip_{1}.mrc".format(input, name)
        )

    maskfile = "{0}_mask.mrc".format(input)
    if os.path.exists(maskfile):
        masking = "--multfile={0}".format(maskfile)
    else:
        masking = ""

    if args.sym:
        sym = "--sym=%s" % args.sym.lower()
        newname = newname + "_" + args.sym.upper()
    else:
        sym = ""

    if args.lowpass == 1:
        filtering = ""
    else:
        filtering = "--process=filter.lowpass.gauss:cutoff_freq={0}:sigma={1}".format(
            args.lowpass, args.sigma
        )
        # filtering = '--process=filter.lowpass.tophat:cutoff_abs={0}:sigma={1}'.format( args.lowpass, args.sigma )
        # filtering = '--process=filter.lowpass.gauss:cutoff_freq={0}'.format( args.lowpass )

    if args.upsample > 1:
        sym += " --trans={0},{0},{0}".format(-args.upsample / 2)

    command = "{8}; e2proc3d.py {0}_clip_{1}.mrc {0}_{1}_{2}.mrc {3} {4} --apix={6} {5} --origin=0,0,0 {7}".format(
        input,
        name,
        "F%.2f_S%.2f" % (args.lowpass, args.sigma) + newname,
        masking,
        pdb,
        filtering,
        pixel_size,
        sym,
        utils.eman_load_command(),
    )
    logger.info(
        "\nApplying mask (if any), structure factor (if any), and setting origin to 0...\n %s",
        command,
    )
    logger.info(subprocess.getoutput(command))

    result = "{0}_{1}_{2}".format(
        input, name, "F%.2f_S%.2f" % (args.lowpass, args.sigma) + newname
    )

    # set start to 0
    data = mrc.read(result + ".mrc")
    h = mrc.readHeaderFromFile(result + ".mrc")
    h["nxstart"] = h["nystart"] = h["nzstart"] = 0
    headerbytes = mrc.makeHeaderData(h)
    f = open(result + ".mrc", "wb")
    f.write(headerbytes)
    mrc.appendArray(data, f)
    f.close()

    command = """
%s/bin/alterheader << EOF
%s.mrc
del
%s,%s,%s
done
EOF
""" % (
        utils.get_imod_path(),
        result,
        pixel_size,
        pixel_size,
        pixel_size,
    )
    [output, error] = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()

    # produce structure factor
    plotsf(result, pixel_size)

    if os.path.exists(".dummy.mrc"):
        os.remove(".dummy.mrc")

    # clean up
    if not args.keep:
        try:
            if os.path.splitext(os.path.split(args.pdb)[-1])[-1] == ".gz":
                os.remove(pdbname + ".pdb")
            os.remove("{0}_clip_{1}.mrc".format(input, name))
            if args.threshold:
                os.remove("{0}_mask.mrc".format(input))
            os.remove("{0}_clip.mrc".format(input))
            if len(pdb_path) > 1:
                os.remove("{0}.mrc".format(pdbname))
                os.remove("{0}.sf".format(pdbname))
        except OSError:
            pass

    logger.info("\nFinal map saved in %s.mrc", result)
