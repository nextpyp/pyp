#!/usr/bin/env python

import argparse
import os
import subprocess
import sys

import numpy as np

from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Particle coordinate converter")
    parser.add_argument(
        "-mod2cryolo",
        help="Convert imod model to cryolo box file.",
        action="store_true",
    )
    parser.add_argument(
        "-cryolo2mod",
        help="Convert cryolo box file to imod model file.",
        action="store_true",
    )
    parser.add_argument("-input", help="Input filename.", type=str)
    parser.add_argument("-output", help="Output filename.", type=str)
    parser.add_argument(
        "-boxsize",
        help="Box size in pixel of particle observed in pyp tomogram. (10)",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-z", help="Z height of cryolo tomogram in pixel. (256)", default=256, type=int
    )
    parser.add_argument(
        "-s",
        help="Scaling factor of cryolo tomogram with respect to pyp tomogram. (1)",
        default=1,
        type=int,
    )

    args = parser.parse_args()

    return args


def read_mod(spk_file):
    modfile = subprocess.getoutput(
        "{0}/bin/imodinfo -a {1}".format(get_imod_path(), spk_file)
    ).split("contour")
    indexes = []
    if len(modfile) > 1:
        for c in range(1, len(modfile)):
            points_in_contour = int(modfile[c].split()[2])
            for point in range(points_in_contour):
                indexes.append(
                    np.array(modfile[c].split("\n")[point + 1].split(), dtype=float)
                )
    return np.array(indexes)


def read_3dbox(filename):
    return [
        np.array(line.split())
        for line in open(filename, "r")
        if not line.strip().startswith("PTL")
    ]


def read_box(cbox):
    spikes = []
    with open(cbox) as f:
        for line in f.readlines():
            if line.startswith("_") or len(line.split()) < 3:
                continue
            else:
                spikes.append(list(map(float, line.split()[:3])))
    return np.array(spikes)


def cryolo2mod(cbox, mod, size, scaling, z, write=True):
    """
    This function converts cryolo box file to IMOD model 
    @cbox - cryolo box file (input)
    @mod - IMOD model file (output)
    @size - particle box size in pixel observed in pyp tomogram 
    @scaling - scaling factor of cryolo tomogram with respect to pyp binned tomogram, which is 512x512x256
    @z - z height of cryolo tomogram
    
    """
    if not os.path.exists(cbox):
        logger.error("Input %s does not exist")
        sys.exit()
    spikes = read_box(cbox)
    ret_spikes = []

    if write:
        modtxt = open(cbox + ".txt", "w")

    for spk in range(spikes.shape[0]):
        spike_x, spike_y, spike_z = list([x / scaling for x in spikes[spk][0:3]])
        spike_z = spike_z - (z / (2 * (scaling))) + (256 / 2)
        if write:
            modtxt.write("%.1f\t%.1f\t%.1f\n" % (spike_x, spike_z, spike_y))
        ret_spikes.append([spike_x, spike_y, spike_z])

    if write:
        modtxt.close()
        os.system(
            "%s/bin/point2model -scat -sphere %d %s %s"
            % (get_imod_path(), size, cbox + ".txt", mod)
        )
        os.system("%s/bin/imodtrans -Y -T %s %s" % (get_imod_path(), mod, mod))
        # clean up
        os.remove(cbox + ".txt")
        os.remove("%s~" % mod)

    return np.array(ret_spikes)


def mod2cryolo(next, cbox, size, scaling, z):
    """
    This function converts IMOD model to cryolo box file
    @mod - IMOD model file (input)
    @cbox - cryolo box file (output)
    @size - particle box size in pixel observed in pyp binned tomogram 
    @scaling - scaling factor of cryolo tomogram with respect to pyp binned tomogram, which is 512x512x256
    @z - z height of cryolo tomogram
    """

    # read coordinates from model
    if not os.path.exists(next):
        logger.error("Input %s does not exist" % next)
        sys.exit()
    spikes = np.loadtxt(next, ndmin=2)

    cboxf = open(cbox, "w")
    # write header info
    cboxf.write(
        "data_global\n\n_cbox_format_version 1.0\n\ndata_cryolo\n\nloop_\n_CoordinateX #1\n_CoordinateY #2\n_CoordinateZ #3\n_Width #4\n_Height #5\n_Depth #6\n_EstWidth #7\n_EstHeight #8\n_Confidence #9\n_NumBoxes #10\n"
    )

    cryolo_box = size * scaling

    slices = []

    for spk in range(spikes.shape[0]):

        spike_x, spike_y, spike_z = spikes[spk][0:3]
        spike_z = spike_z - (256 / 2) + (z / (2 * scaling))

        spike_x, spike_y, spike_z = list(
            [x * scaling for x in [spike_x, spike_y, spike_z]]
        )
        cboxf.write(
            "%.1f %.1f %.1f %.1f %.1f 1.0 <NA> <NA> 1.0 <NA>\n"
            % (
                spike_x - cryolo_box/2,
                spike_y - cryolo_box/2,
                spike_z,
                cryolo_box,
                cryolo_box,
            )
        )

        slices.append(spike_z)
    # add slices info
    cboxf.write("\ndata_cryolo_include\n\nloop_\n_slice_index #1\n")
    for zslice in slices:
        cboxf.write(str(zslice) + "\n")
    # add 10 slices (at most) for negative controls
    count = 10
    half = count / 2
    min_slice, max_slice = int(min(slices)), int(max(slices))
    negatives = []
    min_dist, max_dist = 128, 128
    while not min_dist <= cryolo_box * 2 and count > half:
        min_dist = (min_slice - 0) / 2
        min_slice -= min_dist
        count -= 1
        negatives.append(min_slice)
    while not max_dist <= cryolo_box * 2 and count > 0:
        max_dist = (255 - max_slice) / 2
        max_slice += max_dist
        count -= 1
        negatives.append(max_slice)
    for n in negatives:
        cboxf.write(str(n) + "\n")

    cboxf.close()


if __name__ == "__main__":

    args = parse_arguments()

    if not args.input:
        logger.error("Please provide an input")
        sys.exit()
    if not args.output:
        logger.error("Please provide an output")
        sys.exit()

    if args.mod2cryolo:

        mod2cryolo(args.input, args.output, args.boxsize, args.s, args.z)

    elif args.cryolo2mod:

        cryolo2mod(args.input, args.output, args.boxsize, args.s, args.z)

    else:
        logger.error(
            "Please specify one of these conversions: -mod2cryolo, -cryolo2mod"
        )
