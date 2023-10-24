#!/usr/bin/env python

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from pyp.inout.metadata import isfrealignx
from pyp.inout.metadata.frealign_parfile import Parameters
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


"""
Draw the particle trajectory over tilts 
you have to be in the frealign/ directory to correctly get the .film, mod/*.spk. Alternatively, you can change the relative path in the code
@Hsuan Fu
"""


def read_parfile(parfile, tiltseries):

    # build a data structure (dictionary) storing the particle information
    # the key of dictionary is the name of tiltseries
    # the elements (lists) are several particles
    # the xy-shift pairs are the xy translation in each tilt

    ret = {}
    for name in tiltseries:
        ret[name] = []

    if isfrealignx(parfile):
        ptlind_col = 18 - 1
        scanor_col = 21 - 1
    else:
        ptlind_col = 17 - 1
        scanor_col = 20 - 1

    input = Parameters.from_file(parfile).data

    for line in range(input.shape[0]):
        parline = input[line, :]

        shift_x = float(parline[4])
        shift_y = float(parline[5])
        film = int(parline[7])
        name = tiltseries[film]
        ptlidx = int(parline[ptlind_col])
        scnord = int(parline[scanor_col])

        while ptlidx > len(ret[name]) - 1:
            ret[name].append([])
        while scnord > len(ret[name][ptlidx]) - 1:
            ret[name][ptlidx].append([])

        ret[name][ptlidx][scnord].append(shift_x)
        ret[name][ptlidx][scnord].append(shift_y)

    """
    with open(parfile) as f:
        for line in f.readlines():
            if not line.startswith('C'):
                parline = line.split()
            
                shift_x = float(parline[4])
                shift_y = float(parline[5])
                film = int(parline[7])
                name = tiltseries[film]
                ptlidx = int(parline[ptlind_col])
                scnord = int(parline[scanor_col])
            
                while ptlidx > len(ret[name])-1:
                    ret[name].append( [ ] )
                while scnord > len(ret[name][ptlidx]):
                    ret[name][ptlidx].append( [ ] )

                ret[name][ptlidx][scnord-1].append( shift_x )
                ret[name][ptlidx][scnord-1].append( shift_y )
    """

    return ret


def read_framefile(framefile):
    # obtain the frame index in the parfile corresponding to tiltseries
    ret = []

    with open(framefile) as f:
        for line in f.readlines():
            ret.append(line.strip())

    return ret


def read_boxfile(framefile):
    # get the xy coordinates from a picked model
    ret = {}

    tiltseries = read_framefile(framefile)

    for name in tiltseries:

        if not os.path.exists("../mod/{0}.txt".format(name)):
            os.system(
                "{0}/bin/model2point ../mod/{1}.spk ../mod/{1}.txt".format(
                    get_imod_path(), name
                )
            )

        with open("../mod/{0}.txt".format(name)) as f:

            for line in f.readlines():

                x = float(line.split()[0])
                y = float(line.split()[2])

                if name not in ret:
                    ret[name] = []

                ret[name].append((x, y))
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plot drift trajectories for tomography"
    )
    parser.add_argument("-films", help="Path to .films file", required=True, type=str)
    parser.add_argument(
        "-parfile", help="Input parameter file", required=True, type=str
    )
    parser.add_argument("-binning", help="Binning factor", type=int, default=8)
    parser.add_argument(
        "-trajectory_scale",
        help="Scaling of trajectories for proper display",
        type=int,
        default=6,
    )
    args = parser.parse_args()

    tiltseries = read_framefile(args.films)

    ptlcoor = read_boxfile(args.films)

    ptlshifts = read_parfile(args.parfile, tiltseries)

    # iterate through each tiltseries
    for name in tiltseries:
        fig, ax = plt.subplots(1, 1, figsize=(30, 25), dpi=200)

        logger.info("Now processing %s", name)
        for ptl_idx, ptl in enumerate(ptlcoor[name]):

            x, y = ptl[0] * args.binning, ptl[1] * args.binning

            if ptl_idx + 1 < len(ptlshifts[name]):
                shifts_x = [
                    x + (t[0] * args.trajectory_scale)
                    for i, t in enumerate(ptlshifts[name][ptl_idx])
                    if len(t) != 0 and i % 5 == 0
                ]
                shifts_y = [
                    y + (t[1] * args.trajectory_scale)
                    for i, t in enumerate(ptlshifts[name][ptl_idx])
                    if len(t) != 0 and i % 5 == 0
                ]
            else:
                continue

            ax.plot(
                shifts_x, shifts_y, linewidth=2, alpha=0.5, color="gray", linestyle="-"
            )

            im = ax.scatter(
                shifts_x, shifts_y, s=30, c=np.arange(len(shifts_x)), cmap="viridis_r"
            )
        fig.colorbar(im, ax=ax, aspect=40)
        ax.axis("off")
        plt.savefig("scratch/{0}_trajectory_tomo.png".format(name), bbox_inches="tight")
