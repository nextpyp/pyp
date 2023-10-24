#!/usr/bin/env python

import collections
import math
import multiprocessing
import os
import shutil

import numpy

from pyp.inout.metadata import isfrealignx
from pyp.inout.metadata.frealign_parfile import Parameters
from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_csp_path, get_frealign_paths, get_shell_multirun_path
from pyp.utils import get_relative_path

# relative_path = str(get_relative_path(__file__))
# logger = initialize_pyp_logger(log_name=relative_path)
logger = initialize_pyp_logger()


def plot_score_profile(parfile, tomo=True):

    """ Plot score profile per tilt-angle.
    """

    if isfrealignx(parfile):
        score_col = 16 - 1
        tilt_col = 19 - 1
        field = 21 - 1
    else:
        score_col = 15 - 1
        tilt_col = 18 - 1
        field = 20 - 1

    # input = numpy.array( [line.split() for line in file( parfile ) if not line.startswith('C') ], dtype=float )
    input = Parameters.from_file(parfile).data
    tilts = int(input[:, field].max())

    values = numpy.ones([tilts, 3]) * numpy.nan

    for i in range(int(input[:, field].min()), int(input[:, field].max())):
        actual_values = input[input[:, field] == i]
        non_zeros = numpy.nonzero(actual_values[:, tilt_col])
        if tomo:

            values[i, 0] = numpy.mean(actual_values[non_zeros, tilt_col])
        else:
            values[i, 0] = i
        # values[i,0]=(input[input[:,field]==i][:,field].mean())
        non_zeros = numpy.nonzero(actual_values[:, score_col])
        values[i, 1] = numpy.mean(actual_values[non_zeros, score_col])
        values[i, 2] = numpy.std(actual_values[non_zeros, score_col])

    import seaborn as sns

    sns.set(style="white")
    import os

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("Agg")
    a = plt.gca()
    a.set_frame_on(False)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(values[:, 0], values[:, 1], "b.")
    ax.errorbar(
        values[:, 0], values[:, 1], yerr=values[:, 2], fmt="o", uplims=True, lolims=True
    )
    # ax.set_ylim((values[:,1].min(),values[:,1].max()))
    plt.title("Average scores per tilt-angle\n %s" % parfile)
    if tomo:
        plt.xlabel("Tilt (degrees)")
    else:
        plt.xlabel("SCANORD")
    plt.ylabel("Score")
    plt.savefig(os.path.splitext(parfile)[0] + "_scores.png", bbox_inches="tight")


def plot_score_profile_frames(parfile):

    """ Plot score profile per tilt-angle.
    """

    # input = numpy.array( [line.split() for line in file( parfile ) if not line.startswith('C') ], dtype=float )
    input = Parameters.from_file(parfile).data
    tilts = int(input[:, 19].max()) + 1

    values = numpy.empty([tilts, 2])
    for i in range(int(input[:, 19].min()), int(input[:, 19].max())):
        values[i, 0] = i
        actual_values = input[input[:, 19] == i]
        values[i, 1] = numpy.mean(numpy.nonzero((actual_values[:, 14])))

    import os

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values[:, 0], values[:, 1], "b.")
    ax.set_ylim((values[:, 1].min(), values[:, 1].max()))
    plt.title("Score per tilt-angle\n %s" % parfile)
    plt.xlabel("Tilt angle (degrees)")
    plt.ylabel("Average score")
    plt.savefig(os.path.splitext(parfile)[0] + "_scores.png")
