#!/usr/bin/env python

import argparse

import numpy as np

from pyp.analysis import fit, plot, scores
from pyp.system.logging import initialize_pyp_logger

logger = initialize_pyp_logger(log_name=__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="shape_pr_values")
    parser.add_argument("-mode", help="Operation mode = (plot,shape)", required=True)
    parser.add_argument("-input", help="Input .par file", required=True)
    parser.add_argument("-output", help="Output file", required=True)
    parser.add_argument(
        "-stack", help="Input stack file (required if mode=stack)", required=False
    )
    parser.add_argument(
        "-angle_groups",
        help="Number of groups based on Euler angle assignemnts",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-defocus_groups",
        help="Number of groups based on defocus values",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-cutoff",
        help="Fraction of images to use for reconstruction (as percentage)",
        default=1,
        type=float,
    )
    parser.add_argument(
        "-mindefocus", help="Ignore defocuses below this value", default=0, type=float
    )
    parser.add_argument(
        "-maxdefocus",
        help="Ignore defocuses above this value",
        default=100000,
        type=float,
    )
    parser.add_argument(
        "-mintilt", help="Ignore tilt-angles below this value", default=-90, type=float
    )
    parser.add_argument(
        "-maxtilt", help="Ignore tilt-angles above this value", default=90, type=float
    )
    parser.add_argument(
        "-minazh", help="Ignore tilt-angles below this value", default=0, type=float
    )
    parser.add_argument(
        "-maxazh", help="Ignore tilt-angles above this value", default=180, type=float
    )
    parser.add_argument(
        "-minscore",
        help="Ignore particles with scores below this value",
        default=0,
        type=float,
    )
    parser.add_argument(
        "-maxscore",
        help="Ignore particles with scores above this value",
        default=1,
        type=float,
    )
    parser.add_argument(
        "-firstframe", help="Ignore frames before this value", default=0, type=int
    )
    parser.add_argument(
        "-lastframe", help="Ignore frames after this value", default=-1, type=int
    )
    parser.add_argument(
        "-ref_par",
        help="Reference par file used for regularization",
        default="",
        type=str,
    )
    parser.add_argument(
        "-binning",
        help="Apply binning factor to coordinate shifts",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "-reverse", help="Reverse PR polarity", action="store_true", default=False
    )
    parser.add_argument(
        "-consistency",
        help="Check for alignment consistency",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-odd",
        help="Use for odd half-map reconstruction",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-even",
        help="Use for even half-map reconstruction",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-scores",
        help="Parameters contain scores instead of phase residuals",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-frealignx",
        help="Specify frealignx metric",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-dump",
        help="Dump scores histograms and fmatch stacks per group",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-rotational",
        help="Perform rotational regularization",
        action="store",
        default=False,
    )
    parser.add_argument(
        "-translational",
        help="Perform translational regularization",
        action="store",
        default=False,
    )
    parser.add_argument(
        "-spatial_sigma",
        help="Sigma used for spatial regularization in pixels (500)",
        type=float,
        default=500,
    )
    parser.add_argument(
        "-time_sigma",
        help="Sigma used for time regularization in pixels (5)",
        type=float,
        default=5,
    )
    parser.add_argument(
        "-saveplots",
        help="Save orientation plots for each particle",
        action="store",
        default=False,
    )
    parser.add_argument(
        "-rot_method",
        choices=["AB1", "AB2", "XD"],
        help="Choice of rotational regularization method",
        default="AB2",
    )
    parser.add_argument(
        "-trans_method",
        choices=["AB", "XD", "spline"],
        help="Choice of translational regularization method",
        default="AB",
    )
    args = parser.parse_args()

    if args.mode == "plot":
        plot.generate_plots(
            args.input,
            args.angle_groups,
            args.defocus_groups,
            args.scores,
            args.frealignx,
            args.dump,
        )

    elif args.mode == "shape":
        scores.shape_phase_residuals(
            args.input,
            args.angle_groups,
            args.defocus_groups,
            np.fabs(args.cutoff),
            args.mindefocus,
            args.maxdefocus,
            args.firstframe,
            args.lastframe,
            args.mintilt,
            args.maxtilt,
            args.minazh,
            args.maxazh,
            args.minscore,
            args.maxscore,
            args.binning,
            args.reverse,
            args.consistency,
            args.scores,
            args.frealignx,
            args.odd,
            args.even,
            args.output,
        )

    elif args.mode == "stack":
        scores.generate_cluster_stacks(
            args.stack, args.input, args.angle_groups, args.defocus_groups
        )

    elif args.mode == "regularize":
        fit.regularize(
            args.input,
            args.ref_par,
            args.output,
            args.rotational,
            args.translational,
            args.rot_method,
            args.trans_method,
            args.saveplots,
            args.spatial_sigma,
            args.time_sigma,
        )

    else:
        logger.error("Mode {} not supported.".format(args.mode))
