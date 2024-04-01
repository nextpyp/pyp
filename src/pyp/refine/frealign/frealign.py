import argparse
import collections
import datetime
import glob
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import re
import time

import matplotlib

from pyp.system.singularity import run_pyp

matplotlib.use("Agg")

from pathlib import Path, PosixPath

import numpy as np
from scipy.optimize import minimize

from pyp import analysis, postprocess
from pyp.analysis import plot, statistics
from pyp.analysis.occupancies import occupancies, occupancy_extended
from pyp.inout.image import mrc, writepng, img2webp
from pyp.inout.metadata import create_curr_iter_par, frealign_parfile, isfrealignx
from pyp.inout.metadata.cistem_star_file import *
from pyp.refine.csp import cspty
from pyp.system import local_run, mpi, project_params, slurm, user_comm
from pyp.system.db_comm import save_classes_to_website
from pyp.system.logging import initialize_pyp_logger
from pyp.system.singularity import get_mpirun_command, run_pyp
from pyp.system.utils import (
    eman_load_command,
    get_frealign_paths,
    get_multirun_path,
    get_imod_path,
)
from pyp.utils import get_relative_path
from pyp.utils import timer, symlink_relative

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def parse_def_arguments():
    parser = argparse.ArgumentParser(description="Defocus refinement")
    parser.add_argument(
        "-parfile", "--parfile", help="Parameter file", required=True, type=str
    )
    parser.add_argument("-film", "--film", help="Film number", type=int, required=True)
    parser.add_argument(
        "-scanor", "--scanor", help="Scan order", type=int, required=True
    )
    parser.add_argument(
        "-tolerance", "--tolerance", help="Defocus tolerance", type=int, default=1000,
    )

    return parser.parse_args()


def parse_def_split_arguments():
    parser = argparse.ArgumentParser(description="Defocus refinement")
    parser.add_argument(
        "-parfile", "--parfile", help="Parameter file", required=True, type=str
    )
    parser.add_argument(
        "-def", "--def", help="Refine both defocus", type=int, required=False
    )
    parser.add_argument(
        "-df1", "--df1", help="Refine defocus 1", type=int, required=False
    )
    parser.add_argument(
        "-df2", "--df2", help="Refine defocus 2", type=int, required=False
    )
    parser.add_argument(
        "-tolerance",
        "--tolerance",
        help="Defocus tolerance",
        type=int,
        required=False,
        default=1000,
    )

    return parser.parse_args()


def parse_def_merge_arguments():
    parser = argparse.ArgumentParser(description="Defocus refinement")
    parser.add_argument(
        "-tolerance", "--tolerance", help="Defocus tolerance", type=int, default=1000,
    )
    parser.add_argument(
        "-parfile", "--parfile", help="Parameter file", required=True, type=str
    )

    return parser.parse_args()


def parse_ref_arguments():
    parser = argparse.ArgumentParser(description="Pre-processing daemon")
    parser.add_argument(
        "-iteration", "--iteration", help="Sample name", type=int, required=True
    )
    parser.add_argument(
        "-ref", "--ref", help="Class to process", type=int, required=True
    )
    parser.add_argument(
        "-first", "--first", help="Session name", type=int, required=True
    )
    parser.add_argument(
        "-last", "--last", help="File to process", type=int, required=False
    )
    parser.add_argument(
        "-metric",
        "--metric",
        help="version of frealign",
        type=str,
        required=False,
        default="cc3m",
    )
    return parser.parse_args()


def parse_rec_arguments():
    parser = argparse.ArgumentParser(description="FREALIGN reconstruction mode")
    parser.add_argument(
        "-iteration", "--iteration", help="Sample name", required=True, type=int
    )
    parser.add_argument(
        "-alignment_option",
        "--alignment_option",
        help="Alignment option (=0 after rsample, =1 normal refinement, =2 first pass using cclin, =3 second pass using v9)",
        type=int,
        default=1,
    )

    return parser.parse_args()


def parse_rec_split_arguments():
    parser = argparse.ArgumentParser(description="Pre-processing daemon")
    parser.add_argument(
        "-iteration", "--iteration", help="Sample name", required=True, type=int
    )
    parser.add_argument(
        "-ref", "--ref", help="Class to process", type=int, required=True
    )
    parser.add_argument(
        "-first", "--first", help="First particle", type=int, required=True
    )
    parser.add_argument(
        "-last", "--last", help="Last particle", type=int, required=False
    )
    parser.add_argument(
        "-count", "--count", help="Group number", type=int, required=False
    )

    return parser.parse_args()


def parse_rec_merge_arguments():
    parser = argparse.ArgumentParser(description="Pre-processing daemon")
    parser.add_argument(
        "-iteration", "--iteration", help="Sample name", required=True, type=int
    )

    return parser.parse_args()


def parse_arguments(skip=False):

    # parse arguments
    parser = argparse.ArgumentParser(description="FREALIGN refinement")
    parser.add_argument("-dataset", "--dataset", help="Name of dataset")
    parser.add_argument("-symmetry", "--symmetry", help="Specify symmetry (C1)")
    parser.add_argument(
        "-rbfact",
        "--rbfact",
        help="B-factor to apply to particle image projections before orientation determination or refinement (0)",
        type=str,
    )
    parser.add_argument(
        "-ffilt",
        "--ffilt",
        help="Apply single particle Wiener filter to final reconstruction (T)",
        type=str,
    )
    parser.add_argument(
        "-fbfact",
        "--fbfact",
        help="Determine and apply B-factor to final reconstruction (F)",
        type=str,
    )
    parser.add_argument(
        "-xstd",
        "--xstd",
        help="number of standard deviations above mean for masking of input low-pass filtered 3D model (0)",
        type=str,
    )
    parser.add_argument("-mode", "--mode", help="Mode key 1-4 (4:4:4:4:1)", type=str)
    parser.add_argument(
        "-dang",
        "--dang",
        help="Angular step size for the angular search used in modes 3,4 (200)",
        type=str,
    )
    parser.add_argument(
        "-itmax",
        "--itmax",
        help="Number of cycles of randomised search/refinement used in modes 2,4 (50)",
        type=str,
    )
    parser.add_argument(
        "-ipmax",
        "--ipmax",
        help="Number of potential matches in a search that should be tested further in a subsequent local refinement (10)",
        type=str,
    )
    parser.add_argument(
        "-target", "--target", help="Threshold PR for refinement (90)", type=str
    )
    parser.add_argument(
        "-rlref",
        "--rlref",
        help="Refinement low resolution limit in Angstroms (100)",
        type=str,
    )
    parser.add_argument(
        "-rhref",
        "--rhref",
        help="Refinement high resolution limit in Angstroms (16:12:8:4)",
        type=str,
    )
    parser.add_argument(
        "-rhcls",
        "--rhcls",
        help="Classification high resolution limit in Angstroms (8)",
        type=str,
    )
    parser.add_argument(
        "-threc",
        "--threc",
        help="Phase residual threshold for reconstruction (0)",
        type=str,
    )
    parser.add_argument(
        "-rrec",
        "--rrec",
        help="Resolution of reconstruction (0=half-Nyquist)",
        type=str,
    )
    parser.add_argument(
        "-radrec", "--radrec", help="Radius of reconstruction (0=no masking)", type=str
    )
    parser.add_argument(
        "-fmag", "--fmag", help="Magnification refinement for each film (F)", type=str
    )
    parser.add_argument(
        "-fastig", "--fastig", help="Astigmatism refinement (F)", type=str
    )
    parser.add_argument(
        "-fpart",
        "--fpart",
        help="Defocus refinement for individual particles (F)",
        type=str,
    )
    parser.add_argument("-fdef", "--fdef", help="Defocus refinement (F)", type=str)
    parser.add_argument(
        "-dfsig", "--dfsig", help="Defocus uncertainty in Angstroms (150)", type=str
    )
    parser.add_argument(
        "-pbc",
        "--pbc",
        help="Phase residual / pseudo-B-factor conversion Constant",
        type=str,
    )
    parser.add_argument("-boff", "--boff", help="Average phase residual", type=str)
    parser.add_argument(
        "-iewald", "--iewald", help="Ewald correction (0/1/2/-1/-2)", type=str
    )
    parser.add_argument(
        "-fboost",
        "--fboost",
        help="Set to -1 to allow potential overfitting during refinement (0)",
        type=str,
    )
    parser.add_argument(
        "-fssnr", "--fssnr", help="Use SSNR table while evaluating metric (T)", type=str
    )
    parser.add_argument(
        "-lblur",
        "--lblur",
        help="Use likelihood blurring during reconstruction (F)",
        type=str,
    )
    parser.add_argument(
        "-lblur_start",
        "--lblur_start",
        help="Starting angle for blurring with respect to the optimal orientation in degrees (-10)",
        type=str,
    )
    parser.add_argument(
        "-lblur_step",
        "--lblur_step",
        help="Step size of angular sampling to use for blurring in degrees (1)",
        type=str,
    )
    parser.add_argument(
        "-lblur_nrot",
        "--lblur_nrot",
        help="Number of discrete rotations to use for blurring (21)",
        type=str,
    )
    parser.add_argument(
        "-lblur_range",
        "--lblur_range",
        help="LogP range to use for blurring (20)",
        type=str,
    )
    parser.add_argument(
        "-iblow", "--iblow", help="Padding factor for reference structure (1)", type=str
    )
    parser.add_argument(
        "-imem",
        "--imem",
        help="Memory usage and padding factor for reference structure 0/1/2/3. V9 only (0)",
        type=int,
    )
    parser.add_argument(
        "-interp",
        "--interp",
        help="Interpolation scheme used for 3D reconstruction 0=nearest, 1=linear.  V9 only (1)",
        type=int,
    )
    parser.add_argument(
        "-fmatch",
        "--fmatch",
        help="Write out matching projections after the refinemen (F)",
        type=str,
    )
    parser.add_argument(
        "-mask",
        "--mask",
        help="0/1 mask to exclude parameters from refinement (1,1,1,1,1)",
        type=str,
    )
    parser.add_argument(
        "-cutoff",
        "--cutoff",
        help="Fraction of images to use for reconstruction (=0)",
        type=str,
    )
    parser.add_argument(
        "-iter", "--iter", help="First refinement iteration (2)", type=int
    )
    parser.add_argument(
        "-maxiter", "--maxiter", help="Last refinement iteration (8)", type=int
    )
    parser.add_argument(
        "-maskth", "--maskth", help="Threshold for shape mask", type=str
    )
    parser.add_argument(
        "-mask_weight", "--mask_weight", help="Set weight outside mask (=0)", type=str
    )
    parser.add_argument(
        "-agroups",
        "--agroups",
        help="Number of angular groups used for sorting by PR (1)",
        type=str,
    )
    parser.add_argument(
        "-dgroups",
        "--dgroups",
        help="Number of defocus groups used for sorting by PR (1)",
        type=str,
    )
    parser.add_argument(
        "-mindef",
        "--mindef",
        help="Minimum defocus to use for reconstruction (0)",
        type=str,
    )
    parser.add_argument(
        "-maxdef",
        "--maxdef",
        help="Maximum defocus to use for reconstruction (100000)",
        type=str,
    )
    parser.add_argument(
        "-mintilt",
        "--mintilt",
        help="Minimum desired tilt-angle to use for reconstruction (-90)",
        type=str,
    )
    parser.add_argument(
        "-maxtilt",
        "--maxtilt",
        help="Maximum desired tilt-angle to use for reconstruction (90)",
        type=str,
    )
    parser.add_argument(
        "-minazh",
        "--minazh",
        help="Minimum desired azhimut-angle to use for reconstruction (0)",
        type=str,
    )
    parser.add_argument(
        "-maxazh",
        "--maxazh",
        help="Maximum desired azhimut-angle to use for reconstruction (180)",
        type=str,
    )
    parser.add_argument(
        "-minscore",
        "--minscore",
        help="Minimum desired score to use for reconstruction (0)",
        type=str,
    )
    parser.add_argument(
        "-maxscore",
        "--maxscore",
        help="Maximum desired score to use for reconstruction (1)",
        type=str,
    )
    parser.add_argument(
        "-firstframe",
        "--firstframe",
        help="First frame to use for reconstruction (0)",
        type=str,
    )
    parser.add_argument(
        "-lastframe",
        "--lastframe",
        help="Last frame to use for reconstruction (-1)",
        type=str,
    )
    parser.add_argument(
        "-shapr",
        "--shapr",
        help="Advanced options for shaping based on PR (-reverse, -consistency)",
        type=str,
    )
    parser.add_argument(
        "-daemon",
        "--daemon",
        help="Minimum number of particles to start processing (min2D,inc2D,min3D,inc3D)",
        type=str,
    )
    parser.add_argument(
        "-classes",
        "--classes",
        help="Number of references for multi-reference refinement (1:1:1:1:1:1:1)",
        type=str,
    )
    parser.add_argument(
        "-refineshifts",
        "--refineshifts",
        help="Refine translations every this many iterations (2)",
        type=str,
    )
    parser.add_argument(
        "-refineeulers",
        "--refineeulers",
        help="Refine Euler angles every this many iterations (3)",
        type=str,
    )
    parser.add_argument(
        "-queue", "--queue", help="Use specific queue to run jobs (" ")", type=str
    )
    parser.add_argument(
        "-model", "--model", help="Initial model to use for refinement", type=str
    )
    parser.add_argument("-debug", "--debug", help="Keep refinement log files", type=str)
    parser.add_argument(
        "-metric", "--metric", help="Metric used for alignment (cclin:cc3m)", type=str
    )
    parser.add_argument(
        "-weights", "--weights", help="Use multiple weights (F)", type=str
    )
    parser.add_argument(
        "-fmodel", "--fmodel", help="Model to use for evaluating fit ()", type=str
    )
    parser.add_argument(
        "-fmodelres", "--fmodelres", help="Resolution to evaluate fit (3)", type=str
    )
    parser.add_argument(
        "-fmodelscale",
        "--fmodelscale",
        help="Scale reconstruction for fit evaluation (1)",
        type=str,
    )
    parser.add_argument(
        "-fmodelpixel",
        "--fmodelpixel",
        help="Unbinned calibrated pixel size ()",
        type=str,
    )
    parser.add_argument(
        "-fmodelclip",
        "--fmodelclip",
        help="Clip model to this size for fit evaluation (384)",
        type=str,
    )
    parser.add_argument(
        "-fmodelflip",
        "--fmodelflip",
        help="Flip model for fit evaluation (F)",
        type=str,
    )

    # v9.11 options
    parser.add_argument(
        "-fboostlim",
        "--fboostlim",
        help="Resolution limit for using signed correlation coefficient. Set to 0.0 for maximum resolution (0)",
        type=str,
    )
    parser.add_argument(
        "-bsc",
        "--bsc",
        help="Discriminate particles with different scores during reconstruction. Small values (0 - 10) discriminate less than large values (10 - 20) (2)",
        type=str,
    )
    parser.add_argument(
        "-norm",
        "--norm",
        help="Set to T to normalize input particle images for reconstruction (F)",
        type=str,
    )
    parser.add_argument(
        "-crop",
        "--crop",
        help="Set to T to crop input particle images for faster reconstruction. Slightly decreases reconstruction quality (F)",
        type=str,
    )
    parser.add_argument(
        "-adjust",
        "--adjust",
        help="Set to T to adjust particle scored for defocus dependence during reconstruction (T)",
        type=str,
    )
    parser.add_argument(
        "-invert",
        "--invert",
        help="T or F. Set to T if particles are dark on bright background, otherwise set to F. (F)",
        type=str,
    )
    parser.add_argument(
        "-srad",
        "--srad",
        help="Radius of spherical particle mask applied during global search (in Angstrom). 0.0 = set to 1.5 x mask radius used for local refinement (0)",
        type=str,
    )
    parser.add_argument(
        "-searchx",
        "--searchx",
        help="Search range along the X-axis (in Angstrom). 0.0 = set to search mask radius (0)",
        type=str,
    )
    parser.add_argument(
        "-searchy",
        "--searchy",
        help="Search range along the Y-axis (in Angstrom). 0.0 = set to search mask radius (0)",
        type=str,
    )
    parser.add_argument(
        "-focusmask",
        "--focusmask",
        help="Four numbers (in Angstroms) describing a spherical mask (X, Y, Z for mask center and R for mask radius) (0,0,0,0)",
        type=str,
    )

    parser.add_argument(
        "-beamtilt", "--beamtilt", help="Beam tilt refinement (F)", type=str
    )
    parser.add_argument(
        "-nodes", "--nodes", help="Nodes to use for refinement (5)", type=str
    )

    # custom options for frealignX
    parser.add_argument(
        "-score_weighting",
        "--score_weighting",
        help="Use score weighting in refinement (F)",
        type=str,
    )
    parser.add_argument(
        "-dose_weighting",
        "--dose_weighting",
        help="Use dose weighting in refinement (F)",
        type=str,
    )
    parser.add_argument(
        "-dose_weighting_multiply",
        "--dose_weighting_multiply",
        help="Mutiply dose weighting parameters by the number of frames (T)",
        type=str,
    )
    parser.add_argument(
        "-per_particle_splitting",
        "--per_particle_splitting",
        help="Place all frames from the same particle into the same half map (T)",
        type=str,
    )

    parser.add_argument(
        "-rotreg", "--rotreg", help="Regularize rotational parametrs (F)", type=str
    )
    parser.add_argument(
        "-transreg",
        "--transreg",
        help="Regularize translational parameters (F)",
        type=str,
    )
    parser.add_argument(
        "-saveplots",
        "--saveplots",
        help="Save angular plots to visualize angular trajectories on unit sphere (F)",
        type=str,
    )
    parser.add_argument(
        "-num_frames",
        "--num_frames",
        help="Number of frames per movie (for per frame reconstruction) (1)",
        type=str,
    )
    parser.add_argument(
        "-merge_normalize",
        "--merge_normalize",
        help="Normalize the voxel and CTF in merge3d by the number of frames (T)",
        type=str,
    )

    parser.add_argument(
        "-dose_weighting_fraction",
        "--dose_weighting_fraction",
        help="Number of frames per movie (for per frame reconstruction) (8)",
        type=str,
    )
    parser.add_argument(
        "-dose_weighting_transition",
        "--dose_weighting_transition",
        help="Number of frames per movie (for per frame reconstruction) (0.75)",
        type=str,
    )
    parser.add_argument(
        "-ref_par", "--ref_par", help="Reference par used for regularization", type=str
    )
    parser.add_argument(
        "-spatial_sigma",
        "--spatial_sigma",
        help="Spatial regularization in pixels (500)",
        type=float,
        default=500,
    )
    parser.add_argument(
        "-time_sigma",
        "--time_sigma",
        help="Time regularization in frames (21)",
        type=float,
        default=21,
    )
    parser.add_argument(
        "-rotreg_method",
        "--rotreg_method",
        choices=["AB1", "AB2", "XD"],
        help="Choice of rotational regularization method",
        default="AB2",
    )
    parser.add_argument(
        "-transreg_method",
        "--transreg_method",
        choices=["AB", "XD", "spline"],
        help="Choice of translational regularization method",
        default="spline",
    )

    parser.add_argument(
        "-scratch_copy_stack",
        "--scratch_copy_stack",
        help="Copy stack file to node /scratch (F)",
        type=str,
    )

    # denoise reconstruction
    parser.add_argument(
        "-denoise_enable",
        "--denoise_enable",
        help="Denoise reconstruction after each iteration",
        type=str,
    )
    parser.add_argument(
        "-denoise_sigma",
        "--denoise_sigma",
        help="Noise sigma between 0 and 1, higher values mean more denoising (0.5)",
        type=float,
    )
    parser.add_argument(
        "-denoise_nsearch",
        "--denoise_nsearch",
        help="Size of area to search for similar patches (11)",
        type=int,
    )
    parser.add_argument(
        "-denoise_patch_size",
        "--denoise_patch_size",
        help="Patch size for denoising, must be power of 2 (4))",
        type=int,
    )

    # XD: refine using the same model
    parser.add_argument(
        "-same_ref",
        "--same_ref",
        help="Use the same model for refinement (do not update the model) (F)",
        type=str,
    )

    # create empty parameter file
    empty_parameters = collections.OrderedDict(
        [
            (
                "dataset",
                os.path.split(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))[
                    -1
                ],
            )
        ]
    )
    empty_parameters.update(
        [("symmetry", "C1"), ("rbfact", 0.0), ("ffilt", "T"), ("fbfact", "F")]
    )
    empty_parameters.update(
        [
            ("xstd", 0),
            ("mode", "4:4:4:4:1"),
            ("dang", 200),
            ("itmax", 50),
            ("ipmax", 10),
        ]
    )
    empty_parameters.update(
        [
            ("rlref", 100.0),
            ("rhref", "16:12:8:4"),
            ("rhcls", "8"),
            ("target", 90.0),
            ("threc", 0.0),
            ("rrec", 0.0),
            ("radrec", 0.0),
        ]
    )
    empty_parameters.update(
        [("fmag", "F"), ("fastig", "F"), ("fpart", "F"), ("fdef", "F"), ("dfsig", 150)]
    )
    empty_parameters.update(
        [
            ("pbc", 100.0),
            ("boff", 0.0),
            ("iewald", 0),
            ("fboost", 0),
            ("fboostlim", 0),
            ("fssnr", "T"),
            ("lblur", "F"),
            ("lblur_start", "-10"),
            ("lblur_step", "1"),
            ("lblur_nrot", "21"),
            ("lblur_range", "20"),
            ("iblow", 1),
            ("imem", 0),
            ("interp", 1),
            ("fmatch", "F"),
            ("mask", "1,1,1,1,1"),
        ]
    )
    empty_parameters.update([("cutoff", "0")])
    empty_parameters.update(
        [("iter", 2), ("maxiter", 8), ("maskth", -1.0), ("mask_weight", 0)]
    )
    empty_parameters.update(
        [
            ("agroups", 1),
            ("dgroups", 1),
            ("mindef", 0.0),
            ("maxdef", 100000.0),
            ("mintilt", -90.0),
            ("maxtilt", 90.0),
            ("minazh", 0.0),
            ("maxazh", 180.0),
            ("minscore", 0.0),
            ("maxscore", 1.0),
            ("firstframe", 0),
            ("lastframe", -1),
        ]
    )
    empty_parameters.update([("shapr", " "), ("daemon", "5000,25000,5000,2000")])
    empty_parameters.update(
        [
            ("classes", "1:1:1:1:1:1:1"),
            ("refineshifts", "2"),
            ("refineeulers", "3"),
            ("queue", ""),
            ("model", ""),
            ("debug", "F"),
            ("metric", "cclin:cc3m"),
            ("weights", "F"),
        ]
    )
    empty_parameters.update(
        [
            ("fmodel", ""),
            ("fmodelres", "3"),
            ("fmodelscale", "1"),
            ("fmodelpixel", ""),
            ("fmodelclip", "384"),
        ]
    )

    # v9.11 options
    empty_parameters.update(
        [
            ("bsc", 2),
            ("norm", "F"),
            ("crop", "F"),
            ("adjust", "T"),
            ("invert", "F"),
            ("srad", 0),
            ("searchx", 0),
            ("searchy", 0),
            ("focusmask", "0,0,0,0"),
        ]
    )

    empty_parameters.update([("beamtilt", "F")])
    empty_parameters.update([("nodes", "5")])
    # XD: current options for frealignX
    empty_parameters.update([("score_weighting", "F"), ("dose_weighting", "F")])
    empty_parameters.update(
        [("dose_weighting_multiply", "T"), ("per_particle_splitting", "T")]
    )
    empty_parameters.update([("rotreg", "F"), ("transreg", "F"), ("saveplots", "F")])
    empty_parameters.update([("num_frames", "1")])
    empty_parameters.update([("merge_normalize", "T")])
    empty_parameters.update([("dose_weighting_fraction", "8")])
    empty_parameters.update([("dose_weighting_transition", ".75")])
    empty_parameters.update([("ref_par", "")])
    empty_parameters.update([("same_ref", "F")])
    empty_parameters.update([("spatial_sigma", "500"), ("time_sigma", "21")])
    empty_parameters.update([("rotreg_method", "AB2"), ("transreg_method", "spline")])
    empty_parameters.update([("scratch_copy_stack", "F")])

    empty_parameters.update(
        [
            ("denoise_enable", "False"),
            ("denoise_sigma", "0.5"),
            ("denoise_nsearch", "11"),
            ("denoise_patch_size", "4"),
        ]
    )

    # load existing parameters
    parameters = project_params.load_parameters()

    if parameters == 0:
        parameters = empty_parameters
    else:
        if len(parameters) is not len(empty_parameters):
            logger.warning("Parameter file format has changed. Adding new entries:")
            for key in list(empty_parameters.keys()):
                if key not in parameters:
                    print("\t", key, empty_parameters[key])
                    parameters[key] = empty_parameters[key]

    if not skip:
        args = parser.parse_args()
        for k, v in vars(args).items():
            if v != None and parameters[k] is not v:
                logger.info("Updating {0} from {1} to {2}".format(k, parameters[k], v))
                parameters[k] = v

    # check required parameters
    # if parameters["dataset"] == 0:
    #    logger.info("-dataset is required.")
    #    sys.exit(1)

    project_params.save_fyp_parameters(parameters)

    return parameters


def frealign_def(fparameters, mparameters, parfile, film, scanor, tolerance):

    # extract relevant lines from current parfile
    # TODO: use Parameters()
    if "frealignx" in fparameters["refine_metric"]:
        input = np.array(
            [
                line.split()
                for line in open(parfile)
                if not line.startswith("C")
                and (int(line[60:65]) == film)
                and (int(line[172:180]) == scanor)
            ],
            dtype=float,
        )
    else:
        input = np.array(
            [
                line.split()
                for line in open(parfile)
                if not line.startswith("C")
                and (int(line[60:65]) == film)
                and (int(line[164:172]) == scanor)
            ],
            dtype=float,
        )

    particles = input.shape[0]

    # reset indexes
    input[:, 0] = np.arange(1, particles + 1, 1)

    name = "%s_T%02d_M%04d" % (fparameters["dataset"], film, scanor)
    # name = '%s_T%02d_M%04d' % ( fparameters['dataset'], film, scanor + 1 )

    res = minimize(
        analysis.scores.eval_phase_residual,
        0,
        args=(mparameters, fparameters, input, name, film, scanor, tolerance,),
        method="nelder-mead",
        tol=1,
        options={
            "xatol": 1e-0,
            "maxiter": 50,
            "fatol": 1e-3,
            "initial_simplex": np.array([0, tolerance]).reshape(2, 1),
            "disp": True,
        },
    )

    minimizer_kwargs = {
        "method": "nelder-mead",
        "args": (mparameters, fparameters, input, name, film, scanor, tolerance,),
        "tol": 1,
    }

    # cleanup
    local_model = os.getcwd() + "/scratch/%s_r01_01.mrc" % (name)
    os.remove(local_model)
    local_stack = "%s_stack.mrc" % (name)
    os.remove(local_stack)

    # save output
    output_file = "scratch/film_%04d_scanor_%04d.defocus" % (film, scanor)
    with open(output_file, "w") as f:
        f.write(str(res.x[0]) + "\n")


def frealign_def_split(fp, parfile, tolerance):

    #  statistics if using v9.11
    if "new" in fp["refine_metric"]:
        com = (
            """grep -A 10000 "C  NO.  RESOL  RING RAD" {0}""".format(parfile)
            + """ | grep -v RESOL | grep -v Average | grep -v Date | grep C | awk '{if ($2 != "") printf "%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f\n", $2, $3, $4, $6, $7, $8, $9}' > """
            + "maps/statistics_r01.txt"
        )
        local_run.run_shell_command(com, verbose=False)

    def_swarm_file = slurm.create_def_swarm_file(fp, parfile, tolerance)

    def_merge_file = slurm.create_def_merge_file(parfile, tolerance)

    # submit jobs to batch system
    id = slurm.submit_jobs(
        "swarm",
        def_swarm_file.replace("swarm/", ""),
        "frd_" + fp["refine_dataset"][-6:],
        fp["slurm_queue"],
        25,
        1,
        10,
        "",
        "",
        "12:00:00",
    )

    # submit merge job
    slurm.submit_jobs(
        "swarm",
        def_merge_file.replace("swarm/", ""),
        "frd_" + fp["refine_dataset"][-6:],
        fp["slurm_queue"],
        0,
        1,
        10,
        "",
        "",
        "12:00:00",
        id,
    )


def frealign_def_merge(fparameters, parfile, tolerance):

    par_obj = frealign_parfile.Parameters.from_file(parfile)

    input = par_obj.data

    films = np.unique(input[:, 7])

    for f in films:

        if "frealignx" in fparameters["refine_metric"]:
            tilts = np.unique(input[input[:, 7] == f][:, 20])
        else:
            tilts = np.unique(input[input[:, 7] == f][:, 19])

        for t in tilts:

            defocus = np.loadtxt(
                "scratch/film_%04d_scanor_%04d.defocus" % (f, t), dtype=float
            )
            if math.fabs(defocus) < 0.9 * tolerance:
                if "frealignx" in fparameters["refine_metric"]:
                    input[
                        np.argwhere(
                            np.logical_and(input[:, 7] == f, input[:, 20] == t)
                        ),
                        8:10,
                    ] += defocus
                else:
                    input[
                        np.argwhere(
                            np.logical_and(input[:, 7] == f, input[:, 19] == t)
                        ),
                        8:10,
                    ] += defocus
                logger.info(f"Adjusting defocus of film {f} tilt {t} by {defocus}")
            # else:
            #    print 'Leaving defocus unchanged: film=', f, ', tilt=', t, 'change=', defocus, ', tolerance=', tolerance

    # save copy of old parameter file
    shutil.move(parfile, parfile + "o")

    # parfile + "o"
    # save new parameter file

    # particles = input.shape[0]

    par_obj.write_file(parfile)

    # with open(parfile, "w") as f:
    #     [f.write(line) for line in open(parfile + "o") if line.startswith("C")]
    #     for i in range(particles):
    #         if "frealignx" in fparameters["metric"]:
    #             f.write(
    #                 frealign_parfile.EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE
    #                 % (tuple(input[i, :]))
    #             )
    #         else:
    #             f.write(
    #                 frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE
    #                 % (tuple(input[i, :]))
    #             )
    #         f.write("\n")

    # cleanup
    [os.remove(f) for f in glob.glob("scratch/film*.defocus")]
    [os.remove(f) for f in glob.glob("scratch/*_r01_02.par_")]

    logger.info("Merged ended successfully.")


def frealign_rec(mparameters, fparameters, iteration, alignment_option):

    # merge parameter files except when skipping alignments

    merge_refinements(mparameters, fparameters, iteration, alignment_option)

    if alignment_option == 2:

        # launch second round of alignments (using current ones as reference)

        # go back to frealign directory
        os.chdir("..")
        logger.info("Now running iteration %i of %s", iteration, fparameters["maxiter"])
        frealign_iterate(mparameters, fparameters, iteration, True)

    else:

        # carry out reconstruction as usual
        classes = int(project_params.param(fparameters["classes"], iteration))

        # re-compute occupancies using LogP values
        if classes > 1 and not alignment_option == 0:
            analysis.occupancies.occupancies(fparameters, iteration, classes)

        dataset = fparameters["dataset"]
        particles = int(
            [
                _f
                for _f in (
                    line.rstrip()
                    for line in open("../{}_01.par".format(dataset))
                    if not line.startswith("C")
                )
                if _f
            ][-1].split()[0]
        )

        # PARALLEL RECONSTRUCTION USING VERSION 9
        if float(project_params.param(fparameters["cutoff"], iteration)) >= 0:

            # input = np.array( [line.split() for line in open( '%s_r01_%02d.par' % ( dataset, iteration ) ) if not line.startswith('C') ], dtype=float )
            input = frealign_parfile.Parameters.from_file(
                "%s_r01_%02d.par" % (dataset, iteration)
            ).data

            # serial reconstruction
            if input.shape[1] == 13:

                mreconstruct(mparameters, fparameters, iteration)

                os.chdir("..")

                # launch refinement and reconstruction for next iteration
                if iteration < int(fparameters["maxiter"]):

                    logger.info(
                        "Now running iteration %i of %s",
                        iteration,
                        fparameters["maxiter"],
                    )
                    frealign_iterate(mparameters, fparameters, iteration + 1)

            # branch out multiple reconstructions
            else:

                os.chdir("..")

                # frealign_rec_merge
                rec_swarm_file = slurm.create_rec_merge_swarm_file(iteration)

                machinefile = "mpirun.mynodes"
                machinerecfile = "mpirun.myrecnodes"

                # Multirun
                if os.path.exists(machinefile) or not "cc" in project_params.param(
                    fparameters["metric"], iteration
                ):
                    # if 'cc' in project_params.param( fparameters['metric'], iteration ) or classes > 1:
                    (
                        machinerecfile,
                        increment,
                        procs,
                    ) = project_params.get_particle_increment_in_rec(
                        fparameters,
                        iteration,
                        particles,
                        classes,
                        dataset,
                        machinefile,
                        machinerecfile,
                    )

                    logger.info(f"We now create multirun file {os.getcwd()}")

                    mpirunfile, count = local_run.create_rec_split_multirun_file(
                        iteration, particles, classes, increment
                    )

                    # override procs with count
                    procs = count

                    scratch_transfer_stack = fparameters["scratch_copy_stack"]

                    if "t" in scratch_transfer_stack.lower():
                        slurm.transfer_stack_to_scratch(dataset)
                    else:
                        logger.info("Directly using stack file\n")

                    # perform frealign_rec_split multirun
                    mpirun = get_mpirun_command()

                    # command = "{0} -machinefile {1} -np {2} {3}/multirun -m {4}/{5} > /dev/null".format(
                    command = "{0} -machinefile {1} -np {2} {3}/multirun -m {4}/{5} > /dev/null".format(
                        mpirun,
                        machinerecfile,
                        procs,
                        get_multirun_path(),
                        os.getcwd(),
                        mpirunfile,
                    )
                    logger.info(command)
                    # flush the output buffers so the command output actually appears after the command
                    sys.stdout.flush()
                    # run the command and wait for it to finish while dumping the output to stdout in real-time
                    # if the process exits with an error code, also throw an exception
                    local_run.run_shell_command(command)

                    # merge reconstructions; frealign_rec_merge
                    command = "cd swarm; export frealign_rec_merge=frealign_rec_merge; . ./{0}".format(
                        rec_swarm_file.replace("swarm/", "")
                    )
                    logger.info(command)
                    # flush the output buffers so the command output actually appears after the command
                    sys.stdout.flush()
                    # run the command and wait for it to finish while dumping the output to stdout in real-time
                    # if the process exits with an error code, also throw an exception
                    local_run.run_shell_command(command)

                    if not machinerecfile == machinefile:
                        try:
                            os.remove(machinerecfile)
                        except:
                            pass

                else:
                    # SWARM

                    # split and dump reconstructions

                    increment, threads = slurm.calculate_rec_swarm_required_resources(
                        mparameters, fparameters, particles
                    )

                    # create frealign_rec_split swarm file
                    rec_split_swarm_file = slurm.create_rec_split_swarm_file(
                        iteration, particles, classes, increment
                    )

                    # submit jobs to batch system
                    id = slurm.submit_jobs(
                        "swarm",
                        rec_split_swarm_file.replace("swarm/", ""),
                        "frs_" + fparameters["dataset"][-6:],
                        fparameters["slurm_queue"],
                        400,
                        threads,
                        58,
                        "",
                        "",
                        "12:00:00",
                    )
                    slurm.submit_jobs(
                        "swarm",
                        rec_swarm_file.replace("swarm/", ""),
                        "frm_" + fparameters["dataset"][-6:],
                        fparameters["slurm_queue"],
                        400,
                        0,
                        58,
                        "",
                        "",
                        "12:00:00",
                        id,
                    )
        else:
            logger.info(
                "Non-positive cutoff threshold: {0}. Skipping reconstruction".format(
                    fparameters["cutoff"]
                )
            )

            # launch next iteration
            if iteration < int(fparameters["maxiter"]):

                shutil.copy2(
                    "../maps/%s_r01_%02d.mrc" % (dataset, iteration - 1),
                    "../maps/%s_r01_%02d.mrc" % (dataset, iteration),
                )

                os.chdir("..")

                logger.info(
                    "Now running iteration %i of %s", iteration, fparameters["maxiter"]
                )
                frealign_iterate(mparameters, fparameters, iteration + 1)


def frealign_rec_merge(mparameters, fparameters, iteration):
    classes = int(project_params.param(fparameters["classes"], iteration))

    # make sure there are no missing reconstructions
    # resubmit if necessary
    rec_merge_check_error_and_resubmit(mparameters, fparameters, iteration)

    # all clear, merge reconstructions
    for ref in range(classes):
        merge_reconstructions(mparameters, fparameters, iteration, ref + 1)

    # if (
    #     not "cc" in project_params.param(fparameters["metric"], iteration)
    #     and classes == 1
    # ):
    #     if False:
    #         logger.info("Removing machine file")
    #         os.remove("../mpirun.mynodes")

    # merge all reconstructions if using weights
    if fparameters["reconstruct_weights"]:

        average = (
            mrc.read(
                "../maps/" + fparameters["dataset"] + "_r%02d_%02d.mrc" % (1, iteration)
            )
            / classes
        )
        for ref in range(1, classes):
            average += (
                mrc.read(
                    "../maps/"
                    + fparameters["dataset"]
                    + "_r%02d_%02d.mrc" % (ref + 1, iteration)
                )
                / classes
            )
        mrc.write(
            average, "../maps/" + fparameters["dataset"] + "_%02d.mrc" % (iteration),
        )

    # collate FSC curves from all references in one plot
    if classes > 1:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, figsize=(8, 8))
        ranking = np.zeros([classes])
        for ref in range(classes):
            fsc_file = (
                "../maps/" + fparameters["dataset"] + "_r%02d_fsc.txt" % (ref + 1)
            )
            FSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)
            ax[0].plot(
                1 / FSCs[:, 0], FSCs[:, iteration - 1], label="r%02d" % (ref + 1),
            )
            ranking[ref] = FSCs[:, iteration - 1].mean()
            par_file = (
                "../maps/"
                + fparameters["dataset"]
                + "_r%02d_%02d.par" % (ref + 1, iteration)
            )
            # input = np.array(
            #     [line.split() for line in open(par_file) if not line.startswith("C")],
            #     dtype=float,
            # )
            input = frealign_parfile.Parameters.from_file(par_file).data
            ax[1].plot(np.sort(input[:, 11])[::-1], label="r%02d" % (ref + 1))
        ax[0].legend(loc="upper right", shadow=True)
        ax[0].set_ylim((-0.1, 1.05))
        ax[0].set_xlim((1 / FSCs[0, 0], 1 / FSCs[-1, 0]))
        dataset = fparameters["dataset"] + "_%02d" % iteration
        ax[0].set_title("%s" % dataset, fontsize=12)
        ax[0].set_xlabel("Frequency (1/A)")
        ax[0].set_ylabel("FSC")
        ax[1].legend(loc="upper right", shadow=True)
        ax[1].set_xlim(0, input.shape[0] - 1)
        ax[1].set_xlabel("Particle Index")
        ax[1].set_ylabel("Occupancy (%)")
        plt.savefig("../maps/%s_classes.png" % dataset)
        plt.close()

        # email result
        user_comm.frealign_rec_merge_notify(
            mparameters, fparameters, dataset, ranking, iteration
        )

    # go back to frealign directory
    os.chdir("..")

    # classification convergence
    converged = rec_merge_check_class_converge(fparameters, classes, iteration)

    # launch next iteration
    if not converged and iteration < int(fparameters["maxiter"]):

        logger.info("Now running iteration %i of %s", iteration, fparameters["maxiter"])
        frealign_iterate(mparameters, fparameters, iteration + 1)

    elif (
        classes < 2
        and int(mparameters["extract_cls"]) == 0
        and mparameters["data_auto"]
    ):

        clean_par_file = (
            os.getcwd()
            + "/scratch/"
            + fparameters["dataset"]
            + "_r01_%02d.par" % (iteration)
        )
        threshold = analysis.statistics.optimal_threshold(
            np.array(
                [
                    line.split()
                    for line in open(clean_par_file)
                    if not line.startswith("C")
                ],
                dtype=float,
            )[:, 14]
        )
        par_file = (
            os.getcwd()
            + "/maps/"
            + fparameters["dataset"]
            + "_r01_%02d.par" % (iteration)
        )
        com = "cd ..; {0}/byp -parfile {1} -extract_cls 1 -threshold {2}; {0}/pyp_main.py -extract_cls 1 -class_par {3} -class_ref {4} -class_num {5}".format(
            os.environ["PYTHONDIR"],
            par_file,
            threshold,
            par_file.replace(".par", "_clean.par"),
            par_file.replace(".par", ".mrc"),
            mparameters["class_num"],
        )
        local_run.run_shell_command(com)


def frealign_iterate(mp, fp, iteration, keep_previous_alignments=False):

    # wait if flag is on
    while os.path.exists("wait"):
        time.sleep(20)

    # reload parameters to pick up changes during a run
    fp = project_params.load_fyp_parameters()

    dataset = fp["refine_dataset"]

    # clean up scratch from previous runs
    null = [
        os.remove(i)
        for i in glob.glob("scratch/*")
        if os.path.isfile(i) and not "pdb" in i
    ]

    # read number of classes
    classes = int(project_params.param(fp["class_num"], iteration))

    frealign_paths = get_frealign_paths()

    if iteration == 2:

        # turn off classification
        classes = 1

        # specify which parameter file and reference to use for this iteration
        if not keep_previous_alignments:
            # use empty parameter file to seed refinement
            # shutil.copy2( '%s_%02d.mrc' % ( dataset, iteration - 1 ), 'maps/%s_r01_%02d.mrc' % ( dataset, iteration - 1 ) )
            initial_model = "maps/%s_r01_%02d.mrc" % (dataset, iteration - 1)
            if os.path.exists(initial_model) or os.path.islink(initial_model):
                os.remove(initial_model)
            symlink_relative(
                os.path.join(os.getcwd(), "%s_%02d.mrc" % (dataset, iteration - 1)), initial_model
            )
            if not os.path.exists("maps/%s_%02d.par" % (dataset, iteration - 1)):
                shutil.copy2(
                    "%s_%02d.par" % (dataset, iteration - 1),
                    "maps/%s_r01_%02d.par" % (dataset, iteration - 1),
                )
        else:
            # use alignments obtained using the cclin metric obtained in the first round of refinements
            shutil.copy2(
                "maps/%s_r01_%02d.par" % (dataset, iteration),
                "maps/%s_r01_%02d.par" % (dataset, iteration - 1),
            )

    # keep track of whether we are using multiple references for the first time
    need_initialize_classification = classes > 1 and not os.path.exists(
        "maps/%s_r02_%02d.mrc" % (dataset, iteration - 1)
    )

    if classes > 1:

        if need_initialize_classification:

            from pyp.system.set_up import initialize_classification

            initialize_classification(mp, fp, iteration, dataset, classes)

        else:

            # re-compute occupancies using LogP values
            occupancy_extended(fp, iteration - 1, classes, "maps")

    # Setup files for refinement
    for ref in range(classes):
        setup_refinement_files(fp, iteration, dataset, frealign_paths, ref)

    # Multiple Alignment Options
    # alignment_option = 0: Skip alignments and go straight to reconstruction (this happens only after running rsample).
    # alignment_option = 1: Do one standard round of alignments (this is the default behaviour during refinement).
    # alignment_option = 2: Do first round of alignments using frealign_v8 (cclin).

    alignment_option = project_params.get_align_option(
        fp, iteration, keep_previous_alignments, dataset, need_initialize_classification
    )

    # Merge results and reconstruct swarm command
    rec_swarm_file = slurm.create_rec_swarm_file(iteration, alignment_option)

    # get total number of particles
    particles = get_particles_from_dataset(dataset)

    metric = project_params.param(fp["refine_metric"], iteration)
    if alignment_option == 2 or "cclin" in project_params.param(
        fp["refine_metric"], iteration
    ):
        metric = "-metric cclin"

    # swarm mode
    machinefile = "mpirun.mynodes"

    if not os.path.isfile(machinefile):

        # skip alignment step if we are only building multiple references for the first time
        if alignment_option > 0:

            # create swarm file
            increment = project_params.get_particle_increment(fp, iteration, particles)

            ref_swarm_file = slurm.create_ref_swarm_file(
                fp, iteration, classes, particles, metric, increment
            )

            # submit jobs to batch system
            id = slurm.submit_jobs(
                "swarm",
                ref_swarm_file.replace("swarm/", ""),
                "frf_" + fp["refine_dataset"][-6:],
                fp["slurm_queue"],
                25,
                1,
                10,
                "",
                "",
                "12:00:00",
            )

        else:

            id = ""

        queue = slurm.get_frealign_queue(mp, fp, iteration)

        slurm.submit_jobs(
            "swarm",
            rec_swarm_file.replace("swarm/", ""),
            "frc_" + fp["refine_dataset"][-6:],
            queue,
            0,
            70,
            690,
            "",
            "",
            "12:00:00",
            id,
        )

    # multirun mode
    else:
        # Alignment
        if alignment_option > 0:
            nodes = len(open(machinefile, "r").read().split())
            if "MYCORES" in os.environ and int(os.environ["MYCORES"]) > 0:
                cores = int(os.environ["MYCORES"])
            else:
                cores = multiprocessing.cpu_count() * nodes
            increment = int(math.ceil(particles / float(cores)))

            # create refine multirun
            mpirunfile = local_run.create_ref_multirun_file(
                iteration, classes, particles, metric, increment, cores
            )

            mpirun = get_mpirun_command()

            # wait for transfer to be done
            stack = fp["refine_dataset"] + "_stack.mrc"
            scratch = Path(os.environ["PYP_SCRATCH"])
            if os.path.isfile(scratch / stack):
                sourcesize = os.path.getsize(stack)
                while sourcesize != os.path.getsize(scratch / stack):
                    time.sleep(5)
                    logger.info(
                        f"waiting for transfer to be done {os.path.getsize(scratch / stack)} of {sourcesize}"
                    )

            if cores < 50:
                local_run.run_shell_command(". {}".format(mpirunfile),)
            else:
                command = "{0} -machinefile {1} -np {2} {3}/multirun -m {4}/{5} > /dev/null".format(
                    mpirun,
                    machinefile,
                    cores,
                    get_multirun_path(),
                    os.getcwd(),
                    mpirunfile,
                )
                command = "( {0} -machinefile {1} -np {2} {3}/multirun -m {4}/{5} )".format(
                    mpirun,
                    machinefile,
                    cores,
                    get_multirun_path(),
                    os.getcwd(),
                    mpirunfile,
                )
                local_run.run_shell_command(command)
                logger.info("%s %s", machinefile, os.path.exists(machinefile))

        # Reconstruction
        command = "cd swarm; export frealign_rec=frealign_rec; . ./{0}".format(
            rec_swarm_file.replace("swarm/", "")
        )
        local_run.run_shell_command(command)


def split_reconstruction(
    mp, first, last, i, ref, count, boff, thresh, dump_intermediate="yes", num_frames=1, run=True
):
    fp = mp

    dataset = fp["refine_dataset"]
    name = dataset + "_r%02d" % (ref)

    parfile = f"../{name}.cistem" # "%s_used.par" % (name)
    reference = parfile.replace(".cistem", ".mrc")

    ranger = "%07d_%07d" % (first, last)

    # TODO - fix this, it takes too much time
    # boff, thresh = mreconstruct_pre(mp, fp, i, ref)

    pixel = float(mp["scope_pixel"]) * float(mp["data_bin"]) * float(mp["extract_bin"])
    dstep = pixel * float(mp["scope_mag"]) / 10000.0

    # do we need to limit the number of threads per reconstruct3d instance?
    cpucount = int(mp["slurm_tasks"])

    os.environ["OMP_NUM_THREADS"] = os.environ["NCPUS"] = "{}".format(cpucount)

    # override resolution with Nyquist frequency
    if float(project_params.param(mp["reconstruct_rrec"], i)) > 0:
        res_rec = project_params.param(mp["reconstruct_rrec"], i)
    else:
        # WARNING: 2.0A is the maximum supported resolution in FREALIGN
        res_rec = max(2.0, 2 * pixel)
        res_rec = 2 * pixel

    if float(project_params.param(mp["reconstruct_radrec"], i)) > 0:
        rad_rec = float(project_params.param(mp["reconstruct_radrec"], i))
    else:
        # produce unmasked reconstruction
        rad_rec = pixel * float(mp["extract_box"]) / 2.0

    # ignore fmatch flag for reconstruction
    fmatch = "F"

    reclogfile = "../log/%s_%s_%s_mreconst_split.log" % (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S"),
        name,
        ranger,
    )
    reclogfile = "../log/%s_%s_mreconst.log" % (name, ranger)
    reclogfile = "%s_%s_mreconst.log" % (name, ranger)
    if os.path.exists("log"):
        reclogfile = reclogfile.replace("../", "")

    stack_dir = "../"
    scratch = "./"

    if os.path.exists(stack_dir + "%s_recstack.mrc" % dataset):
        stack = stack_dir + "%s_recstack.mrc" % dataset
        with open(reclogfile, "w") as f:
            f.write("Using particle stack {0}\n".format(stack))
    elif os.path.exists(Path(os.environ["PYP_SCRATCH"]) / f"{dataset}_stack.mrc"):
        stack = str(Path(os.environ["PYP_SCRATCH"]) / f"{dataset}_stack.mrc")
    else:

        scratch_transfer_stack = mp["reconstruct_scratch_copy_stack"]

        if scratch_transfer_stack:
            stack = "/scratch/%s_stack.mrc" % dataset
        else:
            # stack = stack_dir + "%s_stack.mrc" % dataset
            stack = parfile.split("_r")[0] + "_stack.mrc"

        with open(reclogfile, "w") as f:
            f.write("Using particle stack {0}\n".format(stack))
        # start = time.time()
        # logger.info("starting copying the file over")
        # if not os.path.exists( '/scratch/%s_stack.mrc' % dataset ):
        #     shutil.copy2( stack, '/scratch' )

        # XD: stack file is copied over in frealign_rec

        # end = time.time()
        # time_elapsed = end - start

        # logger.info("{} seconds taken to transfer stack file\n".format(time_elapsed))

        # with open( reclogfile, 'w' ) as f:
        #     f.write("{} seconds taken to transfer stack file\n".format(time_elapsed))

    frealign_paths = get_frealign_paths()

    if "cc" in project_params.param(mp["refine_metric"], i).lower():

        # dumping requires an existing non-empty file for 3D output
        shutil.copy2(
            dataset + "_r%02d_%02d.mrc" % (ref, i - 1), name + "_" + ranger + ".mrc"
        )

        command = """
%s/bin/frealign_v9_mp.exe << eot >>%s 2>&1
M,0,F,F,F,F,%s,T,%s,%s,%s,%s,T,%s,%s                                         !CFORM,IFLAG,FMAG,FDEF,FASTIG,FPART,IEWALD,FBEAUT,FFILT,FBFACT,FMATCH,IFSC,FDUMP,IMEM,INTERP
%s,0.,%s,%s,%s,%s,%s,%s,%s,%s,%s                                        !RO,RI,PSIZE,MW,WGH,XSTD,PBC,BOFF,DANG,ITMAX,IPMAX
%s                                                                        !MASK
%i,%i                                                                        !IFIRST,ILAST
%s                                                                        !ASYM symmetry card
1.0, %s, %s, %s, %s, %s, 0., 0.                                                !RELMAG,DSTEP,TARGET,THRESH,CS,AKV,TX,TY
%s, %s, %s, %s, %s, %s                                                        !RREC,RMIN,RMAX,RCLAS,DFSTD,RBFACT
%s
/dev/null
%s
%s_%s.res
%s_%s_dummy.shft
0., 0., 0., 0., 0., 0., 0., 0.                                                !terminator with RELMAG=0.0
%s_%s.mrc
%s_%s_weights
%s_%s_half1.mrc
%s_%s_half2.mrc
%s_%s_phasediffs
%s_%s_pointspread
eot
""" % (
            frealign_paths["cc3m"],
            reclogfile,
            project_params.param(fp["reconstruct_iewald"], i),
            project_params.param(fp["reconstruct_ffilt"], i),
            project_params.param(fp["reconstruct_fbfact"], i),
            fmatch,
            fp["refine_fboost"],
            project_params.param(fp["refine_imem"], i),
            project_params.param(fp["refine_interp"], i),
            rad_rec,
            pixel,
            mp["particle_mw"],
            mp["scope_wgh"],
            project_params.param(fp["refine_xstd"], i),
            project_params.param(fp["refine_pbc"], i),
            boff,
            project_params.param(fp["refine_dang"], i),
            project_params.param(fp["refine_itmax"], i),
            project_params.param(fp["refine_ipmax"], i),
            project_params.param(fp["refine_mask"], i),
            first,
            last,
            project_params.param(fp["particle_sym"], i),
            dstep,
            project_params.param(fp["refine_target"], i),
            thresh,
            mp["scope_cs"],
            mp["scope_voltage"],
            res_rec,
            project_params.param(fp["refine_rlref"], i),
            postprocess.get_rhref(fp, i),
            project_params.param(fp["class_rhcls"], i),
            project_params.param(fp["refine_dfsig"], i),
            project_params.param(fp["refine_rbfact"], i),
            stack,
            parfile,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
        )

    else:

        # v9.11

        # {1} - first particle
        # {2} - last particle
        # {3} - current iteration
        # {4} - class number
        # {5} - group number

        if fp["reconstruct_norm"]:
            normalize = "yes"
        else:
            normalize = "no"
        if "t" in project_params.param(fp["refine_adjust"], i).lower():
            adjust = "yes"
        else:
            adjust = "no"
        if "t" in project_params.param(fp["refine_crop"], i).lower():
            crop = "yes"
        else:
            crop = "no"
        if fp["refine_invert"]:
            invert = "yes"
        else:
            invert = "no"

        # ensure that scores with 0 are used

        # if thresh < 1:

        # thresh = 0.999
        """
        if (
            "new" in project_params.param(fp["refine_metric"], i).lower()
            and not "frealignx" in project_params.param(fp["refine_metric"], i).lower()
            and not fp["dose_weighting_enable"]
            and not fp["reconstruct_lblur"]
        ):
        """
        if False and "spr" in fp["data_mode"] and not "local" in fp["extract_fmt"] and "new" in project_params.param(fp["refine_metric"], i).lower():

            command = (
                "{0}/bin/reconstruct3d << eot >> {1} 2>&1\n".format(
                    frealign_paths["new"], reclogfile
                )
                + "{0}\n{1}\n{2}_map1.mrc\n{2}_map2.mrc\n{2}.mrc\n{2}_n{3}.res\n".format(
                    stack,
                    Path(os.environ["PYP_SCRATCH"]) / parfile,
                    Path(os.environ["PYP_SCRATCH"]) / name,
                    first,
                )
                + project_params.param(fp["particle_sym"], i)
                + "\n"
                + str(first)
                + "\n"
                + str(last)
                + "\n"
                + str(pixel)
                + "\n"
                + str(mp["scope_voltage"])
                + "\n"
                + str(mp["scope_cs"])
                + "\n"
                + str(mp["scope_wgh"])
                + "\n"
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + str(res_rec)
                + "\n"
                + project_params.param(fp["refine_bsc"], i)
                + "\n"
                + "%f\n" % thresh
                + normalize
                + "\n"
                + adjust
                + "\n"
                + invert
                + "\n"
                + crop
                + "\n"
                + dump_intermediate
                + "\n"
                + "{0}_map1_n{1}.mrc\n{0}_map2_n{1}.mrc\n".format(
                    Path(os.environ["PYP_SCRATCH"]) / name, str(count)
                )
                + "eot\n"
            )

        elif True: #"cistem2" in project_params.param(fp["refine_metric"], i).lower():

            res_ref = "0"
            smoothing = "1"
            padding = "1"
            exclude = "no"
            center = "no"
            if fp["reconstruct_lblur"]:
                blurring = "yes"
            else:
                blurring = "no"
            th_input = "no"

            command = (
                "{0}/reconstruct3d << eot >> {1} 2>&1\n".format(
                    frealign_paths["cistem2"], reclogfile
                )
                + f"{stack}\n{parfile}\n{reference}\n{name}_map1.mrc\n{name}_map2.mrc\noutput.mrc\n{name}_n{first}.res\n"
                + str(project_params.param(fp["particle_sym"], i))
                + "\n"
                + str(first)
                + "\n"
                + str(last)
                + "\n"
                + str(pixel)
                + "\n"
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + str(res_rec)
                + "\n"
                + str(res_ref)
                + "\n"
                + str(project_params.param(fp["refine_bsc"], i))
                + "\n"
                + "%f\n" % thresh
                + smoothing
                + "\n"
                + padding
                + "\n"
                + normalize
                + "\n"
                + adjust
                + "\n"
                + invert
                + "\n"
                + exclude
                + "\n"
                + crop
                + "\n"
                + "yes\n"
                + center
                + "\n"
                + blurring
                + "\n"
                + th_input
                + "\n"
                + dump_intermediate
                + "\n"
                + "{0}_map1_n{1}.mrc\n{0}_map2_n{1}.mrc\n1\n".format(
                    Path(os.environ["PYP_SCRATCH"]) / name, str(count)
                )
                + "eot\n"
            )

        else:
            # frealignx
            # this is for the input reconstruction
            prev_name = dataset + "_r%02d_%02d" % (ref, i - 1)

            # {1} - first particle
            # {2} - last particle
            # {3} - current iteration
            # {4} - class number
            # {5} - group number

            if project_params.param(fp["reconstruct_norm"], i):
                normalize = "yes"
            else:
                normalize = "no"
            if "t" in project_params.param(fp["refine_adjust"], i).lower():
                adjust = "yes"
            else:
                adjust = "no"
            if "t" in project_params.param(fp["refine_crop"], i).lower():
                crop = "yes"
            else:
                crop = "no"
            if fp["refine_invert"]:
                invert = "yes"
            else:
                invert = "no"
            if not fp["refine_score_weighting"]:
                score_weighting = "no"
            else:
                score_weighting = "yes"

            if project_params.param(mp["dose_weighting_enable"], i):
                dose_weighting = "yes"
            else:
                dose_weighting = "no"

            if mp["dose_weighting_multiply"]:
                dose_weighting_multiply = "yes"
            else:
                dose_weighting_multiply = "no"

            if not fp["reconstruct_per_particle_splitting"]:
                per_particle_splitting = "no"
            else:
                per_particle_splitting = "yes"

            if fp["reconstruct_lblur"]:
                blurring = "yes"
            else:
                blurring = "no"

            # add blurring parameters
            if blurring == "yes":
                blurring = blurring + "\n" + str(fp["reconstruct_lblur_start"])
                blurring = blurring + "\n" + str(fp["reconstruct_lblur_step"])
                blurring = blurring + "\n" + str(fp["reconstruct_lblur_nrot"])
                blurring = blurring + "\n" + str(fp["reconstruct_lblur_range"])
            """
            if int(project_params.param(fp["reconstruct_num_frames"], i)) > 1:
                num_frames = int(project_params.param(fp["reconstruct_num_frames"], i))
                # infer number of frames directly from parameter file
            else:
                num_frames = 1
            """
            dose_weighting_fraction = fp["dose_weighting_fraction"]

            dose_weighting_transition = fp["dose_weighting_transition"]

            # if "original" in project_params.param(fp["refine_metric"], i).lower():
            if not fp["dose_weighting_enable"] and not fp["reconstruct_lblur"]:
                command = (
                    "{0}/reconstruct3d_stats << eot >> {1} 2>&1\n".format(
                        frealign_paths["frealignx"], reclogfile
                    )
                    + "{0}\n{1}\n{4}.mrc\n{2}_map1.mrc\n{2}_map2.mrc\n{2}.mrc\n{2}_n{3}.res\n".format(
                        stack,
                        parfile,
                        Path(os.environ["PYP_SCRATCH"]) / name,
                        first,
                        Path(os.environ["PYP_SCRATCH"]) / prev_name,
                    )
                    + project_params.param(fp["particle_sym"], i)
                    + "\n"
                    + str(first)
                    + "\n"
                    + str(last)
                    + "\n"
                    + str(pixel)
                    + "\n"
                    + str(mp["scope_voltage"])
                    + "\n"
                    + str(mp["scope_cs"])
                    + "\n"
                    + str(mp["scope_wgh"])
                    + "\n"
                    + str(mp["particle_mw"])
                    + "\n"
                    + "0\n"
                    + str(rad_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + str(project_params.param(fp["refine_bsc"], i))
                    + "\n"
                    + "%f\n" % thresh
                    + "%0.1f\n" % 1
                    + "%d\n" % 2
                    + normalize
                    + "\n"
                    + adjust
                    + "\n"
                    + invert
                    + "\n"
                    + "no"
                    + "\n"
                    + crop
                    + "\n"
                    + "yes"
                    + "\n"
                    + "no\n"
                    + "no\n"
                    + "no\n"
                    + dump_intermediate
                    + "\n"
                    + "{0}_map1_n{1}.mrc\n{0}_map2_n{1}.mrc\n".format(
                        Path(os.environ["PYP_SCRATCH"]) / name, str(count)
                    )
                    + "eot\n"
                )

            elif "ab" in project_params.param(fp["refine_metric"], i).lower():
                command = (
                    "{0}/reconstruct3d_AB1 << eot >> {1} 2>&1\n".format(
                        frealign_paths["frealignx"], reclogfile
                    )
                    + "{0}\n{1}\n{4}.mrc\n{2}_map1.mrc\n{2}_map2.mrc\n{2}.mrc\n{2}_n{3}.res\n".format(
                        stack,
                        parfile,
                        Path(os.environ["PYP_SCRATCH"]) / name,
                        first,
                        Path(os.environ["PYP_SCRATCH"]) / prev_name,
                    )
                    + project_params.param(fp["particle_sym"], i)
                    + "\n"
                    + str(first)
                    + "\n"
                    + str(last)
                    + "\n"
                    + str(pixel)
                    + "\n"
                    + mp["scope_voltage"]
                    + "\n"
                    + mp["scope_cs"]
                    + "\n"
                    + mp["scope_wgh"]
                    + "\n"
                    + mp["particle_mw"]
                    + "\n"
                    + "0\n"
                    + str(rad_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + project_params.param(fp["refine_bsc"], i)
                    + "\n"
                    + score_weighting
                    + "\n"
                    + dose_weighting
                    + "\n"
                    + str(num_frames)
                    + "\n"
                    + str(dose_weighting_fraction)
                    + "\n"
                    + str(dose_weighting_transition)
                    + "\n"
                    + "%f\n" % thresh
                    + "%0.1f\n" % 1.0
                    + "%d\n" % 2
                    + normalize
                    + "\n"
                    + adjust
                    + "\n"
                    + invert
                    + "\n"
                    + "no"
                    + "\n"
                    + crop
                    + "\n"
                    + "yes"
                    + "\n"
                    + "no\n"
                    + blurring
                    + "\n"
                    + "no\n"
                    + dump_intermediate
                    + "\n"
                    + "{0}_map1_n{1}.mrc\n{0}_map2_n{1}.mrc\n".format(
                        Path(os.environ["PYP_SCRATCH"]) / name, str(count)
                    )
                    + "eot\n"
                )

            elif fp["reconstruct_lblur"]:
                blurring = "Yes"
                external_weight = "/scratch/not_provided"
                command = (
                    "{0}/reconstruct3d_stats_DW << eot >> {1} 2>&1\n".format(
                        frealign_paths["frealignx"], reclogfile
                    )
                    + "{0}\n{1}\n{4}.mrc\n{2}_map1.mrc\n{2}_map2.mrc\n{2}.mrc\n{2}_n{3}.res\n".format(
                        stack,
                        parfile,
                        Path(os.environ["PYP_SCRATCH"]) / name,
                        first,
                        Path(os.environ["PYP_SCRATCH"]) / prev_name,
                    )
                    + project_params.param(fp["particle_sym"], i)
                    + "\n"
                    + str(first)
                    + "\n"
                    + str(last)
                    + "\n"
                    + str(pixel)
                    + "\n"
                    + str(mp["scope_voltage"])
                    + "\n"
                    + str(mp["scope_cs"])
                    + "\n"
                    + str(mp["scope_wgh"])
                    + "\n"
                    + str(mp["particle_mw"])
                    + "\n"
                    + "0\n"
                    + str(rad_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + project_params.param(fp["refine_bsc"], i)
                    + "\n"
                    + score_weighting
                    + "\n"
                    + dose_weighting
                    + "\n"
                    + external_weight
                    + "\n"
                    + dose_weighting_multiply
                    + "\n"
                    + per_particle_splitting
                    + "\n"
                    + str(num_frames)
                    + "\n"
                    + str(dose_weighting_fraction)
                    + "\n"
                    + str(dose_weighting_transition)
                    + "\n"
                    + "%f\n" % thresh
                    + "%0.1f\n" % 1
                    + "%d\n" % 2
                    + normalize
                    + "\n"
                    + adjust
                    + "\n"
                    + invert
                    + "\n"
                    + "no"
                    + "\n"
                    + crop
                    + "\n"
                    + "yes"
                    + "\n"
                    + "no\n"
                    + blurring
                    + "\n"
                    + "no\n"
                    + dump_intermediate
                    + "\n"
                    + "{0}_map1_n{1}.mrc\n{0}_map2_n{1}.mrc\n".format(
                        Path(os.environ["PYP_SCRATCH"]) / name, str(count)
                    )
                    + "eot\n"
                )

            else:
                # if dose weighting is enabled, we will go into this block
                weight_files = project_params.resolve_path(mp["dose_weighting_weights"]) if "dose_weighting_weights" in mp else ""
                external_weight = "/scratch/not_provided"

                if ".txt" in weight_files:
                    tag = "_" + name.split("_")[1] + "_" if "*" in weight_files else "txt"
                    files = [f for f in glob.glob(weight_files) if tag in f]
                    external_weight = files[0]

                command = (
                    "{0}/reconstruct3d_stats_DW << eot >> {1} 2>&1\n".format(
                        frealign_paths["frealignx"], reclogfile
                    )
                    + "{0}\n{1}\n{4}.mrc\n{2}_map1.mrc\n{2}_map2.mrc\n{2}.mrc\n{2}_n{3}.res\n".format(
                        stack,
                        parfile,
                        Path(os.environ["PYP_SCRATCH"]) / name,
                        first,
                        Path(os.environ["PYP_SCRATCH"]) / prev_name,
                    )
                    + project_params.param(fp["particle_sym"], i)
                    + "\n"
                    + str(first)
                    + "\n"
                    + str(last)
                    + "\n"
                    + str(pixel)
                    + "\n"
                    + str(mp["scope_voltage"])
                    + "\n"
                    + str(mp["scope_cs"])
                    + "\n"
                    + str(mp["scope_wgh"])
                    + "\n"
                    + str(mp["particle_mw"])
                    + "\n"
                    + "0\n"
                    + str(rad_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + str(res_rec)
                    + "\n"
                    + project_params.param(fp["refine_bsc"], i)
                    + "\n"
                    + score_weighting
                    + "\n"
                    + dose_weighting
                    + "\n"
                    + external_weight
                    + "\n"
                    + dose_weighting_multiply
                    + "\n"
                    + per_particle_splitting
                    + "\n"
                    + str(num_frames)
                    + "\n"
                    + str(dose_weighting_fraction)
                    + "\n"
                    + str(dose_weighting_transition)
                    + "\n"
                    + "%f\n" % thresh
                    + "%0.1f\n" % 1
                    + "%d\n" % 2
                    + normalize
                    + "\n"
                    + adjust
                    + "\n"
                    + invert
                    + "\n"
                    + "no"
                    + "\n"
                    + crop
                    + "\n"
                    + "yes"
                    + "\n"
                    + "no\n"
                    + "no"
                    + "\n"
                    + "no\n"
                    + dump_intermediate
                    + "\n"
                    + "{0}_map1_n{1}.mrc\n{0}_map2_n{1}.mrc\n".format(
                        Path(os.environ["PYP_SCRATCH"]) / name, str(count)
                    )
                    + "eot\n"
                )

    os.environ["OMP_NUM_THREADS"] = os.environ["NCPUS"] = "{}".format(cpucount)

    # run job
    if run:
        if fp["slurm_verbose"]:
            logger.info(command)
        with open(reclogfile, "a") as f:
            f.write(command)
        subprocess.Popen(command, shell=True, text=True).wait()

    return command


def local_merge_reconstruction(name=""):
    """ Locally merge multiple intermediate reconstructions into one to avoid moving all of them to project folder 
        Aim to speed up data transfer, (de)compression and global merge during cspmerge 

        This function searches all files that have keywords in filenames (map1_n*, map2_n*) in the current directory and performs intermediate merge
    """
    logfile = "local_merge3d.log"
    frealign_paths = get_frealign_paths()

    output_1 = "dumpfile_map1.mrc"
    output_2 = "dumpfile_map2.mrc"
    dummy_name = "temp"

    # intermediate reconsutructions should be named in sequence from 1 to N
    map_1 = "(\w+)_map1_n[0-9]+\.mrc$" if len(name) == 0 else f"(\w+){name}_map1_n[0-9]+\.mrc$"
    r_1 = re.compile(map_1)
    map_2 = "(\w+)_map2_n[0-9]+\.mrc$" if len(name) == 0 else f"(\w+){name}_map2_n[0-9]+\.mrc$"
    r_2 = re.compile(map_2)

    sort_by_n = lambda x: int(x.replace(".mrc", "").split("_")[-1].replace("n", ""))
    sort_by_name = lambda x: x.replace(".mrc", "").split("_map")[0]
    matchfiles_1 = sorted(list(filter(r_1.match, os.listdir("."))), key=sort_by_n)
    matchfiles_2 = sorted(list(filter(r_2.match, os.listdir("."))), key=sort_by_n)
    matchfiles_1 = sorted(matchfiles_1, key=sort_by_name)
    matchfiles_2 = sorted(matchfiles_2, key=sort_by_name)

    # do not perform merge if only one dumpfile is present
    if len(matchfiles_1) == 1:
        return 1

    assert (len(matchfiles_1) > 0 and len(matchfiles_1) == len(matchfiles_2)), f"Number of intermediate reconstructions does not match (half1: {len(matchfiles_1)} != half2: {len(matchfiles_2)})"

    renamed_1 = [ f"{dummy_name}_map1_n{idx+1}.mrc" for idx, f in enumerate(matchfiles_1) ]
    renamed_2 = [ f"{dummy_name}_map2_n{idx+1}.mrc" for idx, f in enumerate(matchfiles_2) ]

    [ os.rename(f, newf) for f, newf in zip(matchfiles_1, renamed_1) ]
    [ os.rename(f, newf) for f, newf in zip(matchfiles_2, renamed_2) ] 

    seed = renamed_1[0].replace("n1", "n")

    command = (
        "{0}/local_merge3d << eot >> {1} 2>&1\n".format(
            frealign_paths["cistem2"], logfile
        )
        + output_1 + "\n"
        + output_2 + "\n"
        + seed + "\n"
        + seed.replace("map1", "map2") + "\n"
        + f"{str(len(matchfiles_1))}\n"
        + "eot\n"
    )

    local_run.run_shell_command(command, verbose=False)

    assert (os.path.exists(output_1)), "Local merge3d failed, stopping"
    # remove previous intermediates reconstruction 
    for dump_1, dump_2 in zip(renamed_1, renamed_2):
        os.remove(dump_1)
        os.remove(dump_2)
    # rename the outputs 
    os.rename(output_1, matchfiles_1[0])
    os.rename(output_2, matchfiles_2[0])

    assert (matchfiles_1[0].replace("map1", "map2") == matchfiles_2[0]), f"{matchfiles_1[0]} cannot pair with {matchfiles_2[0]}"

    return len(matchfiles_1)



@timer.Timer(
    "merge3d", text="Merge3d took: {}", logger=logger.info
)
def merge_reconstructions(mp, i, ref):

    """Merge dumped reconstructions in frealign.v9.08.

    """

    fp = mp

    pixel = float(mp["scope_pixel"]) * float(mp["data_bin"]) * float(mp["extract_bin"])
    dstep = pixel * float(mp["scope_mag"]) / 10000.0
    thresh = 90.0
    dataset = fp["refine_dataset"] + "_r%02d" % ref
    classes = fp["class_num"]
    scratch = "./"

    name = dataset + "_%02d" % i

    files = sorted(glob.glob(name + "*_n*.mrc"))

    reclogfile = "../log/%s_mreconst.log" % (name)
    frealign_paths = get_frealign_paths()

    if "cc" in project_params.param(fp["refine_metric"], i).lower():

        command = """
%s/bin/merge_3d_mp.exe << eot >>%s 2>&1
%i
%s
%s.res
%s.mrc
%s_weights
%s_half1.mrc
%s_half2.mrc
%s_phasediffs
%s_pointspread
eot
""" % (
            frealign_paths["cc3m"],
            reclogfile,
            len(files),
            "\n".join([scratch + j for j in files]),
            scratch + name,
            scratch + name,
            scratch + name,
            scratch + name,
            scratch + name,
            scratch + name,
            scratch + name,
        )

    elif False and (
        "new" in project_params.param(fp["refine_metric"], i).lower()
        and "spr" in mp["data_mode"] and (not "local" in mp["extract_fmt"])
    ):

        # v9.11
        if float(project_params.param(fp["reconstruct_radrec"], i)) > 0:
            rad_rec = float(project_params.param(fp["reconstruct_radrec"], i))
        else:
            # produce unmasked reconstruction
            rad_rec = pixel * float(mp["extract_box"]) / 2.0

        if True: # "new" in project_params.param(fp["refine_metric"], i).lower():

            command = (
                "{0}/bin/merge3d << eot >> {1} 2>&1\n".format(
                    frealign_paths["new"], reclogfile
                )
                + "{0}_half1.mrc\n{0}_half2.mrc\n{0}.mrc\n{0}_statistics.txt\n".format(
                    name
                )
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + "{0}_map1_n.mrc\n{0}_map2_n.mrc\n".format(
                    Path(os.environ["PYP_SCRATCH"]) / name
                )
                + "eot\n"
            )

        else:

            num_dump = len(
                [
                    f
                    for f in os.listdir(os.environ["PYP_SCRATCH"])
                    if f.startswith(name + "_map1_n")
                ]
            )

            command = (
                "{0}/merge3d << eot >> {1} 2>&1\n".format(
                    frealign_paths["cistem2"], reclogfile
                )
                + "{0}_half1.mrc\n{0}_half2.mrc\n{0}.mrc\n{0}_statistics.txt\n".format(
                    name
                )
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + "{0}_map1_n.mrc\n{0}_map2_n.mrc\n{1}\n".format(
                    Path(os.environ["PYP_SCRATCH"]) / name, str(num_dump)
                )
                + "eot\n"
            )

    else:
        # number of dump files!
        num_dump = len(
            [
                f
                for f in os.listdir(os.environ["PYP_SCRATCH"])
                if f.startswith(dataset + "_map1_n")
            ]
        )

        # frealignx
        if float(project_params.param(fp["reconstruct_radrec"], i)) > 0:
            rad_rec = float(project_params.param(fp["reconstruct_radrec"], i))
        else:
            # produce unmasked reconstruction
            rad_rec = pixel * float(mp["extract_box"]) / 2.0

        # XD added
        if int(project_params.param(fp["reconstruct_num_frames"], i)) > 1:
            num_frames = int(project_params.param(fp["reconstruct_num_frames"], i))
        else:
            num_frames = 1

        if fp["refine_merge_normalize"]:
            merge_normalize = "yes"
        else:
            merge_normalize = "no"

        # check which binary to use

        if "xd" in project_params.param(fp["refine_metric"], i).lower():
            command = (
                "{0}/merge3d_XD << eot >> {1} 2>&1\n".format(
                    frealign_paths["frealignx"], reclogfile
                )
                + "{0}_half1.mrc\n{0}_half2.mrc\n{0}.mrc\n{0}_statistics.txt\n".format(
                    name
                )
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + "{0}_map1_n.mrc\n{0}_map2_n.mrc\n".format("../scratch/" + name)
                + str(num_dump)
                + "\n"
                + str(num_frames)
                + "\n"
                + merge_normalize
                + "\n"
                + "eot\n"
            )

        #elif "original" in project_params.param(fp["refine_metric"], i).lower():
        elif True:
            command = (
                "{0}/merge3d << eot >> {1} 2>&1\n".format(
                    frealign_paths["cistem2"], reclogfile
                )
                + "{0}_half1.mrc\n{0}_half2.mrc\n{0}.mrc\n{0}_statistics.txt\n".format(
                    name
                )
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + "{0}_map1_n.mrc\n{0}_map2_n.mrc\n".format(
                    Path(os.environ["PYP_SCRATCH"]) / dataset
                )
                + str(num_dump)
                + "\n"
                + "eot\n"
            )

        else:
            command = (
                "{0}/merge3d_XD << eot >> {1} 2>&1\n".format(
                    frealign_paths["frealignx"], reclogfile
                )
                + "{0}_half1.mrc\n{0}_half2.mrc\n{0}.mrc\n{0}_statistics.txt\n".format(
                    name
                )
                + str(mp["particle_mw"])
                + "\n"
                + "0\n"
                + str(rad_rec)
                + "\n"
                + "{0}_map1_n.mrc\n{0}_map2_n.mrc\n".format(
                    Path(os.environ["PYP_SCRATCH"]) / name
                )
                + str(num_dump)
                + "\n"
                + str(num_frames)
                + "\n"
                + merge_normalize
                + "\n"
                + "eot\n"
            )

    #             command = \
    # '{0}/frealignx/merge3d_AB << eot >> {1} 2>&1\n'.format( os.environ['PYP_DIR'], reclogfile ) + \
    # '{0}_half1.mrc\n{0}_half2.mrc\n{0}.mrc\n{0}_statistics.txt\n'.format( name ) + \
    # mp['particle_mw'] + '\n' + \
    # '0\n' + \
    # str(rad_rec) + '\n' + \
    # '{0}_map1_n.mrc\n{0}_map2_n.mrc\n'.format( os.environ['PYP_SCRATCH'] + name ) + \
    # str(num_dump) + '\n' + \
    # str(num_frames) + '\n' + \
    # 'eot\n'

    os.environ["OMP_NUM_THREADS"] = os.environ["NCPUS"] = "{}".format(
        min(math.floor(35/classes), math.floor(multiprocessing.cpu_count()/classes))
    )
    with open(reclogfile, "a") as f:
        f.write(command)
    if mp["slurm_verbose"]:
        logger.info(command)
    subprocess.Popen(command, shell=True, text=True).wait()
    
    # cleanup
    # os.remove( scratch + name + '.res' )
    
    if mp["slurm_verbose"]:
        with open(reclogfile) as log:
            logger.info(log.read())

    try:
        os.remove(scratch + name + "_weights")
    except:
        pass
    [os.remove(t) for t in glob.glob("%s_???????_???????.*" % name)]

    # beam tilt refinement
    if (
        fp["refine_beamtilt"]
        and "cistem" in project_params.param(fp["refine_metric"], i).lower()
    ):
        refine_ctf(mp, i, ref)

    # collate results
    mreconstruct_post(mp, mp, i, ref, scratch, reclogfile)

    # merge log files
    logfilelist = sorted(glob.glob("../log/%s_???????_???????_mreconst.log" % (name)))
    # print logfilelist
    with open(reclogfile, "a") as l:
        for log in logfilelist:
            l.write("".join([line for line in open(log)]))

    if mp["slurm_verbose"]:
        with open(reclogfile) as log:
            logger.info(log.read())

    # remove individual log files
    null = [os.remove(i) for i in logfilelist if os.path.isfile(i)]

    # final cleanup
    null = [
        os.remove(f)
        for f in glob.glob(str(Path(os.environ["PYP_SCRATCH"]) / f"{name}_*"))
        if not f.endswith("_stack.mrc") and not f.endswith("_statistics.txt")
    ]


def mreconstruct(mp, fp, i):

    classes = int(project_params.param(fp["class_num"], i))

    for ref in range(classes):

        dataset = fp["refine_dataset"] + "_r%02d" % (ref + 1)

        parfile = "%s_%02d.par" % (dataset, i)

        boff, thresh = mreconstruct_pre(mp, fp, i, ref + 1)

        first = 1
        last = len([line for line in open(parfile) if not line.startswith("C")])

        pixel = (
            float(mp["scope_pixel"]) * float(mp["data_bin"]) * float(mp["extract_bin"])
        )

        dstep = pixel * float(mp["scope_mag"]) / 10000.0

        os.environ["OMP_NUM_THREADS"] = os.environ["NCPUS"] = "{}".format(
            multiprocessing.cpu_count()
        )

        name = dataset + "_%02d" % i

        # override resolution with Nyquist frequency
        if float(project_params.param(fp["reconstruct_rrec"], i)) > 0:
            res_rec = project_params.param(fp["reconstruct_rrec"], i)
        else:
            # WARNING: 2.0A is the maximum supported resolution in FREALIGN
            res_rec = max(2.0, 2 * pixel)
            res_rec = 2 * pixel

        if float(project_params.param(fp["reconstruct_radrec"], i)) > 0:
            rad_rec = float(project_params.param(fp["reconstruct_radrec"], i))
        else:
            # produce unmasked reconstruction
            rad_rec = pixel * float(mp["extract_box"]) / 2.0

        # ignore fmatch flag for reconstruction
        fmatch = "F"

        reclogfile = "../log/%s_%s_mreconst.log" % (
            datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S"),
            name,
        )

        scratch = Path(os.environ["PYP_SCRATCH"]) / dataset

        if not os.path.isdir(scratch):
            scratch = Path("../scratch")

        stack = "%s_stack.mrc" % dataset
        if os.path.isfile(scratch / stack):
            # set local scratch directory
            stack_dir = scratch
            # wait for transfer to be done
            sourcesize = os.path.getsize(Path("..") / stack)
            while sourcesize != os.path.getsize(scratch / stack):
                time.sleep(5)
                logger.info("waiting for transfer to be done..")
        else:
            stack_dir = Path("..")

        # copy reconstruction from maps directory if using a mask or generating matches
        # if project_params.param(fp['xstd'],i) is not '0' or 't' in project_params.param(fp['fmatch'],i).lower() and project_params.param( fp['mask'], i ) != '0,0,0,0,0':
        if (
            project_params.param(fp["refine_xstd"], i) != "0"
            or "t" in project_params.param(fp["refine_fmatch"], i).lower()
        ):
            shutil.copy2(
                Path("../maps") / f"{dataset}_{(i - 1):02d}.mrc",
                scratch / f"{name}.mrc",
            )

        # rad_rec = mp['particle_rad']
        command = mreconstruct_version(
            mp,
            fp,
            i,
            ref + 1,
            reclogfile,
            fmatch,
            rad_rec,
            pixel,
            boff,
            first,
            last,
            dstep,
            thresh,
            res_rec,
            str(stack_dir),  # TODO: safe to pass Path here?
            str(scratch),  # TODO: safe to pass Path here?
        )

        # print command
        with open(reclogfile, "w") as f:
            f.write(command)
        subprocess.Popen(command, shell=True, text=True).wait()

        # collate results
        mreconstruct_post(
            mp, fp, i, ref + 1, str(scratch), reclogfile
        )  # TODO: safe to pass Path here?


def get_phase_residuals(input,parfile,fp,i):

    # new vs. old style par files
    if input.shape[1] > 13:
        # prs = sorted( [ line[121:128] for line in open( parfile) if not line.startswith('C') and np.isfinite( float(line[121:128]) ) ] )
        if input.shape[1] > 45:
            prs = sorted(
                [
                    float(line[129:136])
                    for line in open(parfile)
                    if not line.startswith("C")
                    and np.isfinite(float(line[129:136]))
                    and float(line[100:107]) > 0
                ]
            )
        else:
            prs = sorted(
                [
                    float(line[121:128])
                    for line in open(parfile)
                    if not line.startswith("C")
                    and np.isfinite(float(line[121:128]))
                    and float(line[92:99]) > 0
                ]
            )
    else:
        prs = sorted(
            [
                float(line[88:94])
                for line in open(parfile)
                if not line.startswith("C") and np.isfinite(float(line[88:94]))
            ]
        )
    return prs

 
@timer.Timer(
    "mreconstruct_pre", text="Reconstruction pre-calculations took: {}", logger=logger.info
)
def mreconstruct_pre(mp, fp, i, ref=1):

    # if "cc" in project_params.param(fp["refine_metric"], i).lower():
    #     parfile = "%s_r%02d_%02d.par" % (fp["refine_dataset"], ref, i)
    # else:
    #     parfile = "%s_r%02d_%02d_used.par" % (fp["refine_dataset"], ref, i)

    # if int(project_params.param(fp["class_num"], i)) > 1:
    #     cutoff = 1
    # else:
    #     cutoff = float(project_params.param(fp["reconstruct_cutoff"], i))

    # # use the middle point between the cutoff and the minimum PR value as BOFF
    # input = frealign_parfile.Parameters.from_file(parfile).data

    # prs = get_phase_residuals(input,parfile,fp,i)

    # if len(prs) > 0:
    #     minpr = float(prs[0])
    #     maxpr = float(prs[-1])
    #     if cutoff < 1 and cutoff > 0:
    #         thresh = prs[min(int(len(prs) *(1 - cutoff)), len(prs) - 1)]
    #     elif cutoff == 1: 
    #         thresh = max(0, minpr)
    #     elif cutoff == 0:
    #         thresh = 0
    # else:
    #     minpr = maxpr = 90
    #     thresh = 90

    # if float(project_params.param(fp["refine_boff"], i)) == 0:
    #     boff = (maxpr + minpr) / 2.0
    # else:
    #     boff = float(project_params.param(fp["refine_boff"], i))

    # # OVERRIDE (input .par file is already shaped by the cutoff)
    # if input.shape[1] > 13:
    #     thresh = max(0, minpr)
    #     if (
    #         "new" in project_params.param(fp["refine_metric"], i)
    #         or "frealignx" in project_params.param(fp["refine_metric"], i)
    #     ) and thresh < 1:
    #         thresh = 0
    # FIXME
    boff = 0.0

    thresh = 0
    return boff, thresh

def build_map_montage( map_file, radius, output ):

    rec = mrc.read(map_file)
    z = rec.shape[0]

    # what if radius is larger than box
    if radius > z / 2:
        logger.warning("Particle radius falls outside box %f > %f", radius, z / 2)
        radius = z / 2 - 1

    #
    lim = int(z / 2 - radius)
    lim = 1 if lim == 0 else lim
    # lim = int(z / 2 * (1 - 0.75))
    nz = z - 2 * lim
    montage = np.zeros([nz * 2, nz * 3])

    # 2D central slices
    i = 0
    j = 0
    I = rec[z // 2, lim:-lim, lim:-lim]
    montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (I - I.mean()) / I.std()
    j = 1
    I = rec[lim:-lim, z // 2, lim:-lim]
    montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (I - I.mean()) / I.std()
    j = 2
    I = rec[lim:-lim, lim:-lim, z // 2]
    montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (I - I.mean()) / I.std()

    # 2D projections
    i = 1
    for j in range(3):
        try:
            I = np.average(rec[lim:-lim, lim:-lim, lim:-lim], j)
            montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = (
                I - I.mean()
            ) / I.std()
        except:
            I = np.average(rec[lim:-lim, lim:-lim, lim:-lim], j)
            montage[i * nz : (i + 1) * nz, j * nz : (j + 1) * nz] = I - I.mean()

    writepng(montage, output )
    return lim

def mreconstruct_post(mp, fp, i, ref, scratch, reclogfile):

    dataset = fp["refine_dataset"] + "_r%02d" % ref
    name = dataset + "_%02d" % i

    prev_name = dataset + "_%02d" % (i - 1)
    prev_mrc_path = prev_name + ".mrc"
    new_mrc_path = "../maps/" + name + ".mrc"
    same_ref = fp["refine_same_ref"]

    if fp["denoise_enable"]:
        try:
            # increase the contrast of virus volume
            volume = scratch + name
            if fp["denoise_method"] == "imod-nad":
                iterations = fp["denoise_iters"] 
                command = f"{get_imod_path()}/bin/nad_eed_3d -n {iterations} {volume}.mrc {volume}_denoised.mrc".format(
                )
                local_run.run_shell_command(command,verbose=fp["slurm_verbose"])
            elif False:
                sigma = fp["denoise_sigma"]
                nsearch = fp["denoise_nsearch"]
                patchsize = fp["denoise_patch_size"]
                command = "{0}/pyp/postprocess/pyp_bm4d.py -input {1} -output {2} -sigma {3} -nsearch {4} -patch_size {5}".format(
                    os.environ["PYP_DIR"],
                    scratch + name + ".mrc",
                    scratch + name + "_denoised.mrc",
                    sigma,
                    nsearch,
                    patchsize,
                )
                local_run.run_shell_command(command)
            elif fp["denoise_method"] == "bm4d":
                with timer.Timer(
                    "BM4D denoising", text = "BM4D denoising took: {}", logger=logger.info
                ):
                    # use new python implementation of BM4D
                    from bm4d import bm4d
                    input = mrc.read(volume + ".mrc")
                    header = mrc.readHeaderFromFile(volume + ".mrc")
                    output = bm4d(input,fp["denoise_sigma"])
                    mrc.write(output,volume + "_denoised.mrc",header=header)
            else:
                logger.error("Unknown denoising method")

            # save raw map to maps directory
            shutil.copy2(volume + ".mrc", "../maps/" + name + "_raw.mrc")
            # overwrite with denoised volume
            shutil.copy2(volume + "_denoised.mrc", volume + ".mrc")
        except:
            logger.error("Denoising failed")
            pass

    # copy reconstruction to maps directory
    if same_ref:
        if os.path.exists(new_mrc_path):
            os.remove(new_mrc_path)
        logger.info("Using symlink to map from previous iteration")
        symlink_relative(prev_mrc_path, new_mrc_path)
    else:
        shutil.copy2(scratch + name + ".mrc", "../maps")

    """
    # remove PSHIFT column if only doing dose-weighting
    if (
        fp["dose_weighting_enable"]
        and "new" in project_params.param(fp["refine_metric"], i).lower()
    ):
        scratch_parfile = "%s.par" % name
        parfile = "../maps/%s.par" % name
        shutil.copy2(scratch_parfile , parfile)

        # read parfile
        comments = [line for line in open(parfile) if line.startswith("C")]
        lines = [line for line in open(parfile) if not line.startswith("C")]

        # overwrite with new one
        with open(parfile, "w") as f:

            [f.write(line) for line in comments]

            for l in lines:
                f.write(l[:92] + l[100:])
    """
    # copy resolution table to maps directory

    # v9.10
    if fp["refine_fssnr"] and os.path.exists(scratch + name + ".res"):
        shutil.copy2(scratch + name + ".res", "../maps")

        # com='cat %s >> %s' % ( scratch + name + '.res', '../maps/' + name + '.par' )
        # commands.getoutput(com)

        # write FREALIGN header to temp file
        com = 'grep -A 2 "C FREALIGN" %s > %s' % (
            "../maps/" + name + ".par",
            scratch + name + ".part",
        )
        try:
            # Note: grep will return an error if the output is empty
            local_run.run_shell_command(com, verbose=False)
        except:
            pass
        # append resolution table to temp file
        com = "cat %s >> %s" % (scratch + name + ".res", scratch + name + ".part")
        try:
            local_run.run_shell_command(com, verbose=False)
        except:
            pass
        # copy body of parameter file
        com = "grep -v C %s >> %s" % (
            "../maps/" + name + ".par",
            scratch + name + ".part",
        )
        try:
            local_run.run_shell_command(com, verbose=False)
        except:
            pass
        # replace original par file with modified file
        com = "mv %s %s" % (scratch + name + ".part", "../maps/" + name + ".par")
        local_run.run_shell_command(com, verbose=False)

    # v9.11
    # if fp["refine_fssnr"] and os.path.exists(scratch + name + "_statistics.txt"):
    #     com = """sed -i -e 's/RING RAD/RING_RAD/g' """ + "../maps/" + name + ".par"
    #     local_run.run_shell_command(com, verbose=False)

    #     if not "cistem" in project_params.param(fp["refine_metric"], i).lower():
    #         # read par file
    #         #current_par_file = "../maps/{0}.par".format(name)
    #         current_par_file = "{0}.par".format(name)
    #         with open(current_par_file) as f:
    #             all_lines = f.read().split("\n")

    #         # remove existing resolution table
    #         with open(current_par_file, "w") as f:
    #             for line in range(len(all_lines) - 1):
    #                 if line < 7 or all_lines[line][0] != "C":
    #                     f.write(all_lines[line] + "\n")

    #     com = """echo "C" >> """ + "../maps/" + name + ".par"
    #     local_run.run_shell_command(com, verbose=False)
    #     if os.path.exists(scratch + name + "_n1.res"):
    #         com = "grep C %s >> %s" % (
    #             scratch + name + "_n1.res",
    #             "../maps/" + name + ".par",
    #         )
    #         local_run.run_shell_command(com, verbose=False)
    #     com = (
    #         """echo "C                                                 sqrt       sqrt" >> """
    #         + "../maps/"
    #         + name
    #         + ".par"
    #     )
    #     local_run.run_shell_command(com, verbose=False)
    #     com = (
    #         """echo "C  NO.  RESOL  RING RAD   FSPR    FSC  Part_FSC  Part_SSNR  Rec_SSNR       CC   EXP. C    SIG C  ERFC  TOTVOX" >> """
    #         + "../maps/"
    #         + name
    #         + ".par"
    #     )
    #     local_run.run_shell_command(com, verbose=False)
    #     com = (
    #         "grep -v C "
    #         + "{0}_statistics.txt".format(scratch + name)
    #         + """ | awk '{printf "C%4d%8.2f%10.4f%7.2f%7.3f%10.3f%11.4f%10.2f%9.4f%9.4f%9.4f%6.2f%8d\\n", $1, $2, $3, 0.0, $4, $5, $6, $7, 0.0, 0.0, 0.0, 0.0, 0}' >> """
    #         + "../maps/"
    #         + name
    #         + ".par"
    #     )
    #     local_run.run_shell_command(com, verbose=False)
    #     com = """echo "C  Averages not calculated" >> """ + "../maps/" + name + ".par"
    #     local_run.run_shell_command(com, verbose=False)

    # RELION postprocessing plots
    # get current pixel size
    pixel_size = float(mp["scope_pixel"]) * float(mp["extract_bin"])

    # keep track of two half maps
    # if int(project_params.param(fp["class_num"], i)) < 2:
    if True:
        shutil.move(scratch + name + "_half1.mrc", "../maps/" + dataset + "_half1.mrc")
        shutil.move(scratch + name + "_half2.mrc", "../maps/" + dataset + "_half2.mrc")

    # GENERATE GRAPHICAL OUTPUTS

    # FSC plots

    # TODO: put these plotting fns in another script

    # extract FSC curve produced internally by FREALIGN
    with open(reclogfile, "r") as f:
        A = f.read()

    # locate FSC table within log file
    if "cc" in project_params.param(fp["refine_metric"], i).lower():
        Afsc = A[A.find("TOTVOX") + 7 : A.find("C  Average") - 1]
        rows = len(Afsc.split("\n"))
        cols = len(Afsc.split()) // rows
        # column 2 has the frequencies and column 5 the FSC values
        current_fsc = (
            np.array(Afsc.split())
            .reshape((rows, cols))[:, list(range(2, 6, 3))]
            .astype("float")
        )
    else:
        Afsc = A[A.find("Rec_SSNR") + 9 : A.find("Merge3D: Normal termination") - 3]
        rows = len(Afsc.split("\n"))
        cols = len(Afsc.split()) // rows
        # column 2 has the frequencies and column 5 the FSC values
        current_fsc = (
            np.array(Afsc.split())
            .reshape((rows, cols))[:, list(range(1, 4, 2))]
            .astype("float")
        )
    fsc_file = "../maps/%s_fsc.txt" % dataset
    if os.path.isfile(fsc_file):
        oldFSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)
        if current_fsc.shape[0] == oldFSCs.shape[0]:
            if oldFSCs.shape[1] < i:
                FSCs = np.zeros([oldFSCs.shape[0], i])
                FSCs[:, : oldFSCs.shape[1]] = oldFSCs
            else:
                FSCs = oldFSCs
        else:
            logger.warning(
                "Size of FSC curve has changed from {0} to {1}. Not plotting past results.".format(
                    current_fsc.shape[0], oldFSCs.shape[0]
                )
            )
            FSCs = np.zeros([current_fsc.shape[0], i])
        FSCs[:, i - 1] = current_fsc[:, 1]
    else:
        FSCs = current_fsc
    np.savetxt(fsc_file, FSCs, fmt="%10.5f")

    if int(project_params.param(fp["class_num"], i)) < 2:

        # keep track of rhref and measured fit
        current_res = np.zeros([1, 4])
        current_res[0, 0] = i
        rhref = float(project_params.param(fp["refine_rhref"], i))
        if rhref > 0:
            current_res[0, 1] = rhref
        else:
            current_res[0, 1] = postprocess.get_rhref(fp, i)

        # evaluate correlation beyond RHREF
        overfitting = current_fsc[current_fsc[:, 0] < current_res[0, 1]][:, 1].sum()

        plots = 1
        if (
            "model_fit" in fp.keys()
            and os.path.exists(fp["model_fit"])
            and fp["model_pixel"] > 0
        ):
            fits = postprocess.measure_score(
                scratch + name + ".mrc",
                fp["model_fit"],
                float(fp["model_res"]),
                int(fp["model_scale"]),
                float(fp["model_pixel"]),
                int(fp["model_clip"]),
                fp["model_flip"],
            )
            current_res[0, 2] = float(fits[0])
            current_res[0, 3] = float(fits[1])
            plots = 2

        res_file = "../maps/%s_res.txt" % dataset
        if os.path.exists(res_file):
            oldRESs = np.loadtxt(res_file, ndmin=2, dtype=float)
            if oldRESs.shape[0] > i - 2:
                oldRESs[i - 2, :] = current_res
                RESs = oldRESs
            else:
                RESs = np.zeros([i - 1, 4])
                RESs[: oldRESs.shape[0], :] = oldRESs
                RESs[-1, :] = current_res
        else:
            RESs = current_res
        np.savetxt(res_file, RESs, fmt="%10.5f")

    else:
        plots = 1
        overfitting = 0

    # plot curves
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(plots, 1, figsize=(8, 5))
    if plots == 1:
        axfsc = ax
    else:
        axfsc = ax[0]
    for iteration in range(1, i):
        if iteration < i - 3:
            use_this_label = "_nolegend_"
        else:
            use_this_label = "Iteration %i" % (iteration + 1)
        if iteration == i - 1:
            line_width = 3
        else:
            line_width = 1
        axfsc.plot(
            1 / FSCs[:, 0], FSCs[:, iteration], label=use_this_label, lw=line_width
        )
    if int(project_params.param(fp["class_num"], i)) < 2:
        axfsc.plot(
            (1.0 / current_res[0, 1], 1.0 / current_res[0, 1]), (-0.1, 1.05), "k-."
        )
        rlref = float(project_params.param(fp["refine_rlref"], i))
        axfsc.plot((1.0 / rlref, 1.0 / rlref), (-0.1, 1.05), "k-.")
        if "new" in project_params.param(fp["refine_metric"], i):
            if fp["refine_fboost"]:
                fboostlim = fp["refine_fboostlim"]
            else:
                fboostlim = 30.0
            if fboostlim > 0:
                axfsc.plot((1.0 / fboostlim, 1.0 / fboostlim), (-0.1, 1.05), "b-.")
    axfsc.plot(1 / FSCs[:, 0], np.zeros(FSCs[:, 1].shape), "k", lw=0.5)

    legend = axfsc.legend(loc="lower left", shadow=True)
    axfsc.set_ylim((-0.1, 1.05))
    if FSCs[0, 0] > 0:
        axfsc.set_xlim((1 / FSCs[0, 0], 1 / FSCs[-1, 0]))
    axfsc.set_title("FSC for %s (%.2f)" % (dataset, overfitting), fontsize=12)
    axfsc.set_xlabel("Frequency (1/A)")
    axfsc.set_ylabel("FSC")
    axfsc.legend(prop={"size": 9})

    """
    if plots > 1:
        ax[1].plot( RESs[:,0], RESs[:,1], '.-', label='rhref (%s)' % overfitting )
        ax[1].set_xlim((2,i))
        ax[1].legend(prop={'size':8}, loc='lower center')
    """

    if plots > 1:
        ax[1].plot(RESs[:, 0], RESs[:, 2], "r.-", label="Unmasked")
        ax[1].set_xlim((2, i))
        ax[1].legend(prop={"size": 10}, loc="lower center")
        ax[1].set_xlabel("Area under FSC vs. model")
        # ax[2].plot( RESs[:,0], RESs[:,3], 'g.-', label='atmft' )
        # ax[2].set_xlim((2,i))
        # ax[2].legend(prop={'size':8}, loc='lower center')

    plt.savefig("../maps/%s_fsc.png" % dataset)
    plt.close()

    if False and os.path.exists(fp["refine_maskth"]):
        shutil.copy(star_file.replace(".star", "_fsc.eps"), "../maps/%s_fsc.eps" % name)

    # reconstruction montage
    rec = mrc.read(scratch + name + ".mrc")

    # what if radius is larger than box
    radius = (
        float(mp["particle_rad"])
        / float(mp["extract_bin"])
        / float(mp["data_bin"])
        / float(mp["scope_pixel"])
    )

    lim = build_map_montage( scratch + name + ".mrc", radius, "../maps/%s_map.png" % name )


    # create composite montage
    command = "montage ../maps/{0}_map.png ../maps/{1}_fsc.png ../maps/{0}_prs.png ../maps/{0}_used_prs.png -geometry 690x460 ../maps/{0}_fyp.png".format(
        name, dataset
    )
    local_run.run_shell_command(command, verbose=False)
    img2webp(f"../maps/{name}_fyp.png",f"../maps/{name}_fyp.webp") 

    rec = mrc.read(scratch + name + ".mrc")
    cropped_volume = rec[ lim:-lim, lim:-lim, lim:-lim ]
    mrc.write( cropped_volume, "../maps/{0}_crop.mrc".format(name) )

    # set correct pixel size in mrc header
    command = """
%s/bin/alterheader << EOF
%s
del
%s,%s,%s
done
EOF
""" % (
        get_imod_path(),
        "../maps/{0}_crop.mrc".format(name),
        pixel_size,
        pixel_size,
        pixel_size,
    )
    local_run.run_shell_command(command)

    img2webp(f"../maps/{name}_map.png",f"../maps/{name}_map.webp","-resize 1024x")

    for i in "../maps/{1}_fsc.png ../maps/{0}_prs.png ../maps/{0}_used_prs.png".format(
        name, dataset
    ).split():
        if os.path.exists(i):
            os.remove(i)

    # email result
    if (
        user_comm.need_reporting()
        and "t" in mp["email"].lower()
        and int(project_params.param(fp["class_num"], iter)) == 1
    ):
        png_plot = "../maps/%s_fyp.webp" % name

        attach = os.getcwd() + "/" + png_plot
        user_comm.notify(name + " (3D)", attach)

    # plot exposure weights
    if int(
        project_params.param(fp["reconstruct_num_frames"], iter)
    ) > 1 and os.path.exists("weights.txt"):

        # get 1D score profile per tilt
        if "tomo" in mp["data_mode"].lower():
            cspty.plot_score_profile(name + "_used.par")
        else:
            cspty.plot_score_profile(name + "_used.par", tomo=False)

        # check tomo or spr then plot
        # # get 1D score profile per tilt
        # cspty.plot_score_profile( name + '_used.par' )

        # plot cistem weights
        command = "{0}/pyp/analysis/plot/pyp_frealign_plot_weights.py -input weights.txt -pixel {1} -boxsize {2} -frames {3}".format(
            os.environ["PYP_DIR"],
            mp["scope_pixel"],
            mp["extract_box"],
            fp["reconstruct_num_frames"],
        )
        local_run.run_shell_command(command)

        # montage png's
        command = "montage {0}_used_scores.png weights_2D_weights.png weights_3D_weights.png -geometry +0+0 -tile 1x3 ../maps/{0}_wgh.png".format(
            name
        )
        local_run.run_shell_command(command, verbose=False)

    # Generate phase residual plots
    # ${SPA_DIR}/frealign/frealign_plt.sh $1 1

    # measurements of resolution
    # printiter=`printf %02d $1`
    # com="${SPA_DIR}/general/fsc_cutoff.sh `ls maps/*_${printiter}_fsc.txt`"
    # echo $com; echo; $com

    # com="${SPA_DIR}/utils/rmeasure.sh maps/${data_input}_${printiter}.mrc `grep pixel_size ../microscope_parameters | awk '{print $2}'`"
    # echo $com; echo; $com

    # arrange matches by PR value
    # if 't' in project_params.param(fp['fmatch'],iter).lower() and project_params.param( fp['mask'], iter ) != '0,0,0,0,0':
    if False and "t" in project_params.param(fp["refine_fmatch"], iter).lower():
        mrc.merge(
            sorted(glob.glob("{0}_match.mrc_*".format(name))),
            str(
                Path(os.environ["PYP_SCRATCH"]) / f"{name}_match.mrc"
            ),  # TODO: safe to pass Path here?
        )  # merge match files into single stack
        for f in glob.glob("{0}_match.mrc_*".format(name)):
            os.remove(f)
        parfile = "../maps/" + name + ".par"
        A = np.loadtxt(parfile, comments="C")  # load .par file
        if False:
            B = A[np.argsort(A[:, 11])]  # sort indexes by PR
        else:
            B = A[np.argsort(-A[:, 14])]
        if float(project_params.param(fp["reconstruct_cutoff"], iter)) > 0:
            last = min(
                int(
                    A.shape[0]
                    * float(project_params.param(fp["reconstruct_cutoff"], iter))
                ),
                A.shape[0] - 1,
            )  # only keep particles below TH
        else:
            last = A.shape[0] - 1
        sorted_indexes = (B[:last, 0] - 1).tolist()  # 0-based indexes
        mrc.extract_slices(
            str(
                Path(os.environ["PYP_SCRATCH"]) / f"{name}_match.mrc"
            ),  # TODO: safe to pass Path here?
            sorted_indexes,
            "../maps/{0}_match.mrc".format(name),
        )  # create new stack
        # shutil.move( '{0}/{1}_match.mrc'.format( os.environ['PYP_SCRATCH'], name ), '../maps/{0}_match_unsorted.mrc'.format( name ) )


def mreconstruct_version(
    mp,
    fp,
    i,
    ref,
    reclogfile,
    fmatch,
    rad_rec,
    pixel,
    boff,
    first,
    last,
    dstep,
    thresh,
    res_rec,
    stack_dir,
    scratch,
):
    frealign_paths = get_frealign_paths()

    dataset = fp["refine_dataset"]
    name = dataset + "_r%02d_%02d" % (ref, i)
    if os.path.exists(stack_dir + "%s_recstack.mrc" % dataset):
        stack = stack_dir + "%s_recstack.mrc" % dataset
        with open(reclogfile, "w") as f:
            f.write("Using {0}\n".format(stack))
    else:
        stack = stack_dir + "%s_stack.mrc" % dataset

    '''
    command = """
%s/bin/frealign_v9_mp.exe << eot >>%s 2>&1
M,0,F,F,F,F,%s,T,%s,%s,%s,%s,F,%s,%s                                        !CFORM,IFLAG,FMAG,FDEF,FASTIG,FPART,IEWALD,FBEAUT,FFILT,FBFACT,FMATCH,IFSC,FSTAT,IMEM,INTERP
%s,0.,%s,%s,%s,%s,%s,%s,%s,%s,%s                                        !RO,RI,PSIZE,MW,WGH,XSTD,PBC,BOFF,DANG,ITMAX,IPMAX
%s                                                                        !MASK
%i,%i                                                                        !IFIRST,ILAST
%s                                                                        !ASYM symmetry card
1.0, %s, %s, %s, %s, %s, 0., 0.                                                !RELMAG,DSTEP,TARGET,THRESH,CS,AKV,TX,TY
%s, %s, %s, %s, %s                                                        !RREC,RMAX1,RMAX2,DFSIG,RBFACT
%s
/dev/null
%s.par
%s_dummy.par
%s_dummy.shft
0., 0., 0., 0., 0., 0., 0., 0.                                                !terminator with RELMAG=0.0
%s.mrc
%s_weights
%s_half1.mrc
%s_half2.mrc
%s_phasediffs
%s_pointspread
eot
""" % (
        frealign_paths['cc3m'],
        reclogfile,
        project_params.param(fp["iewald"], i),
        project_params.param(fp["ffilt"], i),
        project_params.param(fp["fbfact"], i),
        fmatch,
        project_params.param(fp["fboost"], i),
        project_params.param(fp["imem"], i),
        project_params.param(fp["interp"], i),
        rad_rec,
        pixel,
        mp["particle_mw"],
        mp["scope_wgh"],
        project_params.param(fp["xstd"], i),
        project_params.param(fp["pbc"], i),
        boff,
        project_params.param(fp["dang"], i),
        project_params.param(fp["itmax"], i),
        project_params.param(fp["ipmax"], i),
        project_params.param(fp["mask"], i),
        first,
        last,
        project_params.param(fp["symmetry"], i),
        dstep,
        project_params.param(fp["target"], i),
        thresh,
        mp["scope_cs"],
        mp["scope_voltage"],
        res_rec,
        project_params.param(fp["rlref"], i),
        postprocess.get_rhref(fp, i),
        project_params.param(fp["dfsig"], i),
        project_params.param(fp["rbfact"], i),
        stack,
        name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
    )
    '''

    command = """
%s/bin/frealign_v9_mp.exe << eot >>%s 2>&1
M,0,F,F,F,F,%s,T,%s,%s,%s,%s,F,%s,%s                                         !CFORM,IFLAG,FMAG,FDEF,FASTIG,FPART,IEWALD,FBEAUT,FFILT,FBFACT,FMATCH,IFSC,FDUMP,IMEM,INTERP
%s,0.,%s,%s,%s,%s,%s,%s,%s,%s,%s                                        !RO,RI,PSIZE,MW,WGH,XSTD,PBC,BOFF,DANG,ITMAX,IPMAX
%s                                                                        !MASK
%i,%i                                                                        !IFIRST,ILAST
%s                                                                        !ASYM symmetry card
1.0, %s, %s, %s, %s, %s, 0., 0.                                                !RELMAG,DSTEP,TARGET,THRESH,CS,AKV,TX,TY
%s, %s, %s, %s, %s, %s                                                        !RREC,RMIN,RMAX,RCLAS,DFSTD,RBFACT
%s
/dev/null
%s.par
%s.res
%s_dummy.shft
0., 0., 0., 0., 0., 0., 0., 0.                                                !terminator with RELMAG=0.0
%s.mrc
%s_weights
%s_half1.mrc
%s_half2.mrc
%s_phasediffs
%s_pointspread
eot
""" % (
        frealign_paths["cc3m"],
        reclogfile,
        project_params.param(fp["refine_iewald"], i),
        project_params.param(fp["reconstruct_ffilt"], i),
        project_params.param(fp["reconstruct_fbfact"], i),
        fmatch,
        fp["refine_fboost"],
        project_params.param(fp["refine_imem"], i),
        project_params.param(fp["refine_interp"], i),
        rad_rec,
        pixel,
        mp["particle_mw"],
        mp["scope_wgh"],
        project_params.param(fp["refine_xstd"], i),
        project_params.param(fp["refine_pbc"], i),
        boff,
        project_params.param(fp["refine_dang"], i),
        project_params.param(fp["refine_itmax"], i),
        project_params.param(fp["refine_ipmax"], i),
        project_params.param(fp["refine_mask"], i),
        first,
        last,
        project_params.param(fp["particle_sym"], i),
        dstep,
        project_params.param(fp["refine_target"], i),
        thresh,
        mp["scope_cs"],
        mp["scope_voltage"],
        res_rec,
        project_params.param(fp["refine_rlref"], i),
        postprocess.get_rhref(fp, i),
        project_params.param(fp["class_rhcls"], i),
        project_params.param(fp["refine_dfsig"], i),
        project_params.param(fp["refine_rbfact"], i),
        stack,
        name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
        scratch + name,
    )

    return command


def split_refinement(mp, ref, current_path, first, last, i, metric):
    """function to execute refinement using refine3d in parallel

    Parameters
    ----------
    mp : dict 
        parameters for main pyp
    ref : int
        the number of classes
    current_path : str
        project path to find the global parfile
    first : int
        the first particle to be refined (it is usually 1)
    last : int 
        the last particle to be refined 
    i : int 
        the current iteration (it is usually 2)
    metric : str
        alignment metric for refine3d

    Returns
    -------
    [type]
        [description]
    """
    os.chdir("scratch")

    fp = mp

    name = "%s_r%02d" % (fp["refine_dataset"], ref)
    ranger = "%07d_%07d" % (first, last)
    if fp["refine_debug"] or first == 1:
        logfile = "%s_msearch_n.log_%s" % (name, ranger)
    else:
        logfile = "/dev/null"
    scratch = Path(os.environ["PYP_SCRATCH"])

    # if we are in csp using frame refinement mode we can fork each batch into multiple cores
    if mp["csp_no_stacks"] and mp["slurm_tasks"] > 1:

        # This is the output parfile for this function
        short_file_name = name + ".par_%07d_%07d" % (1, last)

        # bypass refine3d if mode == 0
        if int(project_params.param(mp["refine_mode"], i)) == 0:
            shutil.copy2(f"{name}.par", short_file_name)
            return 0

        # create multirun script
        cpus = mp["slurm_tasks"]

        # mpirunfile, count = local_run.create_split_source_file( mp, name, last, cpus, scratch, ref=ref, iteration=i, step="refine3d" )
        commands, count = local_run.create_split_commands(
            mp,
            name,
            last,
            cpus,
            scratch,
            ref=ref,
            current_path=current_path,
            iteration=i,
            step="refine3d",
        )

        # submit jobs to MPI
        mpi.submit_jobs_to_workers(commands, os.getcwd(), verbose=mp["slurm_verbose"])

        # combine all the refined parfile
        short_file_name = name + "_%07d_%07d.cistem" % (1, last)
        all_refined_par = [par for par in glob.glob(name + "_*_*.cistem")]

        # first check if the number of refined par is equal to count
        if len(all_refined_par) != count:
            # attempt to get error information to the user before exiting
            logger.error(commands[0])
            logfile = glob.glob( "*_msearch_n.log_0000001_*" )
            if len(logfile) > 0:
                with open( logfile[0] ) as output:
                    logger.error("\n".join([s for s in output.read().split("\n") if s]))
            raise Exception(
                f"The number of refined parfiles ({len(all_refined_par)}) != the number of jobs ({count})."
            )

        # if "frealignx" in metric:
        #     frealignx = True
        #     short_column = 17
        # else:
        #     frealignx = False
        #     short_column = 16

        # merged_short_par = np.empty((0, short_column))
        # for refined_par in sorted(all_refined_par, key=lambda x: x.split("_")[-1]):
        #     # changed_par = refined_par.replace(".par", "_changed.par")
        #     refined = np.loadtxt(refined_par, ndmin=2, comments="C")
        #     # changed = np.loadtxt(changed_par, ndmin=2, comments="C")
        #     # refined[:, -2] = refined[:, -2] + refined[:, -1]
        #     merged_short_par = np.vstack((merged_short_par, refined))

        # frealign_parfile.Parameters.write_parameter_file(short_file_name, merged_short_par, parx=False, frealignx=frealignx)

        # TODO: Ye, not sure if this is what you wanna do?
        merged_alignment = Parameters.merge(input_files=all_refined_par,
                                            input_extended_files=[])
        merged_alignment.to_binary(short_file_name)
        import sys
        sys.exit()

        """
        sort_allpar = sorted(
            [
                line
                for par in all_refined_par
                for line in open(par)
                if not line.startswith("C")
            ],
            key=lambda x: int(x.split()[0]),
        )

        # write out short allpar (extended metadata will be added later)
        with open(short_file_name, "w") as f:
            for line in sort_allpar:
                f.write(line)
        """
        # cleanup
        [
            os.remove(i)
            for i in glob.glob(str(scratch / f"{name}_*_*.cistem"))
            if i != short_file_name
        ]

    else:

        # this is the regular refinement on a single core
        with timer.Timer(
                    "FREALIGN_nonmpi", text = "Non mpi refine3d took: {}", logger=logger.info
                ):
            command = mrefine_version(
                mp,
                first,
                last,
                i,
                ref,
                current_path,
                name,
                ranger,
                logfile,
                str(scratch),
                metric,  # TODO: safe to pass Path here?
            )

            os.environ["OMP_NUM_THREADS"] = os.environ["NCPUS"] = "1"
            with open(logfile, "w") as f:
                f.write(command)
            subprocess.Popen(command, shell=True, text=True).wait()

            # cleanup
            [
                os.remove(i)
                for i in glob.glob(str(scratch / f"{name}*{ranger}*"))
                if os.path.isfile(i)
            ]


def merge_refinements(mp, fp, iteration, alignment_option):

    classes = int(project_params.param(fp["class_num"], iteration))

    machinefile = "../mpirun.mynodes"
    # make sure there are no missing alignments
    if alignment_option > 0:
        ref_merge_check_error_and_resubmit(fp, iteration, machinefile)

    # TODO: formatting stuff, should put into another function
    #  use Parameters()
    for ref in range(classes):

        dataset = fp["refine_dataset"] + "_r%02d" % (ref + 1)

        if not "cistem" in project_params.param(fp["refine_metric"], iteration).lower():
            parfile = "%s_%02d.par" % (dataset, iteration)
        else:
            parfile = "%s_%02d.star" % (dataset, iteration)

        # merge parameter files only if alignments were done
        if alignment_option > 0:
            # merge log files
            logfilelist = sorted(
                glob.glob("%s_%02d_msearch_n.log_*" % (dataset, iteration))
            )
            logfile = "../log/%s_%s_%02d_msearch.log" % (
                datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S"),
                dataset,
                iteration,
            )
            l = open(logfile, "w")
            for log in logfilelist:
                l.write("".join([line for line in open(log)]))
            l.close()

            # remove individual log files
            # null = [os.remove(i) for i in logfilelist if os.path.isfile(i)]

            # merge parameters into single file
            filelist = sorted(glob.glob("%s_%02d.par_*" % (dataset, iteration)))

            # cistem2 files
            filelist += sorted(
                [
                    f
                    for f in glob.glob(
                        "%s_%02d_???????_???????.star" % (dataset, iteration)
                    )
                ]
            )

            f = open(parfile, "w")
            if "cistem" in project_params.param(fp["refine_metric"], iteration).lower():
                f.write(
                    "".join(
                        [
                            line
                            for line in open(filelist[0])
                            if line.startswith("_")
                            or line.startswith("#")
                            or line.startswith("data")
                            or line.startswith("loop")
                            or line == " \n"
                        ]
                    )
                )
            else:
                f.write(
                    "".join(
                        [line for line in open(filelist[0]) if line.startswith("C")]
                    )
                )

            (
                fieldwidths,
                fieldstring,
                _,
                _,
            ) = frealign_parfile.Parameters.format_from_parfile(filelist[0])

            # collate refined parameters
            for pfile in filelist:
                if (
                    "cistem"
                    in project_params.param(fp["refine_metric"], iteration).lower()
                ):
                    f.write(
                        "".join(
                            [
                                line
                                for line in open(pfile)
                                if not line.startswith("C")
                                and not line.startswith("_")
                                and not line.startswith("#")
                                and not line.startswith("data")
                                and not line.startswith("loop")
                                and not line == " \n"
                            ]
                        )
                    )
                else:
                    # frealign/cistem will actually change the column width in the case of overflow
                    # if.write( ''.join( [line for line in open(pfile) if not line.startswith('C') ] ) )

                    # here we reformat columns to force the standard format (even if that means columns will be joint)
                    input = np.array(
                        [line for line in open(pfile) if not line.startswith("C")]
                    )
                    # input = frealign_parfile.Parameters.from_file(pfile).data

                    # reset occupancies if not doing classification
                    if int(project_params.param(fp["class_num"], iteration)) <= 1:
                        occ = 0
                        if (
                            "frealignx"
                            in project_params.param(
                                fp["refine_metric"], iteration
                            ).lower()
                        ):
                            scores = 15
                            occ = 12
                        else:
                            scores = 14
                            occ = 11

                        for line in input:
                            values = np.array(line.split(), dtype="f")
                            if occ > 0:
                                values[occ] = 100
                            # truncate scores to prevent overflow
                            if values[scores] > 9999:
                                values[scores] = 0
                                values[scores + 1] = 0
                            f.write(fieldstring % tuple(values))
                    else:
                        f.write(
                            "".join(
                                [
                                    line
                                    for line in open(pfile)
                                    if not line.startswith("C")
                                ]
                            )
                        )
            f.close()

            # make sure that all particles were succesfully aligned
            lines = [
                _f
                for _f in (
                    line.rstrip()
                    for line in open(parfile)
                    if not line.startswith("C")
                    and not line.startswith("_")
                    and not line.startswith("#")
                    and not line.startswith("data")
                    and not line.startswith("loop")
                    and not line == " \n"
                )
                if _f
            ]
            nz = mrc.readHeaderFromFile("../" + fp["refine_dataset"] + "_stack.mrc")[
                "nz"
            ]
            if not nz == len(lines):
                raise Exception("{0} missing alignments. Stop.".format(nz - len(lines)))

            # cleanup
            null = [os.remove(i) for i in filelist if os.path.isfile(i)]
            filelist = glob.glob("%s_%02d.shft_*" % (dataset, iteration))
            null = [os.remove(i) for i in filelist if os.path.isfile(i)]
            filelist = glob.glob("%s_%02d*_changes.star" % (dataset, iteration))
            null = [os.remove(i) for i in filelist if os.path.isfile(i)]

        else:
            if os.path.isfile("../maps/%s_%02d.par" % (dataset, iteration - 1)):
                prevpar = "../maps/%s_%02d.par" % (dataset, iteration - 1)
            elif os.path.isfile("../%s_%02d.par" % (dataset, iteration - 1)):
                prevpar = "../%s_%02d.par" % (dataset, iteration - 1)
            else:
                logger.error("Cannot locate parameter file from previous iteration.")
            shutil.copy(prevpar, parfile)

        phases_or_scores = "-scores"
        arg_scores = True
        arg_frealignx = False
        width = 137
        columns = 16

        if "frealignx" in project_params.param(fp["refine_metric"], iteration):
            phases_or_scores = "-frealignx"
            arg_scores = False
            arg_frealignx = True
            width = 145
            columns = 17

        # input = np.array( [line.split() for line in open( parfile ) if not line.startswith('C') and not line.startswith('_') and not line.startswith('#') and not line.startswith('data') and not line.startswith('loop') and not line == " \n" ], dtype=str )
        # TODO: add new methods to extend the file
        input = frealign_parfile.Parameters.from_file(parfile).data

        if alignment_option == 2 or input.shape[1] == 13:
            logger.info(f"NOT USING SCORES? {parfile} {input.shape[1]}")
            phases_or_scores = ""
            arg_scores = False
            arg_frealignx = False
            width = 103
            columns = 13
        else:
            # add extra columns if using CSPT extended .par format
            prevparfile = "../maps/%s_%02d.par" % (dataset, iteration - 1)
            if (
                os.path.isfile(prevparfile)
                and len(
                    [line for line in open(prevparfile) if not line.startswith("C")][
                        0
                    ].split()
                )
                > columns
            ):
                filex = prevparfile
            else:
                filex = "../%s_01.par" % (fp["refine_dataset"])

            # compose extended .parx file
            long_file = [line for line in open(filex) if not line.startswith("C")]

            # if just switching to frealignx or new
            if len(long_file[0]) == 426 or len(long_file[0]) == 421:
                lwidth = 137
            else:
                lwidth = 145

            short_file = [line for line in open(parfile) if not line.startswith("C")]

            comments = [line for line in open(parfile) if line.startswith("C")]

            if (
                len(long_file[0].split()) > columns
                and not len(short_file[0].split()) > columns
                and not "star" in parfile
            ):

                logger.info(f"Merging {filex} with {parfile} into {parfile}")

                with open(parfile, "w") as f:

                    [f.write(line) for line in open(filex) if line.startswith("C")]

                    for i, j in zip(short_file, long_file):
                        f.write(i[: width - 1] + j[lwidth - 1 :])

            logger.info(f"Checking if prev par file {prevparfile} is frealignx")

            if not isfrealignx(prevparfile) and (
                (
                    "frealignx"
                    in project_params.param(fp["reifne_metric"], iteration).lower()
                    and int(project_params.param(fp["refine_mode"], iteration)) == 0
                )
                or (
                    fp["dose_weighting_enable"]
                    and not "frealignx"
                    in project_params.param(fp["refine_metric"], iteration).lower()
                )
                or mp["csp_no_stacks"]
            ):
                logger.info(f"Adding frealignx additional PSHIFT column {filex}")

                new_file = [line for line in open(parfile) if not line.startswith("C")]
                comments = [line for line in open(parfile) if line.startswith("C")]

                with open(parfile, "w") as f:

                    [f.write(line) for line in comments]

                    for i in new_file:
                        f.write(i[:91] + "%8.2f" % 0 + i[91:])

            # overwrite INCLUDE field with FILM field
            if (
                "frealignx"
                in project_params.param(fp["refine_metric"], iteration).lower()
            ):

                new_file = [line for line in open(parfile) if not line.startswith("C")]

                with open(parfile, "w") as f:

                    [f.write(line) for line in comments]

                    for i, j in zip(long_file, new_file):
                        f.write(j[:59] + i[59:65] + j[65:])

                ## append resolution table to parameter file
                # if os.path.exists( '../maps/%s_%02d.res' % (dataset,iteration) ):
                #    com='cat %s >> %s' % ( '../maps/%s_%02d.res' % (dataset,iteration), parfile )
                #    commands.getoutput(com)

        # if using dose weighting we need to use frealignx mode
        if fp["dose_weighting_enable"]:
            phases_or_scores = "-frealignx"
            arg_scores = False
            arg_frealignx = True

        # Save copy of most current parameters to maps directory
        shutil.copy(parfile, "../maps")

        name = "%s_%02d" % (dataset, iteration)

        # arrange matches by PR value

        # if 't' in project_params.param(fp['fmatch'],iter).lower() and project_params.param( fp['mask'], iter ) != '0,0,0,0,0':
        if "t" in project_params.param(fp["refine_fmatch"], iteration).lower():
            mrc.merge(
                sorted(glob.glob("{0}_match.mrc_*".format(name))),
                "../maps/{0}_match_unsorted.mrc".format(name),
            )
            for f in glob.glob("{0}_match.mrc_*".format(name)):
                os.remove(f)
            parfile = "../maps/" + name + ".par"
            A = np.loadtxt(parfile, comments="C")  # load .par file
            if A.shape[1] < 14:
                B = A[np.argsort(A[:, 11])]  # sort indexes by PR
            else:
                B = A[np.argsort(-A[:, 14])]
            if float(project_params.param(fp["reconstruct_cutoff"], iteration)) > 0:
                last = min(
                    int(
                        A.shape[0]
                        * float(
                            project_params.param(fp["reconstruct_cutoff"], iteration)
                        )
                    ),
                    A.shape[0] - 1,
                )  # only keep particles below TH
            else:
                last = A.shape[0]
            sorted_indexes = (B[:last, 0] - 1).tolist()  # 0-based indexes
            mrc.extract_slices(
                "../maps/{0}_match_unsorted.mrc".format(name),
                sorted_indexes,
                "../maps/{0}_match.mrc".format(name),
            )

            # shutil.move( '{0}/{1}_match.mrc'.format(os.environ['PYP_SCRATCH'],name), '../maps/{0}_match_unsorted.mrc'.format(name) )

        # Produce angular and defocus plots
        # plot using all particles
        if float(project_params.param(fp["reconstruct_cutoff"], iteration)) >= 0:
            # command = "{0}/bin/pyp_shape_pr_values -mode plot -input {1}.par -output {1}_used.par -cutoff 1 -angle_groups 25 -defocus_groups 25 {2}".format(
            #     os.environ["PYP_DIR"], name, phases_or_scores
            # )  # -reverse
            # logger.info(command)
            # # flush the output buffers so the command output actually appears after the command
            # sys.stdout.flush()
            # # run the command and wait for it to finish while dumping the output to stdout in real-time
            # # if the process exits with an error code, also throw an exception
            # subprocess.run(command, stderr=subprocess.STDOUT, shell=True, check=True)

            arg_input = f"{name}.par"
            arg_angle_groups = 25
            arg_defocus_groups = 25
            arg_dump = False
            plot.generate_plots(
                arg_input,
                arg_angle_groups,
                arg_defocus_groups,
                arg_scores,
                arg_frealignx,
                arg_dump,
            )

        # create new parameter file only including used particles
        # shape_pr_options = "-mindefocus {0} -maxdefocus {1} -firstframe {2} -lastframe {3} -mintilt {4} -maxtilt {5} {6} {7} -minazh {8} -maxazh {9} -minscore {10} -maxscore {11}".format(
        #     project_params.param(fp["mindef"], iteration),
        #     project_params.param(fp["maxdef"], iteration),
        #     project_params.param(fp["firstframe"], iteration),
        #     project_params.param(fp["lastframe"], iteration),
        #     project_params.param(fp["mintilt"], iteration),
        #     project_params.param(fp["maxtilt"], iteration),
        #     project_params.param(fp["shapr"], iteration),
        #     phases_or_scores,
        #     project_params.param(fp["minazh"], iteration),
        #     project_params.param(fp["maxazh"], iteration),
        #     project_params.param(fp["minscore"], iteration),
        #     project_params.param(fp["maxscore"], iteration),
        # )

        arg_mindefocus = float(
            project_params.param(fp["reconstruct_mindef"], iteration)
        )
        arg_maxdefocus = float(
            project_params.param(fp["reconstruct_maxdef"], iteration)
        )
        arg_firstframe = int(
            project_params.param(fp["reconstruct_firstframe"], iteration)
        )
        arg_lastframe = int(
            project_params.param(fp["reconstruct_lastframe"], iteration)
        )
        arg_mintilt = float(project_params.param(fp["reconstruct_mintilt"], iteration))
        arg_maxtilt = float(project_params.param(fp["reconstruct_maxtilt"], iteration))
        arg_minazh = float(project_params.param(fp["reconstruct_minazh"], iteration))
        arg_maxazh = float(project_params.param(fp["reconstruct_maxazh"], iteration))
        arg_minscore = float(
            project_params.param(fp["reconstruct_minscore"], iteration)
        )
        arg_maxscore = float(
            project_params.param(fp["reconstruct_maxscore"], iteration)
        )

        shapr = project_params.param(fp["reconstruct_shapr"], iteration)
        arg_reverse = False
        arg_consistency = False
        if "reverse" in shapr.lower():
            arg_reverse = True
        if "consistency" in shapr.lower():
            arg_consistency = True

        # use NO cutoff if we are using multiple references
        if int(project_params.param(fp["class_num"], iteration)) > 1:
            cutoff = 1
        else:
            cutoff = project_params.param(fp["reconstruct_cutoff"], iteration)

        # command = "{0}/bin/pyp_shape_pr_values -mode shape -input {1}.par -output {1}_used.par -cutoff {2} -angle_groups {3} -defocus_groups {4} {5}".format(
        #     os.environ["PYP_DIR"],
        #     name,
        #     cutoff,
        #     project_params.param(fp["agroups"], iteration),
        #     project_params.param(fp["dgroups"], iteration),
        #     shape_pr_options,
        # )  # -reverse -consistency
        # local_run.run_shell_command(command)

        arg_input = f"{name}.par"
        arg_angle_groups = int(
            project_params.param(fp["reconstruct_agroups"], iteration)
        )
        arg_defocus_groups = int(
            project_params.param(fp["reconstruct_dgroups"], iteration)
        )
        arg_cutoff = float(cutoff)
        arg_output = f"{name}_used.par"
        arg_binning = 1.0
        arg_odd = False
        arg_even = False

        analysis.scores.shape_phase_residuals(
            arg_input,
            arg_angle_groups,
            arg_defocus_groups,
            arg_cutoff,
            arg_mindefocus,
            arg_maxdefocus,
            arg_firstframe,
            arg_lastframe,
            arg_mintilt,
            arg_maxtilt,
            arg_minazh,
            arg_maxazh,
            arg_minscore,
            arg_maxscore,
            arg_binning,
            arg_reverse,
            arg_consistency,
            arg_scores,
            arg_frealignx,
            arg_odd,
            arg_even,
            arg_output,
        )

        rotreg = False
        transreg = False

        if "t" in project_params.param(fp["reconstruct_rotreg"], iteration).lower():
            rotreg = True

        if "t" in project_params.param(fp["reconstruct_transreg"], iteration).lower():
            transreg = True

        ref_par = project_params.param(fp["refine_ref_par"], iteration)
        if rotreg or transreg:
            if len(ref_par) < 1:
                raise Exception("refine_ref_par is not present")

            if "t" in project_params.param(fp["refine_saveplots"], iteration).lower():
                saveplots = True
            else:
                saveplots = False

            spatial_sigma = float(
                project_params.param(fp["csp_spatial_sigma"], iteration)
            )
            time_sigma = float(project_params.param(fp["csp_time_sigma"], iteration))

            rot_method = fp["csp_rotreg_method"]
            trans_method = fp["csp_transreg_method"]

            # command = "{0}/bin/pyp_shape_pr_values -mode regularize -input {1}_used.par -output {1}_used.par -rotational {2} -translational {3} -saveplots {4} -ref_par {5} -spatial_sigma {6} -time_sigma {7} -rot_method {8} -trans_method {9}".format(
            #     os.environ["PYP_DIR"],
            #     name,
            #     rotreg,
            #     transreg,
            #     saveplots,
            #     ref_par,
            #     spatial_sigma,
            #     time_sigma,
            #     rot_method,
            #     trans_method,
            # )
            # logger.info(command)
            # logger.info(
            #     subprocess.check_output(
            #         command, stderr=subprocess.STDOUT, shell=True, text=True
            #     )
            # )

            arg_input = f"{name}_used.par"
            arg_output = f"{name}_used.par"
            analysis.fit.regularize(
                arg_input,
                ref_par,
                arg_output,
                rotreg,
                transreg,
                rot_method,
                trans_method,
                saveplots,
                spatial_sigma,
                time_sigma,
            )

        # plot using only particles that went into reconstruction
        if float(project_params.param(fp["reconstruct_cutoff"], iteration)) >= 0:
            # command = "{0}/bin/pyp_shape_pr_values -mode plot -input {1}_used.par -output {1}_used.par -cutoff 1 -angle_groups 25 -defocus_groups 25 {2}".format(
            #     os.environ["PYP_DIR"], name, phases_or_scores
            # )  # -reverse
            # local_run.run_shell_command(command)

            arg_input = f"{name}_used.par"
            arg_angle_groups = 25
            arg_defocus_groups = 25
            arg_dump = False
            plot.generate_plots(
                arg_input,
                arg_angle_groups,
                arg_defocus_groups,
                arg_scores,
                arg_frealignx,
                arg_dump,
            )

            # transfer files to maps directory
            shutil.copy2(name + "_prs.png", "../maps/")
            shutil.copy2(name + "_used_prs.png", "../maps/")

        # substitute par file with shaped one
        # if int(project_params.param(fp['classes'],iteration)) == 1 and 'cc' in project_params.param( fp['metric'], iteration ).lower():
        if (
            int(project_params.param(fp["class_num"], iteration)) == 1
            or rotreg
            or transreg
        ):
            shutil.copy2("%s_used.par" % name, "%s.par" % name)

        # Save final parameters to maps directory
        try:
            shutil.copy(parfile, "../maps")
        except:
            logger.warning("File exists %s", parfile)


def mrefine_version(
    mp, first, last, i, ref, current_path, name, ranger, logfile, scratch, metric
):
    """Return frealign refinement command."""

    scratch = str(scratch)

    global_mp = project_params.load_pyp_parameters(current_path)
    global_name = global_mp["data_set"]
    global_par = os.path.join(
        current_path, "frealign", "maps", global_name + "_r%02d_%02d.par" % (ref, i - 1)
    )
    fp = mp

    pixel = float(mp["scope_pixel"]) * float(mp["data_bin"]) * float(mp["extract_bin"])
    dstep = pixel * float(mp["scope_mag"]) / 10000.0
    dataset = fp["refine_dataset"] + "_r%02d" % ref

    if float(project_params.param(mp["reconstruct_rrec"], i)) > 0:
        res_rec = project_params.param(mp["reconstruct_rrec"], i)
    else:
        # WARNING: 2.0A is the maximum supported resolution in FREALIGN
        res_rec = max(2.0, 2 * pixel)
        res_rec = 2 * pixel

    stack_dir = ".."
    if os.path.exists(
        Path(os.environ["PYP_SCRATCH"]) / f'{fp["refine_dataset"]}_stack.mrc'
    ):
        stack_dir = os.environ["PYP_SCRATCH"]

    frealign_paths = get_frealign_paths()

    # if first iteration and mode 4, force metric to cclin
    # if i == 2 and int(project_params.param(mp["refine_mode"], i)) == 4:
    #    metric = "cclin"
   

    # use CCLIN from frealign_v8 for initial orientation assignemnt (mode=4, mask=1,1,1,1,1)
    if "cclin" in metric.lower():
        '''    
        thresh = 90.0
        command = """
%s/frealign_v8.10_intel/bin/frealign_v8_mp_cclin.exe << eot >>%s 2>&1
M, 4, %s, %s, %s, %s, %s, T, %s, %s, %s, 0, F, %s                                !CFORM,IFLAG,FMAG,FDEF,FASTIG,FPART,IEWALD,FBEAUT,FFILT,FBFACT,FMATCH,IFSC,FSTAT,IBLOW
%s, 0., %s, %s, %s, %s, %s, %s, %s, %s                                                !RO,RI,PSIZE,WGH,XSTD,PBC,BOFF,DANG,ITMAX,IPMAX
1,1,1,1,1                                                                                !MASK
%i, %i                                                                                !IFIRST,ILAST
%s
1.0, %s, %s, %s, %s, %s, 0., 0.
%s, %s, %s, %s, %s
%s/%s_stack.mrc
%s_match.mrc_%s
%s.par
%s.par_%s
/dev/null
-100., 0., 0., 0., 0., 0., 0., 0.                                                !terminator with RELMAG=-100.0 to skip 3D reconstruction
%s_%s.mrc
%s_weights_%s_%s
%s_map1_%s.mrc
%s_map2_%s.mrc
%s_phasediffs_%s
%s_pointspread_%s
eot
""" % (
            os.environ["PYP_DIR"],
            logfile,
            project_params.param(fp["fmag"], i),
            project_params.param(fp["fdef"], i),
            project_params.param(fp["fastig"], i),
            project_params.param(fp["fpart"], i),
            project_params.param(fp["iewald"], i),
            project_params.param(fp["ffilt"], i),
            project_params.param(fp["fbfact"], i),
            project_params.param(fp["fmatch"], i),
            project_params.param(fp["iblow"], i),
            mp["particle_rad"],
            pixel,
            mp["scope_wgh"],
            project_params.param(fp["xstd"], i),
            project_params.param(fp["pbc"], i),
            project_params.param(fp["boff"], i),
            project_params.param(fp["dang"], i),
            project_params.param(fp["itmax"], i),
            project_params.param(fp["ipmax"], i),
            first,
            last,
            project_params.param(fp["symmetry"], i),
            dstep,
            10.0,
            thresh,
            mp["scope_cs"],
            mp["scope_voltage"],  # set target PR to a low value
            res_rec,
            project_params.param(fp["rlref"], i),
            postprocess.get_rhref(fp, i),
            project_params.param(fp["dfsig"], i),
            project_params.param(fp["rbfact"], i),
            stack_dir,
            fp["dataset"],
            name,
            ranger,
            name,
            name,
            ranger,
            dataset,
            "%02d" % (i - 1),
            scratch + name,
            "%02d" % i,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
        )
            '''

        # v9.10
        # evaluate scores without refining shifts or angles
        mask = project_params.param(mp["refine_mask"], i)
        mode = project_params.param(mp["refine_mode"], i)
        if metric == "eval":
            mask = "0,0,0,1,1"
            mode = "1"
        thresh = 0.0

        if mp["refine_fboost"]:
            ifsc = -1
        else:
            ifsc = 0
        ifsc = 0

        command = """
%s/bin_debug/frealign_v9_intel_cclin.exe << eot >>%s 2>&1
M, %s, %s, %s, %s, %s, %s, T, %s, %s, %s, %s, F, %s, %s                                !CFORM,IFLAG,FMAG,FDEF,FASTIG,FPART,IEWALD,FBEAUT,FFILT,FBFACT,FMATCH,IFSC,FDUMP,IMEM,INTERP
%s, 0., %s, %s, %s, %s, %s, %s, %s, %s, %s                                        !RO,RI,PSIZE,MW,WGH,XSTD,PBC,BOFF,DANG,ITMAX,IPMAX
%s                                                                                !MASK
%i, %i                                                                                !IFIRST,ILAST
%s
1.0, %s, %s, %s, %s, %s, 0., 0.
%s, %s, %s, %s, %s, %s
%s/%s_stack.mrc
%s_match.mrc_%s
%s.par
%s.par_%s
/dev/null
-100., 0., 0., 0., 0., 0., 0., 0.                                                !terminator with RELMAG=-100.0 to skip 3D reconstruction
%s_%s.mrc
%s_weights_%s_%s
%s_map1_%s.mrc
%s_map2_%s.mrc
%s_phasediffs_%s
%s_pointspread_%s
eot
""" % (
            frealign_paths["cclin"],
            logfile,
            mode,
            project_params.param(mp["refine_fmag"], i),
            project_params.param(mp["refine_fdef"], i),
            project_params.param(mp["refine_fastig"], i),
            project_params.param(mp["refine_fpart"], i),
            project_params.param(mp["reconstruct_iewald"], i),
            project_params.param(mp["reconstruct_ffilt"], i),
            project_params.param(mp["reconstruct_fbfact"], i),
            project_params.param(mp["refine_fmatch"], i),
            ifsc,
            project_params.param(mp["refine_imem"], i),
            project_params.param(mp["refine_interp"], i),
            # mp['particle_rad'], pixel, mp['particle_mw'], mp['scope_wgh'], project_params.param(fp['xstd'],i), project_params.param(fp['pbc'],i), project_params.param(fp['boff'],i), project_params.param(fp['dang'],i), project_params.param(fp['itmax'],i), project_params.param(fp['ipmax'],i),
            mp["particle_rad"],
            pixel,
            mp["particle_mw"],
            mp["scope_wgh"],
            project_params.param(mp["refine_xstd"], i),
            "100.0",
            project_params.param(mp["refine_boff"], i),
            project_params.param(mp["refine_dang"], i),
            project_params.param(mp["refine_itmax"], i),
            project_params.param(mp["refine_ipmax"], i),
            mask,
            first,
            last,
            project_params.param(mp["particle_sym"], i),
            dstep,
            project_params.param(mp["refine_target"], i),
            thresh,
            mp["scope_cs"],
            mp["scope_voltage"],
            res_rec,
            project_params.param(mp["refine_rlref"], i),
            postprocess.get_rhref(mp, i),
            project_params.param(mp["class_rhcls"], i),
            project_params.param(mp["refine_dfsig"], i),
            project_params.param(mp["refine_rbfact"], i),
            stack_dir,
            fp["refine_dataset"],
            name,
            ranger,
            name,
            name,
            ranger,
            dataset,
            "%02d" % (i - 1),
            scratch + name,
            "%02d" % i,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
        )
    elif "cc3m" in metric.lower():

        mask = project_params.param(mp["refine_mask"], i)
        mode = project_params.param(mp["refine_mode"], i)

        # update alignment mask according to settings used in classification
        if int(project_params.param(mp["class_num"], i)) > 1:
            refine_eulers = "0,0,0,"
            if i % int(mp["class_refineeulers"]) == 0:
                refine_eulers = "1,1,1,"
                mode = project_params.param(mp["refine_mode"], i)
            refine_shifts = "0,0"
            if i % int(mp["class_refineshifts"]) == 0:
                refine_shifts = "1,1"
                mode = "1"
            mask = refine_eulers + refine_shifts

        # evaluate scores without refining shifts or angles
        if metric == "eval":
            mask = "0,0,0,1,1"
            mode = "1"

        thresh = 0.0
        command = """
%s/bin/frealign_v9.exe << eot >>%s 2>&1
M, %s, %s, %s, %s, %s, %s, T, %s, %s, %s, %s, F, %s, %s                                !CFORM,IFLAG,FMAG,FDEF,FASTIG,FPART,IEWALD,FBEAUT,FFILT,FBFACT,FMATCH,IFSC,FDUMP,IMEM,INTERP
%s, 0., %s, %s, %s, %s, %s, %s, %s, %s, %s                                        !RO,RI,PSIZE,MW,WGH,XSTD,PBC,BOFF,DANG,ITMAX,IPMAX
%s                                                                                !MASK
%i, %i                                                                                !IFIRST,ILAST
%s
1.0, %s, %s, %s, %s, %s, 0., 0.
%s, %s, %s, %s, %s, %s
%s/%s_stack.mrc
%s_match.mrc_%s
%s.par
%s.par_%s
/dev/null
-100., 0., 0., 0., 0., 0., 0., 0.                                                !terminator with RELMAG=-100.0 to skip 3D reconstruction
%s_%s.mrc
%s_weights_%s_%s
%s_map1_%s.mrc
%s_map2_%s.mrc
%s_phasediffs_%s
%s_pointspread_%s
eot
""" % (
            frealign_paths["cc3m"],
            logfile,
            mode,
            project_params.param(mp["refine_fmag"], i),
            project_params.param(mp["refine_fdef"], i),
            project_params.param(mp["refine_fastig"], i),
            project_params.param(mp["refine_fpart"], i),
            project_params.param(mp["reconstruct_iewald"], i),
            project_params.param(mp["reconstruct_ffilt"], i),
            project_params.param(mp["reconstruct_fbfact"], i),
            project_params.param(mp["refine_fmatch"], i),
            mp["refine_fboost"],
            project_params.param(mp["refine_imem"], i),
            project_params.param(mp["refine_interp"], i),
            mp["particle_rad"],
            pixel,
            mp["particle_mw"],
            mp["scope_wgh"],
            project_params.param(mp["refine_xstd"], i),
            project_params.param(mp["refine_pbc"], i),
            project_params.param(mp["refine_boff"], i),
            project_params.param(mp["refine_dang"], i),
            project_params.param(mp["refine_itmax"], i),
            project_params.param(mp["refine_ipmax"], i),
            mask,
            first,
            last,
            project_params.param(mp["particle_sym"], i),
            dstep,
            project_params.param(mp["refine_target"], i),
            thresh,
            mp["scope_cs"],
            mp["scope_voltage"],
            res_rec,
            project_params.param(mp["refine_rlref"], i),
            postprocess.get_rhref(mp, i),
            project_params.param(mp["class_rhcls"], i),
            project_params.param(mp["refine_dfsig"], i),
            project_params.param(mp["refine_rbfact"], i),
            stack_dir,
            mp["refine_dataset"],
            name,
            ranger,
            name,
            name,
            ranger,
            dataset,
            "%02d" % (i - 1),
            scratch + name,
            "%02d" % i,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
            scratch + name,
            ranger,
        )

    elif "new" in metric or "frealignx" in metric:

        # v9.11

        my_mask = project_params.param(mp["refine_mask"], i).split(",")
        if my_mask[0] == "0":
            psib = "no"
        else:
            psib = "yes"
        if my_mask[1] == "0":
            thetab = "no"
        else:
            thetab = "yes"
        if my_mask[1] == "0":
            phib = "no"
        else:
            phib = "yes"
        if my_mask[3] == "0":
            shxb = "no"
        else:
            shxb = "yes"
        if my_mask[4] == "0":
            shyb = "no"
        else:
            shyb = "yes"

        if (
            mp["refine_fssnr"]
            and os.path.exists("statistics_r%02d.txt" % ref)
            and len(open("statistics_r%02d.txt" % ref).read()) > 0
        ):
            stats = "yes"
        else:
            stats = "no"

        if project_params.param(mp["refine_focusmask"], i) == "0,0,0,0":
            masking = "no"
        else:
            masking = "yes"
        if "t" in project_params.param(mp["refine_fdef"], i).lower():
            defocus = "yes"
        else:
            defocus = "no"
        if "t" in project_params.param(mp["refine_fmatch"], i).lower():
            match = "yes"
        else:
            match = "no"
        if mp["refine_invert"]:
            invert = "yes"
        else:
            invert = "no"

        if project_params.param(mp["refine_mode"], i) == "1":
            globally = "no"
            locally = "yes"
        else:
            globally = "yes"
            locally = "no"

        # set radius for global search
        if project_params.param(mp["refine_srad"], i) == 0:
            srad = str(1.5 * mp["particle_rad"])
        else:
            srad = str(project_params.param(mp["refine_srad"], i))

        boost = "30.0"
        if mp["refine_fboost"]:
            boost = str(mp["refine_fboostlim"])

        mask_2d_x, mask_2d_y, mask_2d_z, mask_2d_rad = project_params.param(
            mp["refine_focusmask"], i
        ).split(",")
        
        #classification using weighted averages when do frames.
        # TODO: should mostly used for tomo
        if mp["class_num"] >1 and "local" in mp["extract_fmt"] and "spr" in mp["data_mode"]:
            stack = "stack_weighted_average.mrc"
        else:
            stack = "stack.mrc"

        if False and "new" in metric:
            command = (
                f"{frealign_paths['new']}/bin/refine3d << eot >>{logfile} 2>&1\n"
                f"{stack_dir}/{fp['refine_dataset']}_{stack}\n"
                f"{name}.par\n{dataset}_{i - 1:02d}.mrc\n"
                f"statistics_r{ref:02d}.txt\n"
                f"{stats}\n"
                f"{name}_match.mrc_{ranger}\n"
                f"{name}.par_{ranger}\n"
                "/dev/null\n"
                f"{project_params.param(fp['particle_sym'], i)}\n"
                f"{first}\n"
                f"{last}\n"
                f"{pixel}\n"
                f"{mp['scope_voltage']}\n"
                f"{mp['scope_cs']}\n"
                f"{mp['scope_wgh']}\n"
                f"{mp['particle_mw']}\n"
                f"{mp['particle_rad']}\n"
                f"{project_params.param(mp['refine_rlref'], i)}\n"
                f"{postprocess.get_rhref(mp, i)}\n"
                f"{boost}\n"
                f"{project_params.param(mp['class_rhcls'], i)}\n"
                f"{srad}\n"
                f"{postprocess.get_rhref(mp, i)}\n"
                f"{project_params.param(mp['refine_dang'], i)}\n"
                "20\n"
                f"{project_params.param(mp['refine_searchx'], i)}\n"
                f"{project_params.param(mp['refine_searchy'], i)}\n"
                f"{mask_2d_x}\n{mask_2d_y}\n{mask_2d_z}\n{mask_2d_rad}\n"
                "500.0\n"
                "50.0\n"
                f"{project_params.param(mp['refine_iblow'], i)}\n"
                f"{globally}\n"
                f"{locally}\n"
                f"{psib}\n"
                f"{thetab}\n"
                f"{phib}\n"
                f"{shxb}\n"
                f"{shyb}\n"
                f"{match}\n"
                f"{masking}\n"
                f"{defocus}\n"
                f"{invert}\n"
                "eot\n"
            )

        elif "frealignx" in metric:

            if project_params.param(mp["reconstruct_norm"], i):
                normalize = "yes"
            else:
                normalize = "no"

            exclude = "no"
            normalize_input = "yes"
            threshold_input = "no"

            if "cistem1" in metric.lower():
                command = (
                    f"{frealign_paths['frealignx']}/refine3d_cistem1 << eot >>{logfile} 2>&1\n"
                    f"{stack_dir}/{fp['refine_dataset']}_stack.mrc\n"
                    f"{name}.par\n{dataset}_{i - 1:02d}.mrc\n"
                    f"statistics_r{ref:02d}.txt\n"
                    f"{stats}\n"
                    f"{name}_match.mrc_{ranger}\n"
                    f"{name}.par_{ranger}\n"
                    "/dev/null\n"
                    f"{project_params.param(fp['particle_sym'], i)}\n"
                    f"{first}\n"
                    f"{last}\n"
                    "1\n"
                    f"{pixel}\n"
                    f"{mp['scope_voltage']}\n"
                    f"{mp['scope_cs']}\n"
                    f"{mp['scope_wgh']}\n"
                    f"{mp['particle_mw']}\n"
                    "0\n"
                    f"{mp['particle_rad']}\n"
                    f"{project_params.param(mp['refine_rlref'], i)}\n"
                    f"{postprocess.get_rhref(mp, i)}\n"
                    f"{boost}\n"
                    f"{project_params.param(mp['class_rhcls'], i)}\n"
                    f"{srad}\n"
                    f"{postprocess.get_rhref(mp, i)}\n"
                    f"{project_params.param(mp['refine_dang'], i)}\n"
                    "20\n"
                    f"{project_params.param(mp['refine_searchx'], i)}\n"
                    f"{project_params.param(mp['refine_searchy'], i)}\n"
                    f"{mask_2d_x}\n{mask_2d_y}\n{mask_2d_z}\n{mask_2d_rad}\n"
                    "500.0\n"
                    "50.0\n"
                    f"{project_params.param(mp['refine_iblow'], i)}\n"
                    f"{globally}\n"
                    f"{locally}\n"
                    f"{psib}\n"
                    f"{thetab}\n"
                    f"{phib}\n"
                    f"{shxb}\n"
                    f"{shyb}\n"
                    f"{match}\n"
                    f"{masking}\n"
                    f"{defocus}\n"
                    "Yes\n"
                    f"{invert}\n"
                    "Yes\n"
                    "Yes\n"
                    "No\n"
                    "eot\n"
                )
            elif "global" in metric.lower():
                command = (
                    "{0}/refine3d_global << eot >>{1} 2>&1\n".format(
                        frealign_paths["frealignx"], logfile
                    )
                    + "{0}/{1}_stack.mrc\n".format(stack_dir, mp["refine_dataset"])
                    + "{0}.par\n{3}\n{1}_{2}.mrc\n".format(
                        name, dataset, "%02d" % (i - 1), global_par
                    )
                    + "statistics_r%02d.txt\n" % ref
                    + stats
                    + "\n"
                    + "{0}_match.mrc_{1}\n{0}.par_{1}\n/dev/null\n".format(name, ranger)
                    + project_params.param(mp["particle_sym"], i)
                    + "\n"
                    + "{0}\n{1}\n1\n".format(first, last)
                    + "{0}\n{1}\n{2}\n{3}\n{4}\n0\n".format(
                        pixel,
                        mp["scope_voltage"],
                        mp["scope_cs"],
                        mp["scope_wgh"],
                        mp["particle_mw"],
                    )
                    + str(mp["particle_rad"])
                    + "\n"
                    + str(project_params.param(mp["refine_rlref"], i))
                    + "\n"
                    + str(postprocess.get_rhref(mp, i))
                    + "\n"
                    + boost
                    + "\n"
                    + str(project_params.param(mp["class_rhcls"], i))
                    + "\n"
                    + srad
                    + "\n"
                    + str(postprocess.get_rhref(mp, i))
                    + "\n"
                    + str(project_params.param(mp["refine_dang"], i))
                    + "\n"
                    + "20\n"
                    + str(project_params.param(mp["refine_searchx"], i))
                    + "\n"
                    + str(project_params.param(mp["refine_searchy"], i))
                    + "\n"
                    + "\n".join(
                        project_params.param(mp["refine_focusmask"], i).split(",")
                    )
                    + "\n"
                    + "500.0\n"
                    + "50.0\n"
                    + str(project_params.param(mp["refine_iblow"], i))
                    + "\n"
                    + globally
                    + "\n"
                    + locally
                    + "\n"
                    + psib
                    + "\n"
                    + thetab
                    + "\n"
                    + phib
                    + "\n"
                    + shxb
                    + "\n"
                    + shyb
                    + "\n"
                    + match
                    + "\n"
                    + masking
                    + "\n"
                    + defocus
                    + "\n"
                    + normalize
                    + "\n"
                    + invert
                    + "\n"
                    + exclude
                    + "\n"
                    + normalize_input
                    + "\n"
                    + threshold_input
                    + "\n"
                    "eot\n"
                )
            else:
                refine3d = "refine3d" if mp["refine_priors"] else "refine3d_no_prior"
                command = (
                    "{0}/{2} << eot >>{1} 2>&1\n".format(
                        frealign_paths["frealignx"], logfile, refine3d
                    )
                    + "{0}/{1}_stack.mrc\n".format(stack_dir, mp["refine_dataset"])
                    + "{0}.par\n{1}_{2}.mrc\n".format(name, dataset, "%02d" % (i - 1))
                    + "statistics_r%02d.txt\n" % ref
                    + stats
                    + "\n"
                    + "{0}_match.mrc_{1}\n{0}.par_{1}\n/dev/null\n".format(name, ranger)
                    + project_params.param(mp["particle_sym"], i)
                    + "\n"
                    + "{0}\n{1}\n1\n".format(first, last)
                    + "{0}\n{1}\n{2}\n{3}\n{4}\n0\n".format(
                        pixel,
                        mp["scope_voltage"],
                        mp["scope_cs"],
                        mp["scope_wgh"],
                        mp["particle_mw"],
                    )
                    + str(mp["particle_rad"])
                    + "\n"
                    + str(project_params.param(mp["refine_rlref"], i))
                    + "\n"
                    + str(postprocess.get_rhref(mp, i))
                    + "\n"
                    + boost
                    + "\n"
                    + str(project_params.param(mp["class_rhcls"], i))
                    + "\n"
                    + srad
                    + "\n"
                    + str(postprocess.get_rhref(mp, i))
                    + "\n"
                    + str(project_params.param(mp["refine_dang"], i))
                    + "\n"
                    + "20\n"
                    + str(project_params.param(mp["refine_searchx"], i))
                    + "\n"
                    + str(project_params.param(mp["refine_searchy"], i))
                    + "\n"
                    + "\n".join(
                        project_params.param(mp["refine_focusmask"], i).split(",")
                    )
                    + "\n"
                    + "500.0\n"
                    + "50.0\n"
                    + str(project_params.param(mp["refine_iblow"], i))
                    + "\n"
                    + globally
                    + "\n"
                    + locally
                    + "\n"
                    + psib
                    + "\n"
                    + thetab
                    + "\n"
                    + phib
                    + "\n"
                    + shxb
                    + "\n"
                    + shyb
                    + "\n"
                    + match
                    + "\n"
                    + masking
                    + "\n"
                    + defocus
                    + "\n"
                    + normalize
                    + "\n"
                    + invert
                    + "\n"
                    + exclude
                    + "\n"
                    + normalize_input
                    + "\n"
                    + threshold_input
                    + "\n"
                    "eot\n"
                )

        else:

            # cistem2
            #
            normalize = "yes"
            edges = "no"
            normalize_rec = "no"
            thresh_rec = "no"

            command = (
                "{0}/refine3d << eot >>{1} 2>&1\n".format(
                    frealign_paths["cistem2"], logfile
                )
                + "{0}/{1}_stack.mrc\n".format(stack_dir, mp["refine_dataset"])
                + "{0}.cistem\n{0}.mrc\n".format(name)
                + "statistics_r%02d.txt\n" % ref
                + stats
                + "\n"
                + "{0}_match.mrc_{1}\n{0}_{1}.cistem\n{0}_{1}_changes.cistem\n".format(
                    name, ranger
                )
                + str(project_params.param(fp["particle_sym"], i))
                + "\n"
                + "{0}\n{1}\n1\n".format(first, last)
                + "{0}\n{1}\n".format(pixel, mp["particle_mw"])
                + "0\n"
                + str(mp["particle_rad"])
                + "\n"
                + str(project_params.param(mp["refine_rlref"], i))
                + "\n"
                + str(postprocess.get_rhref(mp, i))
                + "\n"
                + boost
                + "\n"
                + str(project_params.param(mp["class_rhcls"], i))
                + "\n"
                + srad
                + "\n"
                + str(postprocess.get_rhref(mp, i))
                + "\n"
                + str(project_params.param(mp["refine_dang"], i))
                + "\n"
                + "20\n"
                + str(project_params.param(mp["refine_searchx"], i))
                + "\n"
                + str(project_params.param(mp["refine_searchy"], i))
                + "\n"
                + "\n".join(project_params.param(mp["refine_focusmask"], i).split(","))
                + "\n"
                + "500.0\n" # defocus_search_range
                + "50.0\n" # defocus_step
                + str(project_params.param(mp["refine_iblow"], i))
                + "\n"
                + globally
                + "\n"
                + locally
                + "\n"
                + psib
                + "\n"
                + thetab
                + "\n"
                + phib
                + "\n"
                + shxb
                + "\n"
                + shyb
                + "\n"
                + match
                + "\n"
                + masking
                + "\n"
                + defocus
                + "\n"
                + normalize
                + "\n"
                + invert
                + "\n"
                + edges
                + "\n"
                + normalize_rec
                + "\n"
                + thresh_rec
                + "\neot\n"
            )
    else:

        # frealignx

        my_mask = project_params.param(mp["refine_mask"], i).split(",")
        if my_mask[0] == "0":
            psib = "no"
        else:
            psib = "yes"
        if my_mask[1] == "0":
            thetab = "no"
        else:
            thetab = "yes"
        if my_mask[1] == "0":
            phib = "no"
        else:
            phib = "yes"
        if my_mask[3] == "0":
            shxb = "no"
        else:
            shxb = "yes"
        if my_mask[4] == "0":
            shyb = "no"
        else:
            shyb = "yes"

        if (
            mp["refine_fssnr"]
            and os.path.exists("statistics_r%02d.txt" % ref)
            and len(open("statistics_r%02d.txt" % ref).read()) > 0
        ):
            stats = "yes"
        else:
            stats = "no"

        if project_params.param(mp["refine_focusmask"], i) == "0,0,0,0":
            masking = "no"
        else:
            masking = "yes"
        if "t" in project_params.param(mp["refine_fdef"], i).lower():
            defocus = "yes"
        else:
            defocus = "no"
        if "t" in project_params.param(mp["refine_fmatch"], i).lower():
            match = "yes"
        else:
            match = "no"
        if mp["refine_invert"]:
            invert = "yes"
        else:
            invert = "no"

        if project_params.param(mp["refine_mode"], i) == "1":
            globally = "no"
            locally = "yes"
        else:
            globally = "yes"
            locally = "no"

        # set radius for global search
        if project_params.param(mp["refine_srad"], i) == "0":
            srad = str(1.5 * float(mp["particle_rad"]))
        else:
            srad = project_params.param(mp["refine_srad"], i)

        boost = "30.0"
        if mp["refine_fboost"]:
            boost = mp["refine_fboostlim"]

        if project_params.param(mp["reconstruct_norm"], i):
            normalize = "yes"
        else:
            normalize = "no"

        command = (
            "{0}/refine3d << eot >>{1} 2>&1\n".format(
                frealign_paths["frealignx"], logfile
            )
            + "{0}/{1}_stack.mrc\n".format(stack_dir, fp["refine_dataset"])
            + "{0}.par\n{1}_{2}.mrc\n".format(name, dataset, "%02d" % (i - 1))
            + "statistics_r%02d.txt\n" % ref
            + stats
            + "\n"
            + "{0}_match.mrc_{1}\n{0}.par_{1}\n/dev/null\n".format(name, ranger)
            + project_params.param(mp["particle_sym"], i)
            + "\n"
            + "{0}\n{1}\n".format(first, last)
            + "1.0\n"
            + "{0}\n{1}\n{2}\n{3}\n{4}\n".format(
                pixel,
                mp["scope_voltage"],
                mp["scope_cs"],
                mp["scope_wgh"],
                mp["particle_mw"],
            )
            + "0\n"
            + mp["particle_rad"]
            + "\n"
            + project_params.param(mp["refine_rlref"], i)
            + "\n"
            + str(postprocess.get_rhref(mp, i))
            + "\n"
            + boost
            + "\n"
            + project_params.param(mp["class_rhcls"], i)
            + "\n"
            + srad
            + "\n"
            + str(postprocess.get_rhref(mp, i))
            + "\n"
            + project_params.param(mp["refine_dang"], i)
            + "\n"
            + "20\n"
            + project_params.param(mp["refine_searchx"], i)
            + "\n"
            + project_params.param(mp["refine_searchy"], i)
            + "\n"
            + "\n".join(project_params.param(mp["refine_focusmask"], i).split(","))
            + "\n"
            + "500.0\n"
            + "50.0\n"
            + project_params.param(mp["refine_iblow"], i)
            + "\n"
            + globally
            + "\n"
            + locally
            + "\n"
            + psib
            + "\n"
            + thetab
            + "\n"
            + phib
            + "\n"
            + shxb
            + "\n"
            + shyb
            + "\n"
            + match
            + "\n"
            + masking
            + "\n"
            + defocus
            + "\n"
            + normalize
            + "\n"
            + invert
            + "\n"
            + "yes\n"
            + "yes\n"
            + "no\n"
            + "eot\n"
        )

    return command

def refine2d(
    input_particle_stack: str,
    input_frealign_par: str,
    input_reconstruction: str, # Available only after first iteration
    parameters: dict,
    pngfile: PosixPath,
    new_name: str,
    output_frealign_par: str = "my_refined_parameters.par",
    output_reconstruction: str = "my_refined_classes.mrc",
    logfile: str = "debug.log",
    dump_file: str = "dump_file.dat",
    classes: int = 0,
    first_particle_to_refine: int = 1,
    last_particle_to_refine: int = 0,
    class_fraction: float = 1.0,
    low_res_limit: int = 300,
    high_res_limit: int = 40
):
    # Reference will be use for the next iteration. How about first iteration?

    # Parallelized implementation
    command = (
        f"{get_frealign_paths()['frealignx']}/refine2d << eot > {logfile} 2>&1\n"
        f"{input_particle_stack}\n"
        f"{input_frealign_par}\n"
        f"{input_reconstruction}\n"
        f"{output_frealign_par}\n"
        f"{output_reconstruction}\n" # for next iteration
        f"{classes}\n" # number of classes
        f"{first_particle_to_refine}\n" # First particle to refine
        f"{last_particle_to_refine}\n" # last particle to refine. "0" means use all particles
        f"{class_fraction:.2f}\n" # percentage of particles to use. "1" means all
        f"{parameters['scope_pixel']*parameters['class2d_bin']}\n"
        f"{parameters['scope_voltage']}\n"
        f"{parameters['scope_cs']}\n"
        f"{parameters['scope_wgh']}\n"
        f"{parameters['detect_rad']}\n" # TODO: fix
        f"{low_res_limit}\n"  # low resolution limit
        f"{high_res_limit}\n"  # high resolution limit
        "0\n"
        "0\n"
        "0\n"
        "1\n"
        "2\n"
        "Yes\n"
        "No\n"
        "Yes\n"
        "Yes\n"
        f"{dump_file}\n"
        "eot\n"
    )

    """
            **   Welcome to Refine2D   **

             Version : 1.02
            Compiled : Sep 25 2022
                Mode : Interactive

Input particle images [my_image_stack.mrc]         : test1.mrc
Input Frealign parameter filename
[my_parameters.par]                                : 20221127_152011_test-5_01.par
Input class averages [my_input_classes.mrc]        :
Output parameter file [my_refined_parameters.par]  :
Output class averages [my_refined_classes.mrc]     :
Number of classes (>0 = initialize classes) [0]    :
First particle to refine (0 = first in stack) [1]  :
Last particle to refine (0 = last in stack) [0]    :
Percent of particles to use (1 = all) [1.0]        :
Pixel size of images (A) [1.0]                     :
Beam energy (keV) [300.0]                          :
Spherical aberration (mm) [2.7]                    :
Amplitude contrast [0.07]                          :
Mask radius (A) [100.0]                            :
Low resolution limit (A) [300.0]                   :
High resolution limit (A) [8.0]                    :
Angular step (0.0 = set automatically) [0.0]       :
Search range in X (A) (0.0 = max) [0.0]            :
Search range in Y (A) (0.0 = max) [0.0]            :
Tuning parameter: smoothing factor [1.0]           :
Tuning parameter: padding factor for interpol. [2] :
Normalize particles [Yes]                          :
Invert particle contrast [No]                      :
Exclude images with blank edges [Yes]              :
Dump intermediate arrays (merge later) [No]        :
Output dump filename for intermediate arrays
[dump_file.dat]                                    :
    """

    output, error = local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

    if parameters["slurm_verbose"]:
        with open(logfile) as f:
            logger.info(f.read())

    plot_refine2d_reconstructions(output_reconstruction, new_name, pngfile, parameters)
    


def plot_refine2d_reconstructions(output_reconstruction: str, new_name: str, pngfile: PosixPath, parameters: dict, occ_classes: dict = {}):
    
    cols = 10
    
    # sort the classes by their occ
    if len(occ_classes) > 0:
        class_ind = [(cls, occ_classes[cls]) for cls in occ_classes.keys()]
        class_ind.sort(key=lambda x: x[1], reverse=True)
        class_ind = [_[0]-1 for _ in class_ind] # only store class index (class number - 1) 
    else:
        class_ind = None
    N = plot.contact_sheet(
        mrc.read(output_reconstruction), cols=cols, rescale=True, order=class_ind
    )
    writepng(data=N, pngfile=str(pngfile))
    webpfile = str(pngfile).replace(".png", ".webp")
    img2webp(pngfile, webpfile)

    # notify website that a new set of classes is ready to be displayed
    metadata = {}
    metadata["path"] = str(webpfile)
    save_classes_to_website(new_name, metadata)

    os.remove(pngfile)


def refine2d_mpi(
    input_particle_stack: str,
    input_frealign_par: str,
    input_reconstruction: str, # Available only after first iteration
    name: str,
    parameters: dict,
    classes: int = 0, 
    class_fraction: float = 1.0,
    low_res_limit: int = 300,
    high_res_limit: int = 40, 
):

    # get the number of particles 
    num_particles = frealign_parfile.Parameters.from_file(input_frealign_par).data.shape[0]

    cores = parameters['slurm_class2d_tasks']
    increment = math.floor(num_particles * 1.0 / cores)

    commands = []
    splitted_parfiles = []
    dumpfiles = []

    for idx, first in enumerate(range(1, num_particles+1, increment+1)):
        last = min(first+increment, num_particles)
        ranger = "%07d_%07d" % (first, last)
        logfile = f"{ranger}_refine2d.log" if idx == 0 else "/dev/null"
        output_frealign_par = f"{name}_{ranger}.par"
        output_reconstruction = f"{name}_{ranger}.mrc"
        dumpfile = f"dump_{ranger}.dat" 

        command = (
            f"{get_frealign_paths()['frealignx']}/refine2d << eot > {logfile} 2>&1\n"
            f"{input_particle_stack}\n"
            f"{input_frealign_par}\n"
            f"{input_reconstruction}\n"
            f"{output_frealign_par}\n"
            f"{output_reconstruction}\n" # for next iteration
            f"{classes}\n" # number of classes
            f"{first}\n" # First particle to refine
            f"{last}\n" # last particle to refine. "0" means use all particles
            f"{class_fraction:.2f}\n" # percentage of particles to use. "1" means all
            f"{parameters['scope_pixel']*parameters['class2d_bin']}\n"
            f"{parameters['scope_voltage']}\n"
            f"{parameters['scope_cs']}\n"
            f"{parameters['scope_wgh']}\n"
            f"{parameters['detect_rad']}\n" # TODO: fix
            f"{low_res_limit}\n"  # low resolution limit
            f"{high_res_limit}\n"  # high resolution limit
            "0\n"
            "0\n"
            "0\n"
            "1\n"
            "2\n"
            "Yes\n"
            "No\n"
            "Yes\n"
            "Yes\n"
            f"{dumpfile}\n"
            "eot\n"
        )

        splitted_parfiles.append(output_frealign_par)
        dumpfiles.append(dumpfile)
        commands.append(command)
        
        """
                **   Welcome to Refine2D   **

                Version : 1.02
                Compiled : Sep 25 2022
                    Mode : Interactive

    Input particle images [my_image_stack.mrc]         : test1.mrc
    Input Frealign parameter filename
    [my_parameters.par]                                : 20221127_152011_test-5_01.par
    Input class averages [my_input_classes.mrc]        :
    Output parameter file [my_refined_parameters.par]  :
    Output class averages [my_refined_classes.mrc]     :
    Number of classes (>0 = initialize classes) [0]    :
    First particle to refine (0 = first in stack) [1]  :
    Last particle to refine (0 = last in stack) [0]    :
    Percent of particles to use (1 = all) [1.0]        :
    Pixel size of images (A) [1.0]                     :
    Beam energy (keV) [300.0]                          :
    Spherical aberration (mm) [2.7]                    :
    Amplitude contrast [0.07]                          :
    Mask radius (A) [100.0]                            :
    Low resolution limit (A) [300.0]                   :
    High resolution limit (A) [8.0]                    :
    Angular step (0.0 = set automatically) [0.0]       :
    Search range in X (A) (0.0 = max) [0.0]            :
    Search range in Y (A) (0.0 = max) [0.0]            :
    Tuning parameter: smoothing factor [1.0]           :
    Tuning parameter: padding factor for interpol. [2] :
    Normalize particles [Yes]                          :
    Invert particle contrast [No]                      :
    Exclude images with blank edges [Yes]              :
    Dump intermediate arrays (merge later) [No]        :
    Output dump filename for intermediate arrays
    [dump_file.dat]                                    :
        """

    assert len(commands) > 0, f"{input_frealign_par} does not have particles"
    mpi.submit_jobs_to_workers(commands, os.getcwd(), silent=True)

    return splitted_parfiles, dumpfiles


def merge2d(  
    cycle: int,
    working_directory: PosixPath,
    dumpfiles: list,
    output_reconstruction: str, 
    parameters: dict,
    logfile: str = "merge2d.log"
) -> None:


    num_dump_files = len(dumpfiles)

    # rename the dumpfiles to the format to feed merge2d (suppose dumpfiles is in the current directory)
    seed = "dump_.dat"
    seed_prefix, ext = os.path.splitext(seed)
    [os.rename(f, f"{seed_prefix}{idx+1}{ext}") for idx, f in enumerate(dumpfiles)]
    
    command = (
        f"{get_frealign_paths()['frealignx']}/merge2d << eot > {logfile} 2>&1\n"
        f"{output_reconstruction}\n"
        f"{seed}\n"
        f"{num_dump_files}\n"
        "eot\n"
    )

    """
        **   Welcome to Merge2D   **

            Version : 1.00
           Compiled : Dec 15 2022
    Library Version : 2.0.0-alpha--1--dirty
        From Branch : (HEAD
               Mode : Interactive

Output class averages [my_refined_classes.mrc]     : 
Seed for input dump filenames for intermediate arrays
[dump_file_seed_.dat]                              : 
Number of dump files [1]                           : 

Error : Error: Dump file dump_file_seed_1.dat not found
    """

    output, error = local_run.run_shell_command(command, verbose=False)

    if parameters["slurm_verbose"]:
        with open(logfile) as f:
            logger.info(f.read())

    [os.remove(f) for f in os.listdir(".") if f.endswith(".dat")]


    # get the order of classes based on output occupencies 

    return get_occ_classes()


def get_occ_classes(logfile: str = "merge2d.log"):
    occ_classes = {}
    with open(logfile, "r") as f:
        for line in f.readlines():
            if line.startswith("Class ="):
                data = line.strip().split(",")
                cls = int(data[0].split("=")[-1])
                occ = float(data[1].split("=")[-1])         
                occ_classes[cls] = occ
    return occ_classes

def refine_ctf(mp, fp, i, ref):

    fp = mp

    my_mask = project_params.param(fp["refine_mask"], i).split(",")
    if my_mask[0] == "0":
        psib = "no"
    else:
        psib = "yes"
    if my_mask[1] == "0":
        thetab = "no"
    else:
        thetab = "yes"
    if my_mask[1] == "0":
        phib = "no"
    else:
        phib = "yes"
    if my_mask[3] == "0":
        shxb = "no"
    else:
        shxb = "yes"
    if my_mask[4] == "0":
        shyb = "no"
    else:
        shyb = "yes"

    if (
        fp["refine_fssnr"]
        and os.path.exists("statistics_r%02d.txt" % ref)
        and len(open("statistics_r%02d.txt" % ref).read()) > 0
    ):
        stats = "yes"
    else:
        stats = "no"

    if project_params.param(fp["refine_focusmask"], i) == "0,0,0,0":
        masking = "no"
    else:
        masking = "yes"
    if "t" in project_params.param(fp["refine_fdef"], i).lower():
        defocus = "yes"
    else:
        defocus = "no"

    defocus = "yes"

    if "t" in project_params.param(fp["refine_fmatch"], i).lower():
        match = "yes"
    else:
        match = "no"
    if fp["refine_invert"]:
        invert = "yes"
    else:
        invert = "no"

    if project_params.param(fp["refine_mode"], i) == "1":
        globally = "no"
        locally = "yes"
    else:
        globally = "yes"
        locally = "no"

    # set radius for global search
    if project_params.param(fp["refine_srad"], i) == "0":
        srad = str(1.5 * float(mp["particle_rad"]))
    else:
        srad = project_params.param(fp["refine_srad"], i)

    if fp["refine_beamtilt"]:
        beamtilt = "yes"
    else:
        beamtilt = "no"

    boost = "30.0"
    if fp["refine_fboost"]:
        boost = fp["refine_fboostlim"]

    normalize = "yes"
    edge = "no"
    thresh_rec = "no"
    threads = str(multiprocessing.cpu_count())

    dataset = fp["refine_dataset"]
    name = dataset + "_r01_%02d" % (i)

    first = last = "0"

    pixel = float(mp["scope_pixel"]) * float(mp["data_bin"]) * float(mp["extract_bin"])

    logfile = "../log/%s_%s_%02d_ctf.log" % (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S"),
        dataset,
        i,
    )

    # add column to star file
    p = frealign_parfile.Parameters("cistem")
    p.read_star(name)
    entry = frealign_parfile.ParameterEntry(
        "Reference3DFilename", "REFERENCE_3D_FILENAME", "%10s"
    )
    p.add_column(
        entry, "{0}_r{1}_{2}.mrc".format(dataset, "%02d" % ref, "%02d" % (i - 1))
    )
    p.write_star(name + "_ctf")

    frealign_paths = get_frealign_paths()

    command = (
        "{0}/refine_ctf << eot >>{1} 2>&1\n".format(frealign_paths["cistem2"], logfile)
        + "../{}_stack.mrc\n".format(fp["refine_dataset"])
        + "{0}_ctf.star\n{1}_r{2}_{3}.mrc\n".format(
            name, dataset, "%02d" % ref, "%02d" % (i)
        )
        + "statistics_r%02d.txt\n" % ref
        + stats
        + "\n"
        + "{0}_refine_ctf.star\n{0}_ctf_changes.star\n".format(name)
        + "{0}_phase_difference.mrc\n{0}_beam_tilt_image.mrc\n{0}_difference_image.mrc\n".format(
            name
        )
        + "{0}\n{1}\n".format(first, last)
        + "{0}\n{1}\n".format(pixel, mp["particle_mw"])
        + "0\n"
        + mp["particle_rad"]
        + "\n"
        + project_params.param(fp["refine_rlref"], i)
        + "\n"
        + str(postprocess.get_rhref(fp, i))
        + "\n"
        + "500.0\n"
        + "50.0\n"
        + project_params.param(fp["refine_iblow"], i)
        + "\n"
        + defocus
        + "\n"
        + beamtilt
        + "\n"
        + normalize
        + "\n"
        + invert
        + "\n"
        + edge
        + "\n"
        + normalize
        + "\n"
        + thresh_rec
        + "\n"
        + threads
        + "\n"
        + "eot\n"
    )

    local_run.run_shell_command(command)


"""
        **   Welcome to RefineCTF   **

             Version : 1.00
            Compiled : Mar 27 2020
                Mode : Interactive

Input particle images [../pyp_frames_stack.mrc]    :
Input cisTEM star filename
[pyp_frames_r01_03.star]                           :
Input reconstruction [pyp_frames_r01_02.mrc]       :
Input data statistics [my_statistics.txt]          :
Use statistics [no]                                :
Output star file [my_refined_parameters.star]      :
Output parameter changes
[my_parameter_changes.par]                         :
Output phase difference image
[my_phase_difference.mrc]                          :
Output beam tilt image [my_beamtilt_image.mrc]     :
Output phase diff - beam tilt
[my_difference_image.mrc]                          :
First particle to refine (0 = first in stack) [1]  :
Last particle to refine (0 = last in stack) [0]    :
Pixel size of reconstruction (A) [1.0]             :
Molecular mass of particle (kDa) [1000.0]          :
Inner mask radius (A) [0.0]                        :
Outer mask radius (A) [100.0]                      :
Low resolution limit (A) [300.0]                   :
High resolution limit (A) [8.0]                    :
Defocus search range (A) [500.0]                   :
Defocus step (A) [50.0]                            :
Tuning parameters: padding factor [1.0]            :
Refine defocus [No]                                : yes
Estimate beamtilt [No]                             : yes
Normalize particles [Yes]                          :
Invert particle contrast [No]                      :
Exclude images with blank edges [Yes]              :
Normalize input reconstruction [Yes]               :
Threshold input reconstruction [No]                :
Max. threads to use for calculation [1]            :
"""


def create_initial_model(parameters, actual_pixel, box_size, local_parameters):
    model_box_size = int(mrc.readHeaderFromFile(parameters["class_ref"])["nx"])
    model_pixel_size = float(
        mrc.readHeaderFromFile(parameters["class_ref"])["xlen"]
    ) / float(model_box_size)
    local_pixel = float(local_parameters["scope_pixel"]) * float(
        local_parameters["data_bin"]
    )
    if abs(model_pixel_size - actual_pixel) > 0.01 or int(box_size) != model_box_size:
        #
        # TODO: substitute by from pyp.preprocess import resample_and_reshape
        #
        if False:
            """
            load_eman_cmd = eman_load_command()
            command = "{4}; e2proc3d.py {0} {1} --scale={2} --clip={3}".format(
                parameters["class_ref"],
                os.path.split(parameters["class_ref"])[-1],
                "%.2f" % (model_pixel_size / local_pixel),
                box_size,
                load_eman_cmd,
            )
            local_run.run_shell_command(command)
            """
        else:
            from pyp.preprocess import resample_and_resize

            resample_and_resize(
                input=parameters["class_ref"],
                output=os.path.split(parameters["class_ref"])[-1],
                scale=model_pixel_size / local_pixel,
                size=box_size,
            )

    else:

        local_model = os.path.split(parameters["class_ref"])[-1]

        fparameters = project_params.load_fyp_parameters(
            os.path.split(parameters["class_par"])[0] + "/../"
        )

        if not os.path.exists(fparameters["maskth"]):

            shutil.copy(parameters["class_ref"], local_model)

        else:

            # apply mask to reference

            """
            Image format [M,S,I]?
            Input 3D map?
            Pixel size in A?
            Input 3D mask?
            Width of cosine edge to add (in pixel)?
            Weight for density outside mask?
            Low-pass filter outside (0=no, 1=Gauss, 2=cosine edge)?
            Cosine edge filter radius in A?
            Width of edge in pixels?
            Output masked 3D map?
            """

            frealign_paths = get_frealign_paths()

            if project_params.param(fparameters["mask_weight"], 100) > 0:
                command = """
%s/bin/apply_mask.exe << eot
M
%s
*
%s
3
%s
2
10
10
%s
eot
""" % (
                    frealign_paths["new"],
                    parameters["class_ref"],
                    project_params.param(fparameters["maskth"], 100),
                    project_params.param(fparameters["mask_weight"], 100),
                    local_model,
                )

            else:
                command = """
%s/bin/apply_mask.exe << eot
M
%s
*
%s
3
0
0.0
%s
eot
""" % (
                    frealign_paths["new"],
                    parameters["class_ref"],
                    project_params.param(fparameters["maskth"], 100),
                    local_model,
                )
            local_run.run_shell_command(command)


def setup_refinement_files(fp, iteration, dataset, frealign_paths, ref):
    name = "%s_r%02d" % (dataset, ref + 1)
    previous = "maps/%s_%02d" % (name, iteration - 1)
    current = "%s_%02d" % (name, iteration)

    is_frealignx = isfrealignx(previous + ".par")

    create_curr_iter_par(iteration, name, previous, current, is_frealignx)

    # shape masking?
    if os.path.exists(project_params.param(fp["refine_maskth"], iteration)):
        """
        Image format [M,S,I]?
        Input 3D map?
        Pixel size in A?
        Input 3D mask?
        Width of cosine edge to add (in pixel)?
        Weight for density outside mask?
        Low-pass filter outside (0=no, 1=Gauss, 2=cosine edge)?
        Cosine edge filter radius in A?
        Width of edge in pixels?
        Output masked 3D map?
        """

        if float(project_params.param(fp["refine_mask_weight"], iteration)) > 0:
            # TODO: probably already a similar function
            command = """
%s/bin/apply_mask.exe << eot
M
%s.mrc
*
%s
3
%s
2
10
10
scratch/%s.mrc
eot
""" % (
                frealign_paths["new"],
                previous,
                project_params.param(fp["maskth"], iteration),
                project_params.param(fp["mask_weight"], iteration),
                "%s_r%02d_%02d" % (dataset, ref + 1, iteration - 1),
            )

        else:

            # see if mask is already apodized
            mask = mrc.read(project_params.param(fp["refine_maskth"], iteration))
            if np.where(np.logical_and(mask > 0.0, mask < 1.0), 1, 0).sum() > 0:
                apodization = 0
            else:
                apodization = 3

            command = """
%s/bin/apply_mask.exe << eot
M
%s.mrc
*
%s
%d
0
0.0
scratch/%s.mrc
eot
""" % (
                frealign_paths["new"],
                previous,
                project_params.param(fp["refine_maskth"], iteration),
                apodization,
                "%s_r%02d_%02d" % (dataset, ref + 1, iteration - 1),
            )
        local_run.run_shell_command(command)
        load_eman_cmd = eman_load_command()
        command = "{0}; e2proc3d.py {1}.mrc scratch/{2}.mrc --multfile {3}".format(
            load_eman_cmd,
            previous,
            "%s_r%02d_%02d" % (dataset, ref + 1, iteration - 1),
            project_params.param(fp["maskth"], iteration),
        )
        # print command
        # print commands.getoutput(command)

    # elif float(project_params.param(fp['maskth'],iteration)) == -1:
    elif not os.path.exists(fp["refine_maskth"]):
        if float(project_params.param(fp["reconstruct_cutoff"], iteration)) >= 0:
            shutil.copy(previous + ".mrc", "scratch/")
        else:
            # avoid costly copy operation
            try:
                symlink_relative(
                    os.path.join(os.getcwd(), previous + ".mrc"),
                    "scratch/" + previous.replace("maps/", "") + ".mrc",
                )
            except:
                pass

    else:
        load_eman_cmd = eman_load_command()
        command = "{0}; e2proc3d.py {1}.mrc scratch/{2}.mrc --process=mask.auto3d:radius=200:threshold={3}:nshells=4:nshellsgauss=3:nmaxseed=0".format(
            load_eman_cmd,
            previous,
            "%s_r%02d_%02d" % (dataset, ref + 1, iteration - 1),
            project_params.param(fp["refine_maskth"], iteration),
        )
        local_run.run_shell_command(command)

    # collect FREALIGN statistics
    com = (
        """grep -A 10000 "C  NO.  RESOL  RING RAD" {0}.par""".format(previous)
        + """ | grep -v RESOL | grep -v Average | grep -v Date | grep C | awk '{if ($2 != "") printf "%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f\\n", $2, $3, $4, $6, $7, $8, $9}' > """
        + str(Path(os.environ["PYP_SCRATCH"]) / f"statistics_r{(ref + 1):02d}.txt")
    )
    local_run.run_shell_command(com, verbose=False)

    # smooth part FSC curves
    if project_params.param(fp["refine_metric"], iteration) == "new":

        stats_file_name = (
            Path(os.environ["PYP_SCRATCH"]) / f"statistics_r{(ref + 1):02d}.txt"
        )

        plot_name = "maps/" + current + "_snr.png"

        postprocess.smooth_part_fsc(
            str(stats_file_name), plot_name
        )  # TODO: safe to pass Path here?


def launch_frealign_refinement(parameters, new_name, fp):
    """In pyp_main.py, launch FYP"""
    if "mpi" in parameters["extract_fmt"].lower():

        fp["refine_dataset"] = new_name
        if int(parameters["class_num"]) > 1:
            fp["class_num"] = "1:%s" % parameters["class_num"]
            fp["refine_maxiter"] = 100
            fp["refine_mode"] = "0:1"
            fp["reconstruct_cutoff"] = "1"
        else:
            fp["class_num"] = "1"
            fp["refine_maxiter"] = 8
            fp["particle_sym"] = parameters["particle_sym"]
        fp["refine_iter"] = 2
        fp["refine_model"] = parameters["class_ref"]
        fp["refine_metric"] = "new"
        project_params.save_fyp_parameters(fp, "frealign")
        command = "cd frealign; qsub -l nodes=33:c24 {0}".format(
            run_pyp("fyp", script=False)
        )

    else:

        if int(parameters["class_num"]) > 1 and int(parameters["extract_cls"]) > 0:
            # do only local refinements, classify, no auto-threshold, single-resolution
            classification = (
                "-classes 1:%s -maxiter 24 -cutoff 1 -mode 0:1 -rhref 4"
                % parameters["class_num"]
            )
        else:
            if int(parameters["extract_cls"]) > 0:
                # do only local refinements, no classification, no auto-threshold, single-resolution
                # classification = '-classes 1 -maxiter 8 -cutoff 1 -mode 0:1 -mask 0,0,0,0,0:0,0,0,1,1:0,0,0,1,1:0,0,0,1,1:1,1,1,1,1 -rhref 4'
                classification = "-classes 1 -maxiter 8 -cutoff 1 -mode 0:1 -rhref 4"
                classification = "-classes 1 -maxiter 8 -cutoff 1 -mode 0:1"
                # classification = '-classes 1 -maxiter 2 -cutoff 1 -mode 0:1'
            else:
                # do full refinements, no classification, auto-threshold, multi-resolution
                classification = (
                    "-classes 1 -maxiter 8 -cutoff 0 -mode 4:4:4:4:1 -rhref 16:12:8:4"
                )
                classification = (
                    "-classes 1 -maxiter 2 -cutoff 0 -mode 4:4:4:4:1 -rhref 16:12:8:4"
                )
                if fp == 0:
                    classification = (
                        "-classes 1 -cutoff 0.05:0 -mode 4:4:4:4:1 -rhref 8 -maxiter 8"
                    )
                else:
                    classification = "-classes 1 -cutoff 0.05:0 -mode 4:4:4:4:1 -rhref {0} -maxiter 8".format(
                        str(fp["refine_rhref"]).split(":")[-1]
                    )

        if not os.path.exists("frealign/mpirun.mynodes"):
            # fork out new processes
            command = "cd frealign; unset PBS_O_WORKDIR; unset PBS_NODEFILE; unset SLURM_NODELIST; {0} -dataset {1} -iter 2 {2} -model {3}".format(
                run_pyp("fyp", script=False),
                new_name,
                classification,
                parameters["class_ref"],
            )
        else:
            # continue using present set of nodes
            # command = "cd frealign; {0} -dataset {1} -metric new -iter 2 {2} -model {3}".format(
            command = "cd frealign; {0} -dataset {1} -iter 2 {2} -model {3}".format(
                run_pyp("fyp", script=False),
                new_name,
                classification,
                parameters["class_ref"],
            )
    logger.info(command)
    # flush the output buffers so the command output actually appears after the command
    sys.stdout.flush()
    # run the command and wait for it to finish while dumping the output to stdout in real-time
    # if the process exits with an error code, also throw an exception
    local_run.run_shell_command(command)


def rec_merge_check_class_converge(fparameters, classes, iteration):

    converged = False
    if classes > 1:

        class_convergence = np.zeros(classes, dtype=bool)

        for ref in range(classes):

            # previous iteration
            par_file = (
                "maps/"
                + fparameters["dataset"]
                + "_r%02d_%02d.par" % (ref + 1, iteration - 2)
            )
            if not os.path.exists(par_file):
                logger.error(f"Cannot find {par_file}")
            else:
                logger.info(f"Reading occupancies from {par_file}")
                input = frealign_parfile.Parameters.from_file(par_file).data

                previous = np.sort(input[:, 11])[::-1]

                # current iteration
                par_file = (
                    "maps/"
                    + fparameters["dataset"]
                    + "_r%02d_%02d.par" % (ref + 1, iteration)
                )
                if not os.path.exists(par_file):
                    logger.error(f"Cannot find {par_file}")
                else:
                    logger.info(f"Reading occupancies from {par_file}")
                    input = frealign_parfile.Parameters.from_file(par_file).data

                    current = np.sort(input[:, 11])[::-1]

                    difference = np.fabs(previous - current).mean()
                    logger.info(
                        "\nConvergence difference for class {0} = {1}\n".format(
                            ref, difference
                        )
                    )
                    if difference < 0.25 and iteration > 10:
                        class_convergence[ref] = True

        # converged if no changes in all classes
        converged = class_convergence.prod() > 0

    converged = False

    return converged


def rec_merge_check_error_and_resubmit(mparameters, fparameters, iteration):
    if not os.path.exists("../mpirun.mynodes") and not os.path.exists(
        "../mpirun.myrecnodes"
    ):
        rec_split_swarm_file = "../swarm/frealign_mrecons_split_%02d.swarm" % (
            iteration
        )
    else:
        rec_split_swarm_file = "../swarm/frealign_rec_split.multirun"

    missing_rec_split_swarm_file = rec_split_swarm_file.replace(
        ".swarm", "_missing.swarm"
    )

    try:
        os.remove(missing_rec_split_swarm_file)
    except:
        pass

    count = 0
    if os.path.exists(rec_split_swarm_file):
        for swarm in open(rec_split_swarm_file):
            if "pyp" in swarm:
                missing_reconstructions = False
                ref = int(swarm.split("-ref")[-1].split()[0])
                first = int(swarm.split("--first")[-1].split()[0])
                last = int(swarm.split("--last")[-1].split()[0])
                current_log_file = (
                    "../log/"
                    + fparameters["dataset"]
                    + "_r%02d_%02d_%07d_%07d_mreconst.log"
                    % (ref, iteration, first, last)
                )

                if os.path.exists(current_log_file):
                    if "ERROR" in open(current_log_file).read():
                        logger.error("running FREALIGN")
                        if "t" in mparameters["email"].lower():
                            user_comm.notify(
                                "FREALIGN ERROR in " + fparameters["dataset"],
                                current_log_file,
                            )
                        raise Exception()
                    elif (
                        not "Normal termination, intermediate files dumped"
                        in open(current_log_file).read()
                        and not "Reconstruct3D: Normal termination"
                        in open(current_log_file).read()
                    ):
                        missing_reconstructions = True
                        count += last - first + 1
                else:
                    missing_reconstructions = True
                    count += last - first + 1

                if missing_reconstructions:
                    logger.warning(f"{current_log_file} is missing or incomplete.")
                    with open(missing_rec_split_swarm_file, "a") as f:
                        f.write(swarm)

    if count > 0:
        logger.info(f"Found {count} missing reconstructions.")

        # if all failed, stop
        if missing_reconstructions == (
            len(open(rec_split_swarm_file).read().split("\n")) - 1
        ):
            raise Exception("All reconstructions failed. Stop.")

        # re-launch reconstruction merging
        rec_swarm_file = "frealign_mrecons_%02d.swarm" % (iteration)

        # submit jobs to batch system
        id = slurm.submit_jobs(
            "../swarm",
            missing_rec_split_swarm_file.replace("../swarm/", ""),
            "frs_" + fparameters["dataset"][-6:],
            fparameters["slurm_queue"],
            400,
            0,
            58,
            "",
            "",
            "12:00:00",
        )
        slurm.submit_jobs(
            "../swarm",
            rec_swarm_file.replace("swarm/", ""),
            "frm_" + fparameters["dataset"][-6:],
            fparameters["slurm_queue"],
            400,
            0,
            58,
            "",
            "",
            "12:00:00",
            id,
        )


def ref_merge_check_error_and_resubmit(fp, iteration, machinefile):
    if not os.path.exists(machinefile):
        ali_swarm_file = "../swarm/frealign_msearch_%02d.swarm" % (iteration)
        missing_ali_swarm_file = ali_swarm_file.replace(".swarm", "_missing.swarm")
    else:
        ali_swarm_file = "../swarm/frealign_ref.multirun"
        missing_ali_swarm_file = ali_swarm_file.replace(
            ".multirun", "_missing.multirun"
        )

    first_time_it_fails = True
    if os.path.exists(missing_ali_swarm_file):
        first_time_it_fails = False
    # first_time_it_fails = True

    count = 0
    if os.path.exists(ali_swarm_file):
        for swarm in open(ali_swarm_file):
            if "run/fyp" in swarm:
                missing_alignments = False
                ref = int(swarm.split("--ref")[-1].split()[0])
                first = int(swarm.split("-first")[-1].split()[0])
                last = int(swarm.split("--last")[-1].split()[0])
                if (
                    "cistem"
                    in project_params.param(fp["refine_metric"], iteration).lower()
                ):
                    current_par_file = fp[
                        "refine_dataset"
                    ] + "_r%02d_%02d_%07d_%07d.star" % (ref, iteration, first, last,)
                else:
                    current_par_file = fp[
                        "refine_dataset"
                    ] + "_r%02d_%02d.par_%07d_%07d" % (ref, iteration, first, last,)
                if not os.path.exists(current_par_file):
                    logger.warning(f"{current_par_file} does not exist")
                    missing_alignments = True
                    count += last - first + 1
                else:
                    if (
                        "cistem"
                        in project_params.param(fp["refine_metric"], iteration).lower()
                    ):
                        alignments = len(
                            [
                                line
                                for line in open(current_par_file)
                                if not line.startswith("_")
                                and not line.startswith("#")
                                and not line.startswith("data")
                                and not line.startswith("loop")
                                and not line == " \n"
                            ]
                        )
                    else:
                        alignments = len(
                            [
                                line
                                for line in open(current_par_file)
                                if not line.startswith("C")
                            ]
                        )
                    missing = last - first + 1 - alignments
                    if missing > 0:
                        logger.warning(
                            f"{missing} missing alignments in {current_par_file}"
                        )
                        missing_alignments = True
                        count += missing
                if missing_alignments:
                    logger.warning(
                        "{} is missing or incomplete.".format(
                            os.getcwd() + "/" + current_par_file
                        )
                    )
                    with open(missing_ali_swarm_file, "a") as f:
                        f.write(swarm)

    if count > 0 and first_time_it_fails:

        logger.info(f"Found {count} missing alignments.")

        if not os.path.exists(machinefile):
            # submit jobs to batch system
            rec_swarm_file = "frealign_mrecons_%02d.swarm" % (iteration)
            id = slurm.submit_jobs(
                "../swarm",
                missing_ali_swarm_file.replace("../swarm/", ""),
                "frf_" + fp["refine_dataset"][-6:],
                fp["queue"],
                25,
                1,
                10,
                "",
                "",
                "4:00:00",
            )
            slurm.submit_jobs(
                "../swarm",
                rec_swarm_file,
                "frc_" + fp["refine_dataset"][-6:],
                fp["sliurm_queue"],
                0,
                2,
                58,
                "",
                "",
                "4:00:00",
                id,
            )

        else:

            # read missing lines
            count, cores, mpirunfile = local_run.create_ref_multirun_file_from_missing(
                machinefile, missing_ali_swarm_file
            )

            output = mpi.submit_jobs_file_to_workers(
                os.path.join(os.getcwd(), mpirunfile), os.getcwd()
            )
            logger.info(output)

        # Reconstruction
        rec_swarm_file = "frealign_mrecons_%02d.swarm" % (iteration)
        command = "cd ../swarm; export frealign_rec=frealign_rec; . ./{0}".format(
            rec_swarm_file.replace("swarm/", "")
        )
        local_run.run_shell_command(command)

    else:

        try:
            os.remove(missing_ali_swarm_file)
        except:
            pass

        if count > 0 and not first_time_it_fails:
            raise Exception("Processes failed more than once, aborting.")



