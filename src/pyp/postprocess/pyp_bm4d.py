#!/usr/bin/env python

import argparse
import os
from time import time

import matlab.engine
import numpy as np
import scipy.io

from pyp.inout.image import mrc
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_bm4d_path
from pyp.utils import get_relative_path

# https://www.cs.tut.fi/~foi/GCF-BM3D/README_BM4D.txt


relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Volume denoising by BM4D")
    parser.add_argument(
        "-input", "--input", help="Input volume (mrc file)", required=True
    )
    parser.add_argument(
        "-output", "--output", help="Output volume (mrc file)", required=True
    )
    parser.add_argument(
        "-sigma", "--sigma", help="Noise sigma (0.5)", type=float, default=0.5
    )
    parser.add_argument(
        "-nsearch",
        "--nsearch",
        help="Size of area to search for similar patches (11)",
        type=int,
        default=11,
    )
    parser.add_argument(
        "-patch_size",
        "--patch_size",
        help="Patch size for denoising, must be power of 2 (4))",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-distribution",
        "--distribution",
        help="Model of noise distribution (Gauss))",
        type=str,
        default="Gauss",
    )
    args = parser.parse_args()

    path_work = os.getcwd()

    ta = time()

    # Matlab initialization
    matlab = matlab.engine.start_matlab()
    matlab.addpath(get_bm4d_path(), nargout=0)

    total_time = 0

    # read input volume
    input = mrc.read(args.input)
    scipy.io.savemat(path_work + "/volume.mat", dict(data=input))
    input_volume_mat = matlab.load(path_work + "/volume.mat")
    input_volume_mat = matlab.double(input_volume_mat["data"])

    # Run BM4D
    denoised = matlab.bm4d(
        input_volume_mat,
        args.distribution,
        args.nsearch,
        args.patch_size,
        args.sigma,
        nargout=1,
    )
    denoised = np.asarray(denoised)

    # Save result
    mrc.write(denoised, args.output)

    tb = time()
    total_time = total_time + tb - ta

    logger.info("Execution time of all is %d sec", total_time)

    matlab.exit()
