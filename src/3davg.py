#!/usr/bin/env python

# SBATCH --job-name="3davg"
# SBATCH --output="3davg_%j.out"
# SBATCH --error="3davg_%j.err"

import datetime
import os
import sys
import socket
import shutil
import time

from pyp.inout.image import mrc
from pyp.refine.tomo_avg.sub_tomo_avg import (
    parse_arguments,
    sva_iterate
)
from pyp.system.set_up import prepare_3davg_dir, prepare_3davg_xml
from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger


logger = initialize_pyp_logger(log_name=__name__)

if __name__ == "__main__":
    
    logger.info(f"Running on directory {os.getcwd()}")

    # check if we are in frealign directory
    if "3DAVG" not in os.path.split(os.getcwd())[-1]:
        logger.error("You are not in the 3DAVG directory.")
        sys.exit(1)

    # Create directories if needed
    prepare_3davg_dir()

    # load 3DAVG parameters
    sva_parameters = parse_arguments()

    # prepare xmls in protocol folder if they do not exist
    prepare_3davg_xml(sva_parameters["dataset"])

    # go onother level up to project directory
    mparameters = project_params.load_pyp_parameters("..")

    dataset = sva_parameters["dataset"]

    iter = int(sva_parameters["iter"])
    if iter > 0 and iter <= 7:
        sva_iterate(mparameters, sva_parameters, iter)
    else:
        logger.error(f"Currently only support iter 1 - 7.")
        sys.exit(1)
