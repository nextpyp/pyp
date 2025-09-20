import os
import numpy as np
from pathlib import Path

from pyp.system.logging import logger
from pyp.system import local_run

from pyp.system.utils import get_imod_path

def get_relion_path():
    warptools_path = '/opt/conda/envs/relion'
    return f"export LD_LIBRARY_PATH=/opt/conda/envs/relion/lib:$LD_LIBRARY_PATH; micromamba run -n warp {warptools_path}/bin/"

def relion_refine():
    
    command = f"mpirun -n 3 {get_relion_path()}/build/bin/relion_refine_mpi --o Refine3D/job001/run --auto_refine --split_random_halves --ios matching_optimisation_set.star --ref sphere.mrc --trust_ref_size --ini_high 40 --dont_combine_weights_via_disc --pool 10 --pad 2  --ctf --particle_diameter 130 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym O --low_resol_join_halves 40 --norm --scale  --j 2 --gpu "" --pipeline_control Refine3D/job001"
    local_run.stream_shell_command(command)


def relion_mask_create():
    
    command = f"relion_mask_create --i relion/Refine3D/job002/run_class001.mrc --o m/mask_4apx.mrc --ini_threshold 0.04"
    local_run.stream_shell_command(command)
