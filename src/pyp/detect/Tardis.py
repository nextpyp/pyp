# Tardis
import os
import shutil
import glob
import numpy as np
from pathlib import Path
from pyp.analysis import plot
from pyp.inout.image import mrc
from pyp.system import local_run, project_params

from pyp.system.logging import logger

def get_tardis_path():
    command_base = 'export LD_LIBRARY_PATH=/opt/conda/envs/tardis/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH; export PYTHONPATH=/opt/conda/envs/tardis/lib/python3.11/site-packages:$PYTHONPATH; micromamba run -n tardis /opt/conda/envs/tardis/bin/'
    return command_base

def membrain_preprocessing(parameters, input):

    output = input.replace(".rec", "_preprocessed.rec")

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
    
    if parameters["tomo_mem_pixel"] > 0 and not parameters["tomo_mem_pixel"] == tomo_pixelsize:

        match_pixel = f"match_pixel_size --pixel-size-out {parameters['tomo_mem_pixel']} --pixel-size-in {tomo_pixelsize}"

        output_rescale = input.replace(".rec", "_rescale.rec")
    
        command = f"{get_tardis_path()}tomo_preprocessing {match_pixel} --input-tomogram {input} --output-path {output_rescale}"

        local_run.stream_shell_command(command)

        rescaled = True
        tomo_pixelsize = parameters['tomo_mem_pixel']
    
    else:
        output_rescale = input

    template = project_params.resolve_path(parameters["tomo_mem_target"])
    if parameters["tomo_mem_match_ps"] and os.path.exists(template):
        
        
        output_match_spectrum = input.replace(".rec", "_match_spectrum.rec")
        
        command = f"{get_tardis_path()}tomo_preprocessing extract_spectrum --input-path {template} --output-path ./template_spectrum.mrc"

        local_run.stream_shell_command(command)

        command = f"{get_tardis_path()}tomo_preprocessing match_spectrum --input {output_rescale} --target ./template_spectrum.mrc --output {output_match_spectrum}"

        local_run.stream_shell_command(command)

    else:
        output_match_spectrum = output_rescale

    if parameters["tomo_mem_deconvolve"]:

        command = f"{get_tardis_path()}tomo_preprocessing deconvolve --input {output_match_spectrum} --output {output} --pixel-size {tomo_pixelsize}"

        local_run.stream_shell_command(command)
    else:
        output = output_match_spectrum


    return rescaled, output


def tardis_segmentation(parameters, input, local_output):

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
   
    tm = "tomo_mem"

    inpix = parameters['tomo_mem_pixel'] if parameters['tomo_mem_pixel'] > 0 else tomo_pixelsize
    correct_px = f"--correct_px {inpix}"
    
    debug = ""
    if parameters[tm + "_tardis_mem_debug"]: 
        debug = "--debug True"
    
    common_options = f"--path {input} --output_format mrc_mrc --patch_size {parameters[tm + '_tardis_mem_patch_size']} --rotate {parameters[tm + '_tardis_mem_rotate']} --cnn_threshold {parameters[tm + '_tardis_mem_cnn_threshold']} --dist_threshold {parameters[tm + '_tardis_mem_dist_threshold']} --points_in_patch {parameters[tm + '_tardis_mem_points_in_patch']} {correct_px} {debug}"

    if parameters.get("tomo_mem_use_gpu"):
        common_options += f" --device gpu"
    else:
        common_options += f" --device cpu"

    mt_actin_options = f"--filter_by_length {parameters[tm + '_tardis_filter_by_length']} --connect_splines {parameters[tm + '_tardis_connect_splines']} --connect_cylinder {parameters[tm + '_tardis_connect_cylinder']}"
    
    cnn_model_path = "/opt/pyp/external/models/fnet_attn_32"
    dist_model_path = "/opt/pyp/external/models/dist_triang/3d/model_weights.pth"
    if parameters.get("tomo_mem_method") == "tardis_mem":
        command =f"{get_tardis_path()}tardis_mem {common_options} --cnn_checkpoint {os.path.join(cnn_model_path,'membrane_3d','model_weights.pth')} --dist_checkpoint {dist_model_path}"
    elif parameters.get("tomo_mem_method") == "tardis_mt":
        command =f"{get_tardis_path()}tardis_mt {common_options} {mt_actin_options} --cnn_checkpoint {os.path.join(cnn_model_path,'microtubules_3d','model_weights.pth')} --dist_checkpoint {dist_model_path}"
    elif parameters.get("tomo_mem_method") == "tardis_actin":
        command =f"{get_tardis_path()}tardis_actin {common_options} {mt_actin_options} --cnn_checkpoint {os.path.join(cnn_model_path,'actin_3d','model_weights.pth')} --dist_checkpoint {dist_model_path}"
    else:
        assert False, f"Unknown segmentation method {parameters.get('tomo_mem_method')}"
    local_run.stream_shell_command(command)

    try:
        log_file = glob.glob('Predictions/*_log.txt')[0]
        with open(log_file, 'r') as f:
            logger.debug(f.read())
    except:
        pass

    try:
        segmentation = glob.glob('Predictions/*_semantic.mrc')[0]
        mrc.read(segmentation)
    except:
        raise RuntimeError(f"Tardis failed or produced no output. Please check the logs for errors or decrease the threshold for semantic prediction")

    z_thickness = parameters["tomo_mem_connected_map_z_thickness"]

    if parameters[tm + "_connected_map"] != "none" or z_thickness != -1:

        segmentation = glob.glob('Predictions/*_semantic.mrc')[0]
            
        # create binary mask from binary segmentation volume
        cmask = ( mrc.read(segmentation) > 0 ).astype('uint8')

        # find all connected regions, calculate sizes, and ranking 
        from skimage.measure import label
        label_ids = label(cmask, connectivity=2)

        if z_thickness != -1:
            if not (0.0 <= z_thickness <= parameters["tomo_rec_thickness"]):
                raise ValueError(f"z_thickness must be between 0 and {parameters["tomo_rec_thickness"]}.")

            z_thickness /= float(parameters["tomo_rec_binning"])

            zlen = label_ids.shape[1]

            z0 = int((zlen - z_thickness) / 2)
            z1 = zlen - z0

            outside = np.ones(label_ids.shape, dtype=bool)
            outside[:, z0:z1, :] = False

            bad_labels = np.unique(label_ids[outside])
            bad_labels = bad_labels[bad_labels != 0]

            if bad_labels.size:
                cmask[np.isin(label_ids, bad_labels)] = 0
                label_ids = label(cmask, connectivity=2)

        sizes = np.bincount(label_ids.ravel())
        indexes_by_size = np.argsort(sizes)[::-1]

        clean = np.zeros_like(cmask, dtype='uint8')
        if "mem" in parameters.get("tomo_mem_method"):
            if parameters["tomo_mem_connected_map"] == "number":
                # keep max_number largest components after leaving out min_number largest components
                logger.info(f"Keeping {parameters['tomo_mem_connected_max_number']} largest components after {parameters['tomo_mem_connected_min_number']}")
                for i in range(1+parameters["tomo_mem_connected_min_number"],1+parameters["tomo_mem_connected_min_number"]+parameters["tomo_mem_connected_max_number"]):
                    clean[ label_ids == indexes_by_size[i] ] = 1
            elif parameters["tomo_mem_connected_map"] == "size":
                # remove objects smaller than threshold
                logger.info(f"Removing areas smaller than {parameters['tomo_mem_connected_thres']}")
                for i in range(1, len(sizes)):
                    if sizes[i] > parameters["tomo_mem_connected_thres"]:
                        clean[ label_ids == i ] = 1
            else:
                clean = cmask
        else:
            clean = cmask
    
        # save result
        mrc.write(clean, segmentation)

def run_tardis(name, parameters ):

    local_input = f"./{name}.rec"

    # copy the input tomogram to scratch space
    assert os.path.exists(local_input), f"{local_input} does not exist, please run preprocessing first"

    output = name + "_seg.rec"

    local_output = "seg_out"
    tardis_segmentation(parameters, input=local_input, local_output=local_output)
    
    target = glob.glob(f"./Predictions/*_semantic.mrc")[0]
    shutil.move(target, output)

    # produce poor man's visualization
    reconstruction = mrc.read(local_input)
    segmentation = mrc.read(output)
    max = np.max(reconstruction)
    threshold = segmentation.max()
    # apparently tardis sometimes swaps axes, so we need to check and fix
    if reconstruction.shape != segmentation.shape:
        segmentation = np.swapaxes(segmentation, 0, 1)
        mrc.write(segmentation, output)

    return output