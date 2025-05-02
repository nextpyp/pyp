# MemBrain
import os
import shutil
import glob
import numpy as np
from pathlib import Path
from pyp.analysis import plot
from pyp.inout.image import mrc
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_membrane_path():
    command_base = 'export PYTHONPATH=/opt/conda/envs/membrain/lib/python3.9/site-packages:$PYTHONPATH; micromamba run -n membrain /opt/conda/envs/membrain/bin/'
    return command_base

def membrain_preprocessing(parameters, input):

    output = input.replace(".rec", "_preprocessed.rec")

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
    
    if parameters["tomo_mem_pixel"] > 0 and not parameters["tomo_mem_pixel"] == tomo_pixelsize:

        match_pixel = f"match_pixel_size --pixel-size-out {parameters['tomo_mem_pixel']} --pixel-size-in {tomo_pixelsize}"

        output_rescale = input.replace(".rec", "_rescale.rec")
    
        command = f"{get_membrane_path()}tomo_preprocessing {match_pixel} --input-tomogram {input} --output-path {output_rescale}"

        local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])

        rescaled = True
        tomo_pixelsize = parameters['tomo_mem_pixel']
    
    else:
        output_rescale = input

    template = project_params.resolve_path(parameters["tomo_mem_target"])
    if parameters["tomo_mem_match_ps"] and os.path.exists(template):
        
        
        output_match_spectrum = input.replace(".rec", "_match_spectrum.rec")
        
        command = f"{get_membrane_path()}tomo_preprocessing extract_spectrum --input-path {template} --output-path ./template_spectrum.mrc"

        local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])

        command = f"{get_membrane_path()}tomo_preprocessing match_spectrum --input {output_rescale} --target ./template_spectrum.mrc --output {output_match_spectrum}"

        local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])

    else:
        output_match_spectrum = output_rescale

    if parameters["tomo_mem_deconvolve"]:

        command = f"{get_membrane_path()}tomo_preprocessing deconvolve --input {output_match_spectrum} --output {output} --pixel-size {tomo_pixelsize}"

        local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])
    else:
        output = output_match_spectrum


    return rescaled, output


def membrain_segmentation(parameters, input, local_output):

    model = project_params.resolve_path(parameters["tomo_mem_model"])

    assert os.path.exists(model), "Need a pre-trained model for segmentation inference"

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
   
    logger.info(f"Using model {model} for membrane segmentation inference")
    
    tm = "tomo_mem"

    inpix = parameters['tomo_mem_pixel'] if parameters['tomo_mem_pixel'] > 0 else tomo_pixelsize
    if parameters[tm + "_rescale_patches"]:
        rescale_patches = f"--rescale-patches --in-pixel-size {inpix} --out-pixel-size {parameters[tm + '_patch_pxl']}"
    else:
        rescale_patches = ""
    
    if parameters[tm + "_store_probabilities"]:
        store_p = "--store-probabilities"
    else:
        store_p = "--no-store-probabilities"

    if parameters[tm + "_connected_map"]:
        connected_map = "--store-connected-components"
        connected_map = ""
    else:
        connected_map = "--no-store-connected-components"

    if parameters[tm + "_augmentation"]:
        augment = "--test-time-augmentation"
    else:
        augment = "--no-test-time-augmentation"
    
    command =f"{get_membrane_path()}membrain segment --tomogram-path {input} --ckpt-path {model} --out-folder {local_output} {rescale_patches} {store_p} {connected_map} {augment} --segmentation-threshold {parameters[tm + '_seg_thres']} --sliding-window-size {parameters[tm + '_sliding_wd']}"

    local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])

    if parameters[tm + "_connected_map"] != "none":

        segmentation = glob.glob(local_output+'/*')[0]
            
        if False:
            os.makedirs(f"{local_output}_components", exist_ok=True)
            
            connected_map = f"--connected-component-thres {parameters[tm + '_connected_thres']}"

            command =f"{get_membrane_path()}membrain components --segmentation-path {segmentation} {connected_map} --out-folder {local_output}_components"

            local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])
            
            output = glob.glob(local_output+'_components/*.mrc')[0]
            shutil.move(output, segmentation)
        
        # create binary mask from binary segmentation volume
        cmask = 255 * ( mrc.read(segmentation) > 0 ).astype('uint8')

        # find all connected regions, calculate sizes, and ranking 
        from skimage.measure import label
        label_ids = label(cmask, connectivity=2)
        sizes = np.bincount(label_ids.ravel())
        indexes_by_size = np.argsort(sizes)[::-1]

        clean = np.zeros_like(cmask, dtype='uint8')
        if parameters["tomo_mem_connected_map"] == "number":
            # keep max_number largest components after leaving out min_number largest components
            logger.info(f"Keeping {parameters['tomo_mem_connected_max_number']} largest components after {parameters['tomo_mem_connected_min_number']}")
            for i in range(1+parameters["tomo_mem_connected_min_number"],1+parameters["tomo_mem_connected_min_number"]+parameters["tomo_mem_connected_max_number"]):
                clean[ label_ids == indexes_by_size[i] ] = 255
        elif parameters["tomo_mem_connected_map"] == "size":
            # remove objects smaller than threshold
            logger.info(f"Removing areas smaller than {parameters['tomo_mem_connected_thres']}")
            for i in range(1, len(sizes)):
                if sizes[i] > parameters["tomo_mem_connected_thres"]:
                    clean[ label_ids == i ] = 255
        else:
            clean = cmask
    
        # save result
        mrc.write(clean, segmentation)

def run_membrain(project_dir, name, parameters ):

    # always try to look for tomograms from parent project
    if "data_parent" in parameters and os.path.exists(project_params.resolve_path(parameters["data_parent"])):
        tomogram_source = project_params.resolve_path(parameters["data_parent"])
    else:
        tomogram_source = project_dir
        logger.info("Using current project tomograms for segmentation")

    local_input = f"./{name}.rec"

    # copy the input tomogram to scratch space
    assert os.path.exists(local_input), f"{local_input} dose not exist, please run preprocessing first"

    output = name + "_seg.rec"

    if parameters["tomo_mem_preprocessing"]:
        rescaled, preprocessed = membrain_preprocessing(parameters, input=local_input)
    else:
        rescaled = False
        preprocessed = local_input
    
    local_output = "seg_out"
    membrain_segmentation(parameters, input=preprocessed, local_output=local_output)
    
    if rescaled:
        rescale_input = glob.glob(f"./{local_output}/*.mrc")[0]
        command = f"{get_membrane_path()}tomo_preprocessing match_seg_to_tomo --seg-path {rescale_input} --orig-tomo-path ./{name}.rec --output-path {output}"

        local_run.stream_shell_command(command, verbose=parameters["slurm_verbose"])
    else:
        target = glob.glob(f"./{local_output}/*.mrc")[0]
        shutil.move(target, output)

    # produce poor man's visualization
    reconstruction = mrc.read(local_input)
    segmentation = mrc.read(output)
    max = np.max(reconstruction)
    threshold = segmentation.max()
    visualization = np.where( segmentation == threshold, max, reconstruction )
    mrc.write(visualization,local_input)

    return local_input