# MemBrain
import os
import shutil
import glob
import numpy as np
from pyp.inout.image import mrc
from pyp.inout.metadata import pyp_metadata
from pyp.system import local_run, project_params

from pyp.system.logging import logger

def get_membrane_path():
    command_base = 'export LD_LIBRARY_PATH=/opt/conda/envs/membrain/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH; export PYTHONPATH=/opt/conda/envs/membrain/lib/python3.11/site-packages:$PYTHONPATH; micromamba run -n membrain /opt/conda/envs/membrain/bin/'
    return command_base

def membrain_preprocessing(parameters, input, project_dir, name):

    output = input.replace(".rec", "_preprocessed.rec")

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]

    rescaled = False

    if parameters["tomo_mem_pixel"] > 0 and not parameters["tomo_mem_pixel"] == tomo_pixelsize:

        match_pixel = f"match_pixel_size --pixel-size-out {parameters['tomo_mem_pixel']} --pixel-size-in {tomo_pixelsize}"

        output_rescale = input.replace(".rec", "_rescale.rec")
    
        command = f"{get_membrane_path()}tomo_preprocessing {match_pixel} --input-tomogram {input} --output-path {output_rescale}"

        local_run.stream_shell_command(command)

        rescaled = True
        tomo_pixelsize = parameters['tomo_mem_pixel']
    
    else:
        output_rescale = input

    template = project_params.resolve_path(parameters.get("tomo_mem_target"))
    if parameters["tomo_mem_match_ps"] and os.path.exists(template):
        

        output_match_spectrum = input.replace(".rec", "_match_spectrum.rec")
        
        command = f"{get_membrane_path()}tomo_preprocessing extract_spectrum --input-path {template} --output-path ./template_spectrum.mrc"

        local_run.stream_shell_command(command)

        command = f"{get_membrane_path()}tomo_preprocessing match_spectrum --input {output_rescale} --target ./template_spectrum.mrc --output {output_match_spectrum}"

        local_run.stream_shell_command(command)

    else:
        output_match_spectrum = output_rescale

    if parameters["tomo_mem_deconvolve"]:
        output_deconvolve = input.replace(".rec", "_deconvolve.rec")

        # retrieve defocus from tilt-series metadata
        pkl_file = f"{project_dir}/pkl/{name}.pkl"
        assert os.path.exists(pkl_file), f"There is no meta data for this tomogram, please check the input name: {pkl_file}."
        metadata = pyp_metadata.LocalMetadata(pkl_file, is_spr=False)
        ctf = metadata.data["global_ctf"].to_numpy()
        df = np.squeeze(ctf[0])

        boolean_options = ""
        if parameters["tomo_mem_deconvolve_skip_lowpass"]:
            boolean_options += " --skip-lowpass"
            
        # TODO: Don't use default value for df, add remaining parameters
        command = f"{get_membrane_path()}tomo_preprocessing deconvolve --input {output_match_spectrum} --output {output_deconvolve} --pixel-size {tomo_pixelsize} --df {df} --strength {parameters['tomo_mem_deconvolve_strength']} --falloff {parameters['tomo_mem_deconvolve_falloff']} --hp-fraction {parameters['tomo_mem_deconvolve_hp_fraction']} {boolean_options}"

        local_run.stream_shell_command(command)

        output = output_deconvolve
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

    local_run.stream_shell_command(command)

    try:
        segmentation = glob.glob(local_output+'/*')[0]
        mrc.read(segmentation)
    except:
        raise RuntimeError(f"Membrain-seg failed. Please check the logs for errors or increase the memory per thread allocated to the job.")

    z_thickness = parameters["tomo_mem_connected_map_z_thickness"]

    if parameters[tm + "_connected_map"] != "none" or z_thickness != -1:

        segmentation = glob.glob(local_output+'/*')[0]
            
        if False:
            os.makedirs(f"{local_output}_components", exist_ok=True)
            
            connected_map = f"--connected-component-thres {parameters[tm + '_connected_thres']}"

            command =f"{get_membrane_path()}membrain components --segmentation-path {segmentation} {connected_map} --out-folder {local_output}_components"

            local_run.stream_shell_command(command)
            
            output = glob.glob(local_output+'_components/*.mrc')[0]
            shutil.move(output, segmentation)
        
        # create binary mask from binary segmentation volume
        cmask = ( mrc.read(segmentation) > 0 ).astype('uint8')

        # find all connected regions, calculate sizes, and ranking 
        from skimage.measure import label
        label_ids = label(cmask, connectivity=2)

        if z_thickness != -1:
            if not (0.0 <= z_thickness <= parameters["tomo_rec_thickness"]):
                raise ValueError(f"z_thickness must be between 0 and {parameters["tomo_rec_thickness"]}.")

            z_thickness /= float(parameters["tomo_rec_binning"])
            if parameters["tomo_mem_pixel"] > 0 and not parameters["tomo_mem_pixel"] == tomo_pixelsize:
                z_thickness *= tomo_pixelsize / parameters["tomo_mem_pixel"]

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
        if parameters["tomo_mem_connected_map"] == "number":
            # keep max_number largest components after leaving out min_number largest components
            logger.info(f"Keeping {parameters['tomo_mem_connected_max_number']} largest components after {parameters['tomo_mem_connected_min_number']}")
            for i in range(1+parameters["tomo_mem_connected_min_number"],min(len(sizes), 1+parameters["tomo_mem_connected_min_number"]+parameters["tomo_mem_connected_max_number"])):
                clean[ label_ids == indexes_by_size[i] ] = 1
        elif parameters["tomo_mem_connected_map"] == "size":
            # remove objects smaller than threshold
            logger.info(f"Removing areas smaller than {parameters['tomo_mem_connected_thres']}")
            for i in range(1, len(sizes)):
                if sizes[i] > parameters["tomo_mem_connected_thres"]:
                    clean[ label_ids == i ] = 1
        else:
            clean = cmask
    
        # save result
        mrc.write(clean, segmentation)
    else:
        segmentation = glob.glob(local_output + '/*')[0]
        cmask = (mrc.read(segmentation) > 0).astype('uint8')
        mrc.write(cmask, segmentation)

def run_membrain(project_dir, name, parameters ):

    # always try to look for tomograms from parent project
    if "data_parent" in parameters and os.path.exists(project_params.resolve_path(parameters["data_parent"])):
        tomogram_source = project_params.resolve_path(parameters["data_parent"])
    else:
        tomogram_source = project_dir
        logger.info("Using current project tomograms for segmentation")

    if parameters.get("tomo_mem_use_denoised") and os.path.exists(name + "_den.rec"):
        suffix = "_den"
    else:
        suffix = ""

    local_input = f"./{name}{suffix}.rec"

    # copy the input tomogram to scratch space
    assert os.path.exists(local_input), f"{local_input} does not exist, please run preprocessing first"

    output = name + "_seg.rec"

    if parameters["tomo_mem_preprocessing"]:
        rescaled, preprocessed = membrain_preprocessing(parameters, input=local_input, project_dir=project_dir, name=name)
    else:
        rescaled = False
        preprocessed = local_input
    
    local_output = "seg_out"
    membrain_segmentation(parameters, input=preprocessed, local_output=local_output)
    
    if rescaled:
        rescale_input = glob.glob(f"./{local_output}/*.mrc")[0]
        command = f"{get_membrane_path()}tomo_preprocessing match_seg_to_tomo --seg-path {rescale_input} --orig-tomo-path ./{name}{suffix}.rec --output-path {output}"

        local_run.stream_shell_command(command)
    else:
        target = glob.glob(f"./{local_output}/*.mrc")[0]
        shutil.move(target, output)

    if parameters["tomo_mem_store_probabilities"]:
        target = glob.glob(f"./{local_output}/*_scores.mrc")[0]
        output_scores = name + "_scores.rec"
        shutil.move(target, output_scores)

    return output