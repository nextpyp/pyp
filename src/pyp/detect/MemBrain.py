# MemBrain
import os
import shutil
from pathlib import Path
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_membrane_path():
    config = get_pyp_configuration()
    membrain_path = config["pyp"]["membrain"]
    command_base = f"source activate {membrain_path}; {membrain_path}/bin/"
    return command_base

def membrain_preprocessing(parameters, input):

    output = input.replace(".rec", "_preprocessed.rec")

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
    
    if parameters["tomo_mem_pixel"] > 0 and not parameters["tomo_mem_pixel"] == tomo_pixelsize:

        match_pixel = f"match_pixel_size --pixel-size-out {parameters['tomo_mem_pixel']} --pixel-size-in {tomo_pixelsize}"

        output_rescale = input.replace(".rec", "_rescale.rec")
    
        command = f"{get_membrane_path()}tomo_preprocessing {match_pixel} --input-tomogram {input} --output-path {output_rescale}"

        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

        rescaled = True
        tomo_pixelsize = parameters['tomo_mem_pixel']
    
    else:
        output_rescale = input

    template = project_params.resolve_path(parameters["tomo_mem_target"])
    if parameters["tomo_mem_match_ps"] and os.path.exists(template):
        
        
        output_match_spectrum = input.replace(".rec", "_match_spectrum.rec")
        
        command = f"{get_membrane_path()}tomo_preprocessing extract_spectrum --input-path {template} --output-path ./template_spectrum.mrc"

        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

        command = f"{get_membrane_path()}tomo_preprocessing match_spectrum --input {output_rescale} --target ./template_spectrum.mrc --output {output_match_spectrum}"

        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])

    else:
        output_match_spectrum = output_rescale

    if parameters["tomo_membrane_deconvolve"]:

        command = f"{get_membrane_path()}tomo_preprocessing deconvolve --input {output_match_spectrum} --output {output} --pixel-size {tomo_pixelsize}"

        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])
    else:
        output = output_match_spectrum


    return rescaled, output


def membrain_segmetation(parameters, input, local_output):

    model = project_params.resolve_path(parameters["tomo_mem_model"])

    assert os.path.exists(model), "Need a pre-trained model for segmentation inference"

    tomo_pixelsize = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
   
    logger.info(f"Using model {model} for membrane segmentation inference")
    
    tm = "tomo_mem"

    inpix = parameters['tomo_mem_pixel'] if parameters['tomo_mem_pixel'] > 0 else tomo_pixelsize
    if parameters[tm + "_rescale_patches"]:
        rescale_patches = f"--rescale-patches --in-pixel-size {inpix} --out-pixel-size {parameters[tm + '_patch_pxl']}"
    else:
        rescale_patches = "--no-rescale-patches"
    
    if parameters[tm + "_store_probabilities"]:
        store_p = "--store-probabilities"
    else:
        store_p = "--no-store-probabilities"

    if parameters[tm + "_connected_map"]:
        connected_map = f"--store-connected-components --connected-component-thres {parameters[tm + '_connected_thres']}"
    else:
        connected_map = "--no-store-connected-components"

    if parameters[tm + "_augmentation"]:
        augment = "--test-time-augmentation"
    else:
        augment = "--no-test-time-augmentation"
    
    local_output = "./segmentation.mrc"
    
    command =f"{get_membrane_path()}membrain segment --tomogram-path {input} --ckpt-path {model} --output-folder {local_output} {rescale_patches} {store_p} {connected_map} {augment} --segmentation-threshold {parameters[tm + '_seg_thres']} --sliding-window-size {parameters[tm + '_sliding_wd']}"

    local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])


def run_membrain(project_dir, name):

    os.chdir(project_dir)
    parameters = project_params.load_pyp_parameters()

    # always try to look for tomograms from parent project
    if "data_parent" in parameters and os.path.exists(project_params.resolve_path(parameters["data_parent"])):
        tomogram_source = project_params.resolve_path(parameters["data_parent"])
    else:
        tomogram_source = project_dir
        logger.info("Using current project tomograms for isonet denoising")

    # initialize path
    working_path = Path(os.environ["PYP_SCRATCH"]) / "membrain"

    logger.info(f"Working path: {working_path}")
    
    working_path.mkdir(parents=True, exist_ok=True)
    
    os.chdir(working_path)

    input_tomo = os.path.join(tomogram_source, "mrc", name + ".rec")
    local_input =f"./{name}.rec"

    # copy the input tomogram to scratch space
    assert os.path.exists(input_tomo), "Input tomogram dose not exist, run preprocessing fist"

    shutil.copy2(input_tomo, local_input)

    output = os.path.join(project_dir, "mrc", name + "_segmem.mrc")

    if parameters["tomo_mem_preprocessing"]:
        rescaled, preprocessed = membrain_preprocessing(parameters, input=local_input)
    else:
        rescaled = False
        preprocessed = local_input
    
    local_output =local_input.replace(".rec", "_seg.mrc")
    membrain_segmetation(parameters, input=preprocessed, local_output=local_output)
 
    if rescaled:
        command = f"{get_membrane_path()}tomo_preprocessing match_seg_to_tomo --seg-path {local_output} --orig-tomo-path ./{name}.rec --output-path {output}"

        local_run.run_shell_command(command, verbose=parameters["slurm_verbose"])
    else:
        shutil.move(local_output, output)

    # clean 
    os.chdir(project_dir)
    shutil.rmtree(working_path, "True")
