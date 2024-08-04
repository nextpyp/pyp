# isonet engine
import os
import glob
import shutil
import numpy as np
from pathlib import Path

from pyp.inout.metadata import pyp_metadata
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_isonet_path():
    isonet_path = '/opt/pyp/external'
    command_base = f"export PYTHONPATH={isonet_path}:$PYTHONPATH; {isonet_path}/IsoNet/bin/"
    return command_base

isonet_command = get_isonet_path()

def isonet_generate_star(project_dir, outputname, parameters, name_list):
    """
    Generate star file with tomograms names and defocus
    """

    star_header = """data_

loop_
_rlnIndex #1
_rlnMicrographName #2
_rlnPixelSize #3
_rlnDefocus #4
_rlnNumberSubtomo #5"""

    # all_tomograms = glob.glob(f"{project_dir}/mrc/*.rec")
    # tomograms = [t for t in all_tomograms if not "denoised" in t]

    with open(outputname, 'w') as f:
        f.write(star_header)
        for i, name in enumerate(name_list):
            tomo = os.path.join(project_dir, "mrc", name + ".rec")
            pixel_size = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]

            pkl_file = f"{project_dir}/pkl/{name}.pkl"
            assert os.path.exists(pkl_file), f"There is no meta data for this image, please check the input name: {pkl_file}."
            metadata = pyp_metadata.LocalMetadata(pkl_file, is_spr=False)
            ctf = metadata.data["global_ctf"].to_numpy()
            df = ctf[0]
            
            sub_tomograms = 100
            f.write(f"\n{i + 1}    {tomo}   {pixel_size}    {df}    {sub_tomograms}" )


def isonet_ctf_deconvolve(tomo_star, output, snr_falloff, cs=2.7, voltage=300, hp_nyquist=0.02, ncpu=4, verbose=False):
    """
    CTF deconvolution for the tomograms.
    isonet.py deconv star_file [--deconv_folder] [--snrfalloff] [--deconvstrength] [--highpassnyquist] [--overlap_rate] [--ncpu] [--tomo_idx]
    This step is recommended because it enhances low resolution information for a better contrast. No need to do deconvolution for phase plate data.
    :param deconv_folder: (./deconv) Folder created to save deconvoluted tomograms.
    :param star_file: (None) Star file for tomograms.
    :param voltage: (300.0) Acceleration voltage in kV.
    :param cs: (2.7) Spherical aberration in mm.
    :param snrfalloff: (1.0) SNR fall rate with the frequency. High values means losing more high frequency.
    If this value is not set, the program will look for the parameter in the star file.
    If this value is not set and not found in star file, the default value 1.0 will be used.
    :param deconvstrength: (1.0) Strength of the deconvolution.
    If this value is not set, the program will look for the parameter in the star file.
    If this value is not set and not found in star file, the default value 1.0 will be used.
    :param highpassnyquist: (0.02) Highpass filter for at very low frequency. We suggest to keep this default value.
    :param chunk_size: (None) When your computer has enough memory, please keep the chunk_size as the default value: None . Otherwise, you can let the program crop the tomogram into multiple chunks for multiprocessing and assembly them into one. The chunk_size defines the size of individual chunk. This option may induce artifacts along edges of chunks. When that happen, you may use larger overlap_rate.
    :param overlap_rate: (None) The overlapping rate for adjecent chunks.
    :param ncpu: (4) Number of cpus to use.
    :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
    """
    
    command = isonet_command + f"isonet.py deconv {tomo_star} --snrfalloff {snr_falloff} --deconv_folder {output} --cs {cs} --voltage {voltage} --highpassnyquist {hp_nyquist} --ncpu {ncpu}"
    
    local_run.run_shell_command(command,verbose=verbose)


def isonet_generat_mask(tomo_star, output, d_percent, std_percent, patchsize=4, use_convol="True", z_crop=0.2, verbose=False):
    """
    generate a mask that include sample area and exclude "empty" area of the tomogram. The masks do not need to be precise. In general, the number of subtomograms (a value in star file) should be lesser if you masked out larger area. 
    isonet.py make_mask star_file [--mask_folder] [--patch_size] [--density_percentage] [--std_percentage] [--use_deconv_tomo] [--tomo_idx]
    :param star_file: path to the tomogram or tomogram folder
    :param mask_folder: path and name of the mask to save as
    :param patch_size: (4) The size of the box from which the max-filter and std-filter are calculated.
    :param density_percentage: (50) The approximate percentage of pixels to keep based on their local pixel density.
    If this value is not set, the program will look for the parameter in the star file.
    If this value is not set and not found in star file, the default value 50 will be used.
    :param std_percentage: (50) The approximate percentage of pixels to keep based on their local standard deviation.
    If this value is not set, the program will look for the parameter in the star file.
    If this value is not set and not found in star file, the default value 50 will be used.
    :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
    :param z_crop: If exclude the top and bottom regions of tomograms along z axis. For example, "--z_crop 0.2" will mask out the top 20% and bottom 20% region along z axis.
    :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
    """

    command = isonet_command + f"isonet.py make_mask {tomo_star}  --mask_folder {output} --density_percentage {d_percent} --std_percentage {std_percent} --patch_size {patchsize} --use_deconv_tomo {use_convol} --z_crop {z_crop}"
   
    local_run.run_shell_command(command,verbose=verbose)


def isonet_extract(input_star, output_folder, output_star, cube_size, use_deconv="True", debug=False, verbose=False):

    # extract subtomograms
    """
    Extract subtomograms
    isonet.py extract star_file [--subtomo_folder] [--subtomo_star] [--cube_size] [--use_deconv_tomo] [--tomo_idx]
    :param star_file: tomogram star file
    :param subtomo_folder: (subtomo) folder for output subtomograms.
    :param subtomo_star: (subtomo.star) star file for output subtomograms.
    :param cube_size: (64) Size of cubes for training, should be divisible by 8, eg. 32, 64. The actual sizes of extracted subtomograms are this value adds 16.
    :param crop_size: (None) The size of subtomogram, should be larger then the cube_size The default value is 16+cube_size.
    :param log_level: ("info") level of the output, either "info" or "debug"
    :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
    """
    if not debug:
        log_level = "info"
    else:
        log_level = "debug"

    command = isonet_command + f"isonet.py extract {input_star} --subtomo_folder {output_folder} --subtomo_star {output_star} --cube_size {cube_size} --use_deconv_tomo {use_deconv} --log_level {log_level}"

    local_run.run_shell_command(command,verbose=verbose)


def isonet_refine(input_star, output, parameters):
    """
    train neural network to correct missing wedge
    isonet.py refine subtomo_star [--iterations] [--gpuID] [--preprocessing_ncpus] [--batch_size] [--steps_per_epoch] [--noise_start_iter] [--noise_level]...
    :param subtomo_star: (None) star file containing subtomogram(s).
    :param gpuID: (0,1,2,3) The ID of gpu to be used during the training. e.g 0,1,2,3.
    :param pretrained_model: (None) A trained neural network model in ".h5" format to start with.
    :param iterations: (30) Number of training iterations.
    :param data_dir: (data) Temporary folder to save the generated data used for training.
    :param log_level: (info) debug level, could be 'info' or 'debug'
    :param continue_from: (None) A Json file to continue from. That json file is generated at each iteration of refine.
    :param result_dir: ('results') The name of directory to save refined neural network models and subtomograms
    :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

    ************************Training settings************************

    :param epochs: (10) Number of epoch for each iteraction.
    :param batch_size: (None) Size of the minibatch.If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
    :param steps_per_epoch: (None) Step per epoch. If not defined, the default value will be min(num_of_subtomograms * 6 / batch_size , 200)

    ************************Denoise settings************************

    :param noise_level: (0.05,0.1,0.15,0.2) Level of noise STD(added noise)/STD(data) after the iteration defined in noise_start_iter.
    :param noise_start_iter: (11,16,21,26) Iteration that start to add noise of corresponding noise level.
    :param noise_mode: (None) Filter names when generating noise volumes, can be 'ramp', 'hamming' and 'noFilter'
    :param noise_dir: (None) Directory for generated noise volumes. If set to None, the Noise volumes should appear in results/training_noise

    ************************Network settings************************

    :param drop_out: (0.3) Drop out rate to reduce overfitting.
    :param learning_rate: (0.0004) learning rate for network training.
    :param convs_per_depth: (3) Number of convolution layer for each depth.
    :param kernel: (3,3,3) Kernel for convolution
    :param unet_depth: (3) Depth of UNet.
    :param filter_base: (64) The base number of channels after convolution.
    :param batch_normalization: (True) Use Batch Normalization layer
    :param pool: (False) Use pooling layer instead of stride convolution layer.
    :param normalize_percentile: (True) Normalize the 5 percent and 95 percent pixel intensity to 0 and 1 respectively. If this is set to False, normalize the input to 0 mean and 1 standard dievation.
    """

    isn = "tomo_denoise_isonet"

    iterations = parameters[f"{isn}_iters"]
    model = project_params.resolve_path(parameters[f"{isn}_model"])
    data_dir = "./train"
    continue_from = project_params.resolve_path(parameters[f"{isn}_json"])
    result_dir = output
    ncpu = parameters["slurm_tasks"]

    epochs = parameters[f"{isn}_epochs"]
    batch_size = parameters[f"{isn}_batchsize"]
    steps_per_epoch = parameters[f"{isn}_steps"]

    noise_level = parameters[f"{isn}_nl"]
    noise_start_i = parameters[f"{isn}_ns"]
    noise_mode = parameters[f"{isn}_nm"]
    # noise_dir = parameters[f"{isn}_nd"]
    noise_dir = "./noise_data"

    drop_out = parameters[f"{isn}_dropout"]
    learning_rate = parameters[f"{isn}_lr"]
    convs_per_depth = parameters[f"{isn}_layers"]
    kernel = parameters[f"{isn}_kernel"]
    unet_depth = parameters[f"{isn}_depth"]
    filter_base = parameters[f"{isn}_base"]

    if parameters[f"{isn}_normalization"]:
        normalization = "True"
        if parameters[f"{isn}_threshold"]:
            threshold_norm = "True"
        else:
            threshold_norm = "False"
    else:
        normalization = "False"
        threshold_norm = "False"

    if parameters[f"{isn}_pool"]:
        pool = "True"
    else:
        pool = "False"

    command = isonet_command + f"""isonet.py refine {input_star} \\
--iterations {iterations} \\
--data_dir {data_dir} \\
--continue_from {continue_from} \\
--result_dir {result_dir} \\
--pretrained_model {model} \\
--preprocessing_ncpus {ncpu} \\
--epochs {epochs} \\
--batch_size {batch_size} \\
--steps_per_epoch {steps_per_epoch} \\
--noise_start_iter {noise_start_i} \\
--noise_level {noise_level} \\
--noise_mode {noise_mode} \\
--noise_dir {noise_dir} \\
--drop_out {drop_out} \\
--learning_rate {learning_rate} \\
--convs_per_depth {convs_per_depth} \\
--kernel {kernel} \\
--unet_depth {unet_depth} \\
--filter_base {filter_base} \\
--batch_normalization {normalization} \\
--pool {pool} \\
--normalize_percentile {threshold_norm}
"""
    
    local_run.run_shell_command(command,verbose=parameters["slurm_verbose"])


def isonet_predict_command(input_star, model, output, batch_size, use_deconv, threshold_norm, verbose=False):
    """
    Predict tomograms using trained model
    isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
    :param star_file: star for tomograms.
    :param output_dir: file_name of output predicted tomograms
    :param model: path to trained network model .h5
    :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
    :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
    :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping patch strategy, make this value larger if you see the patchy artifacts
    :param batch_size: The batch size of the cubes grouped into for network predicting, the default parameter is four times number of gpu
    :param normalize_percentile: (True) if normalize the tomograms by percentile. Should be the same with that in refine parameter.
    :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
    :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
    :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
    :raises: AttributeError, KeyError
    """
 
    command = isonet_command + f"""isonet.py predict {input_star} \\
--model {model} \\
--output_dir {output} \\
--batch_size {batch_size} \\
--use_deconv_tomo {use_deconv} \\
--normalize_percentile {threshold_norm}
"""
    
    local_run.run_shell_command(command,verbose=verbose)


def isonet_train(project_dir, output, parameters):
    
    # always try to look for tomograms from parent project
    if "data_parent" in parameters and os.path.exists(project_params.resolve_path(parameters["data_parent"])):
        tomogram_source = project_params.resolve_path(parameters["data_parent"])
    else:
        tomogram_source = project_dir
        logger.info("Using current project tomograms for isonet denoising")

    # get the train list
    train_folder = os.path.join(tomogram_source, "train")
    with open( os.path.join( train_folder, "current_list.txt" ) ) as f:
        list_file = f.read()
    train_name = np.loadtxt(os.path.join( train_folder, list_file + "_images.txt" ), dtype=str, skiprows=1, usecols=0, ndmin=2)

    # initialize path
    working_path = Path(os.environ["PYP_SCRATCH"]) / "isonet"

    logger.info(f"Working path: {working_path}")
    
    working_path.mkdir(parents=True, exist_ok=True)

    os.chdir(working_path)

    # generate input tomo.star
    initial_star = "tomograms.star" 
    isonet_generate_star(tomogram_source, initial_star, parameters, train_name[:, 0])
    
    debug = True if parameters.get("tomo_denoise_isonet_debug", False) else False
        
    # preprocess
    preprocess_star = "tomograms_processed.star"
    ncpu = parameters["slurm_tasks"]
    verbose = parameters["slurm_verbose"]

    # extract parameters
    d_percent = parameters["tomo_denoise_isonet_densityPercent"]
    std_percent = parameters["tomo_denoise_isonet_stdPercent"]
    patchsize = parameters["tomo_denoise_isonet_patchsize"]
    z_crop = parameters["tomo_denoise_isonet_zcrop"]
    
    logger.info("IsoNet preprocessing...")
    
    if parameters["tomo_denoise_isonet_CTFdeconvol"]:

        use_deconvol = "True"
        ctf_convol_star = preprocess_star.replace(".star", "_ctf.star")
        ssnr_falloff = parameters["tomo_denoise_isonet_snrfalloff"]
        cs = parameters["scope_cs"]
        voltage = parameters["scope_voltage"]
        hp_nyquist = parameters["tomo_denoise_isonet_hp"]
        
        isonet_ctf_deconvolve(
            initial_star,
            ctf_convol_star,
            ssnr_falloff,
            cs,
            voltage,
            hp_nyquist,
            ncpu,
            verbose=verbose
            )

        extract_input = ctf_convol_star

    else:
        use_deconvol = "False"
        extract_input = initial_star

    # mask
    isonet_generat_mask(
        extract_input,
        preprocess_star,
        d_percent,
        std_percent,
        patchsize,
        use_deconvol,
        z_crop,
        verbose=verbose
        )

    # extract subvolumes
    logger.info("IsoNet subvolume extraction...")
    cube_size = parameters["tomo_denoise_isonet_cubesize"]
    extracted_folder = os.path.join(working_path, "subtomograms")
    extracted_star = "subtomograms.star"
    isonet_extract(
        preprocess_star,
        extracted_folder,
        extracted_star,
        cube_size,
        use_deconvol,
        debug=debug,
        verbose=verbose
        )
    
    
    # refine (train)
    output_dir = os.path.join(output, "isonet_tran")

    logger.info(f"Running IsoNet refine, model files will be saved in {output_dir}")
    isonet_refine(extracted_star, output_dir, parameters)

    if debug:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Copy each file and directory to the result directory
        for item in os.listdir("./"):
            s = os.path.join("./", item)
            d = os.path.join(output_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True) 
            else:
                shutil.copy2(s, d)
    
    # clean
    os.chdir(project_dir)
    shutil.rmtree(working_path, "True")
    
 
def isonet_predict(project_dir, name):
    
    os.chdir(project_dir)
    parameters = project_params.load_pyp_parameters()
    output = os.path.join(project_dir, "mrc")

    # always try to look for tomograms from parent project
    if "data_parent" in parameters and os.path.exists(project_params.resolve_path(parameters["data_parent"])):
        tomogram_source = project_params.resolve_path(parameters["data_parent"])
    else:
        tomogram_source = project_dir
        logger.info("Using current project tomograms for isonet denoising")

    initial_star = "tomograms.star"
    isonet_generate_star(tomogram_source, initial_star, parameters, name_list=[name])
    # predict
    if parameters["tomo_denoise_isonet_CTFdeconvol"]:
        use_deconvol = "True"
    else:
        use_deconvol = "False"

    use_threshold = parameters["tomo_denoise_isonet_threshold"]
    batch_size = parameters["tomo_denoise_isonet_batchsize"]

    if os.path.exists(project_params.resolve_path(parameters["tomo_denoise_isonet_model"])):
        model = project_params.resolve_path(parameters["tomo_denoise_isonet_model"])
    else:
        logger.warning("Trying to use the most recent trained model for isonet predcition")
        models = glob.glob(os.path.join(project_dir, "mrc", "isonet_train", "model_iter*.h5"))
        # get the most recent model 
        model = max(models, key=os.path.getmtime)

    output_dir = os.path.join(output, "isonet_predict")
    logger.info(f"Running isonet predict, final results will be saved in {output_dir}")

    verbose = parameters["slurm_verbose"]

    isonet_predict_command(
        initial_star,
        model,
        output_dir,
        batch_size,
        use_deconvol,
        use_threshold, 
        verbose=verbose
    )

