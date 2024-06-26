# isonet engine
import os
import glob
import shutil
import numpy as np
from pathlib import Path

from pyp.inout.metadata import pyp_metadata
from pyp.system import local_run
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_isonet_path():
    config = get_pyp_configuration()
    isonet_path = config["pyp"]["isonet"]
    command_base = f"source activate {isonet_path}; {isonet_path}/bin/"
    return command_base

isonet_command = get_isonet_path()

def isonet_generate_star(project_dir, outputname, parameters):
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

    # train_folder = os.path.join(project_dir, "train")
    # with open( os.path.join( train_folder, "current_list.txt" ) ) as f:
    #     train_name = f.read()
    
    tomograms = glob.glob(f"{project_dir}/*.rec")

    with open(outputname, 'w') as f:
        f.write(star_header)
        for i, tomo in enumerate(tomograms):
            name = tomo.replace(".rec", "")
            pixel_size = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]
            metadata = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=False)
            ctf = metadata.data["global_ctf"].to_numpy()
            df = ctf[0]
            
            sub_tomograms = 100
            f.write(f"\n{i}    {tomo}   {pixel_size}    {df}    {sub_tomograms}" )


def isonet_ctf_deconvolve(tomo_star, output, snr_falloff, cs=2.7, voltage=300, hp_nyquist=0.02, process_id="None", ncpu=4, verbose=False):
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
    
    command = isonet_command + f"isonet.py deconv {tomo_star} --snrfalloff {snr_falloff} --deconv_folder {output} --cs {cs} --voltage {voltage} --highpassnyquist {hp_nyquist}--tomo_idx {process_id} --ncpu {ncpu}"
    
    local_run.run_shell_command(command,verbose=verbose)


def isonet_generat_mask(tomo_star, output, d_percent, std_percent, patchsize=4, use_convol="True", z_crop=0.2, process_id="None", verbose=False):
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

    command = isonet_command + f"isonet.py make_mask {tomo_star}  --mask_folder {output} --density_percentage {d_percent} --std_percentage {std_percent} --patch_size {patchsize} --use_deconv_tomo {use_convol} --tomo_idx {process_id} --z_crop {z_crop}"
   
    local_run.run_shell_command(command,verbose=verbose)


def isonet_extract(input_star, output_folder, output_star, cube_size, use_deconv="True", process_id="None", verbose=False):

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
    command = isonet_command + f"isonet.py extract {input_star} --subtomo_folder {output_folder} --subtomo_star {output_star} --cube_size {cube_size} --use_deconv_tomo {use_deconv} --tomo_id {process_id}"

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

    isn = "tomo_rec_isonet"

    iterations = parameters[f"{isn}_iters"]
    model = parameters[f"{isn}_model"]
    data_dir = "./train"
    continue_from = parameters[f"{isn}_json"]
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


def isonet_predict(input_star, model, output, batch_size, use_deconv, threshold_norm, verbose=False):
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


def isonet_run(input_dir, output, parameters, keep=False):
    
    # initialize path
    working_path = Path(os.environ["PYP_SCRATCH"]) / "isonet"

    logger.info(f"Working path: {working_path}")

    if not keep:
        shutil.rmtree(working_path, "True")

    working_path.mkdir(parents=True, exist_ok=True)
    os.chdir(working_path)

    # generate input tomo.star
    initial_star = "tomograms.star" 
    isonet_generate_star(input_dir, initial_star, parameters)
    
    # preprocess
    preprocess_star = "tomograms_processed.star"
    process_id = parameters["tomo_rec_isonet_tomoid"]
    ncpu = parameters["slurm_tasks"]
    verbose = parameters["slurm_verbose"]

    # extract parameters
    d_percent = parameters["tomo_rec_isonet_densityPercent"]
    std_percent = parameters["tomo_rec_isonet_stdPercent"]
    patchsize = parameters["tomo_rec_isonet_patchsize"]
    z_crop = parameters["tomo_rec_isonet_zcrop"]
    
    logger.info("isonet preprocessing...")
    
    if parameters["tomo_rec_isonet_CTFdeconvol"]:

        use_deconvol = "True"
        ctf_convol_star = preprocess_star.replace(".star", "_ctf.star")
        ssnr_falloff = parameters["tomo_rec_isonet_snrfalloff"]
        cs = parameters["scope_cs"]
        voltage = parameters["scope_voltage"]
        hp_nyquist = parameters["tomo_rec_isonet_hp"]
        
        isonet_ctf_deconvolve(
            initial_star,
            ctf_convol_star,
            ssnr_falloff,
            cs,
            voltage,
            hp_nyquist,
            process_id,
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
        process_id,
        verbose=verbose
        )

    # extract subvolumes
    logger.info("isonet subvolumes extraction...")
    cube_size = parameters["tomo_rec_isonet_cubesize"]
    extracted_folder = os.path.join(working_path, "subtomograms")
    extracted_star = "subtomograms.star"
    isonet_extract(
        preprocess_star,
        extracted_folder,
        extracted_star,
        cube_size,
        use_deconvol,
        process_id,
        verbose=verbose
        )
    
    
    # refine (train)
    output_dir = os.path.join(output, "isonet_tran")

    logger.info(f"Running isonet refine, model files will be saved in {output_dir}")
    isonet_refine(extracted_star, output_dir, parameters)

    # predict
    use_threshold = parameters["tomo_rec_isonet_threshold"]
    batch_size = parameters["tomo_rec_isonet_batchsize"]

    models = glob.glob[os.path.join(output_dir, "model_iter*.h5")]
    # get the most recent model 
    model = max(models, key=os.path.getmtime)

    output_dir = os.path.join(output, "isonet_predict")
    logger.info(f"Running isonet predict, final results will be saved in {output_dir}")
    isonet_predict(
        initial_star,
        model,
        output_dir,
        batch_size,
        use_deconvol,
        use_threshold, 
        verbose=verbose
        )

    
 