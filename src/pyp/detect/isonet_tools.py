# isonet engine
import os
import glob
import shutil
import numpy as np
from pathlib import Path

from pyp.analysis import plot
from pyp.inout.metadata import pyp_metadata
from pyp.system import local_run, project_params, mpi
from pyp.system.utils import get_gpu_ids, get_imod_path
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_isonet_path():
    command_base = f"export PATH=/opt/conda/envs/isonet/bin:$PATH; export PYTHONPATH=/opt/pyp/external:/opt/conda/envs/isonet/lib/python3.12/site-packages:$PYTHONPATH; micromamba run -n isonet /opt/pyp/external/IsoNet/bin/"
    return command_base

isonet_command = get_isonet_path()

def isonet_generate_star(project_dir, outputname, parameters, name_list):
    """
    Generate star file with tomograms names and defocus
    """

    star_header = """
data_
loop_
_rlnIndex          #1
_rlnMicrographName #2
_rlnPixelSize      #3
_rlnDefocus        #4
_rlnNumberSubtomo  #5"""

    # all_tomograms = glob.glob(f"{project_dir}/mrc/*.rec")
    # tomograms = [t for t in all_tomograms if not "denoised" in t]

    with open(outputname, 'w') as f:
        f.write(star_header)
        for i, name in enumerate(name_list):
            tomo = os.path.join(os.getcwd(), name + ".rec")
            pixel_size = parameters["scope_pixel"] * parameters["data_bin"] * parameters["tomo_rec_binning"]

            pkl_file = f"{project_dir}/pkl/{name}.pkl"
            assert os.path.exists(pkl_file), f"There is no meta data for this tomogram, please check the input name: {pkl_file}."
            metadata = pyp_metadata.LocalMetadata(pkl_file, is_spr=False)
            ctf = metadata.data["global_ctf"].to_numpy()
            df = np.squeeze(ctf[0])
            
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
    
    local_run.stream_shell_command(command,verbose=verbose)


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
   
    local_run.stream_shell_command(command,verbose=verbose)


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

    local_run.stream_shell_command(command,verbose=verbose)


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
    isonet_parameters = f"--iterations {iterations}"
    
    if parameters.get(f"{isn}_pretrained_model"):
        model = project_params.resolve_path(parameters[f"{isn}_pretrained_model"])
        if len(model) > 1:
            isonet_parameters += f" --pretrained_model {model}"
    
    data_dir = "./train"
    isonet_parameters += f" --data_dir {data_dir}"
    
    continue_from = project_params.resolve_path(parameters[f"{isn}_json"])
    isonet_parameters += f" --continue_from {continue_from}"

    result_dir = output
    isonet_parameters += f" --result_dir {result_dir}"

    ncpu = parameters["slurm_tasks"]
    isonet_parameters += f" --preprocessing_ncpus {ncpu}"

    epochs = parameters[f"{isn}_epochs"]
    isonet_parameters += f" --epochs {epochs}"

    batch_size = parameters[f"{isn}_batchsize"]
    if batch_size != 0:
        isonet_parameters += f" --batch_size {batch_size}"

    steps_per_epoch = parameters[f"{isn}_steps"]
    if steps_per_epoch > 0:
        isonet_parameters += f" --steps_per_epoch {steps_per_epoch}"

    noise_level = parameters[f"{isn}_nl"]
    isonet_parameters += f" --noise_level {noise_level}"

    noise_start_i = parameters[f"{isn}_ns"]
    isonet_parameters += f" --noise_start_iter {noise_start_i}"

    noise_mode = parameters[f"{isn}_nm"]
    isonet_parameters += f" --noise_mode {noise_mode}"

    # noise_dir = parameters[f"{isn}_nd"]
    noise_dir = "./noise_data"
    isonet_parameters += f" --noise_dir {noise_dir}"

    drop_out = parameters[f"{isn}_dropout"]
    isonet_parameters += f" --drop_out {drop_out}"

    learning_rate = parameters[f"{isn}_lr"]
    isonet_parameters += f" --learning_rate {learning_rate}"

    convs_per_depth = parameters[f"{isn}_layers"]
    isonet_parameters += f" --convs_per_depth {convs_per_depth}"

    kernel = parameters[f"{isn}_kernel"]
    isonet_parameters += f" --kernel {kernel}"

    unet_depth = parameters[f"{isn}_depth"]
    isonet_parameters += f" --unet_depth {unet_depth}"

    filter_base = parameters[f"{isn}_base"]
    isonet_parameters += f" --filter_base {filter_base}"

    if parameters[f"{isn}_normalization"]:
        normalization = "True"
        if parameters[f"{isn}_threshold"]:
            threshold_norm = "True"
        else:
            threshold_norm = "False"
    else:
        normalization = "False"
        threshold_norm = "False"
    isonet_parameters += f" --batch_normalization {normalization}"
    isonet_parameters += f" --normalize_percentile {threshold_norm}"

    pool = parameters[f"{isn}_pool"]
    if pool and len(pool):
        isonet_parameters += f" --pool {pool}"

    command = isonet_command + f"""isonet.py refine {input_star} {isonet_parameters} --gpuID {get_gpu_ids(parameters)}"""

    output = []
    def obs(line):
        output.append(line)
    
    local_run.stream_shell_command(command,observer=obs,verbose=parameters["slurm_verbose"])

    # parse output
    loss = [ line.split("loss:")[1].split()[0] for line in output if "ETA:" in line]
    mse = [ line.split("mse:")[1].split()[0] for line in output if "ETA:" in line]
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("dark")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[8, 6], sharex=True)

    ax[0].set_title("IsoNet training loss (refine)")
    ax[0].plot(np.array(loss).astype('f'),".-",color="blue",label="Loss")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(np.array(mse).astype('f'),".-",color="red",label="Mean Squared Error")
    ax[1].set_ylabel("MSE")
    ax[1].set_xlabel("Step")
    ax[1].legend()
    plt.xlabel("Step")
    plt.savefig("training_loss.svgz")
    plt.close()

def isonet_predict_command(input_star, model, output, batch_size, use_deconv, threshold_norm, parameters, verbose=False):
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
--normalize_percentile {threshold_norm} \\
--gpuID {get_gpu_ids(parameters)}
"""
    
    local_run.stream_shell_command(command,verbose=verbose)

def convert_and_transfer_tomograms(train_name,project_dir,parameters):
    # transfer/convert tomograms to local scratch
    commands = []
    for rec in train_name:
        absolute_rec = os.path.join(project_dir, "mrc", rec + ".rec")
        if parameters.get("tomo_rec_depth"):
            command = "{0}/bin/newstack -mode 2 {1} {2} && rm -f {1}~".format(
                get_imod_path(), absolute_rec, rec + ".rec"
            )
        else:
            command = f"cp {absolute_rec} ."
        commands.append(command)
    mpi.submit_jobs_to_workers(commands, os.getcwd())

def isonet_train(project_dir, output, parameters):
    
    # always try to look for tomograms from parent project
    tomogram_source = project_dir

    # get the train list
    train_folder = os.path.join(tomogram_source, "train")
    train_name = np.loadtxt( os.path.join( train_folder, "current_list.txt" ), dtype=str, skiprows=0, usecols=0, ndmin=2)[:, 0]

    # initialize path
    working_path = Path(os.environ["PYP_SCRATCH"]) / "isonet"

    logger.info(f"Working path: {working_path}")
    
    working_path.mkdir(parents=True, exist_ok=True)

    os.chdir(working_path)

    # transfer/convert tomograms to local scratch
    convert_and_transfer_tomograms(train_name,project_dir,parameters)

    # generate input tomo.star
    initial_star = "tomograms.star" 
    isonet_generate_star(tomogram_source, initial_star, parameters, train_name)
    
    # display star file if in verbose mode
    if parameters["slurm_verbose"]:
        with open(initial_star) as f:
            logger.info("Input star file:"+f.read())
    
    debug = True if parameters.get("tomo_denoise_isonet_debug", False) else False
        
    # preprocess
    preprocess_star = "tomograms_processed.star"
    ncpu = parameters["slurm_tasks"]
    verbose = parameters["slurm_verbose"]

    if parameters["tomo_denoise_isonet_CTFdeconvol"]:

        use_deconvol = True
        ssnr_falloff = parameters["tomo_denoise_isonet_snrfalloff"]
        cs = parameters["scope_cs"]
        voltage = parameters["scope_voltage"]
        hp_nyquist = parameters["tomo_denoise_isonet_hp"]
        
        isonet_ctf_deconvolve(
            initial_star,
            "deconv",
            ssnr_falloff,
            cs,
            voltage,
            hp_nyquist,
            ncpu,
            verbose=verbose
            )
    else:
        use_deconvol = False

    # masking
    if parameters["tomo_denoise_isonet_mask"]:
        
        # masking parameters
        d_percent = parameters["tomo_denoise_isonet_densityPercent"]
        std_percent = parameters["tomo_denoise_isonet_stdPercent"]
        patchsize = parameters["tomo_denoise_isonet_patchsize"]
        z_crop = parameters["tomo_denoise_isonet_zcrop"]
        
        isonet_generat_mask(
            initial_star,
            "masked",
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
        initial_star,
        extracted_folder,
        extracted_star,
        cube_size,
        use_deconvol,
        debug=debug,
        verbose=verbose
        )    
    
    # refine (train)
    output_dir = os.path.join(working_path, "refine")

    isonet_refine(extracted_star, output_dir, parameters)
    
    assert len(glob.glob( os.path.join( output_dir, "*.h5") )) > 0, "IsoNet failed to run"
    
    # copy resulting h5 models to project directory
    save_dir = os.path.join( project_dir, "train", "isonet" )
    os.makedirs(save_dir,exist_ok=True)
    shutil.copy2( "training_loss.svgz", os.path.join( project_dir, "train" ) )
    for f in glob.glob( os.path.join( output_dir, "*.h5") ):
        shutil.copy2( f, os.path.join( save_dir, "isonet_" + Path(f).name) )

    if debug:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Copy each file and directory to the result directory
        for item in os.listdir("./"):
            s = os.path.join("./", item)
            d = os.path.join(save_dir, item)
            if os.path.isdir(s) and ( Path(s).stem == "masked" or Path(s).stem == "deconv" ):
                shutil.copytree(s, d, dirs_exist_ok=True) 
            elif Path(s).suffix == ".star":
                shutil.copy2(s, d)
    
    # go back to project directory and clean local scratch
    os.chdir(project_dir)
    shutil.rmtree(working_path, "True")
    
 
def isonet_predict( name, project_dir, parameters ):
    
    # always try to look for tomograms from parent project
    if "data_parent" in parameters and os.path.exists(project_params.resolve_path(parameters["data_parent"])):
        tomogram_source = project_params.resolve_path(parameters["data_parent"])
    else:
        tomogram_source = project_dir
        logger.warning("Using current project tomograms for isonet denoising")

    # transfer/convert tomogram to local scratch
    convert_and_transfer_tomograms([name],project_dir,parameters)

    initial_star = f"{name}_tomograms.star"
    isonet_generate_star( tomogram_source, initial_star, parameters, name_list=[name])

    # predict
    if parameters["tomo_denoise_isonet_CTFdeconvol"]:
        use_deconvol = "True"
    else:
        use_deconvol = "False"

    use_threshold = parameters["tomo_denoise_isonet_threshold"]
    batch_size = parameters["tomo_denoise_isonet_batchsize"]

    if parameters.get("tomo_denoise_isonet_model") == "auto":
        model = sorted(glob.glob( os.path.join( project_params.resolve_path(parameters.get("data_parent")), "train", "isonet", "*.h5" )))[-1]
        parameters["tomo_denoise_isonet_model"] = model

    if os.path.exists(project_params.resolve_path(parameters["tomo_denoise_isonet_model"])):
        model = project_params.resolve_path(parameters["tomo_denoise_isonet_model"])
    else:
        logger.warning("Trying to use the most recent trained model for isonet predcition")
        models = glob.glob(os.path.join(project_dir, "mrc", "isonet_train", "model_iter*.h5"))
        # get the most recent model 
        model = max(models, key=os.path.getmtime)
    
    isonet_predict_command(
        initial_star,
        model,
        os.getcwd(),
        batch_size,
        use_deconvol,
        use_threshold,
        parameters=parameters,
        verbose=parameters["slurm_verbose"]
    )
       
    output = glob.glob( "*_corrected.*" )[0]
    return output