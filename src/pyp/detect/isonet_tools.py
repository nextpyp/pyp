# isonet engine
import os
import glob
import shutil
import numpy as np
from pathlib import Path

from pyp.inout.metadata import pyp_metadata
from pyp.system import local_run, project_params, mpi
from pyp.system.utils import get_gpu_ids, get_imod_path

from pyp.system.logging import logger

def get_isonet_path():
    command_base = f"export PATH=/opt/conda/envs/isonet/bin:$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/isonet/lib:/opt/conda/envs/isonet/lib/python3.9/site-packages/tensorrt_libs; export PYTHONPATH=/opt/pyp/external:/opt/conda/envs/isonet/lib/python3.9/site-packages:$PYTHONPATH; micromamba run -n isonet /opt/pyp/external/IsoNet/bin/"
    return command_base

isonet_command = get_isonet_path()

def get_isonet2_path():
    command_base = f"export PATH=/opt/conda/envs/isonet2/bin:$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/envs/isonet2/lib; export PYTHONPATH=/opt/pyp/external/IsoNet2:/opt/conda/envs/isonet2/lib/python3.10/site-packages:$PYTHONPATH; micromamba run -n isonet2 /opt/pyp/external/IsoNet2/IsoNet/bin/"
    return command_base

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


def isonet_ctf_deconvolve(tomo_star, output, snr_falloff, cs=2.7, voltage=300, hp_nyquist=0.02, ncpu=4):
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
    
    local_run.stream_shell_command(command)


def isonet_generate_mask(tomo_star, output, d_percent, std_percent, patchsize=4, use_convol="True", z_crop=0.2):
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
   
    local_run.stream_shell_command(command)


def isonet_extract(input_star, output_folder, output_star, cube_size, use_deconv="True", debug=False):

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

    local_run.stream_shell_command(command)


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
    
    local_run.stream_shell_command(command,observer=obs)

    # parse output
    loss = np.array([ line.split("loss:")[1].split()[0] for line in output if "ETA:" in line]).astype('f')
    mse = np.array([ line.split("mse:")[1].split()[0] for line in output if "ETA:" in line]).astype('f')

    max_points = 500
    binning_factor = max(loss.shape[0] // max_points,1)
    steps = np.arange(0, loss.shape[0], binning_factor)
    if binning_factor > 1:
        loss = loss[::binning_factor]
        mse = mse[::binning_factor]
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("dark")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[8, 6], sharex=True)

    ax[0].set_title("IsoNet training loss (refine)")
    ax[0].plot(steps,loss,".-",color="blue",label="Loss")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(steps,mse,".-",color="red",label="Mean Squared Error")
    ax[1].set_ylabel("MSE")
    ax[1].set_xlabel("Step")
    ax[1].legend()
    plt.xlabel("Step")
    plt.savefig("training_loss.svgz")
    plt.close()

def isonet_predict_command(input_star, model, output, batch_size, cube_size, crop_size, use_deconv, threshold_norm, parameters):
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
--cube_size {cube_size} \\
--crop_size {crop_size} \\
--use_deconv_tomo {use_deconv} \\
--normalize_percentile {threshold_norm} \\
--gpuID {get_gpu_ids(parameters)}
"""
    
    local_run.stream_shell_command(command)

def convert_and_transfer_tomograms(train_name,project_dir, parameters):
    # transfer/convert tomograms to local scratch
    commands = []
    for rec in train_name:
        absolute_rec = os.path.join(project_dir, "mrc", rec + ".rec")
        if parameters.get("tomo_rec_depth"):
            command = "{0}/bin/newstack -mode 2 {1} {2}".format(
                get_imod_path(), absolute_rec, rec + ".rec"
            )
        else:
            command = f"cp {absolute_rec} ."
        commands.append(command)
    mpi.submit_jobs_to_workers(commands)

def isonet_train(project_dir, parameters):
    
    # get the train list
    train_folder = os.path.join(project_dir, "train")
    train_name = np.loadtxt( os.path.join( train_folder, "current_list.txt" ), dtype=str, skiprows=0, usecols=0, ndmin=2)[:, 0]

    # initialize path
    working_path = Path(os.environ["PYP_SCRATCH"]) / "isonet"

    logger.info(f"Using temporary folder {working_path}")
    
    working_path.mkdir(parents=True, exist_ok=True)

    os.chdir(working_path)

    # transfer/convert tomograms to local scratch
    convert_and_transfer_tomograms(train_name,project_dir,parameters)

    # generate input tomo.star
    initial_star = "tomograms.star" 
    isonet_generate_star(project_dir, initial_star, parameters, train_name)
    
    # display star file if in verbose mode
    with open(initial_star) as f:
        logger.debug("Input star file:"+f.read())
    
    debug = True if parameters.get("tomo_denoise_isonet_debug", False) else False
        
    # preprocess
    preprocess_star = "tomograms_processed.star"
    ncpu = parameters["slurm_tasks"]

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
            ncpu
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
        
        isonet_generate_mask(
            initial_star,
            "masked",
            d_percent,
            std_percent,
            patchsize,
            use_deconvol,
            z_crop
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
        debug=debug
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
    shutil.rmtree(working_path, True)
    
 
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

    use_threshold = parameters["tomo_denoise_isonet_predict_threshold"]
    batch_size = parameters["tomo_denoise_isonet_predict_batchsize"]
    cube_size = parameters["tomo_denoise_isonet_predict_cubesize"]
    crop_size = parameters["tomo_denoise_isonet_predict_cropsize"]

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
        cube_size,
        crop_size,
        use_deconvol,
        use_threshold,
        parameters=parameters
    )
       
    assert len(glob.glob( "*_corrected.*" )) > 0, "IsoNet failed to run"
    output = glob.glob( "*_corrected.*" )[0]
    return output


def isonet2_predict( name, project_dir, parameters ):

    initial_star = f"{name}_tomograms.star"
    isonet2_generate_star( project_dir, initial_star, parameters, name_list=[name])

    if parameters.get("tomo_denoise_isonet2_predict_model") == "auto":
        model = sorted(glob.glob( os.path.join( project_params.resolve_path(parameters.get("data_parent")), "train", "isonet2", "*.pt" )))[-1]
        parameters["tomo_denoise_isonet2_predict_model"] = model

    if os.path.exists(project_params.resolve_path(parameters["tomo_denoise_isonet2_predict_model"])):
        model = project_params.resolve_path(parameters["tomo_denoise_isonet2_predict_model"])
    else:
        logger.warning("Trying to use the most recent trained model for isonet2 prediction")
        models = glob.glob(os.path.join(project_dir, "train", "isonet2", "isonet_network*_full.pt"))
        # get the most recent model 
        model = max(models, key=os.path.getmtime)

    model_base = os.path.basename(model)
    if ("n2n_" not in model_base) and parameters["tomo_denoise_isonet2_predict_input_column"] == "rlnDeconvTomoName":
        isonet2_ctf_deconvolve(initial_star, parameters=parameters)
    
    isonet2_predict_command(
        input_star=initial_star,
        model=model,
        parameters=parameters
    )
       
    assert len(glob.glob( "corrected_tomos/*.mrc" )) > 0, "IsoNet2 failed to run"
    output = glob.glob( "corrected_tomos/*.mrc" )[0]
    return output

def parse_loss(output, max_points=500):
    lines = [l for l in output if "Learning rate:" in l]

    xy = np.array([l.split("[")[0].split()[-1] for l in lines])
    keep = np.r_[False, xy[1:] == xy[:-1]] | np.r_[xy[:-1] != xy[1:], True]

    loss = np.array([l.split("Loss:")[1].split()[0].split(",")[0] for l in lines], dtype="f")[keep]
    x    = np.array([p.split("/")[0] for p in xy], dtype="i")[keep]
    y0   = int(xy[0].split("/")[1])

    epoch_id = np.r_[0, np.cumsum(x[1:] < x[:-1])].astype("i")
    steps = epoch_id * y0 + (x - 1)

    epoch_mean = np.bincount(epoch_id, loss) / np.bincount(epoch_id)
    epoch_mean_per_step = epoch_mean[epoch_id]

    b = max(loss.shape[0] // max_points, 1)
    return steps[::b], loss[::b], epoch_mean_per_step[::b]


def plot_loss(ax, steps, loss, epoch_mean_per_step, title):
    ax.set_title(title)
    ax.plot(steps, loss, ".-", color="blue", label="Loss")
    ax.plot(steps, epoch_mean_per_step, "-", drawstyle="steps-post", color="orange", label="Epoch Average")
    ax.set_ylabel("Loss")
    ax.legend()

def isonet2_train( project_dir, parameters):
    
        # get the train list
    train_folder = os.path.join(project_dir, "train")
    train_name = np.loadtxt( os.path.join( train_folder, "current_list.txt" ), dtype=str, skiprows=0, usecols=0, ndmin=2)[:, 0]
    
    # generate input tomo.star
    initial_star = "tomograms.star" 
    isonet2_generate_star(
        project_dir,
        initial_star, 
        parameters, 
        train_name
        )
    
    # display star file if in verbose mode
    with open(initial_star) as f:
        logger.debug("Input star file:"+f.read())
    
    debug = True if parameters.get("tomo_denoise_isonet_debug", False) else False

    denoise_output = None

    # masking
    if parameters["tomo_denoise_isonet2_mask"]:

        if parameters["tomo_denoise_isonet2_mask_preprocessing"] == "deconv":

            isonet2_ctf_deconvolve(
                initial_star,
                parameters=parameters
                )

        elif parameters["tomo_denoise_isonet2_mask_preprocessing"] == "denoise":

            assert parameters['tomo_denoise_isonet2_denoise_epochs'] >= parameters['tomo_denoise_isonet2_denoise_save_interval'], f"IsoNet2 requires the save interval ({parameters['tomo_denoise_isonet2_denoise_save_interval']}) to be less than number of epochs ({parameters['tomo_denoise_isonet2_denoise_epochs']})!"

            denoise_output = isonet2_denoise(
                initial_star,
                parameters=parameters
                )

            models = glob.glob('denoise/*_full.pt')
            assert len(models) > 0, "IsoNet2 denoising failed"
            model = models[0]

            isonet2_predict_command(
                initial_star,
                model,
                parameters=parameters
            )

        isonet2_generate_mask(
            initial_star,
            parameters=parameters
            )

    with open(initial_star) as f:
        logger.debug("Input star file:" + f.read())

    sub_tomograms_per_tomo = parameters.get("tomo_denoise_isonet2_refine_total_subtomos") // len(train_name)

    v = str(sub_tomograms_per_tomo)
    lines = open(initial_star, "r", encoding="utf-8").read().splitlines(True)
    for i, ln in enumerate(lines):
        t = ln.split()
        if t and t[0].isdigit() and len(t) >= 15:
            t[14] = v
            lines[i] = "\t".join(t) + "\n"
    open(initial_star, "w", encoding="utf-8").writelines(lines)

    with open(initial_star) as f:
        logger.debug("Input star file:" + f.read())

    # refine (train)
    output_dir = os.path.join( os.getcwd(), "isonet_maps")

    assert parameters['tomo_denoise_isonet2_refine_epochs'] >= parameters['tomo_denoise_isonet2_refine_save_interval'], f"IsoNet2 requires the save interval ({parameters['tomo_denoise_isonet2_refine_save_interval']}) to be less than number of epochs ({parameters['tomo_denoise_isonet2_refine_epochs']})!"
    
    refine_output = isonet2_refine(
        initial_star, 
        parameters=parameters
        )
    
    assert len(glob.glob( os.path.join( output_dir, "*.pt") )) > 0, "IsoNet2 failed to run"

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("dark")

    if denoise_output is not None:
        steps_de, loss_de, epoch_mean_per_step_de = parse_loss(denoise_output)
        steps_re, loss_re, epoch_mean_per_step_re = parse_loss(refine_output)

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[8, 6], sharex=True)
        plot_loss(ax[0], steps_de, loss_de, epoch_mean_per_step_de, "IsoNet2 training loss (denoise)")
        plot_loss(ax[1], steps_re, loss_re, epoch_mean_per_step_re, "IsoNet2 training loss (refine)")
    else:
        steps_re, loss_re, epoch_mean_per_step_re = parse_loss(refine_output)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 6], sharex=True)
        plot_loss(ax, steps_re, loss_re, epoch_mean_per_step_re, "IsoNet2 training loss (refine)")

    plt.xlabel("Step")
    plt.savefig("training_loss.svgz")
    plt.close()

    # copy resulting h5 models to project directory
    save_dir = os.path.join( project_dir, "train", "isonet2" )
    os.makedirs(save_dir,exist_ok=True)
    shutil.copy2( "training_loss.svgz", os.path.join( project_dir, "train" ) )
    for f in glob.glob( os.path.join( output_dir, "*.pt") ):
        shutil.copy2( f, os.path.join( save_dir, "isonet_" + Path(f).name) )

    if debug:
        os.makedirs(save_dir, exist_ok=True)

        extensions = [".mrc", ".rec"]
        flip_dirs = {"deconv", "denoise", "corrected_tomos", "mask", "isonet_maps"}

        for item in os.listdir("./"):
            s = os.path.join("./", item)
            d = os.path.join(save_dir, item)

            if os.path.isdir(s) and Path(s).stem in flip_dirs:
                os.makedirs(d, exist_ok=True)
                for f in os.listdir(s):
                    src = os.path.join(s, f)
                    dst = os.path.join(d, f)
                    if Path(f).suffix.lower() in extensions:
                        cmd = f'"{get_imod_path()}/bin/clip" flipyz "{src}" "{dst}"'
                        local_run.run_shell_command(cmd)
                    else:
                        shutil.copy2(src, dst)

            elif Path(s).suffix == ".star":
                shutil.copy2(s, d)
    
    # go back to project directory and clean local scratch
    working_path = os.getcwd()
    os.chdir(project_dir)
    shutil.rmtree(working_path, True)
    
def isonet2_denoise(input_star, parameters, output = "./denoise"):

    # extract subtomograms
    """
    NAME
        isonet.py denoise - Entry point for IsoNet2 training. Use denoise for quicker noise-to-noise (n2n) training workflows for preliminary tomogram testing and mask generation.

    SYNOPSIS
        isonet.py denoise STAR_FILE <flags>

    DESCRIPTION
        Entry point for IsoNet2 training. Use denoise for quicker noise-to-noise (n2n) training workflows for preliminary tomogram testing and mask generation.

    POSITIONAL ARGUMENTS
        STAR_FILE
            Type: str
            STAR file for tomograms. Required parameter.

    FLAGS
        -o, --output_dir=OUTPUT_DIR
            Type: str
            Default: 'denoise'
            Directory to save trained model and results.
        -g, --gpuID=GPUID
            Type: Optional[str]
            Default: None
            GPU IDs to use during training (e.g., "0,1,2,3").
        -n, --ncpus=NCPUS
            Type: int
            Default: 16
            Number of CPUs to use for data processing.
        -a, --arch=ARCH
            Type: str
            Default: 'unet-medium'
            Network architecture string (e.g., unet-small, unet-medium, unet-large). Determines model capacity and VRAM requirements.
        --pretrained_model=PRETRAINED_MODEL
            Type: Optional[str]
            Default: None
            Path to pretrained model to continue training. Previous method, arch, cube_size, CTF_mode, and metrics will be loaded.
        --cube_size=CUBE_SIZE
            Type: int
            Default: 96
            Size in voxels of training subvolumes. Must be compatible with the network (divisible by the network downsampling factors).
        -e, --epochs=EPOCHS
            Type: int
            Default: 50
            Number of training epochs.
        --batch_size=BATCH_SIZE
            Default: 'auto'
            Number of subtomograms per optimization step; if "auto", this is automatically determined by multiplying the number of available GPUs by 2. If the number of GPUs is 1, batch size is 4. Batch size per GPU matters for gradient stability.
        --loss_func=LOSS_FUNC
            Type: str
            Default: 'L2'
            Loss function to use (L2, Huber, L1).
        --save_interval=SAVE_INTERVAL
            Type: int
            Default: 10
            Interval to save model checkpoints.
        --learning_rate=LEARNING_RATE
            Type: float
            Default: 0.0003
            Initial learning rate.
        --learning_rate_min=LEARNING_RATE_MIN
            Type: float
            Default: 0.0003
            Minimum learning rate for scheduler.
        -m, --mixed_precision=MIXED_PRECISION
            Type: bool
            Default: True
            If True, uses float16/mixed precision to reduce VRAM and speed up training.
        -C, --CTF_mode=CTF_MODE
            Type: str
            Default: 'None'
            CTF handling mode: "None": No CTF correction, "phase_only": Phase-only correction, "network": Applies CTF-shaped filter to network input, "wiener": Applies Wiener filter to network target
        -i, --isCTFflipped=ISCTFFLIPPED
            Type: bool
            Default: False
            Whether input tomograms are phase flipped.
        --do_phaseflip_input=DO_PHASEFLIP_INPUT
            Type: bool
            Default: True
            Whether to apply phase flip during training.
        --bfactor=BFACTOR
            Type: float
            Default: 0
            B-factor applied during training/prediction to boost high-frequency content. For cellular tomograms we recommend a b-factor of 0. For isolated samples, you can use a b-factor from 200–300.
        --clip_first_peak_mode=CLIP_FIRST_PEAK_MODE
            Type: float
            Default: 1
            Controls attenuation of overrepresented very-low-frequency CTF peak. Options 2 and 3 might increase low-resolution contrast. 0: none, 1: constant clip, 2: negative sine, 3: cosine
        --snrfalloff=SNRFALLOFF
            Type: float
            Default: 0
            Controls frequency-dependent SNR attenuation applied during deconvolution; larger values reduce high-frequency contribution more aggressively.
        --deconvstrength=DECONVSTRENGTH
            Type: float
            Default: 1
            Scalar multiplier for deconvolution strength; increasing this emphasizes correction and low-frequency recovery. 0.
        -h, --highpassnyquist=HIGHPASSNYQUIST
            Type: float
            Default: 0.02
            Fraction of the Nyquist used as a very-low-frequency high-pass cutoff; use to remove large-scale intensity gradients and drift. 02.
        -w, --with_preview=WITH_PREVIEW
            Type: bool
            Default: True
            If True, run prediction using the final checkpoint(s) after training.
        --prev_tomo_idx=PREV_TOMO_IDX
            Type: str
            Default: 1
            If set, automatically predict only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").

    NOTES
        You can also use flags syntax for POSITIONAL ARGUMENTS
    """
    
    prefix = "tomo_denoise_isonet2_denoise"

    # we always pass these parameters
    values = [ "arch", "cube_size", "epochs", "loss_func", "save_interval", "learning_rate", "learning_rate_min", "mixed_precision", "do_phaseflip_input", "bfactor", "clip_first_peak_mode", "snrfalloff", "deconvstrength", "highpassnyquist", "with_preview" ]
    
    # we only pass these if True
    booleans = [ "isCTFflipped"]

    # we only pass these if not empty
    strings = [ "CTF_mode", "prev_tomo_idx" ]

    isonet_denoise_parameters = build_command_options( parameters, prefix, values, booleans, strings )

    pretrained_model = parameters.get(f"{prefix}_pretrained_model")
    if pretrained_model and os.path.exists( project_params.resolve_path(pretrained_model) ):
        isonet_denoise_parameters += f" --pretrained_model '{project_params.resolve_path(pretrained_model)}'"

    if parameters.get(f"{prefix}_batch_size") > 0:
        isonet_denoise_parameters += f" --batch_size {parameters.get(f'{prefix}_batch_size')}"
        
    command = get_isonet2_path() + f"isonet.py denoise {input_star} --output_dir {output} {isonet_denoise_parameters} --gpuID {get_gpu_ids(parameters)}"

    output = []

    def obs(line):
        output.append(line)

    local_run.stream_shell_command(command, observer=obs)

    return output
    
def isonet2_predict_command(input_star, model, parameters, output = "./corrected_tomos"):
    """
    SYNOPSIS
        isonet.py predict STAR_FILE MODEL <flags>

    DESCRIPTION
        Apply a trained IsoNet model to tomograms to produce denoised or missing-wedge–corrected volumes. Prediction utilizes the model's saved cube size and CTF handling options, but allows for runtime adjustments.

    POSITIONAL ARGUMENTS
        STAR_FILE
            Type: str
            Input STAR describing tomograms to predict. Required parameter.
        MODEL
            Type: str
            Path to trained model checkpoint (.pt) for single-model prediction. Required parameter.

    FLAGS
        --output_dir=OUTPUT_DIR
            Type: str
            Default: './corrected_tomos'
            Folder to save predicted tomograms; outputs are recorded in the STAR as rlnCorrectedTomoName or rlnDenoisedTomoName depending on method. /corrected_tomos".
        -g, --gpuID=GPUID
            Type: Optional[str]
            Default: None
            GPU IDs string (e.g., "0" or "0,1"); use multiple GPUs when available for speed.
        --input_column=INPUT_COLUMN
            Type: str
            Default: 'rlnDeconvTomoName'
            STAR column used for input tomogram paths. This is only relevant if the network model is using method IsoNet2.
        -a, --apply_mw_x1=APPLY_MW_X1
            Type: bool
            Default: True
            If True (default), build and apply the missing-wedge mask to cubic inputs before prediction.
        --isCTFflipped=ISCTFFLIPPED
            Type: bool
            Default: False
            Declare if input tomograms are already phase-flipped; affects CTF handling.
        -p, --padding_factor=PADDING_FACTOR
            Type: float
            Default: 1.5
            Cubic padding factor used during tiling to reduce edge effects; larger padding reduces seams but increases computation. 5.
        -t, --tomo_idx=TOMO_IDX
            Type: Optional[]
            Default: None
            Process a subset of STAR entries by index.
        --output_prefix=OUTPUT_PREFIX
            Type: str
            Default: ''
            Prefix to append to predicted MRC files.
        -s, --save_slices=SAVE_SLICES
            Type: bool
            Default: True
    """
 
    prefix = "tomo_denoise_isonet2_predict"

    # we always pass these parameters
    values = [ "apply_mw_x1", "padding_factor", "save_slices" ]

    # we only pass these if True
    booleans = [ "isCTFflipped", ]

    # we only pass these if not empty
    strings = [ "input_column", "tomo_idx", "output_prefix" ]

    isonet_predict_parameters = build_command_options( parameters, prefix, values, booleans, strings )

    # "input_column", 
    
    command = get_isonet2_path() + f"""isonet.py predict {input_star} \\
{model} \\
--output_dir {output} \\
--gpuID {get_gpu_ids(parameters)} \\
{isonet_predict_parameters}
"""
    
    local_run.stream_shell_command(command)
    
def isonet2_generate_mask(tomo_star, parameters, output = "./mask"):
    """
    NAME
        isonet.py make_mask - Generate masks to prioritize regions of interest. Masks improve sampling efficiency and training stability.

    SYNOPSIS
        isonet.py make_mask STAR_FILE <flags>

    DESCRIPTION
        Generate masks to prioritize regions of interest. Masks improve sampling efficiency and training stability.

    POSITIONAL ARGUMENTS
        STAR_FILE
            Type: str
            Input STAR listing tomograms and acquisition metadata. Required parameter.

    FLAGS
        -o, --output_dir=OUTPUT_DIR
            Type: str
            Default: 'mask'
            Folder to save mask MRCs; rlnMaskName is updated in the STAR.
        -i, --input_column=INPUT_COLUMN
            Type: str
            Default: 'rlnDeconvTomoName'
            STAR column to read tomograms from (default **rlnDeconvTomoName**; falls back to **rlnTomoName** or **rlnTomoReconstructedTomogramHalf1** if absent).
        -p, --patch_size=PATCH_SIZE
            Type: int
            Default: 4
            Local patch size used for max/std local filters; larger values smooth detection of specimen regions; default works for typical pixel sizes.
        -d, --density_percentage=DENSITY_PERCENTAGE
            Type: int
            Default: 50
            Percentage of voxels retained based on local density ranking; lower values create stricter masks (keep fewer voxels).
        -s, --std_percentage=STD_PERCENTAGE
            Type: int
            Default: 50
            Percentage retained based on local standard-deviation ranking; lower values emphasize textured regions.
        -z, --z_crop=Z_CROP
            Type: float
            Default: 0.2
            Fraction of tomogram Z to crop from both ends; masks out top and bottom 10% each when set to 0.2. Use to avoid sampling low-quality reconstruction edges. 2.
        -t, --tomo_idx=TOMO_IDX
            Type: Optional[]
            Default: None
            If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").

    NOTES
        You can also use flags syntax for POSITIONAL ARGUMENTS
    """
    prefix = "tomo_denoise_isonet2_make_mask"

    # we always pass these parameters
    values = [ "patch_size", "density_percentage", "std_percentage", "z_crop" ]
    
    # we only pass these if True
    booleans = [ ]

    # we only pass these if not empty
    strings = [ "tomo_idx" ]

    isonet_make_mask_parameters = build_command_options( parameters, prefix, values, booleans, strings )

    if parameters.get("tomo_denoise_isonet2_mask_preprocessing") == "deconv":
        isonet_make_mask_parameters += " --input_column rlnDeconvTomoName"
    elif parameters.get("tomo_denoise_isonet2_mask_preprocessing") == "denoise":
        isonet_make_mask_parameters += " --input_column rlnDenoisedTomoName"
    else:
        isonet_make_mask_parameters += " --input_column rlnCorrectedTomoName"

    command = get_isonet2_path() + f"isonet.py make_mask {tomo_star}  --output_dir {output} {isonet_make_mask_parameters}"
   
    local_run.stream_shell_command(command)
    
def isonet2_refine(input_star, parameters, output = "./isonet_maps"):
    """
SYNOPSIS
    isonet.py refine STAR_FILE <flags>

DESCRIPTION
    Use refine for IsoNet2 missing-wedge correction (isonet2) or isonet2-n2n combined modes.

POSITIONAL ARGUMENTS
    STAR_FILE
        Type: str
        Input STAR listing tomograms and acquisition metadata. Required parameter.

FLAGS
    -o, --output_dir=OUTPUT_DIR
        Type: str
        Default: 'isonet_maps'
        Directory to save trained model and results.
    -g, --gpuID=GPUID
        Type: Optional[str]
        Default: None
        GPU IDs to use during training (e.g., "0,1,2,3").
    --ncpus=NCPUS
        Type: int
        Default: 16
        Number of CPUs to use for data processing.
    --method=METHOD
        Type: str
        Default: 'isonet2-n2n'
        "isonet2" for single-map missing-wedge correction, "isonet2-n2n" for noise2noise when even/odd halves are present. If omitted, the code auto-detects the method from the STAR columns.
    --arch=ARCH
        Type: str
        Default: 'unet-medium'
        Network architecture string (e.g., unet-small, unet-medium, unet-large, scunet-fast). Determines model capacity and VRAM requirements.
    --pretrained_model=PRETRAINED_MODEL
        Type: Optional[str]
        Default: None
        Path to pretrained model to continue training. Previous method, arch, cube_size, CTF_mode, and metrics will be loaded.
    --cube_size=CUBE_SIZE
        Type: int
        Default: 96
        Size in voxels of training subvolumes. Must be compatible with the network (divisible by the network downsampling factors).
    -e, --epochs=EPOCHS
        Type: int
        Default: 50
        Number of training epochs.
    --input_column=INPUT_COLUMN
        Type: str
        Default: 'rlnDeconvTomoName'
        Column name in STAR file to use as input tomograms.
    --batch_size=BATCH_SIZE
        Type: int
        Default: 'auto'
        Number of subtomograms per optimization step; if None, this is automatically determined by multiplying the number of available GPUs by 2. If the number of GPUs is 1, batch size is 4. Batch size per GPU matters for gradient stability.
    --loss_func=LOSS_FUNC
        Type: str
        Default: 'L2'
        Loss function to use (L2, Huber, L1).
    --learning_rate=LEARNING_RATE
        Type: float
        Default: 0.0003
        Initial learning rate.
    --save_interval=SAVE_INTERVAL
        Type: int
        Default: 10
        Interval to save model checkpoints.
    --learning_rate_min=LEARNING_RATE_MIN
        Type: float
        Default: 0.0003
        Minimum learning rate for scheduler.
    --mw_weight=MW_WEIGHT
        Type: float
        Default: -1
        Weight for missing wedge loss. Higher values correspond to stronger emphasis on missing wedge regions. Disabled by default.
    --apply_mw_x1=APPLY_MW_X1
        Type: bool
        Default: True
        Whether to apply missing wedge to subtomograms at the beginning.
    --mixed_precision=MIXED_PRECISION
        Type: bool
        Default: True
        If True, uses float16/mixed precision to reduce VRAM and speed up training.
    -C, --CTF_mode=CTF_MODE
        Type: str
        Default: 'None'
        CTF handling mode: "None": No CTF correction, "phase_only": Phase-only correction, "network": Applies CTF-shaped filter to network input, "wiener": Applies Wiener filter to network target
    --clip_first_peak_mode=CLIP_FIRST_PEAK_MODE
        Type: int
        Default: 1
        Controls attenuation of overrepresented very-low-frequency CTF peak. Options 2 and 3 might increase low-resolution contrast. 0: none, 1: constant clip, 2: negative sine, 3: cosine
    --bfactor=BFACTOR
        Type: float
        Default: 0
        B-factor applied during training/prediction to boost high-frequency content. For cellular tomograms we recommend a b-factor of 0. For isolated samples, you can use a b-factor from 200–300.
    --isCTFflipped=ISCTFFLIPPED
        Type: bool
        Default: False
        Whether input tomograms are phase flipped.
    --do_phaseflip_input=DO_PHASEFLIP_INPUT
        Type: bool
        Default: True
        Whether to apply phase flip during training.
    --noise_level=NOISE_LEVEL
        Type: float
        Default: 0
        Adds artificial noise during training.
    --noise_mode=NOISE_MODE
        Type: str
        Default: 'nofilter'
        Controls filter applied when generating synthetic noise (None, ramp, hamming).
    -r, --random_rot_weight=RANDOM_ROT_WEIGHT
        Type: float
        Default: 0.2
        Percentage of rotations applied as random augmentation. 2.
    -w, --with_preview=WITH_PREVIEW
        Type: bool
        Default: True
        If True, run prediction using the final checkpoint(s) after training.
    --prev_tomo_idx=PREV_TOMO_IDX
        Type: str
        Default: 1
        If set, automatically predict only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").
    --snrfalloff=SNRFALLOFF
        Type: float
        Default: 0
        Controls frequency-dependent SNR attenuation applied during deconvolution; larger values reduce high-frequency contribution more aggressively and can stabilize deconvolution on noisy data; smaller values preserve more high-frequency content but risk amplifying noise.
    --deconvstrength=DECONVSTRENGTH
        Type: float
        Default: 1
        Scalar multiplier for deconvolution strength; increasing this emphasizes correction and low-frequency recovery but can introduce ringing/artifacts if set too high. 0.
    -h, --highpassnyquist=HIGHPASSNYQUIST
        Type: float
        Default: 0.02
        Fraction of the Nyquist used as a very-low-frequency high-pass cutoff; use to remove large-scale intensity gradients and drift; usually left at default. 02.
    """

    prefix = "tomo_denoise_isonet2_refine"

    # we always pass these parameters
    values = [ "method", "arch", "cube_size", "epochs", "loss_func", "save_interval", "learning_rate", "learning_rate_min", "apply_mw_x1", "mixed_precision", "mw_weight", "bfactor", "clip_first_peak_mode", "do_phaseflip_input", "noise_level", "noise_mode", "random_rot_weight", "with_preview", "snrfalloff", "deconvstrength", "highpassnyquist" ]
    
    # we only pass these if True
    booleans = [ "isCTFflipped" ]

    # we only pass these if not empty
    strings = [ "CTF_mode", "prev_tomo_idx", "input_column" ]

    isonet_refine_parameters = build_command_options( parameters, prefix, values, booleans, strings )

    pretrained_model = parameters.get(f"{prefix}_pretrained_model")
    if pretrained_model and os.path.exists( project_params.resolve_path(pretrained_model) ):
        isonet_refine_parameters += f" --pretrained_model '{project_params.resolve_path(pretrained_model)}'"

    if parameters.get(f"{prefix}_batch_size") > 0:
        isonet_refine_parameters += f" --batch_size {parameters.get(f'{prefix}_batch_size')}"
    """
    if parameters["tomo_denoise_isonet2_mask_preprocessing"] == "deconv":
        isonet_refine_parameters += " --input_column rlnDeconvTomoName"
    elif parameters["tomo_denoise_isonet2_mask_preprocessing"] == "denoise":
        isonet_refine_parameters += " --input_column rlnDenoisedTomoName"
    else:
        isonet_refine_parameters += " --input_column rlnCorrectedTomoName"
    """
    output_dir = "isonet_maps"
    
    command = get_isonet2_path() + f"""isonet.py refine {input_star} --output_dir {output_dir} {isonet_refine_parameters} --gpuID {get_gpu_ids(parameters)} --ncpus {parameters['slurm_tasks']}"""

    output = []
    def obs(line):
        output.append(line)
    
    local_run.stream_shell_command(command,observer=obs)

    return output

def isonet2_generate_star(project_dir, outputname, parameters, name_list):
    """
    Generate star file with tomograms names and defocus
    """

    star_header = """
data_

loop_
_rlnIndex #1
_rlnTomoName #2
_rlnTomoReconstructedTomogramHalf1 #3
_rlnTomoReconstructedTomogramHalf2 #4
_rlnPixelSize #5
_rlnDefocus #6
_rlnVoltage #7
_rlnSphericalAberration #8
_rlnAmplitudeContrast #9
_rlnMaskBoundary #10
_rlnMaskName #11
_rlnTiltMin #12
_rlnTiltMax #13
_rlnBoxFile #14
_rlnNumberSubtomo #15
_rlnCorrectedTomoName #16
_rlnDeconvTomoName #17
_rlnMicrographName #18"""

    # all_tomograms = glob.glob(f"{project_dir}/mrc/*.rec")
    # tomograms = [t for t in all_tomograms if not "denoised" in t]

    sub_tomograms_per_tomo = parameters.get("tomo_denoise_isonet2_denoise_total_subtomos") // len(name_list)

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
            min_tilt = metadata.data['tlt'].to_numpy().min()
            max_tilt = metadata.data['tlt'].to_numpy().max()
            tomo_half1 = os.path.join( os.getcwd(), name + "_half1.rec")
            tomo_half2 = os.path.join( os.getcwd(), name + "_half2.rec")
            
            sub_tomograms = sub_tomograms_per_tomo
            f.write(f"\n{i + 1}    {tomo}   {tomo_half1}    {tomo_half2}    {pixel_size}    {df}    {parameters['scope_voltage']}    {parameters['scope_cs']}    {parameters['scope_wgh']}   None    None    {min_tilt}    {max_tilt}    None {sub_tomograms}   ./corrected_tomos/_n2n_{parameters.get("tomo_denoise_isonet2_denoise_arch")}_{name}_half1.mrc    ./deconv/{name}.rec   {tomo}")


def isonet2_ctf_deconvolve(tomo_star, parameters, output = './deconv'):
    """
    NAME
        isonet.py deconv - CTF deconvolution preprocessing that enhances low-resolution contrast and recovers information attenuated by the microscope contrast transfer function. Recommended for non–phase-plate data; skip for phase-plate data or if intending to use network-based CTF deconvolution.

    SYNOPSIS
        isonet.py deconv STAR_FILE <flags>

    DESCRIPTION
        CTF deconvolution preprocessing that enhances low-resolution contrast and recovers information attenuated by the microscope contrast transfer function. Recommended for non–phase-plate data; skip for phase-plate data or if intending to use network-based CTF deconvolution.

    POSITIONAL ARGUMENTS
        STAR_FILE
            Type: str
            Input STAR listing tomograms and acquisition metadata. Required parameter.

    FLAGS
        --output_dir=OUTPUT_DIR
            Type: str
            Default: './deconv'
            Folder to write deconvolved tomograms (rlnDeconvTomoName entries point here). /deconv".
        -i, --input_column=INPUT_COLUMN
            Type: str
            Default: 'rlnTomoName'
            STAR column used for input tomogram paths.
        -s, --snrfalloff=SNRFALLOFF
            Type: float
            Default: 1
            Controls frequency-dependent SNR attenuation applied during deconvolution; larger values reduce high-frequency contribution more aggressively and can stabilize deconvolution on noisy data; smaller values preserve more high-frequency content but risk amplifying noise. 0.
        -d, --deconvstrength=DECONVSTRENGTH
            Type: float
            Default: 1
            Scalar multiplier for deconvolution strength; increasing this emphasizes correction and low-frequency recovery but can introduce ringing/artifacts if set too high. 0.
        -h, --highpassnyquist=HIGHPASSNYQUIST
            Type: float
            Default: 0.02
            Fraction of the Nyquist used as a very-low-frequency high-pass cutoff; use to remove large-scale intensity gradients and drift; usually left at default. 02.
        -c, --chunk_size=CHUNK_SIZE
            Type: Optional[int]
            Default: None
            If set, tomograms are processed in smaller cubic chunks to reduce memory usage. Useful for very large tomograms or limited RAM/VRAM. May create edge artifacts if chunks are too small.
        --overlap_rate=OVERLAP_RATE
            Type: float
            Default: 0.25
            Fractional overlap between adjacent chunks when chunking; larger overlaps reduce edge artifacts at cost of extra computation. 25.
        -n, --ncpus=NCPUS
            Type: int
            Default: 4
            Number of CPU workers for CPU-bound parts of deconvolution; increase on multi-core systems.
        -p, --phaseflipped=PHASEFLIPPED
            Type: bool
            Default: False
            If True, input is assumed already phase-flipped; otherwise the function uses defocus and CTF info to apply phase handling.
        -t, --tomo_idx=TOMO_IDX
            Type: Optional[str]
            Default: None
            If set, process only the tomograms listed by these indices (e.g., "1,2,4" or "5-10,15,16").

    NOTES
        You can also use flags syntax for POSITIONAL ARGUMENTS
    """
 
 
    prefix = "tomo_denoise_isonet2_deconv"

    # we always pass these parameters
    values = [ "input_column", "snrfalloff", "deconvstrength", "highpassnyquist", "overlap_rate" ]
    
    # we only pass these if True
    booleans = [ "phaseflipped" ]

    # we only pass these if not empty
    strings = [ "tomo_idx" ]

    isonet_deconv_parameters = build_command_options( parameters, prefix, values, booleans, strings )

    if parameters.get(f"{prefix}_chunk_size") > 0:
        isonet_deconv_parameters += f" --chunk_size {parameters.get(f'{prefix}_chunk_size')}"
    
    command = get_isonet2_path() + f"isonet.py deconv {tomo_star} {isonet_deconv_parameters} --ncpus {parameters.get('slurm_tasks')}"
    
    local_run.stream_shell_command(command)


def build_command_options( parameters, prefix, values, booleans, strings):
    
    isonet_parameters = "" 
           
    # we always pass these parameters
    for key in values:
        isonet_parameters += f" --{key} {parameters.get(prefix + '_' + key)}"

    # we only pass these if True
    for key in booleans:
        if parameters.get(prefix + '_' + key):
            isonet_parameters += f" --{key}"
    
    # we only pass these if not empty
    for key in strings:
        parameter = parameters.get(prefix + '_' + key)
        if parameter is not None and len(str(parameter)) > 0:
            isonet_parameters += f" --{key} {parameter}"

    return isonet_parameters