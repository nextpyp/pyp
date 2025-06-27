# cryoDRGN
import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm
from pyp.streampyp.logging import TQDMLogger
from pyp.system import local_run, project_params, mpi
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_cryodrgn_path():
    command_base = f"export PYTHONPATH=/opt/conda/envs/cryodrgn/lib/python3.9/site-packages:$PYTHONPATH; micromamba run -n cryodrgn /opt/conda/envs/cryodrgn/bin/cryodrgn"
    return command_base

def cryodrgn_preprocess(alignment_star, particle_stack_list, output, boxsize, downsample_size, threads=1):
    """prepare CryoDRGN metadata and downsample particles for traning"""

    assert os.path.exists(alignment_star), "The input star dose not exist"

    # cryodrgn pose pkl
    command = f"{get_cryodrgn_path()} parse_pose_star {alignment_star} -o {output}_poses.pkl -D {boxsize}"
    local_run.run_shell_command(command, verbose=True)

    # ctf metadata pkl
    command = f"{get_cryodrgn_path()} parse_ctf_star {alignment_star} -o {output}_ctf.pkl"
    local_run.run_shell_command(command, verbose=True)

    if boxsize > downsample_size:
        logger.info(f"Downsampling particles size to {downsample_size}")
        # downsample particles 
        with tqdm(desc="Progress", total=len(particle_stack_list), file=TQDMLogger()) as pbar:
            for stack in particle_stack_list:            
                command = f"{get_cryodrgn_path()} downsample {stack} -D {int(downsample_size)} -o {stack.replace('.mrcs', '')}_{downsample_size}.mrcs --max-threads {threads}"
                local_run.run_shell_command(command)
                pbar.update(1)

        # edit the input alignment star to replace the new mrcs
        command = f"sed 's/.mrcs/_{downsample_size}.mrcs/g' {alignment_star} > {alignment_star.replace('.star', '_downsample.star')}"
        local_run.run_shell_command(command, verbose=True)

        downsampled = True
    else:
        downsampled = False
        logger.warning("Downsampled size is actually larger than the original box size, skip.")

    return downsampled

def cryodrgn_train(parameters, input_dir, name, output, downsampled=True):
    """Train a VAE for heterogeneous reconstruction with known pose"""

    """
    positional arguments:
    particles             Input particles (.mrcs, .star, .cs, or .txt)

    options:
        -h, --help            show this help message and exit
        -o OUTDIR, --outdir OUTDIR
                    Output directory to save model
        --zdim ZDIM           Dimension of latent variable
        --poses POSES         Image poses (.pkl)
        --ctf pkl             CTF parameters (.pkl)
        --load WEIGHTS.PKL    Initialize training from a checkpoint
        --checkpoint CHECKPOINT
                    Checkpointing interval in N_EPOCHS (default: 1)
        --log-interval LOG_INTERVAL
                    Logging interval in N_IMGS (default: 1000)
        -v, --verbose         Increase verbosity
        --seed SEED           Random seed

        Dataset loading:
        --ind PKL             Filter particle stack by these indices
        --uninvert-data       Do not invert data sign
        --no-window           Turn off real space windowing of dataset
        --window-r WINDOW_R   Windowing radius (default: 0.85)
        --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file
        --lazy                Lazy loading if full dataset is too large to fit in memory
        --shuffler-size SHUFFLER_SIZE
        If non-zero, will use a data shuffler for faster lazy data loading.
        --preprocessed        Skip preprocessing steps if input data is from cryodrgn preprocess_mrcs
        --num-workers NUM_WORKERS
                    Number of subprocesses to use as DataLoader workers. If 0, then use the main process for data loading. (default: 0)
        --max-threads MAX_THREADS
                    Maximum number of CPU cores for data loading (default: 16)

    Tilt series:
        --tilt TILT           Particle stack file (.mrcs)
        --tilt-deg TILT_DEG   X-axis tilt offset in degrees (default: 45)

    Training parameters:
        -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                    Number of training epochs (default: 20)
        -b BATCH_SIZE, --batch-size BATCH_SIZE
                    Minibatch size (default: 8)
        --wd WD               Weight decay in Adam optimizer (default: 0)
        --lr LR               Learning rate in Adam optimizer (default: 0.0001)
        --beta BETA           Choice of beta schedule or a constant for KLD weight (default: 1/zdim)
        --beta-control BETA_CONTROL
                    KL-Controlled VAE gamma. Beta is KL target
        --norm NORM NORM      Data normalization as shift, 1/scale (default: mean, std of dataset)
        --no-amp              Do not use mixed-precision training
        --multigpu            Parallelize training across all detected GPUs

    Pose SGD:
        --do-pose-sgd         Refine poses with gradient descent
        --pretrain PRETRAIN   Number of epochs with fixed poses before pose SGD (default: 1)
        --emb-type {s2s2,quat}
                    SO(3) embedding type for pose SGD (default: quat)
        --pose-lr POSE_LR     Learning rate for pose optimizer (default: 0.0003)

    Encoder Network:
        --enc-layers QLAYERS  Number of hidden layers (default: 3)
        --enc-dim QDIM        Number of nodes in hidden layers (default: 1024)
        --encode-mode {conv,resid,mlp,tilt}
                    Type of encoder network (default: resid)
        --enc-mask ENC_MASK   Circular mask of image for encoder (default: D/2; -1 for no mask)
        --use-real            Use real space image for encoder (for convolutional encoder)

    Decoder Network:
        --dec-layers PLAYERS  Number of hidden layers (default: 3)
        --dec-dim PDIM        Number of nodes in hidden layers (default: 1024)
        --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                    Type of positional encoding (default: gaussian)
        --feat-sigma FEAT_SIGMA
                    Scale for random Gaussian features (default: 0.5)
        --pe-dim PE_DIM       Num frequencies in positional encoding (default: image D/2)
        --domain {hartley,fourier}
                    Volume decoder representation (default: fourier)
        --activation {relu,leaky_relu}
                    Activation (default: relu)                 
    """

    if downsampled:
        particles_input = os.path.join(input_dir, name + "_particles_downsample.star")
    else:
        particles_input = os.path.join(input_dir, name + "_particles.star")
    
    pose_input = os.path.join(input_dir, name + "_poses.pkl")
    ctf_input = os.path.join(input_dir, name + "_ctf.pkl")

    options = f"--checkpoint {parameters['heterogeneity_cryodrgn_train_checkpoint']} \
        --log-interval {parameters['heterogeneity_cryodrgn_train_log_interval']} \
        --num-workers 0 \
        --max-threads {parameters['slurm_tasks']}"

    if parameters.get("heterogeneity_cryodrgn_train_weight") and os.path.exists(parameters["heterogeneity_cryodrgn_train_weight"]):
        options += f" --load {parameters['heterogeneity_cryodrgn_train_weight']}"

    if parameters.get("heterogeneity_cryodrgn_train_seed"):
        options += f" --seed {parameters['heterogeneity_cryodrgn_train_seed']}"
    
    if parameters.get("heterogeneity_cryodrgn_data_ind") and os.path.exists(parameters["heterogeneity_cryodrgn_data_ind"]):
        options += f" --ind {parameters['heterogeneity_cryodrgn_data_ind']}"

    if not parameters["heterogeneity_cryodrgn_data_invert"]:
        options += " --uninvert-data"

    if not parameters["heterogeneity_cryodrgn_data_windowing"]:
        options += " --no-window"
    else:
        options += f" --window-r {parameters['heterogeneity_cryodrgn_data_window_r']}"
    
    if parameters.get("heterogeneity_cryodrgn_data_dir"):
        data_dir = project_params.resolve_path(parameters["heterogeneity_cryodrgn_data_dir"])
        assert os.path.isdir(data_dir), "The data input directory dose not exist"
        options += f" --datadir {data_dir}"

    if parameters["heterogeneity_cryodrgn_data_lazy"]:
        options += " --lazy"

        if parameters['heterogeneity_cryodrgn_data_shufflersize'] > 0:
            options += f" --shuffler-size {parameters['heterogeneity_cryodrgn_data_shufflersize']}"

    if parameters["slurm_verbose"]:
        options += " -v"

    if False and  downsampled:
        options += " --preprocessed"

    training_parameters = f"-n {parameters['heterogeneity_cryodrgn_train_epochs']} \
        -b {parameters['heterogeneity_cryodrgn_train_batch']} \
        --wd {parameters['heterogeneity_cryodrgn_train_wd']} \
        --lr {parameters['heterogeneity_cryodrgn_train_lr']} \
        --pretrain {parameters['heterogeneity_cryodrgn_pretrain']} \
        --emb-type {parameters['heterogeneity_cryodrgn_emd_type']} \
        --pose-lr {parameters['heterogeneity_cryodrgn_pose_lr']} \
        --enc-layers {parameters['heterogeneity_cryodrgn_enc_hl']} \
        --enc-dim {parameters['heterogeneity_cryodrgn_enc_dim']} \
        --encode-mode {parameters['heterogeneity_cryodrgn_enc_mode']} \
        --dec-layers {parameters['heterogeneity_cryodrgn_dec_hl']} \
        --dec-dim {parameters['heterogeneity_cryodrgn_dec_dim']} \
        --pe-type {parameters['heterogeneity_cryodrgn_pe_type']} \
        --domain {parameters['heterogeneity_cryodrgn_dec_domain']} \
        --activation {parameters['heterogeneity_cryodrgn_activation']}\
        "
    
    if parameters.get('heterogeneity_cryodrgn_enc_mask'):
        training_parameters += f" --enc-mask {parameters['heterogeneity_cryodrgn_enc_mask']}"
    if "conv" in parameters['heterogeneity_cryodrgn_enc_mode'] and parameters['heterogeneity_cryodrgn_use_real']:
        training_parameters += " --use_real"
    
    if "gaussian" in parameters['heterogeneity_cryodrgn_pe_type']:
        training_parameters += f" --feat-sigma {parameters['heterogeneity_cryodrgn_feat_sigma']}"

    if parameters.get('heterogeneity_cryodrgn_pe_dim'):
        training_parameters += f" --pe-dim {parameters['heterogeneity_cryodrgn_pe_dim']}"

    if parameters.get('heterogeneity_cryodrgn_train_beta'):
        training_parameters += f" --beta {parameters['heterogeneity_cryodrgn_train_beta']}"
    if parameters.get('heterogeneity_cryodrgn_train_beta_control'):
        training_parameters += f" --beta-control {parameters['heterogeneity_cryodrgn_train_beta_control']}"    

    if not "None" in parameters['heterogeneity_cryodrgn_data_norm']:
        training_parameters += f" --norm {parameters['heterogeneity_cryodrgn_data_norm']}"
    if not parameters['heterogeneity_cryodrgn_train_amp']:
        training_parameters += " --no-amp"
    
    if parameters['heterogeneity_cryodrgn_pose_sgd']:
        training_parameters += " --do-pose-sgd"
    
    if "tomo" in parameters["data_mode"]:
        tomo = f" --encode-mode tilt --dose-per-tilt {parameters['scope_dose_rate']} --ntilts 1"
    else:
        tomo = ""

    # TODO: multigpu option

    command = f"{get_cryodrgn_path()} train_vae {particles_input} --datadir {input_dir} --ctf {ctf_input} --poses {pose_input} --zdim {parameters['heterogeneity_cryodrgn_train_zdim']} --num-epochs {parameters['heterogeneity_cryodrgn_train_epochs']} -o {output} {options} {training_parameters} {tomo}"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])

def cryodrgn_analyze(input_dir, output, parameters, downsampled):
    """cryodrgn analyze""" 
    """usage: cryodrgn analyze [-h] [--device DEVICE] [-o OUTDIR] [--skip-vol] [--skip-umap] [--Apix APIX] [--flip] [--invert] -d DOWNSAMPLE] [--pc PC] [--ksample KSAMPLE] [--vol-start-index VOL_START_INDEX workdir epoch

    Visualize latent space and generate volumes

    positional arguments:
    workdir               Directory with cryoDRGN results
    epoch                 Epoch number N to analyze (0-based indexing, corresponding to z.N.pkl, weights.N.pkl)

    options:
        --device DEVICE       Optionally specify CUDA device
        -o OUTDIR, --outdir OUTDIR
                    Output directory for analysis results (default: [workdir]/analyze.[epoch])
        --skip-vol            Skip generation of volumes
        --skip-umap           Skip running UMAP

        Extra arguments for volume generation:
        --Apix APIX           Pixel size to add to .mrc header (default: 1 A/pix)
        --flip                Flip handedness of output volumes
        --invert              Invert contrast of output volumes
        -d DOWNSAMPLE, --downsample DOWNSAMPLE
                    Downsample volumes to this box size (pixels)
        --pc PC               Number of principal component traversals to generate (default: 2)
        --ksample KSAMPLE     Number of kmeans samples to generate (default: 20)
        --vol-start-index VOL_START_INDEX
                    Default value of start index for volume generation (default: 0)
    """

    options = ''

    if parameters['heterogeneity_cryodrgn_analysis_skipv']:
        options += " --skip-vol"
    if parameters['heterogeneity_cryodrgn_analysis_skipumap']:
        options += " --skip-umap"
    if parameters['heterogeneity_cryodrgn_analysis_flip']:
        options += " --flip"
    if parameters['heterogeneity_cryodrgn_analysis_invert']:
        options += " --invert"
    
    if parameters.get('heterogeneity_cryodrgn_analysis_downsample'):
        options += f" -d {parameters['heterogeneity_cryodrgn_analysis_downsample']}"
    
    original_pixelsize = parameters['scope_pixel'] * parameters["data_bin"] * parameters["extract_bin"]

    if downsampled and parameters.get('heterogeneity_cryodrgn_analysis_downsample'):
        
        output_pixel = original_pixelsize * (parameters["extract_box"] / parameters["heterogeneity_cryodrgn_downsample_size"]) * (parameters["heterogeneity_cryodrgn_downsample_size"]/ parameters["heterogeneity_cryodrgn_analysis_downsample"])
    
    elif parameters.get('heterogeneity_cryodrgn_analysis_downsample'):

        output_pixel = original_pixelsize * (parameters["extract_box"] / parameters["heterogeneity_cryodrgn_analysis_downsample"])
    
    else:
        output_pixel = original_pixelsize

    if parameters['heterogeneity_cryodrgn_analysis_epoch'] == 0 or parameters['heterogeneity_cryodrgn_analysis_epoch'] >= parameters['heterogeneity_cryodrgn_train_epochs']:
        parameters['heterogeneity_cryodrgn_analysis_epoch'] = parameters['heterogeneity_cryodrgn_train_epochs'] - 1

    command = f"{get_cryodrgn_path()} analyze {input_dir} {parameters['heterogeneity_cryodrgn_analysis_epoch']} -o {output} --pc {parameters['heterogeneity_cryodrgn_analysis_pc']} --ksample {parameters['heterogeneity_cryodrgn_analysis_ksample']} --vol-start-index {parameters['heterogeneity_cryodrgn_analysis_istart']} {options} --Apix {output_pixel}"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])


# TODO - only run training tasks
def run_cryodrgn_train(project_dir, parameters):

    # scratch space
    working_path = Path(os.environ["PYP_SCRATCH"]) / "cryodrgn"

    logger.info(f"Working path: {working_path}")

    working_path.mkdir(parents=True, exist_ok=True)
    os.chdir(working_path)

    input_path =os.path.join(
        project_params.resolve_path(parameters["data_parent"]), "relion", "stacks",
    )
    name = parameters['data_set']
    starfile = name + "_particles.star"

    input_star = Path(project_params.resolve_path(parameters["heterogeneity_input_star"]))
    assert input_star.exists(), "Can not find input star file, run the extract stacks first."

    # this is the input star and partilces stacks folder
    
    (working_path / "input_data").mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_star, working_path / "input_data" / starfile)

    particles_stacks = glob.glob(os.path.join(input_path, "*_stack.mrcs"))

    tasks = []
    for stack in particles_stacks:
        
        command = f"cp {stack} {os.getcwd()}/input_data/"
        tasks.append(command)

    mpi.submit_jobs_to_workers(tasks, os.getcwd())
    
    particle_stack_list = [os.path.basename(p) for p in particles_stacks]

    # do preprocessing in input_data folder
    os.chdir("input_data")
    
    boxsize = parameters['extract_box']
    downsample_size = parameters['heterogeneity_cryodrgn_downsample_size']

    downsampled = cryodrgn_preprocess(starfile, particle_stack_list, name, boxsize, downsample_size, threads=parameters['slurm_tasks'])

    os.chdir("../")

    (working_path / "train_output").mkdir(parents=True, exist_ok=True)

    # train
    cryodrgn_train(parameters, "input_data", name, "train_output", downsampled=downsampled)

    if (working_path / "train_output" / "weights.pkl").exists() and (working_path / "train_output" / "z.pkl").exists():
        saved_folder = os.path.join(project_dir, "train")
        logger.info(f"Training finished successfully, saving results to {saved_folder}")

        shutil.copytree((working_path / "train_output"), Path(saved_folder), dirs_exist_ok=True)
    else:
        raise Exception("Training did not finish successfully")

    # analyze
    logger.info("Running CryoDRGN analyze")
    cryodrgn_analyze("train_output", "analyze_output", parameters, downsampled)

    final_output = os.path.join(project_dir, "train", "heterogeneity_cryodrgn_analyze_" + str(parameters["heterogeneity_cryodrgn_analysis_epoch"]))
    
    if not os.path.exists(final_output):
        Path(final_output).mkdir()

    logger.info(f"Saving the final results to {final_output}")

    shutil.copytree((working_path / "analyze_output"), Path(final_output), dirs_exist_ok=True)

# TODO - only run evaluation tasks    
def run_cryodrgn_eval(project_dir, parameters):

    # scratch space
    working_path = Path(os.environ["PYP_SCRATCH"]) / "cryodrgn"

    logger.info(f"Working path: {working_path}")

    working_path.mkdir(parents=True, exist_ok=True)
    os.chdir(working_path)

    input_path =os.path.join(
        project_params.resolve_path(parameters["data_parent"]), "relion", "stacks",
    )
    name = parameters['data_set']
    starfile = name + "_particles.star"

    input_star = Path(project_params.resolve_path(parameters["heterogeneity_input_star"]))
    assert input_star.exists(), "Can not find input star file, run the extract stacks first."

    # this is the input star and partilces stacks folder
    
    (working_path / "input_data").mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_star, working_path / "input_data" / starfile)

    particles_stacks = glob.glob(os.path.join(input_path, "*_stack.mrcs"))

    tasks = []
    for stack in particles_stacks:
        
        command = f"cp {stack} {os.getcwd()}/input_data/"
        tasks.append(command)

    mpi.submit_jobs_to_workers(tasks, os.getcwd())
    
    particle_stack_list = [os.path.basename(p) for p in particles_stacks]

    # do preprocessing in input_data folder
    os.chdir("input_data")
    
    boxsize = parameters['extract_box']
    downsample_size = parameters['heterogeneity_cryodrgn_downsample_size']

    downsampled = cryodrgn_preprocess(starfile, particle_stack_list, name, boxsize, downsample_size, threads=parameters['slurm_tasks'])

    os.chdir("../")

    (working_path / "train_output").mkdir(parents=True, exist_ok=True)

    # train
    cryodrgn_train(parameters, "input_data", name, "train_output", downsampled=downsampled)

    if (working_path / "train_output" / "weights.pkl").exists() and (working_path / "train_output" / "z.pkl").exists():
        saved_folder = os.path.join(project_dir, "train")
        logger.info(f"Training finished successfully, saving results to {saved_folder}")

        shutil.copytree((working_path / "train_output"), Path(saved_folder), dirs_exist_ok=True)
    else:
        raise Exception("Training did not finish successfully")

    # analyze
    logger.info("Running CryoDRGN analyze")
    cryodrgn_analyze("train_output", "analyze_output", parameters, downsampled)

    final_output = os.path.join(project_dir, "train", "heterogeneity_cryodrgn_analyze_" + str(parameters["heterogeneity_cryodrgn_analysis_epoch"]))
    
    if not os.path.exists(final_output):
        Path(final_output).mkdir()

    logger.info(f"Saving the final results to {final_output}")

    shutil.copytree((working_path / "analyze_output"), Path(final_output), dirs_exist_ok=True)

