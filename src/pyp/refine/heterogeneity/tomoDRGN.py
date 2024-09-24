# tomodrgn
import os
import shutil
import glob
from pathlib import Path
from pyp.system import local_run, project_params, mpi
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_tomodrgn_path():
    command_base = f". activate tomodrgn; /opt/conda/envs/tomodrgn/bin/tomodrgn"
    return command_base


def tomodrgn_preprocess(alignment_star, particle_stack_list, output, boxsize, downsample_size):
    """prepare tomodrgn metadata and downsample particles for traning"""

    assert os.path.exists(alignment_star), "The input star dose not exist"

    # tomodrgn pose pkl
    command = f"{get_tomodrgn_path()} parse_pose_star {alignment_star} -o {output}_poses.pkl -D {boxsize}"
    local_run.run_shell_command(command, verbose=True)

    # ctf metadata pkl
    command = f"{get_tomodrgn_path()} parse_ctf_star {alignment_star} -o {output}_ctf.pkl"
    local_run.run_shell_command(command, verbose=True)

    if boxsize > downsample_size:
        logger.info(f"Downsampling particles size to {downsample_size}")
        # downsample particles 
        tasks = []
        for stack in particle_stack_list:
            
            command = f"{get_tomodrgn_path()} downsample {stack} -D {downsample_size} -o {stack.replace('.mrcs', '')}_{downsample_size}.mrcs"
            tasks.append(command)
        
        mpi.submit_jobs_to_workers(tasks, os.getcwd(), verbose=False)

        # edit the input alignment star to replace the new mrcs
        command = f"sed 's/.mrcs/_{downsample_size}.mrcs/g' {alignment_star} > {alignment_star.replace('.star', '_downsample.star')}"
        local_run.run_shell_command(command, verbose=True)

        downsampled = True
    else:
        downsampled = False
        logger.warning("Downsampled size is actually larger than the original box size, skip.")

    return downsampled

def tomodrgn_train(parameters, input_dir, name, output):
    """Train a VAE for heterogeneous reconstruction with known pose"""

    if False:
        particles_input = os.path.join(input_dir, name + "_particles_downsampled.star")
    else:
        particles_input = os.path.join(input_dir, name + "_particles.star")
    
    # pose_input = name + "_pose.pkl"

    # ctf_input = name + "_ctf.pkl"

    options = f"--checkpoint {parameters['heterogeneity_tomodrgn_train_checkpoint']} --log-interval {parameters['heterogeneity_tomodrgn_train_log_interval']} --num-workers 0"
    
        # --max-threads {parameters['slurm_tasks']}

    if parameters.get("heterogeneity_tomodrgn_train_weight") and os.path.exists(parameters["heterogeneity_tomodrgn_train_weight"]):
        options += f" --load {parameters['heterogeneity_tomodrgn_train_weight']}"

    if parameters.get("heterogeneity_tomodrgn_train_seed"):
        options += f" --seed {parameters['heterogeneity_tomodrgn_train_seed']}"
    
    if parameters.get("heterogeneity_tomodrgn_data_ind") and os.path.exists(parameters["heterogeneity_tomodrgn_data_ind"]):
        options += f" --ind {parameters['heterogeneity_tomodrgn_data_ind']}"

    if not parameters["heterogeneity_tomodrgn_data_invert"]:
        options += " --uninvert-data"

    if not parameters["heterogeneity_tomodrgn_data_windowing"]:
        options += " --no-window"
    else:
        options += f" --window-r {parameters['heterogeneity_tomodrgn_data_window_r']} --window-r-outer {parameters['heterogeneity_tomodrgn_data_window_r_outer']}"
    
    if parameters.get("heterogeneity_tomodrgn_data_dir"):
        data_dir = project_params.resolve_path(parameters["heterogeneity_tomodrgn_data_dir"])
        assert os.path.isdir(data_dir), "The data input directory dose not exist"
        options += f" --datadir {data_dir}"

    if parameters["heterogeneity_tomodrgn_data_lazy"]:
        options += " --lazy"

    if parameters["slurm_verbose"]:
        options += " -v"

    if False:
        options += " --preprocessed"
    
    # --pretrain {parameters['heterogeneity_tomodrgn_pretrain']} \
    # --emb-type {parameters['heterogeneity_tomodrgn_emd_type']} \
    # --pose-lr {parameters['heterogeneity_tomodrgn_pose_lr']} \

    training_parameters = f"-n {parameters['heterogeneity_tomodrgn_train_epochs']} -b {parameters['heterogeneity_tomodrgn_train_batch']} --wd {parameters['heterogeneity_tomodrgn_train_wd']} --lr {parameters['heterogeneity_tomodrgn_train_lr']} --enc-layers-A {parameters['heterogeneity_tomodrgn_enc_lya']} --enc-dim-A {parameters['heterogeneity_tomodrgn_enc_dima']} --out-dim-A {parameters['heterogeneity_tomodrgn_out_dima']} --enc-layers-B {parameters['heterogeneity_tomodrgn_enc_lyb']} --enc-dim-B {parameters['heterogeneity_tomodrgn_enc_dimb']} --dec-layers {parameters['heterogeneity_tomodrgn_dec_hl']} --dec-dim {parameters['heterogeneity_tomodrgn_dec_dim']} --pe-type {parameters['heterogeneity_tomodrgn_pe_type']} --pooling-function {parameters['heterogeneity_tomodrgn_pool']} --activation {parameters['heterogeneity_tomodrgn_activation']} --num-seeds {parameters['heterogeneity_tomodrgn_num_seeds']} --num-heads {parameters['heterogeneity_tomodrgn_num_heads']} --l-extent {parameters['heterogeneity_tomodrgn_l_ext']}"

    if parameters["heterogeneity_tomodrgn_layer_norm"]:
        training_parameters += " --layer-norm"

    if parameters.get('heterogeneity_tomodrgn_enc_mask'):
        training_parameters += f" --enc-mask {parameters['heterogeneity_tomodrgn_enc_mask']}"
    
    if False and "gaussian" in parameters['heterogeneity_tomodrgn_pe_type']:
        training_parameters += f" --feat_sigma {parameters['heterogeneity_tomodrgn_feat_sigma']}"

    if parameters.get('heterogeneity_tomodrgn_pe_dim'):
        training_parameters += f" --pe-dim {parameters['heterogeneity_tomodrgn_pe_dim']}"

    if parameters.get('heterogeneity_tomodrgn_train_beta'):
        training_parameters += f" --beta {parameters['heterogeneity_tomodrgn_train_beta']}"
    if parameters.get('heterogeneity_tomodrgn_train_beta_control'):
        training_parameters += f" --beta-control {parameters['heterogeneity_tomodrgn_train_beta_control']}"    

    if not "None" in parameters['heterogeneity_tomodrgn_data_norm']:
        training_parameters += f" --norm {parameters['heterogeneity_tomodrgn_data_norm']}"
    if not parameters['heterogeneity_tomodrgn_train_amp']:
        training_parameters += " --no-amp"
    
    # if parameters['heterogeneity_tomodrgn_pose_sgd']:
    #     training_parameters += " --do-pose-sgd"
    
    tomo = ""
    if parameters["heterogeneity_tomodrgn_tilt_weight"]:
        tomo += " --recon-tilt-weight"
    elif parameters["heterogeneity_tomodrgn_dose_weight"]:
        tomo += " --recon-dose-weight"
    else:
        pass 
    tomo += " --sort-ptcl-imgs dose_ascending"

    if parameters["heterogeneity_tomodrgn_dose_mask"]:
        tomo += " --l-dose-mask"

    if parameters["heterogeneity_tomodrgn_dose"] > 0:
        tomo += f" --dose-override {parameters['heterogeneity_tomodrgn_dose']}"

    if parameters["heterogeneity_tomodrgn_use_ptl"] > 0:
        tomo += f" --use-first-nptcls {parameters['heterogeneity_tomodrgn_use_ptl']}"
    
    if parameters["heterogeneity_tomodrgn_sequential_order"]:
        tomo += " --sequential-tilt-sampling"
    
    if parameters["heterogeneity_tomodrgn_use_firstn"] > 0:
        tomo += f" --use-first-ntilts {parameters['heterogeneity_tomodrgn_use_firstn']}"

    # TODO: multigpu option
    
    workers = ""
    if parameters["heterogeneity_tomodrgn_num_workers"] > 0:
        workers += f" --num-workers {parameters['heterogeneity_tomodrgn_num_workers']}"
    else:
        workers += f" --num-workers {parameters['slurm_tasks']}"

    if parameters["heterogeneity_tomodrgn_prefetch_factor"] > 0:
        workers += f" --prefetch-factor {parameters['heterogeneity_tomodrgn_prefetch_factor']}"

    if parameters["heterogeneity_tomodrgn_pst_wkrs"]:
        workers += " --persistent-workers"
    
    if parameters["heterogeneity_tomodrgn_pin_mem"]:
        workers += " --pin-memory"

    # --ctf {ctf_input} --pose {pose_input}

    command = f"{get_tomodrgn_path()} train_vae {particles_input} --datadir {input_dir} --zdim {parameters['heterogeneity_tomodrgn_train_zdim']} -o {output} {options} {training_parameters} {tomo} {workers}"

    local_run.run_shell_command(command, verbose=parameters['slurm_verbose'])

def tomodrgn_analyze(input_dir, output, parameters):
    """tomodrgn analyze""" 
    """usage: tomodrgn analyze [-h] [--device DEVICE] [-o OUTDIR] [--skip-vol] [--skip-umap] [--Apix APIX] [--flip] [--invert] -d DOWNSAMPLE] [--pc PC] [--ksample KSAMPLE] [--vol-start-index VOL_START_INDEX workdir epoch

    Visualize latent space and generate volumes

    positional arguments:
    workdir               Directory with tomodrgn results
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

    if parameters['heterogeneity_tomodrgn_analysis_skipv']:
        options += " --skip-vol"
    if parameters['heterogeneity_tomodrgn_analysis_skipumap']:
        options += " --skip-umap"
    if parameters['heterogeneity_tomodrgn_analysis_flip']:
        options += " --flip"
    if parameters['heterogeneity_tomodrgn_analysis_invert']:
        options += " --invert"
    if parameters["heterogeneity_tomodrgn_analysis_pc_ondata"]:
        options += " --pc-ondata"
    
    if parameters.get('heterogeneity_tomodrgn_analysis_downsample'):
        options += f" -d {parameters['heterogeneity_tomodrgn_analysis_downsample']}"
    
    original_pixelsize = parameters['scope_pixel'] * parameters["data_bin"] * parameters["extract_bin"]

    if parameters.get('heterogeneity_tomodrgn_analysis_downsample'):
        
        output_pixel = original_pixelsize * (parameters["extract_box"] / parameters["heterogeneity_tomodrgn_analysis_downsample"])
    else:
        output_pixel = original_pixelsize

    if parameters['heterogeneity_tomodrgn_analysis_epoch'] == 0 or parameters['heterogeneity_tomodrgn_analysis_epoch'] >= parameters['heterogeneity_tomodrgn_train_epochs']:
        parameters['heterogeneity_tomodrgn_analysis_epoch'] = parameters['heterogeneity_tomodrgn_train_epochs'] - 1

    command = f"{get_tomodrgn_path()} analyze {input_dir} --epoch {parameters['heterogeneity_tomodrgn_analysis_epoch']} -o {output} --pc {parameters['heterogeneity_tomodrgn_analysis_pc']} --ksample {parameters['heterogeneity_tomodrgn_analysis_ksample']} {options}"

    local_run.run_shell_command(command, verbose=parameters['slurm_verbose'])


def run_tomodrgn(project_dir, parameters):

    # scratch space
    working_path = Path(os.environ["PYP_SCRATCH"]) / "tomodrgn"

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

    logger.info(f"Copying {len(particles_stacks):,} particle stacks to local scratch:")
    mpi.submit_jobs_to_workers(tasks, os.getcwd(), verbose=False)
        
    # particle_stack_list = [os.path.basename(p) for p in particles_stacks]

    # do preprocessing in input_data folder
    # os.chdir("input_data")
    
    # boxsize = parameters['extract_box']
    # downsample_size = parameters['heterogeneity_tomodrgn_downsample_size']

    # downsampled = tomodrgn_preprocess(starfile, particle_stack_list, name, boxsize, downsample_size)

    # os.chdir("../")

    (working_path / "train_output").mkdir(parents=True, exist_ok=True)

    # train
    tomodrgn_train(parameters, "input_data", name, "train_output")

    if (working_path / "train_output" / "weights.pkl").exists() and (working_path / "train_output" / "z.train.pkl").exists():
        saved_folder = os.path.join(project_dir, "train")
        logger.info(f"Training finished successfully, saving results to {saved_folder}")

        shutil.copytree((working_path / "train_output"), Path(saved_folder), dirs_exist_ok=True)
    else:
        raise Exception("Training did not finish successfully")

    # convergence
    logger.info("Running tomoDRGN convergence_vae")
    convergence_vae(parameters, "train_output", "analyze_output")

    final_output = os.path.join(project_dir, "train", "heterogeneity_tomodrgn_analyze_" + str(parameters["heterogeneity_tomodrgn_analysis_epoch"]))
    
    if not os.path.exists(final_output):
        Path(final_output).mkdir()

    logger.info(f"Saving the final results to {final_output}")

    shutil.copytree((working_path / "analyze_output"), Path(final_output), dirs_exist_ok=True)


def train_nn(parameters, input_dir, name, output):

    """Train a decoder-only network to learn a homogeneous structure"""

    particles_input = os.path.join(input_dir, name + "_particles.star")

    options = f"--checkpoint {parameters['heterogeneity_tomodrgn_train_checkpoint']} \
        --log-interval {parameters['heterogeneity_tomodrgn_train_log_interval']} \
        --num-workers 0"

    if parameters.get("heterogeneity_tomodrgn_train_weight") and os.path.exists(parameters["heterogeneity_tomodrgn_train_weight"]):
        options += f" --load {parameters['heterogeneity_tomodrgn_train_weight']}"

    if parameters.get("heterogeneity_tomodrgn_train_seed"):
        options += f" --seed {parameters['heterogeneity_tomodrgn_train_seed']}"
    
    if parameters.get("heterogeneity_tomodrgn_data_ind") and os.path.exists(parameters["heterogeneity_tomodrgn_data_ind"]):
        options += f" --ind {parameters['heterogeneity_tomodrgn_data_ind']}"

    if not parameters["heterogeneity_tomodrgn_data_invert"]:
        options += " --uninvert-data"

    if not parameters["heterogeneity_tomodrgn_data_windowing"]:
        options += " --no-window"
    else:
        options += f" --window-r {parameters['heterogeneity_tomodrgn_data_window_r']} --window-r-outer {parameters['heterogeneity_tomodrgn_data_window_r_outer']}"
    
    if parameters.get("heterogeneity_tomodrgn_data_dir"):
        data_dir = project_params.resolve_path(parameters["heterogeneity_tomodrgn_data_dir"])
        assert os.path.isdir(data_dir), "The data input directory dose not exist"
        options += f" --datadir {data_dir}"

    if parameters["heterogeneity_tomodrgn_data_lazy"]:
        options += " --lazy"

        if False and parameters['heterogeneity_tomodrgn_data_shufflersize'] > 0:
            options += f" --shuffler-size {parameters['heterogeneity_tomodrgn_data_shufflersize']}"

    if parameters["slurm_verbose"]:
        options += " -v"

    if False:
        options += " --preprocessed"
    
    # --pretrain {parameters['heterogeneity_tomodrgn_pretrain']} \
    # --emb-type {parameters['heterogeneity_tomodrgn_emd_type']} \
    # --pose-lr {parameters['heterogeneity_tomodrgn_pose_lr']} \
    # --enc-layers-A {parameters['heterogeneity_tomodrgn_enc_lya']} \
    # --enc-dim-A {parameters['heterogeneity_tomodrgn_enc_dima']} \
    # --out-dim-A {parameters['heterogeneity_tomodrgn_out_dima']} \
    # --enc-layers-B {parameters['heterogeneity_tomodrgn_enc_lyb']} \
    # --enc-dim-B {parameters['heterogeneity_tomodrgn_enc_dimb']} \

    training_parameters = f"-n {parameters['heterogeneity_tomodrgn_train_epochs']} -b {parameters['heterogeneity_tomodrgn_train_batch']} --wd {parameters['heterogeneity_tomodrgn_train_wd']} --lr {parameters['heterogeneity_tomodrgn_train_lr']} \
        --layers {parameters['heterogeneity_tomodrgn_layers']} --dim {parameters['heterogeneity_tomodrgn_dim']} --pe-dim {parameters['heterogeneity_tomodrgn_pe_type']} --dec-layers {parameters['heterogeneity_tomodrgn_dec_hl']} \
        --pe-type {parameters['heterogeneity_tomodrgn_pe_type']} --activation {parameters['heterogeneity_tomodrgn_activation']} --l-extent {parameters['heterogeneity_tomodrgn_l_ext']}"

    if parameters["heterogeneity_tomodrgn_layer_norm"]:
        training_parameters += " --layer-norm"

    if parameters.get('heterogeneity_tomodrgn_enc_mask'):
        training_parameters += f" --enc-mask {parameters['heterogeneity_tomodrgn_enc_mask']}"
    
    if "gaussian" in parameters['heterogeneity_tomodrgn_pe_type']:
        training_parameters += f" --feat-sigma {parameters['heterogeneity_tomodrgn_feat_sigma']}"

    if parameters.get('heterogeneity_tomodrgn_pe_dim'):
        training_parameters += f" --pe-dim {parameters['heterogeneity_tomodrgn_pe_dim']}"

    if parameters.get('heterogeneity_tomodrgn_train_beta'):
        training_parameters += f" --beta {parameters['heterogeneity_tomodrgn_train_beta']}"
    if parameters.get('heterogeneity_tomodrgn_train_beta_control'):
        training_parameters += f" --beta-control {parameters['heterogeneity_tomodrgn_train_beta_control']}"    

    if not "none" in parameters['heterogeneity_tomodrgn_data_norm']:
        training_parameters += f" --norm {parameters['heterogeneity_tomodrgn_data_norm']}"
    if not parameters['heterogeneity_tomodrgn_train_amp']:
        training_parameters += " --no-amp"
    
    # if parameters['heterogeneity_tomodrgn_pose_sgd']:
    #     training_parameters += " --do-pose-sgd"
    # --zdim {parameters['heterogeneity_tomodrgn_train_zdim']}
    
    tomo = ""
    if parameters["heterogeneity_tomodrgn_tilt_weight"]:
        tomo += " --recon-tilt-weight"
    elif parameters["heterogeneity_tomodrgn_dose_weight"]:
        tomo += " --recon-dose-weight"
    else:
        pass 

    if parameters["heterogeneity_tomodrgn_dose_mask"]:
        tomo += " --l-dose-mask"

    if parameters["heterogeneity_tomodrgn_dose"] > 0:
        tomo += f" --dose-override {parameters['heterogeneity_tomodrgn_dose']}"

    if parameters["heterogeneity_tomodrgn_sample_tilt"] > 0:
        tomo += f" --sample-ntilts {parameters['heterogeneity_tomodrgn_sample_tilt']}"
    
    if parameters["heterogeneity_tomodrgn_sequential_order"]:
        tomo += " --sequential-tilt-sampling"
    
    if parameters["heterogeneity_tomodrgn_use_firstn"] > 0:
        tomo += f" --use-first-ntilts {parameters['heterogeneity_tomodrgn_use_firstn']}"

    # TODO: multigpu option
    
    workers = ""
    if parameters["heterogeneity_tomodrgn_num_workers"] > 0:
        workers += f" --num-workers {parameters['heterogeneity_tomodrgn_num_workers']}"
    else:
        workers += f" --num-workers {parameters['slurm_tasks']}"

    if parameters["heterogeneity_tomodrgn_prefetch_factor"] > 0:
        workers += f" --prefetch-factor {parameters['heterogeneity_tomodrgn_prefetch_factor']}"

    if parameters["heterogeneity_tomodrgn_pst_wkrs"]:
        workers += " --persistent-workers"
    
    if parameters["heterogeneity_tomodrgn_pin_mem"]:
        workers += " --pin-memory"

    # --ctf {ctf_input} --pose {pose_input}

    command = f"{get_tomodrgn_path()} train_nn {particles_input} --datadir {input_dir} -o {output} {options} {training_parameters} {tomo} {workers}"

    local_run.run_shell_command(command, verbose=parameters['slurm_verbose'])


def convergence_nn(parameters, input_dir):
    
    """
    Assess convergence of a VAE model after 30 epochs training using internal / self-consistency heuristics
    """
    
    ref = parameters["heterogeneity_tomodrgn_ref"]

    option = ""
    if parameters["heterogeneity_tomodrgn_dc"]:
        option += " --include-dc"
    
    if not "none" in parameters["heterogeneity_tomodrgn_fscmask"]:
        option += f" --fsc-mask {parameters['heterogeneity_tomodrgn_fscmask']}"

    command = f"{get_tomodrgn_path()} convergence_nn {input_dir} {ref} --max-epoch {parameters['heterogeneity_tomodrgn_max_epoch']} {option}"

    local_run.run_shell_command(command, verbose=parameters['slurm_verbose'])


def convergence_vae(parameters, input_dir, output):

    """
    Assess convergence of a VAE model after 30 epochs training using internal / self-consistency heuristics
    """
    pixelsize = parameters['scope_pixel'] * parameters["data_bin"] * parameters["extract_bin"]

    options = f" --epoch-interval {parameters['heterogeneity_tomodrgn_epoch_interval']} --subset {parameters['heterogeneity_tomodrgn_subset']} --random-state {parameters['heterogeneity_tomodrgn_randomstate']} --n-bins {parameters['heterogeneity_tomodrgn_nbins']} --pruned-maxima {parameters['heterogeneity_tomodrgn_pruned_maxima']} --radius {parameters['heterogeneity_tomodrgn_radius']} --final-maxima {parameters['heterogeneity_tomodrgn_final_maxima']} --thresh {parameters['heterogeneity_tomodrgn_thresh']} --dilate {parameters['heterogeneity_tomodrgn_dilate']} --dist {parameters['heterogeneity_tomodrgn_dist']}"
    
    options += f" --smooth {parameters['heterogeneity_tomodrgn_smooth']} --smooth-width {parameters['heterogeneity_tomodrgn_smooth_width']}"

    if parameters["heterogeneity_tomodrgn_force_umapcpu"]:
        options += " --force-umap-cpu"
    if parameters['heterogeneity_tomodrgn_randomseed'] > 0:
        options += f" --random-seed {parameters['heterogeneity_tomodrgn_randomseed']}"  

    if parameters["heterogeneity_tomodrgn_skip_umap"]:
        options += " --skip-umap"

    if parameters['heterogeneity_tomodrgn_analysis_flip']:
        options += " --flip"
    if parameters['heterogeneity_tomodrgn_analysis_invert']:
        options += " --invert"

    if parameters.get('heterogeneity_tomodrgn_analysis_downsample'):
        options += f" -d {parameters['heterogeneity_tomodrgn_analysis_downsample']}"

    if parameters["heterogeneity_tomodrgn_skip_vgen"]:
        options += " --skip-volgen"
    
    if not "None" in parameters["heterogeneity_tomodrgn_gt"]:
        options += f" --ground-truth {parameters['heterogeneity_tomodrgn_gt']}"

    command = f"{get_tomodrgn_path()} convergence_vae {input_dir} --epoch {parameters['heterogeneity_tomodrgn_epoch_index']} -o {output} {options}"

    local_run.run_shell_command(command, verbose=parameters['slurm_verbose'])