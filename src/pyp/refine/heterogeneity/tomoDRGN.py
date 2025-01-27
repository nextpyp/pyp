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
    command_base = f"micromamba run -n tomodrgn /opt/conda/envs/tomodrgn/bin/tomodrgn"
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

    command = f"{get_tomodrgn_path()} train_vae {particles_input} --datadir {input_dir} --source-software nextpyp --zdim {parameters['heterogeneity_tomodrgn_train_zdim']} -o {output} {options} {training_parameters} {tomo} {workers}"

    output = []
    def obs(line):
        output.append(line)
        if '# =====> Epoch:' in line:
            epoch = int(line.split('# =====> Epoch:')[-1].split('Average gen loss')[0])
            # TODO: do something useful here one day

    local_run.stream_shell_command(command, observer=obs, verbose=parameters['slurm_verbose'])

def tomodrgn_analyze(parameters,input_dir, output):
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
        options += f" --downsample {parameters['heterogeneity_tomodrgn_analysis_downsample']}"
    
    original_pixelsize = parameters['scope_pixel'] * parameters["data_bin"] * parameters["extract_bin"]

    if parameters.get('heterogeneity_tomodrgn_analysis_downsample'):
        
        output_pixel = original_pixelsize * (parameters["extract_box"] / parameters["heterogeneity_tomodrgn_analysis_downsample"])
    else:
        output_pixel = original_pixelsize

    if parameters['heterogeneity_tomodrgn_analysis_epoch'] == 0 or parameters['heterogeneity_tomodrgn_analysis_epoch'] >= parameters['heterogeneity_tomodrgn_train_epochs']:
        parameters['heterogeneity_tomodrgn_analysis_epoch'] = parameters['heterogeneity_tomodrgn_train_epochs'] - 1

    command = f"{get_tomodrgn_path()} analyze {input_dir} --epoch {parameters['heterogeneity_tomodrgn_analysis_epoch']} -o {output} --pc {parameters['heterogeneity_tomodrgn_analysis_pc']} --ksample {parameters['heterogeneity_tomodrgn_analysis_ksample']} {options} --plot-format svgz"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])


# TODO - only run training tasks
def run_tomodrgn_train(project_dir, parameters):

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

    logger.info(f"Copying {len(particles_stacks):,} particle stack(s) to local scratch:")
    mpi.submit_jobs_to_workers(tasks, working_path=os.getcwd(), verbose=False)
        
    # particle_stack_list = [os.path.basename(p) for p in particles_stacks]

    # do preprocessing in input_data folder
    # os.chdir("input_data")
    
    # boxsize = parameters['extract_box']
    # downsample_size = parameters['heterogeneity_tomodrgn_downsample_size']

    # downsampled = tomodrgn_preprocess(starfile, particle_stack_list, name, boxsize, downsample_size)

    # os.chdir("../")

    train_folder = Path(project_dir) / "train"
    train_folder.mkdir(parents=True, exist_ok=True)

    # train
    if parameters['heterogeneity_tomodrgn_mode'] == "backproject_voxel":

        backproject_voxel(parameters, "input_data", name, train_folder / f"{name}.mrc")
        
    elif parameters['heterogeneity_tomodrgn_mode'] == "train_nn":

        train_nn(parameters, "input_data", name, train_folder)
        
        # convergence
        logger.info("Running tomoDRGN convergence_nn")

        convergence_folder = Path(project_dir) / "train" / f"convergence_{parameters['heterogeneity_tomodrgn_train_epochs']}"
        convergence_folder.mkdir(parents=True, exist_ok=True)

        local_convergence_folder = Path(working_path) / "convergence"
        local_convergence_folder.mkdir(parents=True, exist_ok=True)
        convergence_nn(parameters, train_folder)
        logger.info(f"convergence_nn finished successfully, results saved to {convergence_folder}")

        shutil.copytree(local_convergence_folder, convergence_folder, dirs_exist_ok=True)

    else:
        tomodrgn_train(parameters, "input_data", name, train_folder)

        if (train_folder / "weights.pkl").exists() and (train_folder / "z.train.pkl").exists():
            logger.info(f"Training finished successfully, results saved to {train_folder}")

            shutil.copytree(train_folder, (working_path / "train_output"), dirs_exist_ok=True)
        else:
            raise Exception("Training did not finish successfully")

        # convergence
        logger.info("Running tomoDRGN convergence_vae")

        convergence_folder = Path(project_dir) / "train" / f"convergence_{parameters['heterogeneity_tomodrgn_train_epochs']}"
        convergence_folder.mkdir(parents=True, exist_ok=True)

        local_convergence_folder = Path(working_path) / "convergence"
        local_convergence_folder.mkdir(parents=True, exist_ok=True)
        convergence_vae(parameters, train_folder, local_convergence_folder)
        logger.info(f"convergence_vae finished successfully, results saved to {convergence_folder}")

        shutil.copytree(local_convergence_folder, convergence_folder, dirs_exist_ok=True)

# TODO - only run evaluation tasks
def run_tomodrgn_eval(project_dir, parameters):

    # scratch space
    working_path = Path(os.environ["PYP_SCRATCH"]) / "tomodrgn"

    logger.info(f"Working path: {working_path}")

    working_path.mkdir(parents=True, exist_ok=True)
    os.chdir(working_path)

    parent_parameters = project_params.load_pyp_parameters(project_params.resolve_path(parameters["data_parent"]))
    input_path = os.path.join(
        project_params.resolve_path(parent_parameters["data_parent"]), "relion", "stacks",
    )
    name = parameters['data_set']
    starfile = name + "_particles.star"

    input_star = Path(project_params.resolve_path(parent_parameters["heterogeneity_input_star"]))
    assert input_star.exists(), "Can not find input star file, run the extract stacks first."

    # this is the input star and particles stacks folder
    
    (working_path / "input_data").mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_star, working_path / "input_data" / starfile)

    particles_stacks = glob.glob(os.path.join(input_path, "*_stack.mrcs"))

    tasks = []
    for stack in particles_stacks:
        
        command = f"cp {stack} {os.getcwd()}/input_data/"
        tasks.append(command)

    logger.info(f"Copying {len(particles_stacks):,} particle stack(s) to local scratch:")
    mpi.submit_jobs_to_workers(tasks, os.getcwd(), verbose=False)
        
    # particle_stack_list = [os.path.basename(p) for p in particles_stacks]

    # do preprocessing in input_data folder
    # os.chdir("input_data")
    
    # boxsize = parameters['extract_box']
    # downsample_size = parameters['heterogeneity_tomodrgn_downsample_size']

    # downsampled = tomodrgn_preprocess(starfile, particle_stack_list, name, boxsize, downsample_size)

    # os.chdir("../")

    (working_path / "train_output").mkdir(parents=True, exist_ok=True)
    
    preprocessed = glob.glob(os.path.join(project_params.resolve_path(parameters["data_parent"]), "train", "*_preprocessed.star"))[0]
    
    with open(os.path.join(project_params.resolve_path(parameters["data_parent"]), "train","run.log")) as f:
        run_log = f.readlines()
    for line in run_log:
        if "Loading dataset from" in line:
            old_working_path = Path(line.split()[-1]).parents[1]
            break
    (old_working_path / "train_output").mkdir(parents=True, exist_ok=True)
    shutil.copy2(preprocessed, old_working_path / "train_output")
    os.symlink(working_path/"input_data", old_working_path/"input_data")

    drgn_path = Path(os.path.join(project_params.resolve_path(parameters["data_parent"])), "train")

    # analyze
    saved_folder = Path(project_dir) / "train"
    tomodrgn_analyze(parameters, drgn_path, saved_folder)

    if not saved_folder.exists():
        raise Exception("tomodrgn analyze failed")

    # eval_vol
    if parameters["heterogeneity_tomodrgn_eval_vol"]:
        logger.info("Running tomoDRGN eval_vol")
        final_output = os.path.join(project_dir, "train", "eval_vols")
        Path(final_output).mkdir(parents=True, exist_ok=True)
        tomodrgn_eval_vol(parameters, final_output, parent=os.path.join(project_params.resolve_path(parameters["data_parent"]), "train"))
        
        if parameters["heterogeneity_tomodrgn_eval_vol_analyze"]:
            final_output_analysis = os.path.join(project_dir, "train", "all_vols_analysis")
            tomodrgn_analyze_volumes(
                parameters=parameters, 
                output_dir=final_output_analysis, 
                vol_dir=final_output, 
                parent=os.path.join(project_params.resolve_path(parameters["data_parent"]),"train")
            )

def backproject_voxel(parameters, input_dir, name, output):

    """
    usage: backproject_voxel [-h] --output OUTPUT [--plot-format {png,svgz}]
                         [--source-software {auto,warp,cryosrpnt,nextpyp,cistem,warptools,relion}]
                         [--ind-ptcls PKL] [--ind-imgs IND_IMGS]
                         [--sort-ptcl-imgs {unsorted,dose_ascending,random}]
                         [--use-first-ntilts USE_FIRST_NTILTS]
                         [--use-first-nptcls USE_FIRST_NPTCLS]
                         [--uninvert-data] [--datadir DATADIR] [--lazy]
                         [--recon-tilt-weight] [--recon-dose-weight]
                         [--lowpass LOWPASS] [--flip]
                         particles
    """
    particles_input = os.path.join(input_dir, name + "_particles.star")

    options = f"--use-first-nptcls {parameters['heterogeneity_tomodrgn_use_first_nptcls']} --sort-ptcl-imgs {parameters['heterogeneity_tomodrgn_sort_ptcl_imgs']}"

    if parameters["heterogeneity_tomodrgn_lowpass"] > 0:
        options += f" --lowpass {parameters['heterogeneity_tomodrgn_lowpass']}"

    if parameters["heterogeneity_tomodrgn_data_lazy"]:
        options += " --lazy"
        
    if parameters["heterogeneity_tomodrgn_flip"]:
        options += " --flip"

    tomo = ""
    if parameters["heterogeneity_tomodrgn_tilt_weight"]:
        tomo += " --recon-tilt-weight"
    elif parameters["heterogeneity_tomodrgn_dose_weight"]:
        tomo += " --recon-dose-weight"
    else:
        pass 

    if parameters["heterogeneity_tomodrgn_use_firstn"] > 0:
        tomo += f" --use-first-ntilts {parameters['heterogeneity_tomodrgn_use_firstn']}"

    command = f"{get_tomodrgn_path()} backproject_voxel {particles_input} --datadir {input_dir} --source-software nextpyp --output {output} {options} {tomo}"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])

def train_nn(parameters, input_dir, name, output):

    """Train a decoder-only network to learn a homogeneous structure"""

    particles_input = os.path.join(input_dir, name + "_particles.star")

    options = f"--checkpoint {parameters['heterogeneity_tomodrgn_train_checkpoint']} --log-interval {parameters['heterogeneity_tomodrgn_train_log_interval']} --num-workers 0"

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
        options += " --verbose"

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

    training_parameters = f"-n {parameters['heterogeneity_tomodrgn_train_epochs']} -b {parameters['heterogeneity_tomodrgn_train_batch']} --wd {parameters['heterogeneity_tomodrgn_train_wd']} --lr {parameters['heterogeneity_tomodrgn_train_lr']} --layers {parameters['heterogeneity_tomodrgn_layers']} --dim {parameters['heterogeneity_tomodrgn_dim']} --pe-type {parameters['heterogeneity_tomodrgn_pe_type']} --activation {parameters['heterogeneity_tomodrgn_activation']} --l-extent {parameters['heterogeneity_tomodrgn_l_ext']}"

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

    """
    usage: train_nn [-h] --outdir OUTDIR [--load WEIGHTS.PKL]
                    [--checkpoint CHECKPOINT] [--log-interval LOG_INTERVAL]
                    [--verbose] [--seed SEED] [--plot-format {png,svgz}]
                    [--source-software {auto,warp,cryosrpnt,nextpyp,cistem,warptools,relion}]
                    [--ind-ptcls PKL] [--ind-imgs IND_IMGS]
                    [--sort-ptcl-imgs {unsorted,dose_ascending,random}]
                    [--use-first-ntilts USE_FIRST_NTILTS]
                    [--use-first-nptcls USE_FIRST_NPTCLS] [--uninvert-data]
                    [--no-window] [--window-r WINDOW_R]
                    [--window-r-outer WINDOW_R_OUTER] [--datadir DATADIR] [--lazy]
                    [--sequential-tilt-sampling] [--recon-tilt-weight]
                    [--recon-dose-weight] [--l-dose-mask] [-n NUM_EPOCHS]
                    [-b BATCH_SIZE] [--wd WD] [--lr LR] [--norm NORM NORM]
                    [--no-amp] [--multigpu] [--layers LAYERS] [--dim DIM]
                    [--l-extent L_EXTENT]
                    [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}]
                    [--pe-dim PE_DIM] [--activation {relu,leaky_relu}]
                    [--feat-sigma FEAT_SIGMA] [--num-workers NUM_WORKERS]
                    [--prefetch-factor PREFETCH_FACTOR] [--persistent-workers]
                    [--pin-memory]
                    particles
    """
    
    command = f"{get_tomodrgn_path()} train_nn {particles_input} --datadir {input_dir} --source-software nextpyp --outdir {output} {options} {training_parameters} {tomo} {workers}"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])


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

    """
    usage: convergence_nn [-h] [--max-epoch MAX_EPOCH] [--include-dc]
                        [--fsc-mask {none,sphere,tight,soft}]
                        [--plot-format {png,svgz}]
                        training_directory reference_volume
    """
    
    command = f"{get_tomodrgn_path()} convergence_nn {input_dir} {ref} --max-epoch {parameters['heterogeneity_tomodrgn_max_epoch']} {option} --plot-format svgz"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])


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

    command = f"{get_tomodrgn_path()} convergence_vae {input_dir} --epoch {parameters['heterogeneity_tomodrgn_epoch_index']} -o {output} {options} --plot-format svgz"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])
    
def tomodrgn_eval_vol(parameters, output_dir, parent):
    
    """
    Generate volumes from corresponding latent embeddings using a pretrained train_vae model (i.e. evaluating decoder module only)
    
    usage: eval_vol [-h] -w WEIGHTS -c CONFIG -o OUTDIR [--prefix PREFIX]
                [--zfile ZFILE] [--flip] [--invert] [--downsample DOWNSAMPLE]
                [--lowpass LOWPASS] [-b BATCH_SIZE] [--no-amp] [--multigpu]
    """
    batch_size = parameters['heterogeneity_tomodrgn_analysis_batch']

    options = ""
    if parameters.get('heterogeneity_tomodrgn_analysis_downsample'):
        options += f" --downsample {parameters['heterogeneity_tomodrgn_analysis_downsample']}"

    if parameters["heterogeneity_tomodrgn_analysis_epoch"] == -1:
        epoch = parameters['heterogeneity_tomodrgn_train_epochs'] - 1
    else:
        epoch = parameters['heterogeneity_tomodrgn_analysis_epoch'] - 1
    
    command = f"{get_tomodrgn_path()} eval_vol -o {output_dir} --weights {parent}/weights.{epoch}.pkl --config {parent}/config.pkl -b {batch_size} {options} --zfile {parent}/z.{epoch}.train.pkl"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])

def tomodrgn_analyze_volumes(parameters, output_dir, vol_dir, parent):
    
    """
    Dimensionality reduction and clustering of a volume ensemble.
    
    usage: analyze_volumes [-h] --voldir VOLDIR --config CONFIG [--outdir OUTDIR]
                        [--num-pcs NUM_PCS] [--ksample KSAMPLE]
                        [--plot-format {png,svgz}] [--mask-path MASK_PATH]
                        [--mask {none,sphere,tight,soft}] [--thresh THRESH]
                        [--dilate DILATE] [--dist DIST]
    """
    options = f"--ksample {parameters['heterogeneity_tomodrgn_analysis_ksample']} --mask {parameters['heterogeneity_tomodrgn_eval_vol_mask']}"
 
    """
    tomodrgn analyze_volumes \
    --voldir 03_heterogeneity-1_train_vae/all_vols \
    --config 03_heterogeneity-1_train_vae/config.pkl \
    --outdir 03_heterogeneity-1_train_vae/all_vols_analysis \
    --ksample 100 \
    --mask soft
    """
    command = f"{get_tomodrgn_path()} analyze_volumes --outdir {output_dir} --config {parent}/config.pkl --voldir {vol_dir} {options}"

    local_run.stream_shell_command(command, verbose=parameters['slurm_verbose'])