import os, sys
import shutil
import numpy as np
from pathlib import Path
import json
import glob

from pyp.analysis import plot
from pyp import preprocess, merge
from pyp.inout.metadata import pyp_metadata
from pyp.inout.image import get_image_dimensions, mrc
from pyp.inout.image.core import generate_aligned_tiltseries
from pyp.system import local_run, project_params, mpi, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system.utils import get_imod_path, get_gpu_ids
from pyp.system.db_comm import load_tomo_results, load_config_files
from pyp.system.singularity import get_pyp_configuration

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def get_cryocare_path():
    cryocare_path = '/opt/conda/envs/cryocare'
    return f"export LD_LIBRARY_PATH={cryocare_path}/lib:/opt/conda/envs/isonet/lib:$LD_LIBRARY_PATH; micromamba run -n cryocare {cryocare_path}/bin/"

def cryocare_train(project_dir, output, parameters):
    """
    cryoCARE training
    Will take all the *half1.rec from mrc folder as list to train and run denoise
    """

    half1_list = glob.glob(os.path.join(project_dir, "train", "*half1.rec"))
    half2_list = [f.replace("half1.rec", "half2.rec") for f in half1_list]
    
    # train_data.json
    config = {
    "even": half1_list,
    "odd": half2_list,
    "tilt_axis": "Y",
    "path": f"./train_data"
    }

    config["patch_shape"] = [parameters["tomo_denoise_cryocare_patch"]] * 3
    config["num_slices"] = parameters["tomo_denoise_cryocare_slices"]
    config["split"] = parameters["tomo_denoise_cryocare_split"]
    config["n_normalization_samples"] = parameters["tomo_denoise_cryocare_samples"]

    if not "0" in parameters["tomo_denoise_masksize"]:
        mask_shape = [int(i) for i in parameters["tomo_denoise_masksize"].split(",")]
        train_mask = "./train_mask.mrc"
        mrc.generate_cuboid_mask(half1_list[0], mask_shape, outputname=train_mask)

        config["mask"] = train_mask

    data_config = "train_data_config.json"
    with open(data_config, 'w') as file:
        json.dump(config, file, indent=4)

    command = get_cryocare_path() + f"cryoCARE_extract_train_data.py --conf {data_config}"
    local_run.stream_shell_command(command,verbose=parameters["slurm_verbose"])

    # train.json
    train_config = {
    "train_data": "./train_data",
    "model_name": "cryocare_model",
    "path": "./train_model",
    "gpu_id": get_gpu_ids(parameters)
    }
    
    train_config["epochs"] = parameters["tomo_denoise_cryocare_epochs"]
    train_config["steps_per_epoch"] = parameters["tomo_denoise_cryocare_steps"]
    train_config["batch_size"] = parameters["tomo_denoise_cryocare_batchsize"]
    train_config["unet_kern_size"] = parameters["tomo_denoise_cryocare_kern"]
    train_config["unet_n_depth"] = parameters["tomo_denoise_cryocare_depth"]
    train_config["unet_n_first"] = parameters["tomo_denoise_cryocare_nfirst"]
    train_config["learning_rate"] = parameters["tomo_denoise_cryocare_lr"]
    
    train_config_file = "train_config.json"
    with open(train_config_file, 'w') as file:
        json.dump(train_config, file, indent=4)

    output = []
    def obs(line):
        output.append(line)

    command = f"{get_cryocare_path()}cryoCARE_train.py --conf {train_config_file}"
    local_run.stream_shell_command(command,observer=obs,verbose=parameters["slurm_verbose"])

    # parse output
    loss = [ line.split("loss:")[1].split()[0] for line in output if "ETA:" in line]
    mse = [ line.split("mse:")[1].split()[0] for line in output if "ETA:" in line]
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("dark")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[8, 6], sharex=True)

    ax[0].set_title("cryoCARE training loss")
    ax[0].plot(np.array(loss).astype('f'),".-",color="blue",label="Loss")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(np.array(mse).astype('f'),".-",color="red",label="Mean Squared Error")
    ax[1].set_ylabel("MSE")
    ax[1].set_xlabel("Step")
    ax[1].legend()
    plt.xlabel("Step")
    plt.savefig(os.path.join(project_dir,"train","training_loss.svgz"))
    plt.close()

    # save trained model to project folder
    shutil.copy2("./train_model/cryocare_model.tar.gz",os.path.join(project_dir,"train"))

def cryocare_predict(working_path, project_path, name, parameters):
    """
    cryoCARE evaluation
    Will take all the *half1.rec from mrc folder as list to train and run denoise
    """

    half1_list = [os.path.join(working_path, name + "_half1.rec")]
    half2_list = [os.path.join(working_path, name + "_half2.rec")]

    # TODO: create a list to run prediction

    # prediction.json
    predcit_config = {
    "path": project_params.resolve_path(parameters["tomo_denoise_cryocare_overwrite"]),
    "even": half1_list,
    "odd": half2_list,
    "n_tiles": [parameters["tomo_denoise_cryocare_tiles"]] * 3,
    "gpu_id": get_gpu_ids(parameters)
    }

    output_path = os.path.join( working_path, "denoised" )
    if os.path.exists(output_path):
        predcit_config["overwrite"] = True
    else:
        predcit_config["overwrite"] = parameters["tomo_denoise_cryocare_overwrite"]
    
    predcit_config["output"] = output_path
    
    predict_config_file = "predict_config.json"
    with open(predict_config_file, 'w') as file:
        json.dump(predcit_config, file, indent=4)

    if parameters["slurm_verbose"]:
        with open(predict_config_file) as file:
            logger.warning(file.read())

    command = f"{get_cryocare_path()}cryoCARE_predict.py --conf {predict_config_file}"
    local_run.stream_shell_command(command,verbose=parameters["slurm_verbose"])    

def cryocare(working_path, project_path, name, parameters):
    """
    cryoCARE training and prediction
    Will take all the *half1.rec from mrc folder as list to train and run denoise
    """

    # half1_list = glob.glob(os.path.join(project_path, "mrc", "*half1.rec"))
    # half2_list = [f.replace("half1", "half2") for f in half1_list]
    half1_list = [os.path.join(working_path, name + "_half1.rec")]
    half2_list = [os.path.join(working_path, name + "_half2.rec")]

    # train_data.json
    config = {
    "even": half1_list,
    "odd": half2_list,
    "tilt_axis": "Y",
    "path": f"./train_data"
    }

    config["patch_shape"] = [parameters["tomo_denoise_cryocare_patch"]] * 3
    config["num_slices"] = parameters["tomo_denoise_cryocare_slices"]
    config["split"] = parameters["tomo_denoise_cryocare_split"]
    config["n_normalization_samples"] = parameters["tomo_denoise_cryocare_samples"]

    if not "0" in parameters["tomo_denoise_masksize"]:
        mask_shape = [int(i) for i in parameters["tomo_denoise_masksize"].split(",")]
        train_mask = "./train_mask.mrc"
        mrc.generate_cuboid_mask(half1_list[0], mask_shape, outputname=train_mask)

        config["mask"] = train_mask

    data_config = "train_data_config.json"
    with open(data_config, 'w') as file:
        json.dump(config, file, indent=4)

    command = get_cryocare_path() + f"cryoCARE_extract_train_data.py --conf {data_config}"
    local_run.stream_shell_command(command,verbose=parameters["slurm_verbose"])

    # train.json
    train_config = {
    "train_data": "./train_data",
    "model_name": "cryocare_model",
    "path": "./train_model",
    "gpu_id": get_gpu_ids(parameters)
    }
    
    train_config["epochs"] = parameters["tomo_denoise_cryocare_epochs"]
    train_config["steps_per_epoch"] = parameters["tomo_denoise_cryocare_steps"]
    train_config["batch_size"] = parameters["tomo_denoise_cryocare_batchsize"]
    train_config["unet_kern_size"] = parameters["tomo_denoise_cryocare_kern"]
    train_config["unet_n_depth"] = parameters["tomo_denoise_cryocare_depth"]
    train_config["unet_n_first"] = parameters["tomo_denoise_cryocare_nfirst"]
    train_config["learning_rate"] = parameters["tomo_denoise_cryocare_lr"]
    
    train_config_file = "train_config.json"
    with open(train_config_file, 'w') as file:
        json.dump(train_config, file, indent=4)

    command = f"{get_cryocare_path()}cryoCARE_train.py --conf {train_config_file}"
    local_run.stream_shell_command(command,verbose=parameters["slurm_verbose"])

    # TODO: create a list to run prediction

    # prediction.json
    predcit_config = {
    "path": f"./train_model/cryocare_model.tar.gz",
    "even": half1_list,
    "odd": half2_list,
    "n_tiles": [parameters["tomo_denoise_cryocare_tiles"]] * 3,
    "gpu_id": get_gpu_ids(parameters)
    }

    output_path = os.path.join( working_path, "denoised" )
    if os.path.exists(output_path):
        predcit_config["overwrite"] = True
    else:
        predcit_config["overwrite"] = parameters["tomo_denoise_cryocare_overwrite"]
    
    predcit_config["output"] = output_path
    
    predict_config_file = "predict_config.json"
    with open(predict_config_file, 'w') as file:
        json.dump(predcit_config, file, indent=4)

    if parameters["slurm_verbose"]:
        with open(predict_config_file) as file:
            logger.warning(file.read())

    command = f"{get_cryocare_path()}cryoCARE_predict.py --conf {predict_config_file}"
    local_run.stream_shell_command(command,verbose=parameters["slurm_verbose"])    

def tomo_swarm_half( name, project_path, working_path, parameters):
    """
        Generate half tomograms for cryoCARE training
    """

    # use this to save intermediate files generated by NN particle picking
    with open("project_folder.txt", "w") as f:
        f.write(project_params.resolve_path(project_path))

    # retrieve available results
    if "data_set" in parameters:
        dataset = parameters["data_set"]
    elif "stream_session_name" in parameters:
        dataset = parameters["stream_session_name"]
    else:
        raise Exception("Unknown dataset or session name")
    
    load_config_files( dataset, project_path, working_path)
    load_tomo_results( name, parameters, project_path, working_path, verbose=parameters["slurm_verbose"])

    # unpack pkl file
    if os.path.exists(f"{name}.pkl"):
        metadata_object = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=False)
        metadata = metadata_object.data
        # convert metadata to files
        metadata_object.meta2PYP(path=working_path,data_path=os.path.join(project_path,"raw/"))
    else:
        raise Exception("Please run normal preprocessing first")

    # split movies
    frame_list = []      
    frame_list = metadata["frames"] if "frames" in metadata else frame_list

    if not len(frame_list) > 0:
        raise Exception("There are no movies in this dataset, cannot run cryoCARE")

    # copy over all the frames
    arguments = []
    for f in frame_list:
        arguments.append((str(project_path) + "/raw/" + f, f))
    mpi.submit_function_to_workers(shutil.copy2, arguments, verbose=parameters["slurm_verbose"])

    # convert eer files to mrc using movie_eer_reduce and movie_eer_frames parameters (flipping in x is required to match unblur/motioncorr convention)
    if frame_list[0].endswith(".eer"):
        full_frame = get_image_dimensions(frame_list[0])[2]
        valid_averages = np.floor(full_frame / parameters['movie_eer_frames'])
        eer = True
        arguments = []
        for f in frame_list:
            # average eer frames
            command = f"{get_imod_path()}/bin/clip flipx -es {parameters['movie_eer_reduce']-1} -ez {parameters['movie_eer_frames']} {f} {f.replace('.eer','.mrc')}; rm -f {f}"
            arguments.append(command)
        mpi.submit_jobs_to_workers(arguments, os.getcwd())

        raw_image = [ i.replace('.eer','.mrc') for i in frame_list ]
    else:
        raw_image = frame_list
        eer = False

    # convert tif movies to mrc files
    if ".tif" in raw_image[0]:
        logger.info("Converting tif to mrc")
        commands = [] 
        for f in raw_image:
            com = "{0}/bin/newstack -mode 2 {1} {2}".format(
                get_imod_path(), f, f.replace(".tiff", ".mrc").replace(".tif", ".mrc")
            )
            commands.append(com)
        mpi.submit_jobs_to_workers(commands, os.getcwd())
        
    raw_image = [f.replace(".tiff", ".mrc").replace(".tif", ".mrc") for f in raw_image]
 
    # get the dimension first
    dims = get_image_dimensions(raw_image[0])
    if eer:
        z_slices = int(valid_averages) - 1
    else:
        z_slices = dims[2] - 1

    even_list = np.arange(0, z_slices + 1, 2)
    odd_list = np.arange(1, z_slices + 1, 2)

    logger.info("Creating half movies")
    # half 1 movies
    arguments = []
    for i, f in enumerate(raw_image):
        output_half1 = f.replace(name, name + "_half1").replace(".mrc", "")

        drifts = metadata["drift"][i].to_numpy()
        half1_drift = drifts[even_list]
        np.savetxt(output_half1 + ".xf", half1_drift, fmt="%s", delimiter='\t')
        # read half slices 
        command = f"{get_imod_path()}/bin/newstack -input {f} -secs {','.join(map(str, even_list))} -output {output_half1}.mrc"
        arguments.append(command)
    mpi.submit_jobs_to_workers(arguments, os.getcwd())

    # half 2 movies
    arguments = []
    for i, f in enumerate(raw_image):
        output_half2 = f.replace(name, name + "_half2").replace(".mrc", "")

        drifts = metadata["drift"][i].to_numpy()
        half2_drift = drifts[odd_list]
        np.savetxt(output_half2 + ".xf", half2_drift, fmt="%s", delimiter='\t')
        # read half slices and remove the original movie
        command = f"{get_imod_path()}/bin/newstack -input {f} -secs {','.join(map(str, odd_list))} -output {output_half2}.mrc && rm -f {f}"
        arguments.append(command)
    mpi.submit_jobs_to_workers(arguments, os.getcwd())

    # generate half tomograms from half movies    
    for i in [1, 2]:
        logger.info(f"Processing half{i} tomogram ...")
        newname = name + f"_half{i}"
        new_filelist = [file.replace(name, newname) for file in raw_image]
        # copy the tilt aignment and angle files
        shutil.copy2(f"{name}.xf", newname + ".xf")
        shutil.copy2(f"{name}.tlt", newname + ".tlt")
        shutil.copy2(f"{name}.order", newname + ".order")

        # generate averages using existing xf
        preprocess.regenerate_average_quick(
            newname,
            parameters,
            dims,
            new_filelist,
        )

        os.symlink(newname + ".mrc", newname + ".st")

        # actual stack sizes
        headers = mrc.readHeaderFromFile(newname + ".mrc")
        x = int(headers["nx"])
        y = int(headers["ny"])

        # binned reconstruction
        binning = parameters["tomo_rec_binning"]
        zfact = ""

        # tilt-series alignment
        if project_params.tiltseries_align_is_done(metadata):
            logger.info("Using existing tilt-series alignments")
        else:
            raise Exception("run tilt-series alignemnt with full movie frames first")

        # regenerate aligned tilt-series
        generate_aligned_tiltseries(newname, parameters, x, y)

        # Refined tilt angles
        tltfile = f"{newname}.tlt"
        tilt_angles = np.loadtxt(tltfile) if os.path.exists(tltfile) else metadata["tlt"].to_numpy()

        exclude_views = merge.do_exclude_views(newname, tilt_angles)

        # Reconstruction options
        tilt_options = merge.get_tilt_options(parameters,exclude_views)

        # produce binned tomograms
        # erase fiducials if needed
        if parameters["tomo_ali_method"] == "imod_gold" and parameters["tomo_rec_erase_fiducials"]:
            preprocess.erase_gold_beads(newname, parameters, tilt_options, binning, zfact, x, y)
        else:
            merge.reconstruct_tomo(parameters, newname, x, y, binning, zfact, tilt_options, force=True)

        # save the half tomograms to the project folder 
        # shutil.copy2(newname + ".rec", os.path.join(project_path, "mrc", newname + ".rec"))
    
    # run cryoCARE 
    cryocare("./", project_path, name, parameters)
    
    # figure out output file name
    import glob
    output = glob.glob( "denoised/*.*" )[0]
    shutil.move( output, Path(output).name )
    output = Path(output).name
    
    return output

def tomo_swarm_halves( name, project_path, working_path, parameters):
    """
        Generate half tomograms for cryoCARE training
    """
    current_dir = os.getcwd()
    
    # process each tilt-series on its own folder
    os.makedirs(name,exist_ok=True)
    os.chdir(name)
    
    # retrieve available results
    if "data_set" in parameters:
        dataset = parameters["data_set"]
    elif "stream_session_name" in parameters:
        dataset = parameters["stream_session_name"]
    else:
        raise Exception("Unknown dataset or session name")
    
    load_config_files( dataset, project_path, working_path / name)
    load_tomo_results( name, parameters, project_path, working_path / name, verbose=parameters["slurm_verbose"])

    # unpack pkl file
    if os.path.exists(f"{name}.pkl"):
        metadata_object = pyp_metadata.LocalMetadata(f"{name}.pkl", is_spr=False)
        metadata = metadata_object.data
        # convert metadata to files
        metadata_object.meta2PYP(path=working_path / name,data_path=os.path.join(project_path,"raw/"))
    else:
        raise Exception("Please run normal preprocessing first")

    assert "drift" in metadata, "No drift metadata found?"

    # split movies
    frame_list = []      
    frame_list = metadata["frames"] if "frames" in metadata else frame_list

    if not len(frame_list) > 0:
        raise Exception("There are no movies in this dataset, cannot run cryoCARE")

    # copy over all the frames
    arguments = []
    for f in frame_list:
        arguments.append((str(project_path) + "/raw/" + f, f))
    mpi.submit_function_to_workers(shutil.copy2, arguments, verbose=parameters["slurm_verbose"])

    # convert eer files to mrc using movie_eer_reduce and movie_eer_frames parameters (flipping in x is required to match unblur/motioncorr convention)
    if frame_list[0].endswith(".eer"):
        full_frame = get_image_dimensions(frame_list[0])[2]
        valid_averages = np.floor(full_frame / parameters['movie_eer_frames'])
        eer = True
        arguments = []
        for f in frame_list:
            # average eer frames
            command = f"{get_imod_path()}/bin/clip flipx -es {parameters['movie_eer_reduce']-1} -ez {parameters['movie_eer_frames']} {f} {f.replace('.eer','.mrc')}; rm -f {f}"
            arguments.append(command)
        mpi.submit_jobs_to_workers(arguments, os.getcwd())

        raw_image = [ i.replace('.eer','.mrc') for i in frame_list ]
    else:
        raw_image = frame_list
        eer = False

    # convert tif movies to mrc files
    if ".tif" in raw_image[0]:
        commands = [] 
        for f in raw_image:
            com = "{0}/bin/newstack -mode 2 {1} {2}; rm -f {1}".format(
                get_imod_path(), f, f.replace(".tiff", ".mrc").replace(".tif", ".mrc")
            )
            commands.append(com)
        mpi.submit_jobs_to_workers(commands, os.getcwd())
        
    raw_image = [f.replace(".tiff", ".mrc").replace(".tif", ".mrc") for f in raw_image]
 
    # get the dimension first
    dims = get_image_dimensions(raw_image[0])
    if eer:
        z_slices = int(valid_averages) - 1
    else:
        z_slices = dims[2] - 1

    even_list = np.arange(0, z_slices + 1, 2)
    odd_list = np.arange(1, z_slices + 1, 2)

    # half 1 movies
    arguments = []
    for i, f in enumerate(raw_image):
        output_half1 = f.replace(name, name + "_half1").replace(".mrc", "")

        drifts = metadata["drift"][i].to_numpy()
        half1_drift = drifts[even_list]
        np.savetxt(output_half1 + ".xf", half1_drift, fmt="%s", delimiter='\t')
        # read half slices 
        command = f"{get_imod_path()}/bin/newstack -input {f} -secs {','.join(map(str, even_list))} -output {output_half1}.mrc"
        arguments.append(command)
    mpi.submit_jobs_to_workers(arguments, os.getcwd())

    # half 2 movies
    arguments = []
    for i, f in enumerate(raw_image):
        output_half2 = f.replace(name, name + "_half2").replace(".mrc", "")

        drifts = metadata["drift"][i].to_numpy()
        half2_drift = drifts[odd_list]
        np.savetxt(output_half2 + ".xf", half2_drift, fmt="%s", delimiter='\t')
        # read half slices and remove the original movie
        command = f"{get_imod_path()}/bin/newstack -input {f} -secs {','.join(map(str, odd_list))} -output {output_half2}.mrc && rm -f {f}"
        arguments.append(command)
    mpi.submit_jobs_to_workers(arguments, os.getcwd())

    # generate half tomograms from half movies    
    for i in [1, 2]:
        newname = name + f"_half{i}"
        new_filelist = [file.replace(name, newname) for file in raw_image]
        # copy the tilt aignment and angle files
        shutil.copy2(f"{name}.xf", newname + ".xf")
        shutil.copy2(f"{name}.tlt", newname + ".tlt")
        shutil.copy2(f"{name}.order", newname + ".order")

        # generate averages using existing xf
        preprocess.regenerate_average_quick(
            newname,
            parameters,
            dims,
            new_filelist,
        )

        os.symlink(newname + ".mrc", newname + ".st")

        # actual stack sizes
        headers = mrc.readHeaderFromFile(newname + ".mrc")
        x = int(headers["nx"])
        y = int(headers["ny"])

        # binned reconstruction
        binning = parameters["tomo_rec_binning"]
        zfact = ""

        # tilt-series alignment
        if not project_params.tiltseries_align_is_done(metadata):
            raise Exception("run tilt-series alignemnt with full movie frames first")

        # regenerate aligned tilt-series
        generate_aligned_tiltseries(newname, parameters, x, y)

        # Refined tilt angles
        tltfile = f"{newname}.tlt"
        tilt_angles = np.loadtxt(tltfile) if os.path.exists(tltfile) else metadata["tlt"].to_numpy()

        exclude_views = merge.do_exclude_views(newname, tilt_angles)

        # Reconstruction options
        tilt_options = merge.get_tilt_options(parameters,exclude_views)

        # produce binned tomograms
        if parameters["tomo_ali_method"] == "imod_gold" and parameters["tomo_rec_erase_fiducials"]:
            # erase fiducials if needed
            preprocess.erase_gold_beads(newname, parameters, tilt_options, binning, zfact, x, y)
        else:
            merge.reconstruct_tomo(parameters, newname, x, y, binning, zfact, tilt_options, force=True)

        # save the half tomogram to the train/ folder 
        shutil.move(newname + ".rec", os.path.join(project_path, "train", newname + ".rec"))
        
    # go up one level and cleanup
    os.chdir(current_dir)
    shutil.rmtree(name)