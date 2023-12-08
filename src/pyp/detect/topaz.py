import datetime
import os
import shutil
import time
import numpy as np
from pathlib import Path

from pyp.system import local_run, project_params, utils, mpi
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.detect import joint

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def sprtrain(args):

    train_folder = os.path.join( os.getcwd(), "train" )
    with open( os.path.join( train_folder, "current_list.txt" ) ) as f:
        train_name = f.read()
    train_coords = os.path.join( train_folder, train_name + "_coordinates.txt" )

    # generate binned versions of images
    files = np.loadtxt( os.path.join( "train", train_name + "_images.txt"), comments="image_name", dtype="str", ndmin=2)[:,0]
    binning = args["detect_nn2d_bin"]

    number_of_labels = np.loadtxt( train_coords, dtype='str', comments="image_name", ndmin=2).shape[0]

    # setup local scratch area
    scratch_train = os.path.join( os.environ["PYP_SCRATCH"], "train" )
    os.makedirs(scratch_train)

    if binning > 1:

        logger.info(f"Binning coordinates ({number_of_labels} labels)")
        train_coords_bin = train_coords.replace(".txt", f"_bin{binning:02d}.txt")
        # substitute coordinate file with binned positions
        bin_next_coordinates(train_coords,train_coords_bin,binning,verbose=args["slurm_verbose"])
        train_coords = train_coords_bin

    logger.info(f"Binning {len(files)} micrographs")
    # bin images and save to local scratch
    arguments = []
    for f in files:
        filename = f.split()[0] + ".mrc"
        arguments.append(
                (
                    os.path.join( os.getcwd(), "mrc", filename ),
                    os.path.join( scratch_train, filename ),
                    binning,
                    args["slurm_verbose"]
                )
            )
    mpi.submit_function_to_workers(joint.bin_image, arguments, verbose=args["slurm_verbose"])


    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join( train_folder, time_stamp )
    os.makedirs( output_folder )

    # go to scratch directory
    os.chdir(scratch_train)

    # topaz train boolean parameters
    pretrained = "--pretrained" if args["detect_nn2d_topaz_pretrained"] else "--no-pretrained"
    batchnorm = "on" if args['detect_nn2d_topaz_bn'] else "off"

    logger.info(f"Training model")
    command = f"{utils.get_topaz_path()}/topaz train \
-n {args['detect_nn2d_num_particles']} \
--num-workers={args['slurm_tasks']} \
--train-images {scratch_train} \
--train-targets {train_coords} \
{pretrained} \
--method {args['detect_nn2d_topaz_train_method']}\
--num-epochs {args['detect_nn2d_topaz_epochs']} \
--radius {args['detect_nn2d_topaz_train_rad']} \
--slack {args['detect_nn2d_topaz_train_slack']} \
--autoencoder {args['detect_nn2d_topaz_train_autoencoder']} \
--l2 {args['detect_nn2d_topaz_train_reg']} \
--learning-rate {args['detect_nn2d_topaz_train_learn_rate']} \
--minibatch-size {args['detect_nn2d_topaz_train_batchsize']} \
--minibatch-balance {args['detect_nn2d_topaz_train_batchbalance']} \
--epoch-size {args['detect_nn2d_topaz_train_epochsize']} \
--save-prefix=topaz_train \
-o model_training.txt"
# --model {args['detect_nn2d_topaz_model']} \
# --units {args['detect_nn2d_topaz_units']} \
# --dropout {args['detect_nn2d_topaz_dropout']} \
# --bn {batchnorm} \
# --pooling {args['detect_nn2d_topaz_pooling']} \
# --unit-scaling {args['detect_nn2d_topaz_unit_scale']} \
# --ngf {args['detect_nn2d_topaz_network_unit_scale']} \
    local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # check for failure if not output was produced
    if len(list(Path(os.getcwd()).rglob('*.sav'))) == 0:
        raise Exception("Failed to run training module")

    # move trained models to project folder
    logger.info(f"Copying results to {output_folder}")
    for path in Path(os.getcwd()).rglob('*.sav'):
        new_name = path.name.replace(".sav", ".training")
        shutil.copy2(path, Path(output_folder) / f"{new_name}")
    shutil.copy2("model_training.txt", output_folder)


def spreval(args,name):

    binning = args["detect_nn2d_bin"]
    joint.bin_image(name + ".avg", name + "_bin.mrc", binning, args["slurm_verbose"])

    coordinates_file = f"{name}_predicted_particles_all_upsampled.txt"

    if 'detect_nn2d_ref' in args.keys() and os.path.isfile(project_params.resolve_path(args['detect_nn2d_ref'])):
        logger.info(f"Evaluating using model: {Path(project_params.resolve_path(args['detect_nn2d_ref'])).name}")
        model_arg = f"-m {project_params.resolve_path(args['detect_nn2d_ref'])}"
    else:
        logger.info(f"Evaluating using pre-trained model")
        model_arg = f"--model {args['detect_nn2d_topaz_pretrained_model']}"

    command = f"{utils.get_topaz_path()}/topaz extract \
-r {args['detect_nn2d_topaz_extract_rad']} \
{model_arg} \
--threshold {args['detect_nn2d_topaz_extract_thres']} \
--assignment-radius {args['detect_nn2d_topaz_extract_assign_rad']} \
--min-radius {args['detect_nn2d_topaz_extract_min_rad']} \
--max-radius {args['detect_nn2d_topaz_extract_max_rad']} \
--step-radius {args['detect_nn2d_topaz_extract_step_rad']} \
-x {binning} \
-o {coordinates_file} \
{Path().cwd() / f'{name}_bin.mrc'}"

    local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # use this to save intermediate files generated by NN particle picking
    if args["slurm_verbose"]:
        with open("project_folder.txt") as f:
            project_folder = f.read()
        logger.info("Now saving " + coordinates_file)
        logger.info("To " + os.path.join(project_folder, "train"))
        shutil.copy2( coordinates_file, os.path.join(project_folder, "train") )

    if not os.path.exists(coordinates_file):
        raise Exception("Failed to run inference module")

    with open(coordinates_file) as f:
        lines = len(f.readlines()) - 1

    if lines > 0:
        try:
            coordinates = np.loadtxt( coordinates_file, dtype=str, comments="image_name", ndmin=2)

            # display total number of positions
            logger.info(str(len(coordinates)) + " total positions")

            # threshold positions 
            boxes = coordinates.copy()[:,1:].astype('f')
            coordinates = boxes[ boxes[:,-1] > args["detect_thre"] ]
            logger.info(str(len(coordinates)) + " positions with confidence greater than " + str(args["detect_thre"]))

            if len(coordinates) > 0:
                return coordinates[:,:2].astype('i')
            else:
                logger.warning("No particles found")
                return np.array([])
        except:
            logger.warning("No particles found.")
            return np.array([])
    else:
        logger.warning("No particles found")
        return np.array([])

def bin_next_coordinates(coordinates,coordinates_bin, binning, verbose=False):
    command = f"{utils.get_topaz_path()}/topaz convert -s {binning} -o {coordinates_bin} {coordinates}"
    local_run.run_shell_command(command, verbose=verbose)


