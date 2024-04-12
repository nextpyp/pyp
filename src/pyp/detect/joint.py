import datetime
import math
import os
import shutil
import socket
import glob
import time
import numpy as np
from pathlib import Path

from pyp.inout.metadata import pyp_metadata
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.utils.timer import Timer
from pyp.system import local_run, mpi
from pyp.system.utils import get_imod_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def bin_image( input, output, binning, verbose ):
    command = f"{get_imod_path()}/bin/newstack '{input}' '{output}' -bin {binning}; rm -f '{output}~'"
    local_run.run_shell_command(command, verbose=verbose)

def sprtrain(args):

    train_folder = os.path.join( os.getcwd(), "train" )
    with open( os.path.join( train_folder, "current_list.txt" ) ) as f:
        train_name = f.read()
    train_images = os.path.join( train_folder, train_name + "_images.txt" )
    train_coords = os.path.join( train_folder, train_name + "_coordinates.txt" )
    validation_images = train_images
    validation_coords = train_coords

    # Directory in which the output directory is generated
    runs_dir = "train"

    # generate binned versions of images
    files = np.loadtxt( os.path.join( "train", train_name + "_images.txt"), comments="image_name", dtype="str", ndmin=2)[:,0]
    binning = args["detect_nn2d_bin"]

    number_of_labels = np.loadtxt( train_coords, dtype='str', comments="image_name", ndmin=2).shape[0]
    logger.info(f"Binning coordinates ({number_of_labels} labels)")
    # substitute coordinate file with binned positions
    bin_next_coordinates(train_coords,binning)

    # setup local scratch area
    scratch_train = os.path.join( os.environ["PYP_SCRATCH"], "train" )
    os.makedirs(os.path.join(scratch_train,"log"),exist_ok=True)

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
    mpi.submit_function_to_workers(bin_image, arguments, verbose=args["slurm_verbose"])

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join( train_folder, time_stamp )
    os.makedirs(output_folder, exist_ok=True)

    # go to scratch directory
    os.chdir(scratch_train)

    logger.info(f"Training pyp model")
    command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/spr_pick; python {os.environ['PYP_DIR']}/external/spr_pick/spr_pick/__main__.py train start --algorithm {args['detect_nn2d_algorithm']} --noise_value {args['detect_nn2d_noise_value']} --noise_style {args['detect_nn2d_noise_style']} --tau {args['detect_nn2d_tau']} --runs_dir '{runs_dir}' --train_dataset '{train_images}' --train_label '{train_coords}' --iterations {args['detect_nn2d_iterations']} --alpha {args['detect_nn2d_alpha']} --train_batch_size {args['detect_nn2d_batch_size']} --nms {args['detect_dist']} --num {args['detect_nn2d_num']} --bb {args['detect_nn2d_bb']} --patch_size {args['detect_nn2d_patch_size']} --validation_dataset '{validation_images}' --validation_label '{validation_coords}' 2>&1 | tee {os.path.join(os.getcwd(), 'log', time_stamp + '_spr_pick_train.log')}"
    local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # check for failure if not output was produced
    if len(list(Path(os.getcwd()).rglob('*.training'))) == 0:
        raise Exception("Failed to run training module. Try increasing the memory per task")

    # move trained models to project folder
    logger.info(f"Copying results to {output_folder}")
    for path in Path(os.getcwd()).rglob('*.training'):
        shutil.copy2( path, output_folder )

    if args.get("detect_nn2d_debug"):
        debug_folder = os.path.join( output_folder, "debug" )
        os.makedirs( debug_folder )
        logger.info(f"Saving intermediate results to {debug_folder}")
        for path in Path(os.getcwd()).rglob('*.png'):
            shutil.copy2( path, debug_folder )

def spreval(args,name):

    # bin data by 8
    bin_image( name + ".avg", name + "_bin.mrc", 8, args["slurm_verbose"] )

    imgs_file = "images.txt"
    with open( imgs_file, 'w' ) as f:
        f.write("image_name\tpath\n")
        f.write( name + "_bin\t" + os.path.join( os.getcwd(), name + "_bin.mrc") )

    if 'detect_nn2d_ref' in args.keys() and os.path.exists( project_params.resolve_path(args['detect_nn2d_ref']) ):
        logger.info(f"Evaluating using model: {Path(project_params.resolve_path(args['detect_nn2d_ref'])).name}")
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/spr_pick; python {os.environ['PYP_DIR']}/external/spr_pick/spr_pick/__main__.py eval --model '{project_params.resolve_path(args['detect_nn2d_ref'])}' --dataset '{os.path.join( os.getcwd(), imgs_file)}' --runs_dir '{os.getcwd()}' --num 1"
        local_run.run_shell_command(command, verbose=args['slurm_verbose'])
        results_folder = glob.glob("./*/")[0]

        # use this to save intermediate files generated by NN particle picking
        if args.get("detect_nn2d_debug"):
            with open("project_folder.txt") as f:
                project_folder = f.read()
            for f in glob.glob( results_folder + "eval_imgs/*" ):
                if args["slurm_verbose"]:
                    logger.info("Now saving " + Path(f).name + " to train/ folder")
                shutil.copy2( f, os.path.join( project_folder, "train") )

        # convert coordinates to boxx
        coordinates_file = glob.glob(results_folder+"eval_imgs/*.txt")[0]

        if not os.path.exists(coordinates_file):
            raise Exception("Failed to run inference module")

        with open(coordinates_file) as f:
            lines = len(f.readlines()) - 1

        if lines > 0:
            try:
                coordinates = np.loadtxt( coordinates_file, dtype=str, comments="image_name", ndmin=2)

                # display total number of positions
                logger.info(str(len(coordinates)) + " candidate positions")

                # threshold positions using mean of score distribution
                boxes = coordinates.copy()[:,1:].astype('f')
                # mean = boxes[:,-1].mean()
                # boxes[:,-1] = ( boxes[:,-1] - mean )
                coordinates = boxes[ boxes[:,-1] > args["detect_nn2d_thresh"] ]
                logger.info(str(len(coordinates)) + " positions with confidence greater than " + str(args["detect_nn2d_thresh"]))

                if len(coordinates) > 0:
                    return coordinates[:,:2].astype('i') * 8
                else:
                    logger.warning("No particles found")
                    return np.array([])
            except:
                logger.warning("No particles found.")
                return np.array([])
        else:
            logger.warning("No particles found")
            return np.array([])

    else:
        logger.error("A model is needed for DL-based particle picking")

def bin_next_coordinates(coordinates,binning):

    # read coordinate file written by nextPYP
    coordinate_file = np.loadtxt( coordinates, dtype='str', ndmin=2)
    next_coordinates = coordinate_file[1:,1:].astype("float")

    # apply binning in both dimensions
    next_coordinates /= binning
    coordinate_file[1:,1:] = next_coordinates.astype('str')

    # overwrite original file with pyp coordinates
    np.savetxt(coordinates, coordinate_file, fmt='%s', delimiter='\t')


def coordinates_next2pyp(coordinates,binning,radius=0):

    # read coordinate file written by nextPYP
    coordinate_file = np.loadtxt( coordinates, dtype='str', ndmin=2)
    next_coordinates = coordinate_file.astype("float")

    # store coordinates in pyp format
    if radius > 0:
        pyp_coordinates = np.zeros( [ next_coordinates.shape[0], next_coordinates.shape[1] + 1 ] )
        pyp_coordinates[:,-1] = radius
    else:
        pyp_coordinates = next_coordinates.copy()

    # apply binning in x-dimension
    pyp_coordinates[:,0] = next_coordinates[:,0] / binning

    # apply 2x binning in z-dimension
    pyp_coordinates[:,1] = next_coordinates[:,2] / binning

    # apply binning and flip in y-dimension
    pyp_coordinates[:,2] = next_coordinates[:,1] / binning

    # overwrite original file with pyp coordinates
    np.savetxt(coordinates, pyp_coordinates.astype('int').astype('str'), fmt='%s', delimiter='\t')

def tomotrain(args):
    """Train NN for 3D particle picking.

    Parameters
    ----------
    args : args
        pyp parameters
    """
    train_folder = os.path.join( os.getcwd(), "train" )
    with open( os.path.join( train_folder, "current_list.txt" ) ) as f:
        train_name = f.read()
    train_images = os.path.join( train_folder, train_name + "_images.txt" )
    train_coords = os.path.join( train_folder, train_name + "_coordinates.txt" )
    validation_images = train_images
    validation_coords = train_coords

    files = np.loadtxt( train_images, comments='image_name', dtype="str", ndmin=2)[:,1]

    # substitute coordinate files with binned values
    number_of_labels = np.loadtxt( train_coords, dtype='str', comments="image_name", ndmin=2).shape[0]
    logger.info(f"Binning coordinates ({number_of_labels} labels)")

    # setup local scratch area
    scratch_train = os.path.join( os.environ["PYP_SCRATCH"], "train" )
    os.makedirs(os.path.join(scratch_train,"log"), exist_ok=True)

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join( train_folder, time_stamp )
    os.makedirs( output_folder )

    # make sure all output stays under the train folder
    os.chdir(scratch_train)

    logger.info(f"Training pyp model")

    if args.get("detect_nn3d_debug"):
        debug = "--debug 4"
    else:
        debug = ""

    if args.get("detect_nn3d_compress"):
        compress = "--compress"
    else:
        compress = ""

    command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/main.py semi --down_ratio {args['detect_nn3d_down_ratio']} {compress} --num_epochs {args['detect_nn3d_num_epochs']} --bbox {args['detect_nn3d_bbox']} --contrastive --exp_id test_reprod --dataset semi --arch unet_4 {debug} --val_interval {args['detect_nn3d_val_interval']} --thresh {args['detect_nn3d_thresh']} --cr_weight {args['detect_nn3d_cr_weight']} --temp {args['detect_nn3d_temp']} --tau {args['detect_nn3d_tau']} --K {args['detect_nn3d_max_objects']} --lr {args['detect_nn3d_lr']} --train_img_txt '{train_images}' --train_coord_txt '{train_coords}' --val_img_txt '{validation_images}' --val_coord_txt '{validation_coords}' --test_img_txt '{validation_images}' --test_coord_txt '{validation_coords}' 2>&1 | tee {os.path.join( os.getcwd(), 'log', time_stamp + '_cet_pick_train.log')}"
    [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # display log if available
    try:
        with open( list(Path(os.getcwd()).rglob('log.txt'))[0], ) as f:
            for line in f.readlines():
                logger.info(line.rstrip('\r\n'))
    except:
        logger.warning("No log found for training command")
        pass

    # check for failure if no output model was produced
    if len(list(Path(os.getcwd()).rglob('*.pth'))) == 0:
        raise Exception("Failed to run training module. Try increasing the memory per task")

    # move trained models to project folder
    logger.info(f"Copying results to {output_folder}")
    for path in Path(os.getcwd()).rglob('*.pth'):
        shutil.copy2( path, output_folder )

    if args["detect_nn3d_debug"]:
        debug_folder = os.path.join( output_folder, "debug" )
        os.makedirs( debug_folder )
        logger.info(f"Saving intermediate results to {debug_folder}")
        for path in Path(os.getcwd()).rglob('*.png'):
            shutil.copy2( path, debug_folder )

def tomoeval(args,name):
    # bin data by 8
    # bin_image( name + ".avg", name + "_bin.mrc", 8, args["slurm_verbose"] )

    with open("project_folder.txt") as f:
        project_folder = f.read()

    imgs_file = "images.txt"
    with open( imgs_file, 'w' ) as f:
        f.write("image_name\trec_path\n")
        f.write( name + "\t" + os.path.join( project_folder, 'mrc', name + ".rec") + "\n" )

    test_file = "testing.txt"
    with open( test_file, 'w' ) as f:
        f.write("image_name\tx_coord\ty_coord\tz_coord\n")

    if 'detect_nn3d_ref' in args.keys() and os.path.exists( project_params.resolve_path(args['detect_nn3d_ref']) ):

        logger.info(f"Evaluating using model: {Path(project_params.resolve_path(args['detect_nn3d_ref'])).name}")
        # use option "--gpus -1" to force run on CPU

        if args.get("detect_nn3d_compress"):
            compress = "--compress"
        else:
            compress = ""


        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/test.py semi --gpus -1 --arch unet_4 --dataset semi_test --with_score --exp_id test_reprod --load_model '{project_params.resolve_path(args['detect_nn3d_ref'])}' {compress} --down_ratio 2 --contrastive --K {args['detect_nn3d_max_objects']} --out_thresh {args['detect_nn3d_thresh']} --test_img_txt '{os.path.join( os.getcwd(), imgs_file)}' --test_coord_txt '{os.path.join( os.getcwd(), test_file)}' 2>&1 | tee '{os.path.join(project_folder, 'train', name + '_testing.log')}'"
        [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])
        results_folder = os.getcwd()

        # display log
        try:
            with open( list(Path(os.getcwd()).rglob('log.txt'))[0], ) as f:
                for line in f.readlines():
                    logger.info(line.rstrip('\r\n'))
        except:
            logger.warning("No log found for inference command")
            pass

        # parse output and convert coordinates to boxx format
        coordinates_file = os.path.join(results_folder,"exp/semi/test_reprod/output",name+".txt")

        # check for failure if not output was produced
        if not os.path.exists(coordinates_file):
            raise Exception("Failed to run inference module")

        with open(coordinates_file) as f:
            lines = len(f.readlines()) - 1

        if lines > 0:
            try:
                coordinates = np.loadtxt( coordinates_file, dtype=str, comments="image_name", ndmin=2)

                # threshold positions using mean of score distribution
                boxes = coordinates.copy().astype('f')
                # mean = boxes[:,-1].mean()
                # boxes[:,-1] = ( boxes[:,-1] - mean )
                coordinates = boxes[ boxes[:,-1] > args["detect_nn3d_thresh"] ]
                logger.info(str(len(coordinates)) + " positions with confidence greater than " + str(args["detect_nn3d_thresh"]))

                if len(coordinates) > 0:
                    return coordinates[:,:3].astype('i')
                else:
                    logger.warning("No particles found")
                    return np.array([])

            except:
                logger.warning("No particles found")
                return np.array([])
        else:
            logger.warning("No particles found")
            return np.array([])

    else:
        logger.error("A model is needed for 3D NN-based particle picking")
