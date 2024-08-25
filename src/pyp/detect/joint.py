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
from pyp.inout.image import img2webp
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path, symlink_relative
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
    binning = args["tomo_spk_detect_nn2d_bin"]

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
    command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/spr_pick; python {os.environ['PYP_DIR']}/external/spr_pick/spr_pick/__main__.py train start --algorithm {args['tomo_spk_detect_nn2d_algorithm']} --noise_value {args['tomo_spk_detect_nn2d_noise_value']} --noise_style {args['tomo_spk_detect_nn2d_noise_style']} --tau {args['tomo_spk_detect_nn2d_tau']} --runs_dir '{runs_dir}' --train_dataset '{train_images}' --train_label '{train_coords}' --iterations {args['tomo_spk_detect_nn2d_iterations']} --alpha {args['tomo_spk_detect_nn2d_alpha']} --train_batch_size {args['tomo_spk_detect_nn2d_batch_size']} --nms {args['tomo_spk_detect_dist']} --num {args['tomo_spk_detect_nn2d_num']} --bb {args['tomo_spk_detect_nn2d_bb']} --patch_size {args['tomo_spk_detect_nn2d_patch_size']} --validation_dataset '{validation_images}' --validation_label '{validation_coords}' 2>&1 | tee {os.path.join(os.getcwd(), 'log', time_stamp + '_spr_pick_train.log')}"
    local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # check for failure if not output was produced
    if len(list(Path(os.getcwd()).rglob('*.training'))) == 0:
        raise Exception("Failed to run training module. Try increasing the memory per task")

    # move trained models to project folder
    logger.info(f"Copying results to {output_folder}")
    for path in Path(os.getcwd()).rglob('*.training'):
        shutil.copy2( path, output_folder )

    if args.get("tomo_spk_detect_nn2d_debug"):
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

    if 'tomo_spk_detect_nn2d_ref' in args.keys() and os.path.exists( project_params.resolve_path(args['tomo_spk_detect_nn2d_ref']) ):
        logger.info(f"Evaluating using model: {Path(project_params.resolve_path(args['tomo_spk_detect_nn2d_ref'])).name}")
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/spr_pick; python {os.environ['PYP_DIR']}/external/spr_pick/spr_pick/__main__.py eval --model '{project_params.resolve_path(args['tomo_spk_detect_nn2d_ref'])}' --dataset '{os.path.join( os.getcwd(), imgs_file)}' --runs_dir '{os.getcwd()}' --num 1"
        local_run.run_shell_command(command, verbose=args['slurm_verbose'])
        results_folder = glob.glob("./*/")[0]

        # use this to save intermediate files generated by NN particle picking
        if args.get("tomo_spk_detect_nn2d_debug"):
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
                coordinates = boxes[ boxes[:,-1] > args["tomo_spk_detect_nn2d_thresh"] ]
                logger.info(str(len(coordinates)) + " positions with confidence greater than " + str(args["tomo_spk_detect_nn2d_thresh"]))

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

    if "detect_milo_parquet" in args and os.path.exists(args["detect_milo_parquet"]):

        train_coords = os.path.join(train_folder, 'training_coordinates.txt')
        train_images = os.path.join(train_folder, 'training_images.txt')

        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/interactive_to_training_coords.py --input {args['detect_milo_parquet']} --output {train_coords}"
        [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])
        if os.path.exists(train_coords):

            train_data = np.loadtxt(train_coords, dtype=str, ndmin=2, comments="image_name")
            image_names = np.unique(train_data[:,0])
            if image_names.size > 0:
                with open(train_images, 'w') as f:
                    f.write("image_name\tpath\n")
                    for name in image_names:
                        rec_path = os.path.join( os.getcwd(), "mrc", name+".rec" )
                        f.write(name + "\t" + rec_path)
            else:
                raise Exception("Converted coordinates file is empty, please check the input parquet file")

        else:
            raise Exception("Failed to convert the parquet to coordinates")

    else:
    
        train_name = "particles"
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

    if args.get("tomo_spk_detect_nn3d_debug"):
        debug = "--debug 4"
    else:
        debug = ""

    if args.get("tomo_spk_detect_nn3d_compress"):
        compress = "--compress"
    else:
        compress = ""

    command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/main.py semi --down_ratio {args['tomo_spk_detect_nn3d_down_ratio']} {compress} --num_epochs {args['tomo_spk_detect_nn3d_num_epochs']} --bbox {args['tomo_spk_detect_nn3d_bbox']} --contrastive --exp_id test_reprod --dataset semi --arch unet_4 {debug} --val_interval {args['tomo_spk_detect_nn3d_val_interval']} --thresh {args['tomo_spk_detect_nn3d_thresh']} --cr_weight {args['tomo_spk_detect_nn3d_cr_weight']} --temp {args['tomo_spk_detect_nn3d_temp']} --tau {args['tomo_spk_detect_nn3d_tau']} --K {args['tomo_spk_detect_nn3d_max_objects']} --lr {args['tomo_spk_detect_nn3d_lr']} --train_img_txt '{train_images}' --train_coord_txt '{train_coords}' --val_img_txt '{validation_images}' --val_coord_txt '{validation_coords}' --test_img_txt '{validation_images}' --test_coord_txt '{validation_coords}' 2>&1 | tee {os.path.join( os.getcwd(), 'log', time_stamp + '_cet_pick_train.log')}"
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

    if args["tomo_spk_detect_nn3d_debug"]:
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

    if args['detect_milo_compress']:
        z_compress = "--compress"
    else:
        z_compress = ""
    
    if args['detect_milo_with_score']:
        with_score = "--with_score"
    else:
        with_score = ""

    if args['detect_milo_fiber_mode']:
        fiber = f"--fiber --distance_cutoff {args['detect_milo_distance_cutoff']} --r2_cutoff {args['detect_milo_r2_cutoff']} --curvature_cutoff {args['detect_milo_curvature_cutoff']}"
    else:
        fiber = ""

    if 'tomo_spk_detect_nn3d_ref' in args.keys() and os.path.exists( project_params.resolve_path(args['tomo_spk_detect_nn3d_ref']) ):

        logger.info(f"Evaluating using model: {Path(project_params.resolve_path(args['tomo_spk_detect_nn3d_ref'])).name}")
        # use option "--gpus -1" to force run on CPU

        if args.get("tomo_spk_detect_nn3d_compress"):
            compress = "--compress"
        else:
            compress = ""


        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/test.py semi --gpus -1 --arch unet_4 --dataset semi_test --with_score --exp_id test_reprod --load_model '{project_params.resolve_path(args['tomo_spk_detect_nn3d_ref'])}' {compress} --down_ratio 2 --contrastive --K {args['tomo_spk_detect_nn3d_max_objects']} --out_thresh {args['tomo_spk_detect_nn3d_thresh']} --test_img_txt '{os.path.join( os.getcwd(), imgs_file)}' --test_coord_txt '{os.path.join( os.getcwd(), test_file)}' 2>&1 | tee '{os.path.join(project_folder, 'train', name + '_testing.log')}'"
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
                coordinates = boxes[ boxes[:,-1] > args["tomo_spk_detect_nn3d_thresh"] ]
                logger.info(str(len(coordinates)) + " positions with confidence greater than " + str(args["tomo_spk_detect_nn3d_thresh"]))

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


def milotrain(args):
    """Train NN for 3D cellular mining.

    Parameters
    ----------
    args : args
        pyp parameters
    """
    train_folder = os.path.join( os.getcwd(), "train" )
    train_image_list = os.path.join( train_folder, "current_list.txt" )
    """
    try:
        with open( train)_image_list ) as f:
            train_name = f.read()
    except:
        logger.info("No current list exists")
        train_name = "milo"
    """

    input_list = np.unique( np.loadtxt(train_image_list, dtype='str', ndmin=2))
    train_images = os.path.join( train_folder, "train_images.txt" )

    with open( train_images, 'w' ) as train_images_file:
        train_images_file.write("image_name\trec_path\ttilt_path\tangle_path\n")

        # create list of images and rawtlt files in train/ directory
        #input_list = np.unique(np.loadtxt(train_images, dtype='str', ndmin=2))
        
        for file in input_list:
            # retrieve metadata from pkl file
            pkl_file = os.path.join("pkl", file + ".pkl")

            # unpack pkl file
            metadata = pyp_metadata.LocalMetadata(pkl_file, is_spr=False)

            # save into rawtlt file
            tlt_file = pkl_file.replace("pkl/","train/").replace(".pkl",".rawtlt")
            np.savetxt(tlt_file, metadata.data["tlt"].values, fmt="%.2f")
            
            # check tilt bin.ali file
            if not os.path.exists(os.path.join("mrc", file + "_bin.ali")):
                try:
                    binning = args["tomo_rec_binning"]
                    comm = "{0}/bin/newstack -input mrc/{1}.mrc -output mrc/{1}_bin.ali -mode 2 -origin -linear -bin {2}".format( get_imod_path(), file, binning )

                    [output, error] =local_run.run_shell_command(comm,verbose=args["slurm_verbose"])
                except:
                    raise Exception("Can't find aligned tilt series images")

            train_images_file.write( file + "\t" + os.path.join( os.getcwd(), 'mrc', file + ".rec") + "\t" + os.path.join( os.getcwd(), 'mrc', file + "_bin.ali") + "\t" + os.path.join( os.getcwd(), 'train', file + ".rawtlt") + "\n" )

    # setup local scratch area
    scratch_train = os.path.join( os.environ["PYP_SCRATCH"], "train" )
    os.makedirs(scratch_train)

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join( train_folder, time_stamp )
    os.makedirs( output_folder )

    # make sure all output stays under the train folder
    os.chdir(scratch_train)

    logger.info(f"Training MiLoPYP's exploration module")
    if 'detect_milo_mode' in args and '2d' in args['detect_milo_mode']:
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_main.py simsiam2d3d --num_epochs {args['detect_milo_num_epochs']} --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam2d3d --arch simsiam2d3d_18  --nclusters {args['detect_milo_num_clusters']} --lr {args['detect_milo_lr']} --train_img_txt {train_images} --batch_size {args['detect_milo_batch_size']} --val_intervals {args['detect_milo_val_interval']} --save_all --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} 2>&1 | tee {os.path.join( train_folder, time_stamp + '_train.log')}"

        output_path = Path(os.getcwd() + "/exp/simsiam2d3d/test_sample")
    else:
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_main.py simsiam3d --num_epochs {args['detect_milo_num_epochs']} --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam2d3d --arch simsiam2d_18  --nclusters {args['detect_milo_num_clusters']} --lr {args['detect_milo_lr']} --train_img_txt {train_images} --batch_size {args['detect_milo_batch_size']} --val_intervals {args['detect_milo_val_interval']} --save_all --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} 2>&1 | tee {os.path.join( train_folder, time_stamp + '_train.log')}"

        output_path = Path(os.getcwd() + "/exp/simsiam3d/test_sample")

    [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # check for failure if not output was produced
    if len(list(output_path.rglob('*.pth'))) == 0:
        raise Exception("Failed to run training module")

    # move trained models to project folder
    logger.info(f"Copying results to {output_folder}")
    for path in output_path.rglob('*.pth'):
        shutil.copy2( path, output_folder )

def miloeval(args):

    train_folder = os.path.join( os.getcwd(), "train" ) 
    imgs_file = os.path.join( project_params.resolve_path(args.get("data_parent")), "train",  "train_images.txt" )

    if 'detect_milo_model' in args.keys() and os.path.exists( project_params.resolve_path(args['detect_milo_model']) ):

        logger.info(f"Evaluating MiLoPYP's exploration module using {Path(project_params.resolve_path(args['detect_milo_model'])).name}")
        
        # setup local scratch area
        scratch_train = os.path.join( os.environ["PYP_SCRATCH"], "eval" )
        os.makedirs(scratch_train)

        # make sure all output stays under the train folder
        os.chdir(scratch_train)

        input_model = project_params.resolve_path(args['detect_milo_model'])

        if '2d' in args['detect_milo_mode']:
            command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_test_hm_2d3d.py simsiam2d3d --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam2d3d --arch simsiam2d3d_18 --test_img_txt {imgs_file} --load_model {input_model} --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} 2>&1 | tee {train_folder + '_testing.log'}"

            output_file = Path(os.getcwd() + "/exp/simsiam2d3d/test_sample/all_output_info.npz")

        else:
            command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_test_hm_3d.py simsiam3d --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam3d --arch simsiam2d_18 --test_img_txt {imgs_file} --load_model {input_model} --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} 2>&1 | tee {train_folder + '_testing.log'}"

            output_file = Path(os.getcwd() + "/exp/simsiam3d/test_sample/all_output_info.npz")

        [ output, error ] = local_run.run_shell_command(command, verbose=False)
        
        if args.get('slurm_verbose'):
            with open(train_folder + '_testing.log') as f:
                logger.info("\n".join([s for s in f.read().split("\n") if not s.startswith('No param') and not s.startswith('Drop parameter layer')]))
        
        # check for failure if no output was produced
        if not os.path.isfile(output_file):
            raise Exception("Failed to run inference module")
        
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join( train_folder, "display", time_stamp )
        os.makedirs( output_folder )
        
        # move trained models to project folder
        logger.info(f"Copying results to {output_folder}")  
        shutil.copy2( output_file, output_folder )

        # generate 2D visualization plots
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/plot_2d.py --input {output_file} --n_cluster {args['detect_milo_num_clusters']} --num_neighbor 40 --mode umap --path {output_folder} --min_dist_vis 1.3e-3 2>&1 | tee {train_folder +  '_plot2d.log'}"

        [ output, error ] = local_run.run_shell_command(command, verbose=False)
        if args.get('slurm_verbose'):
            with open(train_folder + '_plot2d.log') as f:
                logger.info("\n".join([s for s in f.read().split("\n") if s]))                

        # convert the png to webp
        img2webp(f"{output_folder}/2d_visualization_labels.png", f"{output_folder}/2d_visualization_labels.webp")
        img2webp(f"{output_folder}/2d_visualization_out.png", f"{output_folder}/2d_visualization_out.webp")
        os.remove(f"{output_folder}/2d_visualization_labels.png")
        os.remove(f"{output_folder}/2d_visualization_out.png")
        
        # create symlinks to latest results
        symlink_relative( os.path.join(output_folder, 'all_output_info.npz'), os.path.join( train_folder, 'all_output_info.npz' ) )
        symlink_relative( os.path.join(output_folder, '2d_visualization_labels.webp'), os.path.join( train_folder, "2d_visualization_labels.webp") )
        symlink_relative( os.path.join(output_folder, '2d_visualization_out.webp'), os.path.join( train_folder, "2d_visualization_out.webp") )

        """
        # generate 3D tomogram visualization plots
        commmand = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick//visualize_3dhm.py --input {outputfile} --color exp/simsiam2d3d/test_sample/all_colors.npy --dir_simsiam exp/simsiam2d3d/test_sample/ --rec_dir sample_data/ 2>&1 | tee {train_folder + '_plot3d.log'}"
        [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])
        if args.get('slurm_verbose'):
            with open(train_folder + '_plot3d.log') as f:
                logger.info("\n".join([s for s in output.read().split("\n") if s]))                
        """

    else:
        raise Exception("Model to run MiLoPYP inference is missing")
