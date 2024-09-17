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
    train_name = "particles"
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

    train_coords = os.path.join(train_folder, 'training_coordinates.txt')
    train_images = os.path.join(train_folder, 'training_images.txt')

    scratch_folder = os.path.join( os.environ["PYP_SCRATCH"], "milopyp")
    os.makedirs( scratch_folder, exist_ok=True)
    if args.get("detect_nn3d_milo_import") != "none":

        if_double = "--if_double" if args.get("detect_nn3d_compress") else ""
        # select classes manually
        if args.get("detect_nn3d_milo_import") == "classes":
            if len(args.get("detect_nn3d_milo_classes").split(",")) == 0:
                raise Exception("Please specify a list of classes to select")
            else:

                # extract specific classes
                command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/select_sublabels.py --input {project_params.resolve_path(args.get('data_parent'))}/train/interactive_info_parquet.gzip --out_path {scratch_folder} {if_double} --use_classes {args.get('detect_nn3d_milo_classes').replace(' ','')}"
                [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])
                
                # extract coordinates
                command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/utils/generate_train_file.py --dir {scratch_folder} --out {os.path.join(scratch_folder,'test')}"
                [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])

                shutil.copy2( os.path.join( scratch_folder, "test_train_coords.txt"), train_coords)
                
                train_data = np.loadtxt(train_coords, dtype=str, ndmin=2, comments="image_name")
                image_names = np.unique(train_data[:,0])
                if image_names.size > 0:
                    with open(train_images, 'w') as f:
                        f.write("image_name\trec_path\n")
                        for name in image_names:
                            rec_path = os.path.join( os.getcwd(), "mrc", name+".rec" )
                            f.write(name + "\t" + rec_path + "\n")

                number_of_coordinates = np.loadtxt(train_coords, comments='image_name', ndmin=2, dtype='str').shape[0]
                if number_of_coordinates == 0:
                    raise Exception(f"Class selection {args.get('detect_nn3d_milo_classes')} contains no particles")
                logger.info(f"Selecting class IDs: {args.get('detect_nn3d_milo_classes').replace(' ','')} containing {number_of_coordinates:,} particles")
        
        # load classes from Phoenix
        elif args.get("detect_nn3d_milo_import") == "phoenix":
            if not os.path.exists( project_params.resolve_path(args["detect_nn3d_milo_parquet"])):
                raise Exception("Please specify the location of a .parquet file")
            else:
                command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/interactive_to_training_coords.py --input {project_params.resolve_path(args['detect_nn3d_milo_parquet'])} {if_double} --output {train_coords}"
                [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])
                if os.path.exists(train_coords):

                    train_data = np.loadtxt(train_coords, dtype=str, ndmin=2, comments="image_name")
                    image_names = np.unique(train_data[:,0])
                    if image_names.size > 0:
                        with open(train_images, 'w') as f:
                            f.write("image_name\trec_path\n")
                            for name in image_names:
                                rec_path = os.path.join( os.getcwd(), "mrc", name+".rec" )
                                f.write(name + "\t" + rec_path + "\n")
                    else:
                        raise Exception("Converted coordinates file is empty, please check the input parquet file")
                else:
                    raise Exception("Failed to convert the parquet to coordinates")
    else:
    
        train_name = "particles"
        train_images = os.path.join( train_folder, train_name + "_images.txt" )
        train_coords = os.path.join( train_folder, train_name + "_coordinates.txt" )
        
        # bin coordinates since website is now saving unbinned coordiantes
        coordinates = np.loadtxt(train_coords, dtype='str',ndmin=2)
        coordinates[1:,1:] = (coordinates[1:,1:].astype('int') / args['tomo_rec_binning']).astype('int').astype('str')
        np.savetxt( train_coords, coordinates, delimiter="\t", fmt="%s" )

    validation_images = train_images
    validation_coords = train_coords

    files = np.loadtxt( train_images, comments='image_name', dtype="str", ndmin=2)[:,1]

    # substitute coordinate files with binned values
    number_of_labels = np.loadtxt( train_coords, dtype='str', comments="image_name", ndmin=2).shape[0]
    logger.info(f"Binning coordinates ({number_of_labels:,} labels)")

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
        log_file = list(Path(os.getcwd()).rglob('log.txt'))[0]

        with open(log_file) as f:
            for line in f.readlines():
                logger.info(line.rstrip('\r\n'))

        with open(log_file) as f:
            output = f.read()

            loss = [ line.split("loss")[1].split()[0] for line in output.split("\n") if len(line)]
            hmloss = [ line.split("hm_loss")[1].split()[0] for line in output.split("\n") if len(line)]
            crloss = [ line.split("cr_loss")[1].split()[0] for line in output.split("\n") if len(line)]
            consisloss = [ line.split("consis_loss")[1].split()[0] for line in output.split("\n") if len(line)]
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("dark")

            fig, ax = plt.subplots(nrows=4, ncols=1, figsize=[8, 6], sharex=True)            
        
            epochs = np.arange(1,len(loss)+1)
            ax[0].set_title("Training loss")
            ax[0].plot(epochs,np.array(loss).astype('f'),".-",color="blue",label="Total loss")
            ax[0].set_ylabel("Total")
            ax[0].legend()
            ax[1].plot(epochs,np.array(hmloss).astype('f'),".-",color="green",label="Heatmap loss")
            ax[1].set_ylabel("Heatmap")
            ax[1].legend()
            ax[2].plot(epochs,np.array(crloss).astype('f'),".-",color="red",label="Contrastive loss")
            ax[2].set_ylabel("Contrastive")
            ax[2].legend()
            ax[3].plot(epochs,np.array(consisloss).astype('f'),".-",color="orange",label="Consistency loss")
            ax[3].set_ylabel("Consistency")
            ax[3].legend()
            plt.xlabel("Epoch")
            plt.savefig( os.path.join( train_folder, "training_loss.svgz"))
            plt.close()
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

    if args['detect_nn3d_with_score']:
        with_score = "--with_score"
    else:
        with_score = ""

    if args['detect_nn3d_fiber_mode']:
        fiber = f"--fiber --distance_cutoff {args['detect_nn3d_distance_cutoff']} --r2_cutoff {args['detect_nn3d_r2_cutoff']} --curvature_cutoff {args['detect_nn3d_curvature_cutoff']} --distance_scale {args['detect_nn3d_curvature_sampling']}"
    else:
        fiber = ""

    if 'detect_nn3d_ref' in args.keys() and os.path.exists( project_params.resolve_path(args['detect_nn3d_ref']) ):

        logger.info(f"Evaluating using model: {Path(project_params.resolve_path(args['detect_nn3d_ref'])).name}")
        # use option "--gpus -1" to force run on CPU

        if args.get("detect_nn3d_compress"):
            compress = "--compress"
        else:
            compress = ""

        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/test.py semi --gpus -1 --arch unet_4 --dataset semi_test {with_score} --exp_id test_reprod --load_model '{project_params.resolve_path(args['detect_nn3d_ref'])}' {compress} {fiber} --down_ratio 2 --contrastive --K {args['detect_nn3d_max_objects']} --out_thresh {args['detect_nn3d_thresh']} --test_img_txt '{os.path.join( os.getcwd(), imgs_file)}' --test_coord_txt '{os.path.join( os.getcwd(), test_file)}' 2>&1 | tee '{os.path.join(project_folder, 'train', name + '_testing.log')}'"
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

    if args.get("detect_milo_compress"):
        compress = "--compress"
    else:
        compress = ""

    logger.info(f"Training MiLoPYP's exploration module")
    if 'detect_milo_mode' in args and '2d' in args['detect_milo_mode']:
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_main.py simsiam2d3d --num_epochs {args['detect_milo_num_epochs']} --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam2d3d --arch simsiam2d3d_18  --nclusters {args['detect_milo_num_clusters']} --lr {args['detect_milo_lr']} --train_img_txt {train_images} --batch_size {args['detect_milo_batch_size']} --val_intervals {args['detect_milo_val_interval']} --save_all --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} {compress} 2>&1 | tee {os.path.join( train_folder, time_stamp + '_train.log')}"

        output_path = Path(os.getcwd() + "/exp/simsiam2d3d/test_sample")
    else:
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_main.py simsiam3d --num_epochs {args['detect_milo_num_epochs']} --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam2d3d --arch simsiam2d_18  --nclusters {args['detect_milo_num_clusters']} --lr {args['detect_milo_lr']} --train_img_txt {train_images} --batch_size {args['detect_milo_batch_size']} --val_intervals {args['detect_milo_val_interval']} --save_all --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} {compress} 2>&1 | tee {os.path.join( train_folder, time_stamp + '_train.log')}"

        output_path = Path(os.getcwd() + "/exp/simsiam3d/test_sample")

    [ output, error ] = local_run.run_shell_command(command, verbose=args['slurm_verbose'])

    # check for failure if not output was produced
    if len(list(output_path.rglob('*.pth'))) == 0:
        raise Exception("Failed to run training module")

    # move trained models to project folder
    logger.info(f"Copying results to {output_folder}")
    for path in output_path.rglob('*.pth'):
        shutil.copy2( path, output_folder )
    for path in output_path.rglob('log.txt'):
        shutil.copy2( path, os.getcwd() )

    # parse output
    with open("log.txt") as f:
        output = f.read()
    closs = [ line.split("cosine_loss")[1].split()[0] for line in output.split("\n") if len(line)]
    std = [ line.split("output_std")[1].split()[0] for line in output.split("\n") if len(line)]
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("dark")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=[8, 6], sharex=True)

    epochs = np.arange(1,len(closs)+1)
    ax[0].set_title("Training loss")
    ax[0].plot(epochs,np.array(closs).astype('f'),".-",color="blue",label="Cosine loss")
    ax[0].set_ylabel("Cosine")
    ax[0].legend()
    ax[1].plot(epochs,np.array(std).astype('f'),".-",color="red",label="Variation in learned embedding")
    ax[1].set_ylabel("STD")
    ax[1].legend()
    plt.xlabel("Epoch")
    plt.savefig( os.path.join( train_folder, "milo_training.svgz"))
    plt.close()


def miloeval(args):

    train_folder = os.path.join( os.getcwd(), "train" ) 
    rec_folder = os.path.join( os.getcwd(), "mrc" ) 
    imgs_file = os.path.join( project_params.resolve_path(args.get("data_parent")), "train",  "train_images.txt" )

    # if training file doesn't exist, create one from the list of micrographs
    if not os.path.exists(imgs_file):
        micrographs = f"{args['data_set']}.micrographs"
        if not os.path.exists(micrographs):
            raise Exception('No micrographs file in ' + os.getcwd())
        else:
            imgs_file = os.path.join( os.getcwd(), "train",  "train_images.txt" )
            input_list = [line.strip() for line in open(micrographs)]

            with open(imgs_file, 'w') as f:
                f.write("image_name\trec_path\ttilt_path\tangle_path\n")
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

                    f.write( file + "\t" + os.path.join( os.getcwd(), 'mrc', file + ".rec") + "\t" + os.path.join( os.getcwd(), 'mrc', file + "_bin.ali") + "\t" + os.path.join( os.getcwd(), 'train', file + ".rawtlt") + "\n" )                    

    if 'detect_milo_model' in args.keys() and os.path.exists( project_params.resolve_path(args['detect_milo_model']) ):

        logger.info(f"Evaluating MiLoPYP's exploration module using {Path(project_params.resolve_path(args['detect_milo_model'])).name}")
        
        # setup local scratch area
        scratch_train = os.path.join( os.environ["PYP_SCRATCH"], "eval" )
        os.makedirs(scratch_train)

        # make sure all output stays under the train folder
        os.chdir(scratch_train)

        if args.get("detect_milo_compress"):
            compress = "--compress"
        else:
            compress = ""

        input_model = project_params.resolve_path(args['detect_milo_model'])

        if '2d' in args['detect_milo_mode']:
            command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_test_hm_2d3d.py simsiam2d3d --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam2d3d --arch simsiam2d3d_18 --test_img_txt {imgs_file} --load_model {input_model} --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} {compress} 2>&1 | tee {scratch_train + '_testing.log'}"

            output_file = Path(os.getcwd() + "/exp/simsiam2d3d/test_sample/all_output_info.npz")

        else:
            command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/simsiam_test_hm_3d.py simsiam3d --exp_id test_sample --bbox {args['detect_milo_bbox']} --dataset simsiam3d --arch simsiam2d_18 --test_img_txt {imgs_file} --load_model {input_model} --gauss {args['detect_milo_gauss']} --dog {args['detect_milo_dog']} {compress} 2>&1 | tee {scratch_train + '_testing.log'}"

            output_file = Path(os.getcwd() + "/exp/simsiam3d/test_sample/all_output_info.npz")

        local_run.run_shell_command(command, verbose=args["slurm_verbose"])
        
        # check for failure if no output was produced
        if not os.path.isfile(output_file):
            raise Exception("Failed to run inference module")
        
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join( os.getcwd(), "display", time_stamp )
        os.makedirs( output_folder )
        
        # generate 2D visualization plots
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick/plot_2d.py --input {output_file} --n_cluster {args['detect_milo_num_clusters']} --num_neighbor 40 --mode umap --path {output_folder} --min_dist_vis 1.3e-3 2>&1 | tee {scratch_train +  '_plot2d.log'}"

        local_run.run_shell_command(command, verbose=args["slurm_verbose"])             

        # copy results to project folder
        for file in [ 'interactive_info_parquet.gzip', "2d_visualization_labels.webp", "2d_visualization_out.webp" ]:
            target = os.path.join( train_folder, file )
            if os.path.exists(target) or os.path.islink(target):
                os.remove(target)
            shutil.copy2( os.path.join(output_folder, file), target )
            
        # pack images and metadata into single file
        command = f"cd {output_folder}; tar cvfz {train_folder}/milopyp_interactive.tbz interactive_info_parquet.gzip imgs/"
        local_run.run_shell_command(command, verbose=False)

        # TODO: generate 3D tomogram visualization plots
        color_file = os.path.join(output_folder,'all_colors.npy')
        command = f"export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; export PYTHONPATH=$PYTHONPATH:$PYP_DIR/external/cet_pick; python {os.environ['PYP_DIR']}/external/cet_pick/cet_pick//visualize_3dhm.py --input {output_file} --color {color_file} --dir_simsiam exp/simsiam2d3d/test_sample/ --rec_dir {rec_folder} 2>&1 | tee {scratch_train + '_plot3d.log'}"
        local_run.run_shell_command(command, verbose=args['slurm_verbose'])

        import matplotlib.image
        first = True
        rec_files = glob.glob('exp/simsiam2d3d/test_sample/*_rec3d.npy')

        for rec_file in rec_files:
            name = Path(rec_file).stem.replace('_rec3d','')
            
            # get corresponding volume slice from pyp reconstruction
            slice_rec = matplotlib.image.imread( os.path.join( Path(train_folder).parents[0] / 'webp', name + '.webp' ))
    
            hm_file = glob.glob(f'exp/simsiam2d3d/test_sample/{name}_hm3d*.npy')[0]
            hm = np.load(hm_file)
            
            # extract middle slice from colormap (considering binning)
            med_slice_index = 4 if args.get("detect_milo_compress") else 2
            slice_color = hm[int(hm.shape[0]/med_slice_index),:,:,:]

            # produce weighted image
            weight = args.get('detect_milo_blend_ratio')
            slice = ( 1 - weight ) * slice_rec + weight * np.flip(slice_color,0)
            
            # save result as webp
            if first:
                matplotlib.image.imsave(os.path.join(train_folder,'3d_visualization_out.webp'), slice.astype('uint8') )
                first = False
            matplotlib.image.imsave(os.path.join(train_folder,name + '_3d_visualization.webp'), slice.astype('uint8') )

    else:
        raise Exception("Model to run MiLoPYP inference is missing")
