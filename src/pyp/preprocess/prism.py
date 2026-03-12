import datetime
import logging
import os
import shutil
import glob
import time
import numpy as np
from pathlib import Path

from pyp.detect.isonet_tools import build_command_options
from pyp.utils import symlink_force
from pyp.system import local_run, project_params
from pyp.system.logging import logger

milopyp_path = '/opt/pixi/prismpyp/.pixi/envs/default'

PRISM_INIT_COMMAND = f"export LD_LIBRARY_PATH={milopyp_path}/lib:{milopyp_path}/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH; export PYTHONPATH={milopyp_path}/lib/python3.11/site-packages:$PYP_DIR/external/prismpyp/src:$PYTHONPATH; export TERM=xterm; export PATH={milopyp_path}/bin:$PATH; python -u {os.environ['PYP_DIR']}/external/prismpyp/src/prismpyp/__main__.py"

def run(args):

    os.makedirs("webp",exist_ok=True)
    os.makedirs("train/real",exist_ok=True)
    os.makedirs("train/fft",exist_ok=True)    

    preprocessing(args)
    
    # Real domain
    if args.get('prism_train_real'):
        train(args)
        eval2d(args)
        eval3d(args)
    else:
        logger.info(f"Skipping training using real space images")
    
    # Fourier domain
    if args.get('prism_train_fft'):
        train(args,real_domain=False)
        eval2d(args,real_domain=False)
        eval3d(args,real_domain=False)
    else:
        logger.info(f"Skipping training using power spectra")

def preprocessing(args):

    cs_path = args.get("prism_ice_thicknkess")
    ice_thickness = f"--cryosparc-path {cs_path}" if cs_path and os.path.exists(cs_path) else ""
    
    log_file = os.path.join('train','prismpyp_preprocesing.log')
    command = f"{PRISM_INIT_COMMAND} metadata_nextpyp --pkl-path {os.path.join(os.getcwd(),"pkl")} --output-dir={os.path.join(os.getcwd(),"train")} {ice_thickness} 2>&1 | tee '{log_file}'"
    local_run.stream_shell_command(command)    

def train(args,real_domain=True):
    """
    # train real/fft

    prismpyp train \
    --output-path output_dir/real \
    --metadata-path metadata_from_nextpyp \
    -a resnet50 \
    --epochs 100 \
    --batch-size 512 \
    --workers 1 \
    --dim 512 \
    --pred-dim 256 \
    --lr 0.05 \
    --resume pretrained_weights/checkpoint_0099.pth.tar \
    --multiprocessing-distributed \
    --dist-url 'tcp://localhost:10057' \
    --world-size 1 \
    --rank 0
    --use-fft
    """

    prefix = "prism_train"

    # we always pass these parameters
    values = [ "epochs", "batch_size", "lr", "workers", "momentum", "weight_decay", "print_freq", "dim", "pred_dim", "min_dist_umap" ]
    
    # we only pass these if True
    booleans = [ "use_fft", "add_datetime", "zip_images", "num_neighbors", "fix_pred_lr" ]

    # we only pass these if not empty
    strings = [ "resume", "feature_extractor_weights", "classifier_weights", "seed", "downsample", "scope_pixel", "size", "conf_thresh", "n_components", "n_clusters" ]

    prism_train_parameters = build_command_options( args, prefix, values, booleans, strings, style="-" )

    output = 'real' if real_domain else 'fft'
     
    logger.info(f"Training prism model")

    log_file = os.path.join('train','prismpyp_training.log')
    command = f"{PRISM_INIT_COMMAND} train --metadata-path {os.path.join(os.getcwd(),"train")} --output-path {os.path.join(os.getcwd(),"train",output)} {prism_train_parameters} --svgz 2>&1 | tee '{log_file}'"
    local_run.stream_shell_command(command)
    
def eval2d(args,real_domain=True):
    """
    # eval2d real/fft

    prismpyp eval2d \
    --output-path output_dir/real \
    --metadata-path metadata_from_nextpyp \
    -a resnet50 \
    --dist-url "tcp://localhost:10059" \
    --world-size 1 \
    --rank 0 \
    --batch-size 512 \
    --workers 1 \
    --gpu 0 \
    --fix-pred-lr \
    --feature-extractor-weights output_dir/real/checkpoints/model_best.pth.tar \
    --evaluate \
    --dim 512 \
    --pred-dim 256 \
    --n-clusters 10 \
    --num-neighbors 10 \
    --min-dist-umap 0
    --use-fft
    """

    prefix = "prism_train"

    # we always pass these parameters
    values = [ "pred_dim", "min_dist_umap", "matrix_num_references", "matrix_num_neighbors" ]
    
    # we only pass these if True
    booleans = [ "use_fft", "fix_pred_lr" ]

    # we only pass these if not empty
    strings = [ "embedding_path", "feature_extractor_weights", "num_neighbors", "n_components", "dim", "n_clusters" ]

    prism_eval2d_parameters = build_command_options( args, prefix, values, booleans, strings, style="-" )

    output = 'real' if real_domain else 'fft'

    logger.info(f"Evaluating prism model in 2D")
    log_file = os.path.join('train','prismpyp_eval2d.log')
    command = f"{PRISM_INIT_COMMAND} eval2d --evaluate --feature-extractor-weights {os.path.join(os.getcwd(),'train',output,'checkpoints','model_last.pth.tar')} --metadata-path {os.path.join(os.getcwd(),"train")} --output-path {os.path.join(os.getcwd(),"train",output)} {prism_eval2d_parameters} --svgz 2>&1 | tee '{log_file}'"
    local_run.stream_shell_command(command)
    
    list_of_useful_files = [ "scatter_plot_UMAP.svgz", "nearest_neighbors_matrix.svgz", "data_for_export.parquet" ]
    if real_domain:
        list_of_useful_files.append("thumbnail_plot_umap_mg.svgz")
    else:
        list_of_useful_files.append("thumbnail_plot_umap_ps.svgz")
    for f in list_of_useful_files:
        source = os.path.join(os.getcwd(),"train",output,"inference",f)
        target = os.path.join(os.getcwd(),"train",output,Path(f).name)
        if os.path.exists(target):
            os.remove(target)
        if os.path.exists(source):
            shutil.move(source,target)
    
    shutil.rmtree(os.path.join(os.getcwd(),"train",output,"inference"))
    shutil.rmtree(os.path.join(os.getcwd(),"train",output,"runs"))
    
    if real_domain:
        source = os.path.join(os.getcwd(),"train",output,"thumbnail_plot_umap_mg.svgz")
        target = source.replace('_mg.svgz','.svgz')
        symlink_force(source,target)
    else:
        source = os.path.join(os.getcwd(),"train",output,"thumbnail_plot_umap_ps.svgz")
        target = source.replace('_ps.svgz','.svgz')
        symlink_force(source,target)

def eval3d(args,real_domain=True):
    """
    # eval3d real/fft

    prismpyp eval3d \
    --output-path output_dir/real \
    --metadata-path metadata_from_nextpyp \
    --embedding-path output_dir/real/inference/embeddings.pth \
    -a resnet50 \
    --dist-url 'tcp://localhost:10038' \
    --world-size 1 \
    --rank 0 \
    --batch-size 512 \
    --workers 1 \
    --gpu 0 \
    --fix-pred-lr \
    --feature-extractor-weights output_dir/real/checkpoints/model_best.pth.tar \
    --evaluate \
    --dim 512 \
    --pred-dim 256 \
    --n-clusters 10 \
    --num-neighbors 10 \
    --min-dist-umap 0
    --use-fft
    """

    prefix = "prism_train"

    # we always pass these parameters
    values = [ "dim", "pred_dim", "min_dist_umap" ]
    
    # we only pass these if True
    booleans = [ "use_fft", "evaluate", "num_neighbors", "n_components", "fix_pred_lr" ]

    # we only pass these if not empty
    strings = [ "embedding_path", "feature_extractor_weights", "n_clusters" ]

    prism_eval3d_parameters = build_command_options( args, prefix, values, booleans, strings, style="-" )

    output = 'real' if real_domain else 'fft'
    
    logger.info(f"Evaluating prism model in 3D")
    log_file = os.path.join('train','prismpyp_eval3d.log')
    command = f"{PRISM_INIT_COMMAND} eval3d --evaluate --feature-extractor-weights {os.path.join(os.getcwd(),'train',output,'checkpoints','model_last.pth.tar')} --metadata-path {os.path.join(os.getcwd(),"train")} --output-path {os.path.join(os.getcwd(),"train",output)} {prism_eval3d_parameters} --svgz"
    if not os.path.exists(zipped_images):
        command += " --zip-images"
    command += f" 2>&1 | tee '{log_file}'"
    local_run.stream_shell_command(command)
    
    for f in glob.glob(os.path.join(os.getcwd(),"train",output,"inference","*")):
        target = os.path.join(os.getcwd(),"train",output,Path(f).name)
        if os.path.exists(target):
            os.remove(target)
        shutil.move(f,os.path.join(os.getcwd(),"train",output))

def intersect(args,good_real_classes,good_fft_classes):
    
    # intersection

    """
    prismpyp intersect \
        --parquet-files output_dir/fft/fft_good_export.parquet output_dir/real/real_good_export.parquet \
        --output-folder intersection \
        --link-type soft \
        --data-path example_data/webp
    """

    prism_intersect_parameters = f"--output-folder {os.getcwd()} --link-type soft"

    parent_block_path = project_params.resolve_path(args.get("data_parent"))

    real_parquet_file = os.path.join( parent_block_path, "train", "real", "data_for_export.parquet")
    fft_parquet_file = os.path.join( parent_block_path, "train", "fft", "data_for_export.parquet")
    prism_intersect_parameters += f" --real-parquet-file {real_parquet_file} --fft-parquet-file {fft_parquet_file}"
    
    bypass_filtering = False
    if len(good_real_classes) + len(good_fft_classes) > 0:
        if len(good_real_classes) > 0:
            prism_intersect_parameters += f" --good-real-classes {' '.join(good_real_classes)}"

        if len(good_fft_classes) > 0:
            prism_intersect_parameters += f" --good-fft-classes {' '.join(good_fft_classes)}"
    elif os.path.exists(good_real_parquet) or os.path.exists(good_fft_parquet):
        good_real_parquet = os.path.join( parent_block_path, "train", "real", "micrographs.parquet" )
        good_fft_parquet = os.path.join( parent_block_path, "train", "fft", "micrographs.parquet" )
        if os.path.exists(good_real_parquet):
            prism_intersect_parameters += f" --real-parquet-file {good_real_parquet}"
        if os.path.exists(good_fft_parquet):
            prism_intersect_parameters += f" --fft-parquet-file {good_fft_parquet}"
    else:
        bypass_filtering = True
        logger.warning('No selection specified for prismPYP!')
        
    if not bypass_filtering:
        logger.info(f"Intersecting prismPYP results")
        log_file = os.path.join('train','prismpyp_intersect.log')
        command = f"{PRISM_INIT_COMMAND} intersect {prism_intersect_parameters} 2>&1 | tee '{log_file}'"
        local_run.stream_shell_command(command)
