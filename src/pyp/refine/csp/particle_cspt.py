import datetime
import fcntl
import glob
import json
import math
import os
import re
import shutil
import sys
import time
from ast import Or
from pathlib import Path
from re import L, T
from xml.sax import make_parser
from tqdm import tqdm

import numpy as np
import pickle

from pyp import align, postprocess
from pyp.analysis import plot, statistics
from pyp.analysis.geometry import divide2regions, findSpecimenBounds, get_tomo_binning
from pyp.analysis.geometry.pyp_convert_coord import read_3dbox
from pyp.analysis.occupancies import occupancies, occupancy_extended
from pyp.analysis.scores import call_shape_phase_residuals
from pyp.analysis.plot import pyp_frealign_plot_weights
from pyp.inout.image import mrc, img2webp
from pyp.inout.metadata import frealign_parfile, isfrealignx, pyp_metadata, generate_ministar
from pyp.inout.metadata.cistem_star_file import *
from pyp.refine.frealign import frealign
from pyp.streampyp.web import Web
from pyp.streampyp.logging import TQDMLogger
from pyp.system import local_run, mpi, project_params, slurm
from pyp.system.db_comm import save_reconstruction_to_website, save_refinement_bundle_to_website
from pyp.system.logging import initialize_pyp_logger
from pyp.system.set_up import prepare_frealign_dir
from pyp.system.singularity import run_pyp
from pyp.system.utils import get_imod_path
from pyp.utils import get_relative_path, symlink_force, timer, symlink_relative
from pyp_main import csp_split

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def clean_tomo_particles(par_data, boxes3d_file, metric="new"):
    """Clean particles by setting occ to zero based on boxes3d files

    Parameters
    ----------
    par_data : numpy array 
        Array that stores particle information in parfile
    boxes3d_file : str
        The filename of boxes3d file
    metric : str, optional
        The alignment metric for parfile formatting, by default "new"

    Returns
    -------
    numpy array 
        A new numpy array where occ of some particles is set to zero
    """
    if metric == "new":
        film_col = 7
        score_col = 14
        ptlidx_col = 16
        scanord_col = 19
        occ_col = 11
    else:
        logger.error("Currently not support other metrics except metric new")
        sys.exit()

    try:
        clean_boxes3d = read_3dbox(boxes3d_file)
    except:
        logger.warning(f"{boxes3d_file} canot be read successfully.")

    # Modify OCCs on a 3d sub-volume basis depending on the Keep_CSP in 3d box file
    for par in par_data:
        ptlind = int(par[ptlidx_col])
        if "n" in clean_boxes3d[ptlind][5].lower():
            par[occ_col] = 0.0
        elif "y" in clean_boxes3d[ptlind][5].lower():
            pass
        else:
            logger.warning(
                f"Cannot recognize if keeping this sub-volume or not. {clean_boxes3d[ptlind][5]}"
            )

    return par_data


def sort_particles_regions(
    particle_coordinates, corners_squares, squaresize, per_particle=False
):
    """ Sort particles by sub-regions 

    Parameters
    ----------
    particle_coordinates : list[str]
        Particle 3D coordinates from 3dboxes file 
    corners_squares : list[list[float]]
        List that stores 3D coordinate of the bottom-left corner of every divided squares
    squaresize : list[float]
        List stores the size of squares

    Returns
    -------
    list[ list[str] ]
        List that stores the sorted particle indexes in squares
    """
    # let's have one more region to store particle completely out of bound (should be first checked by csp_tomo_swarm)
    ret = []
    if not per_particle:
        ret = [[] for i in range(len(corners_squares) + 1)]

    for idx, par in enumerate(particle_coordinates):

        x, y, z = float(par[1]), float(par[2]), float(par[3])

        if per_particle:
            ret.append([int(par[0])])

        else:
            find_square = False
            for idx_square, square in enumerate(corners_squares):

                # find which squares the particle belongs to
                if (
                    x >= square[0]
                    and y >= square[1]
                    and z >= square[2]
                    and x <= square[0] + squaresize[0]
                    and y <= square[1] + squaresize[1]
                    and z <= square[2] + squaresize[2]
                ):

                    # add particle index to the list
                    ret[idx_square].append(int(par[0]))
                    find_square = True
                    break

            if not find_square:
                ret[-1].append(int(par[0]))
                logger.debug(
                    "Particle [x = %f, y = %f, z = %f] is possibly out of bound."
                    % (x, y, z)
                )

    # sort the list based on the number of particles
    ret = sorted(ret, key=lambda x: len(x))

    return ret


def merge_alignment_parameters(
    parameter_file: str, mode: int, output_pattern: str = "_region????_????_????"
):
    """ Merge the splitted outputs from CSP into a main parfile

    Parameters
    ----------
    alignment_parameters : str
        The path of the main parfile
    output_pattern : str, optional
        The pattern used to search CSP results, by default "_region????_????_????"

    Returns
    -------
    numpy array
        Particle metadata in numpy array (before written to parfile)
    """
    extended_parameter_file = parameter_file.replace(".cistem", "_extended.cistem")
    outputs = sorted(
        [
            file
            for file in glob.glob(
                parameter_file.strip(".cistem") + output_pattern + ".cistem"
            )
        ]
    )
    outputs_extended = sorted(
        [
            file
            for file in glob.glob(
                extended_parameter_file.strip("_extended.cistem") + output_pattern + "_extended.cistem"
            ) + [extended_parameter_file] 
        ]
    )

    assert (len(outputs) > 0), "No output parameter file is generated."

    if mode == 5 or mode == 4:
        assert (len(outputs_extended) > 0), "No extended parameter file is generated." 
    elif mode == 7 or mode == 2:
        assert (len(outputs_extended) > 0), "No extended parameter file is generated." 

    return Parameters.merge(input_files=outputs, input_extended_files=outputs_extended)    


def split_parx_particle_cspt(p_object, main_parxfile, regions_list, metric="new"):
    """ Before frame refinement (CSP mode 5 & 6), this function splits the main parfile, which contains all particles in a tilt-series,
    into several sub-parfiles based on their 3D locations

    Parameters
    ----------
    p_object : Parameters
        Frealign parameter file object
    main_parxfile : str
        The relative path of the main_parxfile
    regions_list : list[list]
        A nested list containing particle indexes in squares
    metric : str, optional
        Alignment metric, by default "new"

    Returns
    -------
    list
        A list containing the names of splitted sub-parfiles that each will be read by CSP binary
    """
    if metric == "new":
        film_col = 7
        score_col = 14
        ptlidx_col = 16
        scanord_col = 19
        occ_col = 11
    else:
        logger.error("Currently not support other metrics except metric new")
        sys.exit()

    # first read the main parfile in numpy array
    par_data = p_object.data

    parinfo_regions = []
    
    # go through each square to find parlines based on the particle index
    for region_idx, region in enumerate(regions_list):
        # if this square is not empty
        if len(region) > 0:
            # filter based on the list of particle index in this region
            parlines_filter = np.isin(par_data[:, ptlidx_col], region)
            parlines_region = par_data[:][parlines_filter]

            if parlines_region.size != 0:
                parinfo_regions.append(parlines_region)

    split_files_list = []

    # write out splitted parfiles
    for idx, parinfo in enumerate(parinfo_regions):

        split_filename = main_parxfile.replace(".parx", "_region%04d.parx" % (idx))
        p_object.data = parinfo
        p_object.write_file(split_filename)

        split_files_list.append(
                (split_filename, np.unique(parinfo[:, ptlidx_col].astype("int")), np.unique(parinfo[:, scanord_col].astype("int")))
        )

    return split_files_list


def prepare_particle_cspt(
    name, dataset, main_parxfile, main_stackfile, parx_object, mode, cpus, parameters, grids=[1,1,1], use_frames=False
):
    """ This function prepares stuffs for frame refinement (CSP mode 5 & 6)
        1. Compute specimen bounds in xyz
        2. Divide specimen into multiple (overlapped) grids
        3. Sort particles into different grids 
        4. Prepare stack files for CSP 
        5. Split parfile by grids

    Parameters
    ----------
    name : str
        The name of the tilt-series
    dataset : str
        The name of the dataset
    main_parxfile : str
        The path of the main parfile
    main_stackfile : str
        The path of the main stack file
    parx_object : Parameters
        Frealign parameter object
    cpus : int
        The number of cpus/threads 
        
    Returns
    -------
    list
        list containing the names of splitted sub-parfiles that each will be independently read by CSP processes
    """
    
    ptlidx_col = 16
    metafile = "{}.pkl".format(name)
    if not os.path.exists(metafile):
        raise Exception(f"Metadata is required to run patch-based local refinement")
    metaobj= pyp_metadata.LocalMetadata(metafile)
    metadata = metaobj.data        


    if "tomo" in parameters["data_mode"].lower():
        # First read 3D particle coordinates
        boxes3d = "{}_boxes3d.txt".format(name)
        if os.path.exists(boxes3d):
            coord_3d = read_3dbox(boxes3d)
        else:
            logger.error("{} not found".format(os.path.join(os.getcwd(), boxes3d)))
            logger.error("Frame refinement mode 5 & 6 require boxes3d files!")
            return None

        # Figure out the dimension of un-binned tomogram (use the exact same way in csp_tomo_swarm)
        micrographsize_x, micrographsize_y = metadata["image"].at[0, "x"], metadata["image"].at[0, "y"]
        # binning = get_tomo_binning(micrographsize_x, micrographsize_y, int(parameters["tomo_rec_size"]), squared_image=parameters["tomo_rec_square"])
        binning = parameters["tomo_rec_binning"]
        tomox, tomoy, tomoz = metadata["tomo"].at[0, "x"] * binning, metadata["tomo"].at[0, "y"] * binning, metadata["tomo"].at[0, "z"] * binning


        # find out the bounds of the specimen (where particles are actually located)
        bottom_left_corner, top_right_corner = findSpecimenBounds(
            coord_3d, [tomox, tomoy, tomoz]
        )

        # divide the specimen into several sub-regions
        corners, size_region = divide2regions(
            bottom_left_corner, top_right_corner, split_x=grids[0], split_y=grids[1], split_z=grids[2],
        )

        # sort particles into sub-regions
        if parameters["csp_frame_refinement"] and use_frames:
            per_particle = True
        else: 
            per_particle = False
        
        ptlidx_regions_list = sort_particles_regions(
            coord_3d, corners, size_region, per_particle
        )

    else:
        ptlind_col = 16
        
        boxes = np.loadtxt(f"{name}.allboxes", ndmin=2)
        parfile = parx_object.data
        parfile = parfile[
                    np.unique(parfile[:, ptlind_col].astype("int"), return_index=True)[1], :
                    ]
        
        # compose cooridate list to [[ptlind, x, y, z],...[]] like tomo 
        boxes = [[int(parline[ptlind_col])] + list(box[:2]) + [0.0] 
                    for box, parline in zip(boxes, parfile)]

        imagex, imagey = metadata["image"].at[0, "x"], metadata["image"].at[0, "y"]

        corners, size_region = divide2regions(
            bottom_left_corner=[0,0,0], 
            top_right_corner=[imagex, imagey, 0], 
            split_x=grids[0], 
            split_y=grids[1], 
            split_z=1,
        )
            
        ptlidx_regions_list = sort_particles_regions(
            boxes, corners, size_region, per_particle=use_frames # do per particle only for frame refinement 
        )

    # split the main parxfile into several sub-parxfile based on the regions
    split_parx_list = split_parx_particle_cspt(
        parx_object, main_parxfile, ptlidx_regions_list
    )

    return split_parx_list



@timer.Timer(
    "csp_local_merge", text="Total time elapsed (csp_local_merge): {}", logger=logger.info
)
def merge_movie_files_in_job_arr(
    movie_file,
    par_file,
    ordering_file,
    project_dir_file,
    output_basename,
    save_stacks=False,
):
    """
    When running multiple sprswarm runs in a single array job, merge the output
    """

    with open(movie_file) as f:
        movie_list = [line.strip() for line in f]
    with open(par_file) as f:
        par_list = [line.strip() for line in f]

    # workaround for case when job array has only one component
    if "_" not in output_basename:
        output_basename += "_1"

    # check that all files are present
    assert len(movie_list) == len(
        par_list
    ), "movie files and par files must be of equal length"
    for movie, parfile in zip(movie_list, par_list):
        not_found = "{} is not found"
        if not os.path.exists(movie):
            logger.info(not_found.format(movie))
        if not os.path.exists(parfile):
            logger.info(not_found.format(parfile))

    with timer.Timer(
        "merge_stack", text="Merging particle stack took: {}", logger=logger.info
    ):
        if len(movie_list) > 1:
            logger.info("Merging movie files")
            mrc.merge_fast(movie_list,f"{output_basename}_stack.mrc",remove=True)
        else:
            os.rename(movie_list[0], str(output_basename) + "_stack.mrc")

    project_path = open(project_dir_file).read()
    mp = project_params.load_pyp_parameters(project_path)
    fp = mp

    if "frealignx" in mp["refine_metric"]:
        is_frealignx = True
    else:
        is_frealignx = False

    if fp["extract_stacks"]:
        shutil.copy2(
            output_basename + "_stack.mrc",
            os.path.join(
                project_path, "frealign", output_basename + "_stack.mrc"
            ),
        )
        save_stacks = True

    logger.info("Merging parameter files")

    iteration = fp["refine_iter"]

    convert_parfile = False

    if iteration == 2:
        classes = 1
        mp["class_num"] = 1 
    else:
        classes = int(project_params.param(fp["class_num"], iteration))

    for class_index in range(classes):

        current_class = class_index + 1

        merged_par_file = str(output_basename) + "_r%02d.cistem" % (current_class)
        new_par_list = []
        for p in par_list:
            new_par_list.append(p.replace("_r01_", "_r%02d_" % current_class))

        metric = project_params.param(mp["refine_metric"], iteration)
        


    #     frealign_parfile.Parameters.merge_parameters(
    #         new_par_list, merged_par_file, metric, update_film=True, parx=True, frealignx=is_frealignx
    #     )

        # FIXME: merge all the cistem binary files in a bundle
        # now I simply copy it over 
        shutil.copy2(new_par_list[0], merged_par_file)
        shutil.copy2(new_par_list[0].replace(".cistem", "_extended.cistem"), merged_par_file.replace(".cistem", "_extended.cistem"))

        shutil.copy2(
            par_list[0]
            .replace("/maps/", "/scratch/")
            .replace(".cistem", ".mrc")
            .replace("_r01_", "_r%02d_" % current_class),
            str(output_basename) + "_r%02d.mrc" % (current_class),
        )
    #     # if dose weighting is enabled and we are not in metric frealignx, we need to add PSHIFT column, film column start from 0 is OK
    #     current_pardata = frealign_parfile.Parameters.from_file(merged_par_file).data
    #     if current_pardata.shape[-1] < 46:

    #         new_file = [
    #             line for line in open(merged_par_file) if not line.startswith("C")
    #         ]
    #         header = frealign_parfile.EXTENDED_FREALIGNX_PAR_HEADER
    #         with open(merged_par_file, "w") as f:
    #             [f.write(line) for line in header]
    #             for i in new_file:
    #                 f.write(i[:91] + "%8.2f" % 0 + i[91:])
    #         if current_class == classes:
    #             is_frealignx = True
    #             convert_parfile = True
    #         logger.debug("Reconstruction using frealignx format")

    #     else:
    #         is_frealignx = True
    #         if current_pardata[0, 7] == 1:
    #             # film id start from 0 for reconstruction
    #             current_pardata[:, 7] -= 1 
    #             frealign_parfile.Parameters.write_parameter_file(merged_par_file, current_pardata, parx=True, frealignx=True)

    logger.info("Running reconstruction")

    # change occupancy after refinement
    if classes > 1 and not mp["refine_skip"]:

        # keep copies of original par files before updating occupancies
        for class_index in range(classes):
            par_file = str(output_basename) + "_r%02d_%02d.par" % (
                class_index + 1,
                iteration,
            )
            shutil.copy2(par_file, par_file.replace(".par", "_pre.par"))

        logger.info("Updating occupancies after local merge")
        # update occupancies using LogP values
        occupancy_extended(mp, output_basename, iteration, classes, ".", is_frealignx=is_frealignx, local=True)

    for class_index in range(classes):

        current_class = class_index + 1

        # # append other rows to the current block from previous iteration
        # film_col = 7
        # current_block_par = "%s_r%02d_%02d.par" % (
        #     output_basename,
        #     current_class,
        #     iteration,
        # )

        # # generate tsv files for Artix display
        # if "tomo" in mp["data_mode"]:
        #     star_output = os.path.join(project_path, "frealign", "artiax")
        #     binning = mp["tomo_rec_binning"]
        #     z_thicknes = mp["tomo_rec_thickness"]
        #     generate_ministar(current_block_par, movie_list, z_thicknes, binning, cls=current_class, output_path=star_output)

        # if classes > 1:
        #     # backup the current par for later recovery
        #     shutil.copy(current_block_par, current_block_par.replace(".par", ".paro"))

        # current_block_data = frealign_parfile.Parameters.from_file(
        #     current_block_par
        # ).data

        # # ptl_id_now = current_block_data[-1, 0]
        # film_id_now = current_block_data[-1, film_col]
        # film_total = np.loadtxt(os.path.join(project_path, mp["data_set"] + ".films"), dtype=str, ndmin=2)
        # if classes > 1 and film_total.shape[0] > 1 and iteration > 2:

        #     frealign_parfile.Parameters.add_lines_with_statistics(
        #         current_block_par, 
        #         current_class, 
        #         project_path, 
        #         is_frealignx=is_frealignx, 
        #         )
        
        # else:
        #     logger.info("Skip modifying metadata for 1 film only")
        # run reconstructions
        run_reconstruction(
            output_basename,
            mp,
            "merged_recon" + "_r%02d" % current_class,
            "../output" + "_r%02d" % current_class,
            save_stacks,
            current_class,
            iteration,
        )

        # move back the original par with current rows
        if classes > 1 and film_total.shape[0] > 1 and iteration > 2:
             with timer.Timer(
                "remove_hacking", text="Removing additional lines from parfile took: {}", logger=logger.info
            ):
                shutil.copy(current_block_par.replace(".par", ".paro"), current_block_par)
                # in case used par from score_phase function is not tha same as par 
                hack_film = film_id_now + 1
                used_hackpar = current_block_par.replace(".par", "_used.par")
                used_hackdata = frealign_parfile.Parameters.from_file(used_hackpar).data
                used_data = used_hackdata[used_hackdata[:, film_col] < hack_film]
                version = project_params.param(mp["refine_metric"], iteration).split("_")[0]
                frealign_par = frealign_parfile.Parameters(version=version)
                if used_data.shape[1] > 16:
                    is_parx = True
                else:
                    is_parx = False
                # write used par without hack lines
                frealign_par.write_parameter_file(
                    used_hackpar, used_data, parx=is_parx, frealignx=is_frealignx
                )
                # shutil.copy(current_block_par.replace(".par", ".paro"), current_block_par.replace(".par", "_used.par"))

        t = timer.Timer(text="Saving logs and reconstruction took: {}", logger=logger.info)
        t.start()

        # save logs
        mypath = os.path.join(
            "merged_recon" + "_r%02d" % current_class, "log/*_0000001_*.log"
        )
        log_files = glob.glob(mypath)
        if len(log_files) > 0:
            target_name = (
                mp["data_set"]
                + "_r%02d_%02d" % (current_class, iteration)
                + "_mreconst.log"
            )
            target = os.path.join(project_path, "frealign", "log", target_name)
            shutil.copy2(log_files[0], target)
            # send output to interface
            if mp['slurm_verbose']:
                with open(log_files[0]) as f:
                    logger.info(f.read())

        # convert it back to metric new format
        if True:
            if mp['slurm_verbose']:
                logger.info("Change back to NEW format parfile")
            for par in glob.glob("merged_recon" + "_r%02d/*.par" % current_class):
                new_file = [
                    line for line in open(par) if not line.startswith("C")
                ]
                comments = [line for line in open(par) if line.startswith("C")]

                with open(par, "w") as f:

                    [f.write(line) for line in comments]

                    for i in new_file:
                        f.write(i[:91] + i[99:])

        # copy recon over to project dir
        if mp["refine_parfile_compress"]:
            compressed = True
        else:
            compressed = False

        save_reconstruction(
            output_basename, 
            project_path,
            iteration,
            output_folder="output" + "_r%02d" % current_class,
            threads=mp["slurm_tasks"],
            compress=compressed
        )
        t.stop()

    # keep track of movies that have been processed successfully
    for movie in movie_list:
        local_movie = os.path.basename(movie).replace("_stack.mrc", "")
        path = Path(
            os.path.join(project_path, "frealign", "scratch", ".{}".format(local_movie))
        )
        path.touch()

        # save metadata  
        convert = True

        if "spr" in mp["data_mode"]:
            is_spr = True
        else:
            is_spr = False
        
        if convert:
            cwd = os.getcwd()
            os.chdir(local_movie)
            metaname = local_movie + ".pkl"
            metadata = pyp_metadata.LocalMetadata(metaname, is_spr=is_spr)
            metadata.loadFiles()

            # copy the metadata to project folder.
            meta_path = os.path.join(project_path, "pkl")
            if not os.path.isdir(meta_path):
                os.mkdir(meta_path)
            if not os.path.isfile(os.path.join(meta_path, metaname)):  
                shutil.copy2(metaname, meta_path)
            os.chdir(cwd)

def save_ordering(project_path, basename, order_number, ordering_file="ordering.txt"):
    save_file = os.path.join(project_path, "frealign", "scratch", ordering_file)
    dest_path = os.path.dirname(save_file)
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except:
            pass
    with open(save_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("{} {}\n".format(basename, order_number))
        fcntl.flock(f, fcntl.LOCK_UN)
    logger.info("ordering information written")


def save_reconstruction(
    output_basename, project_path, iteration, output_folder="output", threads=1, compress=True
):
    logger.info("Copying reconstruction to project folder")
    dest_path = os.path.join(project_path, "frealign", "scratch")
    log_path = os.path.join(project_path, "frealign")
    weight_path = os.path.join(project_path, "frealign", "weights")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    all_files = os.listdir(output_folder)

    # output_name = Path(
    #     [f for f in all_files if f.endswith(".par") and not "_used" in f][0]
    # ).stem
    output_name = output_basename
    
    # move over all the binary files (each represent on tilt-series) in a given bundle to output folder
    [os.rename(binary_file, Path(output_folder) / Path(binary_file).name) for binary_file in glob.glob("*/frealign/maps/*.cistem")]

    # copy only dumped mrc and .par files
    saved_path = os.getcwd()
    os.chdir(output_folder)

    for file in glob.glob("*.svgz"):
        shutil.copy(file, log_path)

    for file in glob.glob("*_weights.svgz"):
        # convert image to webp format and save to target directory: log_path
        shutil.copy(file,os.path.join(log_path, file.replace(".svgz","_local.svgz")))

        if not os.path.exists(weight_path): 
            os.makedirs(weight_path)
        try:
            shutil.copy2(file.replace("_weights.svgz", "_scores.txt"), os.path.join(weight_path, file.replace("_weights.svgz", "_scores.txt")))
        except:
            logger.warning("No scores.txt exist from the reconstruction results, probably using lblur")

        # save weights to website
        save_refinement_bundle_to_website(file.replace("_weights.svgz",""), iteration)

    # delete the broken links that have already been deleted by local merge
    local_run.run_shell_command("find . -xtype l -delete", verbose=False)
    dumpfiles = glob.glob("*_???????_???????.mrc") + glob.glob("*_map?_n*.mrc") + glob.glob("*.cistem")
    if compress:
        compressed_file = output_name + ".bz2"
        frealign_parfile.Parameters.compress_parameter_file(
            " ".join(dumpfiles), os.path.join(dest_path, compressed_file + "."), threads
        )
        # in case file starts to be decompressed when undergoing compression (due to live decompression)
        os.rename(os.path.join(dest_path, compressed_file + "."), os.path.join(dest_path, compressed_file))
        os.chdir(saved_path)
        logger.info("Compressing intermediate files to scratch folder")
    else:
        for file in dumpfiles:
            shutil.copy2(file, os.path.join(dest_path, file + '.'))
        for file in dumpfiles:
            os.rename(os.path.join(dest_path, file + '.'), os.path.join(dest_path, file))
        os.chdir(saved_path)



def get_number_of_intermediate_reconstructions(mp):
    """Figure out how to manage resources for 3D reconstruction

    Parameters
    ----------
    mp : parameters
        pyp parameters

    Returns
    -------
    groups, cpus_per_group
        Number of reconstructions and threads to use for each process
    """
    frealign_cpus_per_group = 1
    # use multiple threads
    cpus_per_group = frealign_cpus_per_group
    groups = int(mp["slurm_tasks"] / cpus_per_group)
    return groups, cpus_per_group


@timer.Timer(
    "run_reconstruction", text="Running reconstruction took: {}", logger=logger.info
)
def run_reconstruction(
    name,
    mp,
    merged_recon_dir="merged_recon",
    output_folder="../output",
    save_stacks=False,
    ref=1,
    iteration=2,
):
    fp = mp

    scratch = os.environ["PYP_SCRATCH"] = ""

    # remove dirs if exist
    shutil.rmtree(merged_recon_dir, ignore_errors=True)

    if not os.path.exists(merged_recon_dir):
        os.makedirs(merged_recon_dir)

    header = mrc.readHeaderFromFile(name + "_stack.mrc")
    frames = header["nz"]

    fp["refine_dataset"] = name

    current_suffix = "_r%02d_%02d" % (ref, iteration)
    previous_suffix = "_r%02d_%02d" % (ref, iteration - 1)

    parameter_file = name + "_r%02d.cistem" % (ref)
    alignment_parameters = Parameters.from_file(parameter_file)

    # necessary symlink for reconstruction
    # create shaped _used par file
    # par_obj = call_shape_phase_residuals(
    #     name + current_suffix + ".par",
    #     name + current_suffix + "_used.par",
    #     os.path.join(merged_recon_dir, name + current_suffix + "_scores.png"),
    #     fp,
    #     iteration,
    # )
 
    # os.symlink(
    #     "../" + name + current_suffix + "_used.par",
    #     os.path.join(merged_recon_dir, name + current_suffix + "_used.par"),
    # )

    # os.symlink(
    #     "../" + name + current_suffix + ".par",
    #     os.path.join(merged_recon_dir, name + current_suffix + ".par"),
    # )

    # if mp["class_num"] > 1 and not fp["refine_skip"]:
    #     os.symlink(
    #         "../" + name + current_suffix + "_pre.par",
    #         os.path.join(merged_recon_dir, name + current_suffix + "_pre.par"),
    #     )

    # os.symlink(
    #     "../" + name + previous_suffix + ".mrc",
    #     os.path.join(merged_recon_dir, name + current_suffix + ".mrc"),
    # )

    # os.symlink(
    #     "../" + name + previous_suffix + ".mrc",
    #     os.path.join(merged_recon_dir, name + previous_suffix + ".mrc"),
    # )

    # # if needed save the stack files
    # if save_stacks:
    #     os.symlink(
    #         "../" + name + "_stack.mrc",
    #         os.path.join(merged_recon_dir, name + "_stack.mrc"),
    #     )

    curr_dir = os.getcwd()

    os.chdir(merged_recon_dir)

    # AB - Create directories if needed
    prepare_frealign_dir()

    # Split reconstruction into several jobs to run reconstruct3d in parallel
    groups, cpus_per_group = get_number_of_intermediate_reconstructions(mp)

    # get the number of frames
    num_tilts = alignment_parameters.get_extended_data().get_num_tilts()
    frames_per_tilt = alignment_parameters.get_num_frames()

    commands, count = local_run.create_split_commands(
        mp,
        name,
        frames,
        groups,
        scratch,
        step="reconstruct3d",
        num_frames=num_tilts,
        ref=ref,
        iteration=iteration,
    )

    # make sure reconstruct3d runs in parallel
    commands[0] = (
        "export OMP_NUM_THREADS={0}; export NCPUS={0}; ".format(cpus_per_group)
        + commands[0]
    )

    recon_st = str(datetime.datetime.now())
    recon_S = time.perf_counter()

    mpi.submit_jobs_to_workers(commands, os.getcwd())
    recon_E = time.perf_counter()
    recon_T = recon_E - recon_S
    timer.Timer.timers.update({"reconstruct3d_splitcom" :{"elapsed_time": recon_T, "start_time": recon_st, "end_time": str(datetime.datetime.now())}})


    if mp["dose_weighting_enable"]:
        if os.path.exists("weights.txt"):
            pyp_frealign_plot_weights.plot_weights(name, "weights.txt", num_tilts, frames_per_tilt, mp["extract_box"], mp["scope_pixel"] * mp["extract_bin"])
        else:
            logfile = commands[0].splitlines()[0].split(" ")[-2].replace(" ","")
            if os.path.exists(logfile):
                with open(logfile) as f:
                    errors = f.read()
                    logger.warning(errors)
                    if "caught" in errors:
                        raise Exception(errors)

    # files that will be saved to /nfs
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder)

    files_to_keep = glob.glob(name + "*")
    for f in files_to_keep:
        symlink_relative(os.path.join(os.getcwd(), f), os.path.join(output_folder, f))

    # perform local merge
    if "cc" not in mp["refine_metric"].lower() and groups > 2:
        frealign.local_merge_reconstruction()

    # save the parameter files
    project_params.save_pyp_parameters(mp, output_folder)
    # project_params.save_fyp_parameters(fp, output_folder)

    # change back to working dir
    os.chdir(curr_dir)


@timer.Timer(
    "merge_check_err_and_resubmit", text="Check error and resubmit took: {}", logger=logger.info
)
def merge_check_err_and_resubmit(
    parameters, input_dir, micrographs, iteration=2
):
    """ Re-submit cspswarm for missing tilt-series due to slurm errors, missing files ... etc.  

    Parameters
    ----------
    parameters : dict
        PYP configuration parameters
    input_dir : str, optional
        The directory storing reconstruct3d dumped files, parfiles ...etc, by default "scratch"


    """
    # we're supposed to be in frealign/{input_dir} directory, go back to project dir
    cur_dir = os.getcwd()
    os.chdir(input_dir)
    os.chdir("../../")

    # get the number of tilt-series to be processed
    movies = [line.strip() for line in micrographs.keys()]
    num_movies = len(movies)

    # read list of processed movies
    # movies_done = [ line.strip() for line in open(orderfile, "r") ]

    # movies_resubmit = [ movie for movie in movies if movie not in movies_done ]

    movies_resubmit = []
    for movie in movies:
        if not os.path.exists(os.path.join(input_dir, "." + movie)):
            movies_resubmit.append(movie)

    if len(movies_resubmit) > 0:
        logger.warning("The following micrographs/tiltseries failed and will be re-submitted")
        fail_movies = ",".join(movies_resubmit)
        logger.info(f"{fail_movies}")
        
        # resubmit jobs (code borrowed from csp_split)

        os.chdir("swarm")

        if not os.path.exists(".cspswarm_retry"):
            slurm.launch_csp(micrograph_list=movies_resubmit,
                            parameters=parameters,
                            swarm_folder=Path().cwd(),
                            )
            message = "Successfully re-submitted failed jobs"

            # save flag to indicate failure
            Path(".csp_current_fail").touch()
            Path(".cspswarm_retry").touch()

        else:
            logger.error("Giving up retrying...")
            os.remove(".cspswarm_retry")
            message = "Stop re-submitting failed jobs"

        raise Exception(message)


    else:
        logger.info(
            "All series were successfully processed, start merging reconstructions"
        )
        os.chdir(cur_dir)


@timer.Timer(
    "run_mpi_reconstruction", text="Function with merge3d took: {}", logger=logger.info
)
def run_mpi_reconstruction(
    ref, pattern, dataset_name, iteration, mp, fp, input_dir, orderings,
):
    local_input_dir = os.getcwd()

    # check if there are any symlinks from previous iterations and remove them
    curr_files = [f for f in os.listdir(os.getcwd()) if dataset_name in f]
    [os.remove(f) for f in curr_files if os.path.islink(f)]

    metric = project_params.param(fp["refine_metric"], iteration)

    # performing symlinks and sorting
    _ = rename_csp_local_files(
        dataset_name, local_input_dir, orderings, pattern, metric
    )

    iteration = mp["refine_iter"]

    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(mp["class_num"], iteration))

    if not mp["refine_skip"] and classes > 1:
        new_preocc_pars = rename_par_local_files(
            dataset_name,
            local_input_dir,
            orderings,
            old_pattern=r"(\w+)_%s_pre.par" % pattern,
            new_pattern="{0}_{1:04d}_%s_pre.par" % pattern,
        )

    new_parfiles = rename_par_local_files(
        dataset_name,
        local_input_dir,
        orderings,
        old_pattern=r"(\w+)_%s.par" % pattern,
        new_pattern="{0}_{1:04d}_%s.par" % pattern,
    )

    new_used_parfiles = rename_par_local_files(
        dataset_name,
        local_input_dir,
        orderings,
        old_pattern=r"(\w+)_%s_used.par" % pattern,
        new_pattern="{0}_{1:04d}_%s_used.par" % pattern,
    )

    # merge the par files and save to maps folder
    merged_parfile = dataset_name + ".par"
    output_merge_parfile = "../maps/" + dataset_name + ".par"
    merge_used_parfile = dataset_name + "_used.par"

    output_parfile = os.path.join(os.path.dirname(input_dir), "maps", merged_parfile)
    if False and "frealignx" in metric: # Always saving NEW format
        frealignx = True
    else:
        frealignx = False

    if False and "frealignx" in project_params.param(fp["refine_metric"], iteration): # Always saving NEW format
        saved_frealignx = True
    else:
        saved_frealignx = False

    # save new iteration par without changing occ to frealign/maps
    with timer.Timer(
        "global_merge_parfile", text = "Merge all parfiles took: {}", logger=logger.info
    ):
        arguments = []

        if not mp["refine_skip"] and classes > 1:

            frealign_parfile.Parameters.merge_parameters(
                new_preocc_pars,
                output_merge_parfile,
                metric,
                update_film=True,
                parx=True,
                frealignx=saved_frealignx,
            )
            frealign_parfile.Parameters.merge_parameters(
                new_parfiles,
                merged_parfile,
                metric,
                update_film=True,
                parx=True,
                frealignx=frealignx,
            )

        else:

            frealign_parfile.Parameters.merge_parameters(
                new_parfiles,
                output_merge_parfile,
                metric,
                update_film=True,
                parx=True,
                frealignx=frealignx,
            )
            symlink_force(output_merge_parfile, merged_parfile)

        frealign_parfile.Parameters.merge_parameters(
            new_used_parfiles,
            merge_used_parfile,
            metric,
            update_film=True,
            parx=True,
            frealignx=frealignx,
        )


    # symlink_force(output_par_path, output_merge_parfile)
    # symlink_force(output_merge_used_parfile, output_merge_parfile)

    # create prs.png for output image
    phases_or_scores = "-scores"
    arg_scores = True
    arg_frealignx = False
    width = 137
    columns = 16

    # Plot only recognize NEW format
    if False and "frealignx" in project_params.param(fp["refine_metric"], iteration):
        phases_or_scores = "-frealignx"
        arg_scores = False
        arg_frealignx = True
    """
    if fp["dose_weighting_enable"]:
        phases_or_scores = "-frealignx"
        arg_scores = False
        arg_frealignx = True
    """
    with timer.Timer(
        "plot_used_png", text = "Plot used particles pngs took: {}", logger=logger.info
    ):

        if float(project_params.param(fp["reconstruct_cutoff"], iteration)) >= 0:
            # creat bild file from used.par file
            mpi_funcs, mpi_args = [], []

            mpi_funcs.append(plot.par2bild)
            bild_output = os.path.join(os.path.dirname(input_dir),"maps",f"{dataset_name}.bild")
            mpi_args.append( [( merge_used_parfile, bild_output, fp)] )

            # plot using all particles
            arg_input = f"{dataset_name}.par"
            arg_angle_groups = 25
            arg_defocus_groups = 25
            arg_dump = False
            mpi_funcs.append(plot.generate_plots)
            mpi_args.append([(
                arg_input,
                arg_angle_groups,
                arg_defocus_groups,
                arg_scores,
                arg_frealignx,
                arg_dump,
            )])

            # plot using used particles
            arg_input = f"{dataset_name}_used.par"
            arg_angle_groups = 25
            arg_defocus_groups = 25
            arg_dump = False
            mpi_funcs.append(plot.generate_plots)
            mpi_args.append([(
                arg_input,
                arg_angle_groups,
                arg_defocus_groups,
                arg_scores,
                arg_frealignx,
                arg_dump,
            )])

            mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=fp["slurm_verbose"])
            # transfer files to maps directory
            for file in glob.glob(dataset_name + "*_prs.png"):
                shutil.move(file, "../maps")

    # combine 2D plots from used particles and global statistics for histograms
    # read saved pickle files 
    with open(f"{dataset_name}_temp.pkl", 'rb') as f1:
        plot_outputs = pickle.load(f1)
    with open(f"{dataset_name}_meta_temp.pkl", 'rb') as f2:
        metadata = pickle.load(f2)
    with open(f"{dataset_name}_used_temp.pkl", 'rb') as f3:
        plot_outputs_used = pickle.load(f3)
    with open(f"{dataset_name}_used_meta_temp.pkl", 'rb') as f4:
        metadata_used = pickle.load(f4)

    consolidated_plot_outputs = plot_outputs.copy()
    consolidated_plot_outputs["def_rot_histogram"] = plot_outputs_used["def_rot_histogram"]
    consolidated_plot_outputs["def_rot_scores"] = plot_outputs_used["def_rot_scores"]

    consolidated_metadata = metadata.copy()
    consolidated_metadata["particles_used"] = metadata_used["particles_used"]
    consolidated_metadata["phase_residual"] = metadata_used["phase_residual"]

    pardata = frealign_parfile.Parameters.from_file(f"{dataset_name}.par").data
    samples = np.array(frealign.get_phase_residuals(pardata,f"{dataset_name}.par",fp,2))
    threshold = 1.075 * statistics.optimal_threshold(
        samples=samples, criteria="optimal"
    )
    consolidated_metadata["phase_residual"] = threshold

    # perform final merge
    # hack os.environ['PYP_SCRATCH']
    local_scratch = os.environ["PYP_SCRATCH"]
    os.environ["PYP_SCRATCH"] = local_input_dir
    
    logger.info("Merging intermediate reconstructions")
    frealign.merge_reconstructions(mp, iteration, ref)

    os.environ["PYP_SCRATCH"] = local_scratch

    with timer.Timer(
        "output reconstruction results", text = "Final output reconstructions took: {}", logger=logger.info
    ):
        # copy log and png files

        # write compressed file to maps directory
        output_folder = os.path.join(os.path.dirname(input_dir), "maps")
        if fp["refine_parfile_compress"]:
            compressed_file = os.path.join(
                output_folder, Path(merged_parfile).name + ".bz2"
            )
            saved_path = os.getcwd()
            os.chdir(Path(output_merge_parfile).parent)
            frealign_parfile.Parameters.compress_parameter_file(
                Path(output_merge_parfile).name, compressed_file, fp["slurm_merge_tasks"]
            )
            os.chdir(saved_path)
        else:
            shutil.copy2(output_merge_parfile, output_folder)

        # append merge log
        reclogfile = "../log/%s_mreconst.log" % (dataset_name)
        outputlogfile = os.path.join(os.path.dirname(input_dir), "log/%s_mreconst.log" % (dataset_name))
        with open(outputlogfile, "a") as fw:
            with open(reclogfile) as fr:
                fw.write(fr.read())

        # save statistics file
        stats_file_name = os.path.join(output_folder, f"{dataset_name}_statistics.txt")
        res_file_name = f"{dataset_name}.res"

        # smooth part FSC curves
        """
        if project_params.param(mp["refine_metric"], iteration) == "new" and os.path.exists(stats_file_name):

            plot_name = os.path.join(
                output_folder, "%s_%02d_snr.png" % (dataset_name, iteration)
            )

            postprocess.smooth_part_fsc(str(stats_file_name), plot_name)
        """

        if os.path.exists(res_file_name):
            com = (
                """grep -A 10000 "C  NO.  RESOL  RING RAD" {0}""".format(res_file_name)
                + """ | grep -v RESOL | grep -v Average | grep -v Date | grep C | awk '{if ($2 != "") printf "%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f\\n", $2, $3, $4, $6, $7, $8, $9}' > """
                + str(stats_file_name)
            )
            local_run.run_shell_command(com, verbose=False)

        elif os.path.exists(f"{dataset_name}_statistics.txt"):
            shutil.copy2(f"{dataset_name}_statistics.txt", stats_file_name)


        # save what is worth to original frealing/maps
        for file in (
            ["../maps/" + dataset_name + "_fyp.webp", "../maps/" + dataset_name + "_map.webp", "../maps/" + dataset_name + ".mrc", "../maps/" + dataset_name + "_raw.mrc"]
            + glob.glob(f"../maps/*_r{ref:02d}_???.txt")
            + glob.glob("../maps/*half*.mrc")
            + glob.glob("../maps/*crop.mrc")
            + glob.glob("../maps/*scores.svgz")
        ):
            if os.path.exists(file):
                shutil.copy2(file, output_folder)

        if True:
            fsc_file = os.path.join("../maps", fp["refine_dataset"] + "_r%02d_fsc.txt" % ref)
            FSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)

            # send reconstruction to website
            save_reconstruction_to_website(
                dataset_name, FSCs, consolidated_plot_outputs, consolidated_metadata
            )

@timer.Timer(
    "run_merge", text="Total time elapsed: {}", logger=logger.info
)
def run_merge(input_dir="scratch", ordering_file="ordering.txt"):

    # we are originally in the project directory
    project_dir = os.getcwd()

    # now switch to frealign/scratch directory where all the .bz2 files are
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(input_dir)

    # load pyp params from main folder
    mp = project_params.load_pyp_parameters("../..")
    fp = mp

    if not (fp["class_num"] > 1 and fp["refine_iter"] > 2):
        # cspswarm -> cspmerge
        csp_class_merge(class_index=1, input_dir=input_dir)
    else: 
        # cspswarm -> classmerge -> cspmerge
        if not classmerge_succeed(fp):
            raise Exception("One or more classmerge job(s) failed")

    iteration = fp["refine_iter"]

    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(fp["class_num"], iteration))

    fp["refine_dataset"] = mp["data_set"]
    metric = project_params.param(mp["refine_metric"], iteration).lower()

    # initialize directory structure to replicate frealign folders
    local_scratch = os.environ["PYP_SCRATCH"]
    local_frealign = os.path.join(local_scratch, "frealign")
    local_frealign_scratch = os.path.join(local_frealign, "scratch")
    for dir in ["maps", "scratch", "log"]:
        os.makedirs(os.path.join(local_frealign, dir), exist_ok=True)
    # copy frealign metadata
    for file in glob.glob(os.path.join(input_dir, "../maps/*.txt")):
        shutil.copy2(file, os.path.join(local_frealign, "maps"))

    # copy pyp metadata to scracth space
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".micrographs"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".films"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], ".pyp_config.toml"), local_frealign_scratch
    )

    micrographs = {}
    all_micrographs_file = "../../" + mp["data_set"] + ".films"
    with open(all_micrographs_file) as f:
        index = 0
        for line in f.readlines():
            micrographs[line.strip()] = index
            index += 1

    with timer.Timer(
        "plot fsc and clean par", text = "Plot FSC and producing clean par file took: {}", logger=logger.info
    ):
        # collate FSC curves from all references in one plot
        if classes > 1 and not Web.exists:

            metadata = {}
            logger.info("Creating plots for visualizing classes OCCs and FSCs")
            # plot class statistics
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, figsize=(8, 8))
            ranking = np.zeros([classes])
            for ref in range(classes):
                fsc_file = "../maps/" + fp["refine_dataset"] + "_r%02d_fsc.txt" % (ref + 1)
                FSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)

                if ref == 0:
                    metadata["frequency"] = FSCs[:, 0].tolist()
                metadata["fsc_%02d" % ( ref + 1 ) ] = FSCs[:, iteration - 1].tolist()

                ax[0].plot(
                    1 / FSCs[:, 0], FSCs[:, iteration - 1], label="r%02d" % (ref + 1),
                )
                ranking[ref] = FSCs[:, iteration - 1].mean()
                par_file = (
                    fp["refine_dataset"]
                    + "_r%02d_%02d.par" % (ref + 1, iteration)
                )
                
                parfile = Path().cwd().parent / "maps" / par_file
                compressed_parfile = Path().cwd().parent / "maps" / str(par_file+".bz2")

                if compressed_parfile.exists():
                    parfile = frealign_parfile.Parameters.decompress_parameter_file(str(compressed_parfile), mp["slurm_merge_tasks"])
                elif not parfile.exists():
                    assert Exception(f"{parfile} does not exist. Please check")

                input = frealign_parfile.Parameters.from_file(parfile).data

                if input[0].shape[0] > 45:
                    sortedocc = np.sort(input[:, 12])[::-1]
                else:
                    sortedocc = np.sort(input[:, 11])[::-1]
                ax[1].plot(sortedocc, label="r%02d" % (ref + 1))
                metadata["occ_%02d" % ( ref + 1 ) ] = sortedocc.tolist()
            ax[0].legend(loc="upper right", shadow=True)
            ax[0].set_ylim((-0.1, 1.05))
            ax[0].set_xlim((1 / FSCs[0, 0], 1 / FSCs[-1, 0]))
            dataset = fp["refine_dataset"] + "_%02d" % iteration
            ax[0].set_title("%s" % dataset, fontsize=12)
            ax[0].set_xlabel("Frequency (1/A)")
            ax[0].set_ylabel("FSC")
            ax[1].legend(loc="upper right", shadow=True)
            ax[1].set_xlim(0, input.shape[0] - 1)
            ax[1].set_xlabel("Particle Index")
            ax[1].set_ylabel("Occupancy (%)")
            plt.savefig(os.path.join(input_dir, "../maps/%s_classes.png" % dataset))
            plt.close()

            with open( "../maps/%s_classes.json" % dataset, 'w' ) as f:
                json.dump( metadata, f )

        else:
            if len(glob.glob(f"{local_frealign_scratch}/*_scores.png")) > 0:
                command = "montage {0}/*_{1}_scores.png -geometry +0+0 {3}/../maps/{2}_{1}_scores.png".format(
                    local_frealign_scratch,
                    "r%02d_%02d" % (1, iteration),
                    fp["refine_dataset"],
                    input_dir,
                )
                local_run.run_shell_command(command, verbose=mp["slurm_verbose"])

    # remove the directory
    shutil.rmtree(input_dir)

    # go back to project directory
    os.chdir(project_dir)
    # remove decompressed par
    for ref in range(classes):
        name = "%s_r%02d" % (mp["data_set"], ref + 1)
        decompressed_par = "frealign/maps/" + name + "_%02d.par" % (iteration -1)
        if os.path.exists(decompressed_par) and fp["refine_parfile_compress"]:
            os.remove(decompressed_par)
    # denoising using sidesplitter with half maps
    if False and classes > 1 and "refine_maskth" in mp and os.path.exists(project_params.resolve_path(project_params.param(mp["refine_maskth"], iteration))): # default denoising but could be parameterized 
        logger.info("Denoising using sidesplitter")
        for ref in range(classes):
            name = "%s_r%02d" % (mp["data_set"], ref + 1)
            halfmap1 = "frealign/maps/" + name + "_half1.mrc"
            halfmap2 = halfmap1.replace("half1", "half2")
            mask = project_params.resolve_path(project_params.param(mp["refine_maskth"], iteration))
            sidesplitter = "sidesplitter/SIDESPLITTER/build/sidesplitter"
            command = sidesplitter + " " + "--v1 {0} --v2 {1} --mask {2}".format(halfmap1, halfmap2, mask)
            local_run.run_shell_command(command)
            newhalf1 = halfmap1.replace(".mrc", "_sidesplitter.mrc")
            newhalf2 = halfmap2.replace(".mrc", "_sidesplitter.mrc")
            newmap = "frealign/maps/" + name + "_%02d.mrc" % iteration
            os.rename(newmap, newmap.replace(".mrc", "_ori.mrc"))
            command =   "{0}/bin/clip add {1} {2} {3}".format(get_imod_path(), newhalf1, newhalf2, newmap)
            local_run.run_shell_command(command)

    # update iteration number
    maxiter = fp["refine_maxiter"]
    fp["refine_iter"] = iteration + 1
    fp["refine_dataset"] = mp["data_set"]

    if "refine_skip" in fp.keys() and fp["refine_skip"] and fp["class_num"] > 1:
        fp["refine_skip"] = False

    fp["slurm_merge_only"] = False

    project_params.save_parameters(fp, ".")

    # launch next iteration if needed
    if iteration < maxiter:
        logger.info("Now launching iteration " + str(iteration + 1))
        csp_split(fp, iteration + 1)


def rename_csp_local_files(dataset_name, input_dir, ordering, pattern, metric):
    import re

    curr_dir = os.getcwd()
    os.chdir(input_dir)
    files = os.listdir(os.getcwd())

    if "new" in metric or "frealignx" in metric:
        # p = re.compile(r"(\w+)_r01_02_(map\d)_n1.mrc")
        p = re.compile(r"(\w+)_%s_map1_(n\d+).mrc" % pattern)

        new_files = []
        order = 1

        for f in files:
            if p.match(f):
                match = p.match(f)
                old_name = match.group()
                old_name2 = old_name.replace("map1", "map2")
                # job_name, map_no, dump_num = match.groups()
                # job_name, dump_num = match.groups()

                # order = next((i for i, v in enumerate(ordering, 1) if v[0] == job_name), -1)
                # new_name = "{0}_{1}_n{2}.mrc".format(dataset_name, map_no, order)
                new_name = "{0}_map1_n{1}.mrc".format(dataset_name, order)
                new_name2 = "{0}_map2_n{1}.mrc".format(dataset_name, order)

                # logger.info("symlinking from {} to {}".format(old_name, new_name))
                os.rename(old_name, new_name)
                os.rename(old_name2, new_name2)

                new_files.append(new_name)
                new_files.append(new_name2)

                order += 1
    elif "cc" in metric:
        p = re.compile(r"(\w+)_%s_(\w+).mrc" % pattern)

        new_files = []
        order = 1

        for f in files:
            if p.match(f):
                match = p.match(f)
                old_name = match.group()
                job_name, dump_num = match.groups()

                new_name = "{0}_n{1}.mrc".format(dataset_name, order)

                # logger.info("symlinking from {} to {}".format(old_name, new_name))
                symlink_force(old_name, new_name)

                new_files.append(new_name)

                order += 1

    os.chdir(curr_dir)
    return new_files


def rename_par_local_files(
    dataset_name,
    input_dir,
    ordering,
    old_pattern=r"(\w+)_r01_02.par",
    new_pattern="{0}_{1:04d}_r01_02.par",
):
    import re

    curr_dir = os.getcwd()
    os.chdir(input_dir)
    files = os.listdir(os.getcwd())
    p = re.compile(old_pattern)

    new_files = []
    for f in files:
        if p.match(f):
            match = p.match(f)
            old_name = match.group()
            job_name = match.groups()[0]

            # order = next((i for i, v in enumerate(ordering, 1) if v[0] == job_name), -1)
            order = [i for i in range(len(ordering)) if ordering[i] == job_name][0]
            new_name = new_pattern.format(dataset_name, order)

            # logger.info("symlinking from {} to {}".format(old_name, new_name))
            symlink_force(old_name, new_name)
            new_files.append(new_name)

    os.chdir(curr_dir)
    new_files.sort()
    return new_files


@timer.Timer(
    "live_decompress_and_merge", text="Live decompress and merge took: {}", logger=logger.info
)
def live_decompress_and_merge(class_index, input_dir, parameters, micrographs, all_jobs, merge=True):
    """ Perform live bz2 file decompression and intermediate merging once single cspswarm completes.  

    Args:
        input_dir (str): Directory where cspswarm stores its compressed files
        parameters (dict): PYP parameters
        micrographs (dict): Micorgraphs derived from .films file
        all_jobs (list): job id list that will be used for later parfile merging 
        merge (boolean): perform intermediate merge or not (now only support metrics other than cc3m, cclin)

    Raises:
        Exception: Check if number of dumpfiles is correct
    """

    # timer will be reset if we get a batch of files
    # set the timer to slurm_merge_walltime - 10 min 
    TIMEOUT =  slurm.get_total_seconds(parameters["slurm_walltime"]) - 10 * 60 
    INTERVAL = 10           # 10 s 
    start_time = time.time()

    path_to_logs = Path(input_dir).parent.parent / "log"
    iteration = parameters["refine_iter"]
    compressed = parameters["refine_parfile_compress"]
    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(parameters["class_num"], iteration))
    if compressed:
        num_dumpfiles_per_bundle = 1 
        num_bundle = math.ceil(len(micrographs.keys()) / parameters["slurm_bundle_size"])
        num_bz2_per_bundle = 1 # classes
        total_num_bz2 = num_bz2_per_bundle * num_bundle

        decompression_threshold = min(parameters["slurm_merge_tasks"] - 1, total_num_bz2) 

        decompressed = set()
        decompression_queue = set()
        processed_micrographs = set()
        class_to_merge = [False for _ in range(classes)]
        dumpfiles_count_class = [1 for _ in range(classes)]
        processed = 0
        succeed = False
        logger.info("Start live-merging intermediate reconstructions")
        while time.time() - start_time < TIMEOUT:

            arguments = []
            finished_micrographs = glob.glob(os.path.join(input_dir, ".*"))

            compressed_files = glob.glob(os.path.join(input_dir, f"*_r{class_index:02d}_*.bz2"))

            # find un-processed bz2 files
            for f in compressed_files:
                filename = os.path.basename(f)
                if filename not in decompressed:
                    decompression_queue.add(filename)

            # decompress them if we find enough files
            if len(decompression_queue) >= decompression_threshold:

                for filename in decompression_queue:

                    arguments.append((os.path.join(input_dir, filename), 1,))
                    decompressed.add(filename)

                    class_ind = int(filename.split("_r")[-1].split("_")[0]) - 1
                    class_to_merge[class_ind] = True

                    jobid = filename.split("_r")[0].split("_")
                    
                    cspswarm_jobid = int(jobid[0])
                    cspswarm_arrid = int(jobid[1])
                    [all_jobs.append([cspswarm_jobid, cspswarm_arrid]) for _ in range(num_dumpfiles_per_bundle)]

                decompression_threshold = min(parameters["slurm_merge_tasks"] - 1, total_num_bz2 - len(decompressed))
                decompression_queue.clear()

            if len(arguments) > 0:
                mpi.submit_function_to_workers(
                    frealign_parfile.Parameters.decompress_file, arguments, verbose=parameters["slurm_verbose"]
                )
                # reset if we get a batch
                start_time = time.time()

                # if they're successfully decompressed we think they're complete
                [processed_micrographs.add(micrograph.split(".")[-1]) for micrograph in finished_micrographs]
                processed += len(arguments)

            if merge:
                # perform intermediate merge on files we just decompressed
                class_ind = class_index - 1
                # only merge decompressed intermediate reconstructions
                if class_to_merge[class_ind]:
                    pattern = "r%02d_%02d" % (class_index, iteration)
                    num_dumpfiles = frealign.local_merge_reconstruction(name=pattern)
                    class_to_merge[class_ind] = False
                    dumpfiles_count_class[class_ind] += num_dumpfiles - 1   # the 1 is output dumpfile

            # done processing all micrographs
            # if len(set(micrographs.keys()) - processed_micrographs) == 0:

            if total_num_bz2 - processed == 0:
                # check the number of dumpfiles is correct
                if merge:
                    assert (dumpfiles_count_class[class_index-1] == num_bundle * num_dumpfiles_per_bundle), f"{dumpfiles_count_class[class_index-1]} dumpfiles in class {class_index} is not {num_bundle * num_dumpfiles_per_bundle}"
                succeed = True
                break

            # check if there's error from logs. If yes, we stop the merge job
            if csp_has_error(path_to_logs, micrographs):
                return False

            time.sleep(INTERVAL)
        logger.info("Done live-merging intermediate reconstructions")

    else:
        pwd = os.getcwd()
        os.chdir(input_dir)
        num_dumpfiles_per_bundle = 1
        num_bundle = math.ceil(len(micrographs.keys()) / parameters["slurm_bundle_size"])
        total_num_dump_perclass =num_dumpfiles_per_bundle * num_bundle

        processed_micrographs = set()
        class_to_merge = [True for _ in range(classes)]
        dumpfiles_count_class = [1 for _ in range(classes)]
        processed = 0
        succeed = False
        while time.time() - start_time < TIMEOUT:
            if merge:
                # only merge decompressed intermediate reconstructions
                if class_to_merge[class_index-1]:
                    pattern = "r%02d_%02d" % (class_index, iteration)
                    num_dumpfiles = frealign.local_merge_reconstruction(name=pattern)
                    dumpfiles_count_class[class_index-1] += num_dumpfiles - 1   # the 1 is output dumpfile
            else:
                # check all the intermediate files
                if class_to_merge[class_index-1]:
                    pattern = "r%02d_%02d.par" % (class_index, iteration)
                    dump_par_num = len(glob.glob("*" + pattern))
                    dumpfiles_count_class[class_index-1] = dump_par_num

            if dumpfiles_count_class[class_index-1] == total_num_dump_perclass:
                class_to_merge[class_index-1] = False
            else:
                class_to_merge[class_index-1] = True

            if not any(class_to_merge):
                succeed = True
                os.chdir(pwd)
                break

            # check if there's error from logs. If yes, we stop the merge job
            if csp_has_error(path_to_logs, micrographs):
                return False

            time.sleep(INTERVAL)

    if not succeed:
        logger.warning("Job reached walltime. Attempting to resubmit remaining jobs but you may need to increase the walltime (merge task)")

        # result is incomplete after TIMEOUT -> need to resubmit failed cspswarm jobs
        if class_index == 1 and len(set(micrographs.keys()) - processed_micrographs) > 0:
            merge_check_err_and_resubmit(parameters, input_dir, micrographs, int(parameters["refine_iter"]))
        return False
    else:
        logger.info(f"Decompression of all micrographs/tilt-series is done, start merging reconstruction and parfiles")

    return True

def csp_has_error(path_to_logs: Path, micrographs: dict) -> bool:

    has_error = False
    ERROR_KEYWORDS = ["PYP (cspswarm) failed"]

    for micrograph in micrographs.keys():
        micrograph_log = Path(path_to_logs, f"{micrograph}_csp.log")
        if micrograph_log.exists():
            # use "grep" to check if log files contain any error message
            command = "grep -E %s %s" % ("'" + "|".join(ERROR_KEYWORDS) + "'", str(micrograph_log))
            [output, error] = local_run.run_shell_command(command, verbose=False)

            if len(output) > 0:
                logger.error(f"{micrograph} fails. Stopping the merge job.")
                logger.error(output)
                has_error = True
                break

    return has_error



def csp_class_merge(class_index: int, input_dir="scratch", ordering_file="ordering.txt"):

    # we are originally in the project directory
    project_dir = os.getcwd()

    # now switch to frealign/scratch directory where all the .bz2 files are
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(input_dir)

    # load pyp params from main folder
    mp = project_params.load_pyp_parameters("../../")
    fp = mp

    iteration = fp["refine_iter"]

    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(fp["class_num"], iteration))

    fp["refine_dataset"] = mp["data_set"]
    metric = project_params.param(mp["refine_metric"], iteration).lower()

    # initialize directory structure to replicate frealign folders
    local_scratch = os.environ["PYP_SCRATCH"]
    local_frealign = os.path.join(local_scratch, "frealign")
    local_frealign_scratch = os.path.join(local_frealign, "scratch")
    for dir in ["maps", "scratch", "log"]:
        os.makedirs(os.path.join(local_frealign, dir), exist_ok=True)
    # copy frealign metadata
    for file in glob.glob(os.path.join(input_dir, "../maps/*.txt")):
        shutil.copy2(file, os.path.join(local_frealign, "maps"))

    # copy pyp metadata to scracth space
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".micrographs"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".films"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], ".pyp_config.toml"), local_frealign_scratch
    )

    micrographs = {}
    all_micrographs_file = "../../" + mp["data_set"] + ".films"
    with open(all_micrographs_file) as f:
        index = 0
        for line in f.readlines():
            micrographs[line.strip()] = index
            index += 1

    total_micrographs = len(micrographs.keys())

    (number_of_reconstructions_per_micrograph, _) = get_number_of_intermediate_reconstructions(mp)

    # each class should have (num_dumpfiles_per_bundle * num_bundle) dumpfiles
    num_dumpfiles_per_bundle = 1 if "cc" not in metric else number_of_reconstructions_per_micrograph - 1
    num_bundle = math.ceil(total_micrographs / mp["slurm_bundle_size"])

    orderings = ["" for _ in range(num_dumpfiles_per_bundle * num_bundle)]

    # decompress all intermediate dump files in local scratch
    os.chdir(local_frealign_scratch)

    all_jobs = []

    collect_all_cspswarm = live_decompress_and_merge(class_index, input_dir, mp, micrographs, all_jobs, merge=(
        (("spr" not in mp["data_mode"] or "local" in mp["extract_fmt"]) and iteration > 2 )
        )
    )

    if not collect_all_cspswarm:
        raise Exception(f"One or more job(s) failed")

    if not fp["refine_parfile_compress"]:
        # copy the mrc files and parfiles to scratch
        with timer.Timer("Copy mrc, par to scratch", text = "Copy file to merge took: {}", logger=logger.info):
            for file in glob.glob(input_dir + "/*mrc") + glob.glob(input_dir + "/*par"):
                symlink_relative(file, os.path.join(local_frealign_scratch, os.path.basename(file)))

    if "cc" not in metric and fp["refine_parfile_compress"]:
        all_jobs = np.array(all_jobs)
    else:
        all_jobs = np.atleast_2d(np.array(
            [
                i.split("_r01_")[0].split("_")
                for i in glob.glob("*_r01_%02d.par" % iteration)
            ],
            dtype=int,
        ))

    # identify the different job IDs
    slurm_job_ids = sorted(np.unique(all_jobs[:, 0]))
    for job in slurm_job_ids:
        # figure out number of empty slots
        zero_indexes = [i for i in range(len(orderings)) if orderings[i] == ""]
        # get the job array IDs for this job
        array_ids = sorted(all_jobs[all_jobs[:, 0] == job][:, 1])
        if len(zero_indexes) >= len(array_ids):
            for id in array_ids:
                orderings[zero_indexes[id - 1]] = str(job) + "_" + str(id)
        else:
            message = "Number of missing jobs ({}) does not match the number of missing movies ({}).".format(
                len(zero_indexes), len(array_ids)
            )
            raise Exception(message)


    with timer.Timer(
        "Parallel run all reconstruction", text = "Run parallel reconstruction took: {}", logger=logger.info
    ):
        ref = class_index
        pattern = "r%02d_%02d" % (ref, iteration)
        dataset_name = fp["refine_dataset"] + "_%s" % pattern

        run_mpi_reconstruction(ref, pattern, dataset_name, iteration, mp, fp, input_dir, orderings)


    os.chdir(project_dir)


def classmerge_succeed(parameters: dict) -> bool: 
    """classmerge_succeed Check if classmerge jobs all succeed. If not, either terminate current cspmerge or relaunch classmerge/cspmerge

    Parameters
    ----------
    parameters : dict
        PYP parameters 

    Returns
    -------
    bool
        Succeed or not
    """
    # see if classmerge resubmits cspwarm by its logs
    # currently in frealign/scratch
    frealign_maps = Path().cwd().parent / "maps"
    swarm_folder = Path().cwd().parent.parent / "swarm"
    cspswarm_fail_tag = swarm_folder / ".csp_current_fail"

    dataset = parameters["data_set"]
    iteration = parameters["refine_iter"]
    num_classes = parameters["class_num"] if parameters["refine_iter"] > 2 else 1
    maps_classes = [f"{dataset}_r{class_idx+1:02d}_{iteration:02d}.mrc" for class_idx in range(num_classes)]
 
    TIMEOUT =  slurm.get_total_seconds(parameters["slurm_merge_walltime"]) - 10 * 60 
    INTERVAL = 10           # 10 s 
    start_time = time.time()

    while time.time() - start_time < TIMEOUT:
        
        if cspswarm_fail_tag.exists():
            # partial cspswarm(s) & classmerge & cspmerge are all resubmitted by classmerge (one or more cspswarm(s) failed)
            # so terminate this cspmerge directly
            os.remove(cspswarm_fail_tag)
            return False

        classmerge_all_complete = True
        for map in maps_classes:
            if not (frealign_maps / map).exists():  
                classmerge_all_complete = False 

        if classmerge_all_complete:
            return True       

        time.sleep(INTERVAL)

    # part of the classmerge jobs failed, resubmit classmerge & cspmerge (w/o cspswarm)
    parameters["slurm_merge_only"] = True
    
    slurm.launch_csp(micrograph_list=[],
                    parameters=parameters,
                    swarm_folder=swarm_folder,
                    )
    return False


