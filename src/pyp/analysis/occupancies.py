import math
import os
import shutil
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from pyp.analysis import statistics
from pyp.inout.metadata import frealign_parfile, cistem_star_file
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_frealign_paths
from pyp.utils import get_relative_path, symlink_force
from pyp.utils import timer
from pyp.inout.metadata import get_particles_tilt_index
relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def get_occupancy_columns():

    prj_obj = cistem_star_file.Parameters()

    film_col = prj_obj.get_index_of_column(cistem_star_file.IMAGE_IS_ACTIVE)
    occ_col = prj_obj.get_index_of_column(cistem_star_file.OCCUPANCY)
    logp_col = prj_obj.get_index_of_column(cistem_star_file.LOGP)
    sigma_col = prj_obj.get_index_of_column(cistem_star_file.SIGMA)
    score_col = prj_obj.get_index_of_column(cistem_star_file.SCORE)
    ptl_col = prj_obj.get_index_of_column(cistem_star_file.PIND)
    scanord_col = prj_obj.get_index_of_column(cistem_star_file.TIND)

    return film_col, occ_col, logp_col, sigma_col, score_col, ptl_col, scanord_col


def occupancies(fparameters, iteration, nclass, path="."):
    """Compute occupancies based on LogP values using calc_occ.exe."""
    dataset = fparameters["refine_dataset"]

    calc_occ = "calc_oc.in"
    with open(calc_occ, "w") as f:
        f.write("%d\n1.0\n" % nclass)
        for i in range(2):
            for res in range(nclass):
                f.write("%s/%s_r%02d_%02d.par\n" % (path, dataset, res + 1, iteration))

    metric = project_params.param(fparameters["refine_metric"], iteration)
    frealign_paths = get_frealign_paths()

    if "cc" in metric.lower():
        command = "{0}/bin/calc_occ.exe < {1}".format(
            frealign_paths["cc3m"], calc_occ, path, iteration
        )  # >& {2}/calc_occ_{3}.log
    elif "new" in metric.lower():
        command = "{0}/bin/calc_occ.exe < {1}".format(
            frealign_paths["new"], calc_occ, path, iteration
        )  # >& {2}/calc_occ_{3}.log
    elif "frealignx" in metric.lower():
        command = "{0}/calc_occ.exe < {1}".format(
            frealign_paths["frealignx"], calc_occ, path, iteration
        )  # >& {2}/calc_occ_{3}.log
    [output, error] = local_run.run_shell_command(command)
    if "error" in output.lower():
        logger.error("In calculating occupancies:")
        print(output)
        sys.exit()


@timer.Timer(
    "occ_function", text="Occupancy calculations took: {}", logger=logger.info
)
def occupancy_extended(parameters, dataset, nclasses, image_list=None, parameter_file_folders=".", local=False):
    """Compute occupancies based on LogP values with extended par file."""

    # load the par files from all the classes and calculate average occupancies
    parx = []
    all_extended_list = []
    class_average_occ = []
    occupancy_change_multiplier = 1
 
    (
        film_col,
        occ_col,
        logp_col,
        sigma_col,
        score_col,
        ptl_col,
        scanord_col, 
    ) = get_occupancy_columns()

    # info needed only
    newfilm_col = 0
    newocc_col = 1
    newlogp_col = 2
    newsigma_col = 3
    newscore_col = 4
    newptlind_col = 5
    newscanord_col = 6
    
    iteration = parameters["refine_iter"] 
    if not local:
        with timer.Timer(
        "read par for occ", text = "Reading occupancies took: {}", logger=logger.info
        ):
            for k in range(nclasses):
                class_k = k + 1
                decompressed_parameter_file_folder = os.path.join(parameter_file_folders, dataset + "_r%02d_%02d" % (class_k, iteration - 1)) 
                binary_list = [os.path.join(decompressed_parameter_file_folder, image + "_r%02d.cistem" % class_k ) for image in image_list]
                park_data, par_extk_dict, _ = cistem_star_file.merge_all_binary_with_filmid(binary_list, read_extend=True)                
                select_data = park_data[:, [film_col, occ_col, logp_col, sigma_col, score_col, ptl_col, scanord_col]]

                all_extended_list.append(par_extk_dict)
                parx.append(select_data)
                all_occ = parx[k][:, newocc_col].ravel()
                class_average_occ.append(mean(all_occ))

                if "tomo" in parameters["data_mode"] and k == 0:
                    
                    index = get_particles_tilt_index(select_data, ptl_col=newptlind_col)

                    film_index = get_particles_tilt_index(select_data, ptl_col = newfilm_col)

            parx_3d = np.array(parx)

    else:
        with open("project_dir.txt") as f:
            project_dir = f.readline().strip("\n")

        for k in range(nclasses):
            
            global_dataset_name = parameters["data_set"]
            decompressed_parameter_file_folder = os.path.join(project_dir, "frealign", "maps", global_dataset_name + "_r%02d_%02d" % (k + 1, iteration - 1)) 
            remote_par_stat = os.path.join(decompressed_parameter_file_folder, global_dataset_name + "_r%02d_stat.cistem" % (k + 1))
            stat = cistem_star_file.Parameters.from_file(remote_par_stat).get_data()
            occmean = stat[0, occ_col]
            class_average_occ.append(occmean)

            # reload local par files
            par_binary = os.path.join(parameter_file_folders, dataset + "_r%02d.cistem" % (k + 1) )
            park_data = cistem_star_file.Parameters.from_file(par_binary)
            select_data = park_data.get_data()[:, [film_col, occ_col, logp_col, sigma_col, score_col, ptl_col, scanord_col]]
            parx.append(select_data)
            all_extended_list.append(park_data.get_extended_data().get_tilts())

            if "tomo" in parameters["data_mode"] and k == 0:
                
                index = get_particles_tilt_index(select_data, ptl_col=newptlind_col)
        
        parx_3d = np.array(parx)
    
    if "tomo" in parameters["data_mode"]:

        if parameters["refine_score_weighting"]: # score weights
            logger.info("Weighting logp with score averages")
            scoreavg_tilt = statistics.get_class_score_weight(parx_3d, newscore_col, newscanord_col)
            
            for k in range(0, nclasses):
                for i in index:
                    per_particle_scoreweight(parx_3d[k, :,:], newlogp_col, newscanord_col, scoreavg_tilt, i)

        else: # tilt gaussian weights
            logger.info("Weighting LogP with tilt distribution")
            for k in range(0, nclasses):
                for i in index:
                    per_particle_tiltweight(parx_3d[k, :,:], all_extended_list[k], newlogp_col, i)

    else:
        logger.info("Not using tilt weighting to change OCC")

    # recalculate occ from logp
    all_frame, col_num = parx[0].shape

    ptl_logp_array = parx_3d[:, :, newlogp_col]
    # ptl_occ_array = parx_3d[:, :, newocc_col]
    ptl_sigma_array = parx_3d[:, :, newsigma_col]
    maxlogp_array = np.amax(ptl_logp_array, axis=0)
    sum_pp_array = np.full((all_frame,), 0.0)

    for k in range(nclasses):
        delta_logp_array = maxlogp_array - ptl_logp_array[k, :]
        logp_mask_array = delta_logp_array < 10
        # delta_logp_array = np.where(logp_mask_array, delta_logp_array, 0)
        delta_array = np.negative(delta_logp_array)
        sum_pp_array += np.where(
            logp_mask_array, np.exp(delta_array) * class_average_occ[k], 0
        )

    average_sigma = np.full((all_frame,), 0.0)
    new_occ = np.full((nclasses, all_frame), 0.0)
    for k in range(nclasses):
        delta_logp_array = maxlogp_array - ptl_logp_array[k, :]
        logp_mask_array = delta_logp_array < 10
        # delta_logp_array = np.where(logp_mask_array, delta_logp_array, 0)
        delta_array = np.negative(delta_logp_array)
        new_occ[k, :] += np.where(
            logp_mask_array,
            np.exp(delta_array) * class_average_occ[k] * 100 / sum_pp_array,
            0,
        )

        # new_occ[k, :] = (
        #     occupancy_change_multiplier * (new_occ[k, :] - ptl_occ_array[k, :])
        #     + ptl_occ_array[k, :]
        # )
        average_sigma += ptl_sigma_array[k, :] * new_occ[k, :] / 100

    # updating columns in parx array
    for k in range(nclasses):
        occdata = new_occ[k, :]
        sigmadata = average_sigma
        if not local:
            for film_id, image_name in enumerate(image_list):
                decompressed_parameter_file_folder = os.path.join(parameter_file_folders, dataset + "_r%02d_%02d" % (k + 1, iteration - 1)) 
                # film_id += 1 # film index from 0
                class_binary = os.path.join(decompressed_parameter_file_folder, image_name + "_r%02d.cistem" % (k + 1))
                image_data = cistem_star_file.Parameters.from_file(class_binary)
                image_array = image_data.get_data()
                image_array[:, occ_col] = occdata[film_index[film_id][0]:film_index[film_id][1]]
                image_array[:, sigma_col] = sigmadata[film_index[film_id][0]:film_index[film_id][1]]
                image_data.set_data(image_array)
                image_data.sync_particle_occ(ptl_to_prj=False)
                image_data.to_binary(class_binary, extended_output=class_binary.replace(".cistem", "_extended.cistem"))
        else:
            class_binary = os.path.join(parameter_file_folders, dataset + "_r%02d.cistem" % (k + 1))
            image_data = cistem_star_file.Parameters.from_file(class_binary)
            image_array = image_data.get_data()
            image_array[:, occ_col] = occdata
            image_array[:, sigma_col] = sigmadata
            image_data.set_data(image_array)
            image_data.to_binary(class_binary)


def random_sample_occ(
    parameters, iteration, dataset, classes, metric="frealignx", parameters_only=False
):

    parfile = "maps/%s_r01_%02d.par" % (dataset, iteration - 1)
    par_data = frealign_parfile.Parameters.from_file(parfile).data
    max_occ = 100 / int(classes)
    total_col = par_data.shape[1]
    total_row = par_data.shape[0]

    if "frealignx" in metric or total_col > 45:
        occ_col = 12
        is_parx = True
        is_frealignx = True
    elif total_col <= 16:
        occ_col = 11
        is_parx = False
    else:
        occ_col = 11
        is_parx = True
        is_frealignx = False

    version = project_params.param(parameters["refine_metric"], iteration).split("_")[0]
    frealign_par = frealign_parfile.Parameters(version=version)

    for k in range(classes):
        new_parfile = parfile.replace("_r01_", "_r%02d_" % (k + 1))

        random_col = np.random.random_sample((total_row,))
        random_occ = random_col * max_occ
        par_data[:, occ_col] = random_occ

        frealign_par.write_parameter_file(
            new_parfile, par_data, parx=is_parx, frealignx=is_frealignx
        )

    if not parameters_only:
        for ref in range(1, classes):
            # create initial models and fsc.txt files for each reference
            source = "maps/%s_r01_%02d.mrc" % (dataset, iteration - 1)
            target = "maps/%s_r%02d_%02d.mrc" % (dataset, ref + 1, iteration - 1)
            symlink_force(os.path.join(os.getcwd(), source), target)
            shutil.copy2(
                "maps/%s_r01_fsc.txt" % dataset,
                "maps/%s_r%02d_fsc.txt" % (dataset, ref + 1),
            )


def random_seeding(seeding):
    
    if seeding < 0:
        seeding = -1 * seeding * 127773.0
    
    I1 = seeding /127773.0
    I2 = seeding % 127773.0
    I3 = 16807.0 * I2 - 2836.0 * I1

    if I3 < 0:
        seeding = I3 + 2147483647
    else:
        seeding = I3

    random = seeding/2147483647

    return random


@timer.Timer(
    "occ_initialization", text="OCC initialization took: {}", logger=logger.info
)
def classification_initialization( dataset, classes, iteration, decompressed_parameter_file_folder, image_list, use_frame = False, is_tomo = False, references_only = False, parameters_only = False):

    if not references_only:
        # read all images parameters for statistices and we need to track the film id to recover individual image data
        binary_list = [os.path.join(decompressed_parameter_file_folder, image + "_r01.cistem") for image in image_list]
        par_data = cistem_star_file.merge_all_binary_with_filmid(binary_list, read_extend=False)

        # the column index from cistem2 binary
        (
            film_col,
            occ_col,
            logp_col,
            sigma_col,
            score_col,
            ptlind_col,
            scanord_col, 
        ) = get_occupancy_columns()

        Nrow = par_data.shape[0]
        
        film = np.unique(par_data[:, film_col].ravel())
        
        if not use_frame and not is_tomo:
            N = Nrow
        else:
            CNF = par_data[:, scanord_col]
            maxframe = int(np.amax(CNF)) + 1
            N = 0
            ptl_per_film=[]
            filma = par_data[:, film_col]
            ptla = par_data[:, ptlind_col]
            for m in film:
                filmmask = np.where(filma == m)
                ptls = ptla[filmmask][-1] + 1
                N += ptls
                ptl_per_film.append(ptls)
            N = int(N)
        logger.info("Total number of particles is " + str(N))
       
        for k in range(classes):
            ref = k + 1
            if not use_frame and not is_tomo:
                
                occ = np.zeros(N)            
                occmax = 0
                seed = np.random.rand(N)
                rani = (seed * N).astype(int)
                unique, counts = np.unique(rani, return_counts=True)

                for i, c in zip(unique, counts):
                    occ[i] = occ[i] + c
                
                occmax = np.max(occ)
                occ = occ / occmax * 100
                
                # update par_data and write class par
                par_data[:, occ_col] = occ
            
            else:
                occ = np.zeros(N)
                ref = k + 1
                occmax = 0

                seed = np.random.rand(N)
                rani = (seed * N).astype(int)
                unique, counts = np.unique(rani, return_counts=True)
                
                for i, c in zip(unique, counts):
                    
                    occ[i] = occ[i] + c
                
                occmax = np.max(occ)
                occ = occ / occmax * 100
                Fast = False
                if Fast:
                    occf = np.repeat(occ, maxframe)
                    occf = np.resize(occf, Nrow)
                else:
                    tilt_index = get_particles_tilt_index(par_data, ptl_col=ptlind_col)
                    index = tilt_index.tolist()
                    occf = np.zeros(Nrow)
                    for i, ind in enumerate(index):
                        occf[ind[0]:ind[1]] = occ[i]

                # update par_data and write class par
                par_data[:, occ_col] = occf
           
            class_parameter_file_folder = decompressed_parameter_file_folder.replace("_r01_", "_r%02d_" % ref)
            
            if not os.path.exists(class_parameter_file_folder):
                os.makedirs(class_parameter_file_folder)

            # split the par_data to individual array and save as binary file
            for f, image in enumerate(image_list):
                saved_binary = os.path.join(class_parameter_file_folder, image + "_r%02d.cistem" % ref )
                ext_binary = os.path.join(decompressed_parameter_file_folder, image + "_r01_extended.cistem" )
                ext_data = cistem_star_file.ExtendedParameters.from_file(ext_binary)
                image_data = par_data[par_data[:, film_col] ==  f ]
                image_data[:, film_col] = 0 # reset film id as 0
                image_parameters = cistem_star_file.Parameters()
                image_parameters.set_data(data=image_data, extended_parameters=ext_data)
                image_parameters.sync_particle_occ(ptl_to_prj=False)
                image_parameters.to_binary(saved_binary, extended_output=ext_binary.replace("r01", "r%02d" % ref))

    if not parameters_only:
        for ref in range(1, classes):
            # create initial models and fsc.txt files for each reference
            try:
                source = "maps/%s_r01_%02d.mrc" % (dataset, iteration - 1)
                target = "maps/%s_r%02d_%02d.mrc" % (dataset, ref + 1, iteration - 1)
                symlink_force(os.path.join(os.getcwd(), source), target)
                shutil.copy2(
                    "maps/%s_r01_fsc.txt" % dataset,
                    "maps/%s_r%02d_fsc.txt" % (dataset, ref + 1),
                )
            except:
                pass


@timer.Timer(
    "get avg and variance", text="Geting statistics for refine and reconstruct took: {}", logger=logger.debug
)
def get_statistics_from_par(parfile, statistics_file):

    par_meta = frealign_parfile.Parameters.from_file(parfile).data
    frealign_data = par_meta[:, 1:16]
    averages = np.average(frealign_data, axis=0)
    vars = np.var(frealign_data, axis=0)
    with open(statistics_file, 'w') as f:
        f.write(" ".join(map(str, averages)) + "\n")
        f.write(" ".join(map(str, vars)) + "\n")


def per_particle_tiltweight(target, tltang_dict, logp_col, index):

    ptl_data = target[index[0]:index[1], :]
    
    ptl_logp = statistics.weighted_by_tilt_angle(ptl_data, tltang_dict)
    
    # pardata[index[0]:index[1], occ_col] = ptl_occ
    target[index[0]:index[1], logp_col] = ptl_logp
    # pardata[index[0]:index[1], sigma_col] = ptl_sigma

def per_particle_scoreweight(target, logp_col, scanord_col, scoreavg_tilt, index):

    ptl_logp_scanord = target[index[0]:index[1], logp_col : scanord_col + 1]

    ptl_logp = statistics.weighted_by_scoreavgs(ptl_logp_scanord, scoreavg_tilt)
    
    target[index[0]:index[1], logp_col] = ptl_logp





