import math
import os
import shutil
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from pyp.analysis import statistics
from pyp.inout.metadata import frealign_parfile
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_frealign_paths
from pyp.utils import get_relative_path, symlink_force
from pyp.utils import timer
from pyp.inout.metadata import get_particles_tilt_index
relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def get_occupancy_columns(par_col):
    if  par_col > 45:
        film_col = 7
        occ_col = 12
        logp_col = 13
        sigma_col = 14
        score_col = 15
        ptl_col = 17
        tltan_col = 18
        scanord_col = 20
    else:
        film_col = 7
        occ_col = 11
        logp_col = 12
        sigma_col = 13
        score_col = 14
        ptl_col = 16
        tltan_col = 17
        scanord_col = 19

    return film_col, occ_col, logp_col, sigma_col, score_col, ptl_col, tltan_col, scanord_col


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
    "occ_function", text="OCC calculation took: {}", logger=logger.info
)
def occupancy_extended(parameters, dataset, iteration, nclasses, path=".", is_frealignx=False, local=False):
    """Compute occupancies based on LogP values with extended par file."""
    # dataset = fparameters["dataset"]

    metric = project_params.param(parameters["refine_metric"], iteration)
    # film_col, occ_col, logp_col, sigma_col, ptl_col, tltan_col = get_occupancy_columns(metric)

    # load the par files from all the classes and calculate average occupancies
    parx = []
    class_average_occ = []
    occupancy_change_multiplier = 1
    
    is_parx = True

    if is_frealignx:
        col_num = 46
    else:
        col_num = 45
    (
        film_col,
        occ_col,
        logp_col,
        sigma_col,
        score_col,
        ptl_col,
        tltan_col,
        scanord_col, 
    ) = get_occupancy_columns(col_num)
    # info needed only
   
    newptl_col = 0
    newocc_col = 1
    newlogp_col = 2
    newsigma_col = 3
    newscore_col = 4
    newtltan_col = 5
    newscanord_col = 6
    
    if not local:
        index_file = "../csp/particle_tilt.index"
        with timer.Timer(
        "read par for occ", text = "OCC Read par took: {}", logger=logger.info
        ):
            for k in range(nclasses):
                logger.debug("Processing class {}".format(k + 1))
                parxfile = "%s/%s_r%02d_%02d.par" % (path, dataset, k + 1, iteration)
                headerline = frealign_parfile.EXTENDED_NEW_PAR_HEADER[2].strip("\n").split()
                column_names = np.array(headerline[1:])
                select_data = pd.read_csv(
                    parxfile, 
                    comment='C', 
                    delim_whitespace=True, 
                    names=column_names, 
                    usecols=['NO', 'OCC', 'LOGP','SIGMA', 'SCORE', 'TILTAN', 'SCANOR'],
                    )
                parx.append(select_data.to_numpy())
                all_occ = parx[k][:, newocc_col].ravel()
                class_average_occ.append(mean(all_occ))

            parx_3d = np.array(parx)

    else:
        # for local run, read directly from map/parfile_constrain.txt
        parx = []
        for k in range(nclasses):
            if "new" in parameters["refine_metric"]:
                occindex = 10                                # no id in the constrain file
            elif "frealignx" in parameters["refine_metric"]:
                occindex = 11
            
            with open("project_dir.txt") as f:
                project_dir = f.readline().strip("\n")
            remote_par_stat = os.path.join(project_dir, "frealign", "maps", "parfile_constrain_r%02d.txt" % (k + 1))
            stat = np.loadtxt(remote_par_stat, ndmin=2)
            occmean = stat[0, occindex]
            class_average_occ.append(occmean)

            # reload local par files
            parxfile = "%s/%s_r%02d_%02d.par" % (path, dataset, k + 1, iteration)
            parkdata = np.loadtxt(parxfile, comments="C", ndmin=2)
            occ_only = parkdata[:, [0, occ_col, logp_col, sigma_col, score_col, tltan_col, scanord_col]]
            parx.append(occ_only)
        
        if "tomo" in parameters["data_mode"]:
            get_particles_tilt_index(parxfile, ".")
            index_file = "particle_tilt.index"

        parx_3d = np.array(parx)
    
    if is_parx and "tomo" in parameters["data_mode"]:
        
        tilt_index = np.loadtxt(index_file, dtype='int', ndmin=2)
        index = tilt_index.tolist()

        if parameters["refine_score_weighting"]: # score weights
            logger.info("Weighting logp with score averages")
            scoreavg_tilt = statistics.get_class_score_weight(parx_3d, newscore_col, newscanord_col)
            
            for k in range(0, nclasses):
                for i in index:
                    per_particle_scoreweight(parx_3d[k, :,:], newlogp_col, newscanord_col, scoreavg_tilt, i)

        else: # tilt gaussian weights
            logger.info("Weighting logp with tilt distribution")
            for k in range(0, nclasses):
                for i in index:
                    per_particle_tiltweight(parx_3d[k, :,:], newocc_col, newtltan_col, newlogp_col, i)

    else:
        logger.info("Not using tilt weighting to change OCC")

    # recalculate occ from logp
    all_frame, col_num = parx[0].shape

    ptl_logp_array = parx_3d[:, :, newlogp_col]
    ptl_occ_array = parx_3d[:, :, newocc_col]
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
        new_occ[k, :] = (
            occupancy_change_multiplier * (new_occ[k, :] - ptl_occ_array[k, :])
            + ptl_occ_array[k, :]
        )
        average_sigma += ptl_sigma_array[k, :] * new_occ[k, :] / 100

    # updating columns in parx array
    if not local:
        header1 = "C\nC     1      12         14\nC    NO     OCC      SIGMA"
    else: 
        header1 = "C\nC     1      13         15\nC    NO     OCC      SIGMA"

    for k in range(nclasses):
        occ_sigmafile = "%s/%s_r%02d_occsigma.txt" % (path, dataset, k + 1)
        pid = parx[k][:,0].reshape((-1, 1))
        occdata = new_occ[k, :].reshape((-1, 1)) 
        sigmadata = average_sigma.reshape((-1, 1))
        occ_sigma = np.hstack((pid, occdata, sigmadata))

        np.savetxt(occ_sigmafile, occ_sigma, fmt='%7d%8.2f%11.2f', header=header1, comments="")
    
    # column replacement
    with timer.Timer(
        "write_occ_changed", text = "OCC write par took: {}", logger=logger.info
    ):  
        # bytes numbers for each column
        if is_frealignx:
            pshift = 8
        else:
            pshift = 0
        preocc = 91 + pshift      # columns before occ
        logpstart = 100 + pshift  # logp start position
        logpend = logpstart + 9   # logp end position
        postsigma = 121 + pshift  # after sigma
        parend = 425 + pshift     # end

        for k in range(nclasses):
            occ_sigmafile = "%s/%s_r%02d_occsigma.txt" % (path, dataset, k + 1)
            parxfile = "%s/%s_r%02d_%02d.par" % (path, dataset, k + 1, iteration)
            inputfile = parxfile.replace(".par", ".paro")
            os.rename(parxfile, inputfile)
            
            writecom = f"""/bin/bash -c "paste -d '' <(cut -b 1-{preocc} '{inputfile}') <(cut -b 8-15 '{occ_sigmafile}') <(cut -b {logpstart}-{logpend} '{inputfile}') <(cut -b 16-26 '{occ_sigmafile}') <(cut -b {postsigma}-{parend} {inputfile}) > {parxfile}" """
            [output, error] = local_run.run_shell_command(writecom, verbose=False)
            os.remove(occ_sigmafile)
            os.remove(inputfile)


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
def classification_initialization(parameters, dataset, classes, iteration, use_frame = False, is_tomo = False, references_only = False, parameters_only = False):

    if not references_only:
        #if not parameters["reconstruct_weights"]:
        if True:
            # initialize metadata
            parfile = os.path.join("maps", "%s_r01_%02d.par" % (dataset, iteration - 1))
            par_data = frealign_parfile.Parameters.from_file(parfile).data
            Ncol = par_data.shape[1]
            if Ncol > 16:
                is_parx = True
            else:
                is_parx = False

            # only consider metric new
            if use_frame or is_tomo:
                ptlind_col = 16
            else:
                ptlind_col = 0

            occ_col = 11
            film_col = 7
            scanorder = 19
            Nrow = par_data.shape[0]
            
            film = np.unique(par_data[:, film_col].ravel())
            
            if not use_frame and not is_tomo:
                N = Nrow
            else:
                CNF = par_data[:, scanorder]
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
            # seed = -100
            for k in range(classes):
                if not use_frame and not is_tomo:
                    
                    occ = np.zeros(N)            
                    ref = k + 1
                    occmax = 0
                    """
                    for ptl in range(N):
                        #i = int(N * random_seeding(seed))
                        i = int(N*np.random.random_sample())
                        occ[i] = occ[i] + 1
                    """
                    seed = np.random.rand(N)
                    rani = (seed * N).astype(int)
                    unique, counts = np.unique(rani, return_counts=True)
                    for i, c in zip(unique, counts):
                        occ[i] = occ[i] + c
                    
                    occmax = np.max(occ)
                    occ = occ / occmax * 100
                    
                    # update par_data and write class par
                    par_data[:, occ_col] = occ
                    class_par = parfile.replace("_r01", "_r%02d" % ref)
                    version = project_params.param(parameters["refine_metric"], iteration).split("_")[0]
                    frealign_par = frealign_parfile.Parameters(version=version)
                    frealign_par.write_parameter_file(
                        class_par, par_data, parx=is_parx, frealignx=False
                    )
                
                else:
                    occ = np.zeros(N)
                    ref = k + 1
                    occmax = 0
                    # logger.info("start random seeding")
                    #for ptl in range(int(N)):
                    #    i = int(N*np.random.random_sample())
                    #    logger.info("iterate particle " + str(ptl)+ "\n")
                    """
                        count = 0 
                        for film, ptls in enumerate(ptl_per_film):
                            # filmblock = occ[occ[:, film] == m]
                            # ptls = filmblock[-1, ptlind_col] + 1
                            count += ptls
                            if i <= count:
                                dp = count - i
                                pind = int(ptls - dp - 1)
                                mask = np.logical_and(occ[:, 0] == film, occ[:, 1] == pind)
                                newocc = occ[:, 2] + 1
                                occ[:, 2] = np.where(mask, newocc,  occ[:, 2] )
                                break
                            else:
                                continue
                    """
                    seed = np.random.rand(N)
                    rani = (seed * N).astype(int)
                    unique, counts = np.unique(rani, return_counts=True)
                    
                    for i, c in zip(unique, counts):
                        # occ[:, 3]= np.where(occ[:, 2] == i, occ[:,3] + c, occ[:,3])
                        occ[i] = occ[i] + c
                    
                    occmax = np.max(occ)
                    occ = occ / occmax * 100
                    Fast = False
                    if Fast:
                        occf = np.repeat(occ, maxframe)
                        occf = np.resize(occf, Nrow)
                    else:
                        if "tomo" in parameters["data_mode"] and not os.path.isfile("../csp/particle_tilt.index"):
                            get_particles_tilt_index(parfile, "../csp")
                        index_file = "../csp/particle_tilt.index"
                        tilt_index = np.loadtxt(index_file, dtype='int', ndmin=2)
                        index = tilt_index.tolist()
                        occf = np.zeros(Nrow)
                        for i, ind in enumerate(index):
                            occf[ind[0]:ind[1]] = occ[i]

                    #occmax = np.max(occ[:, 3].ravel())
                    #occ[:, 3] = occ[:, 3] / occmax * 100
                    
                    # update par_data and write class par
                    par_data[:, occ_col] = occf
                    class_par = parfile.replace("_r01", "_r%02d" % ref)
                    version = project_params.param(parameters["refine_metric"], iteration).split("_")[0]
                    frealign_par = frealign_parfile.Parameters(version=version)
                    frealign_par.write_parameter_file(
                        class_par, par_data, parx=is_parx, frealignx=False
                    )
        
        else:
            input_par_file = "maps/%s_r01_%02d.par" % (dataset, iteration - 1)
            for res in range(1, classes):
                mask = ["0", "0", "0", "0", "0"]
                mask[(res - 1) % 5] = "1"
                frealign_parfile.Parameters.generate_par_file(
                    input_par_file,
                    "maps/%s_r%02d_%02d.par" % (dataset, res + 1, iteration - 1),
                    ",".join(mask),
                )

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


def per_particle_tiltweight(target, occ_col, tltan_col, logp_col, index):

    ptl_occ_tltang = target[index[0]:index[1], occ_col : tltan_col + 1]
    
    ptl_logp = statistics.weighted_by_tilt_angle(ptl_occ_tltang)

    # pardata[index[0]:index[1], occ_col] = ptl_occ
    target[index[0]:index[1], logp_col] = ptl_logp
    # pardata[index[0]:index[1], sigma_col] = ptl_sigma

def per_particle_scoreweight(target, logp_col, scanord_col, scoreavg_tilt, index):

    ptl_logp_scanord = target[index[0]:index[1], logp_col : scanord_col + 1]

    ptl_logp = statistics.weighted_by_scoreavgs(ptl_logp_scanord, scoreavg_tilt)
    
    target[index[0]:index[1], logp_col] = ptl_logp