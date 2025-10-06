import os
import numpy as np
from pathlib import Path
import pickle
import glob
import logging
import math

from pyp.system.logging import logger
from pyp.system import local_run
from pyp import analysis
from pyp.analysis import plot
from pyp.system.utils import get_imod_path
from pyp.refine.frealign import frealign
from pyp.inout.image import mrc, img2webp
from pyp.inout.metadata import pyp_metadata
from pyp.inout.metadata.pyp_metadata import Relion
from pyp.inout.metadata.cistem_star_file import *
from pyp.utils import symlink_relative
from pyp.system import project_params, set_up
from pyp.system.db_comm import save_reconstruction_to_website

def get_relion_path():
    return '/usr/local/bin'

def relion_refine(parameters):
    
    radius = (
        float(parameters["particle_rad"])
        / float(parameters["extract_bin"])
        / float(parameters["data_bin"])
        / float(parameters["scope_pixel"])
    )

    # build refine command
    options = ""

    #====== General options =====

    # === Scalars ===
    if parameters['relion_refine_general_iter'] > 0:
        options += f" --iter {parameters['relion_refine_general_iter']}"
    if parameters['relion_refine_general_tau2_fudge'] > 0:
        options += f" --tau2_fudge {parameters['relion_refine_general_tau2_fudge']}"
    options += f" --K {parameters['relion_refine_general_k']}"
    if parameters['relion_refine_general_lowpass'] > 0:
        options += f" --lowpass {parameters['relion_refine_general_lowpass']}"
    if parameters['relion_refine_general_low_resol_join_halves'] > 0:
        options += f" --low_resol_join_halves {parameters['relion_refine_general_low_resol_join_halves']}"

    # === Strings ===
    if parameters['relion_refine_general_tau2_fudge_scheme']:
        options += f" --tau2_fudge_scheme {parameters['relion_refine_general_tau2_fudge_scheme']}"

    # === Booleans ===
    if parameters.get('relion_refine_general_zero_mask',False):
        options += " --zero_mask"
    if parameters.get('relion_refine_general_flatten_solvent',False):
        options += " --flatten_solvent"
    if parameters.get('relion_refine_general_split_random_halves',False):
        options += " --split_random_halves"
    if parameters.get('relion_refine_general_center_classes',False):
        options += " --center_classes"

    # Path parameters
    if parameters.get('relion_refine_general_solvent_mask',False):
        options += f" --solvent_mask {project_params.resolve_path(parameters['relion_refine_general_solvent_mask'])}"
    if parameters.get('relion_refine_general_solvent_mask2',False):
        options += f" --solvent_mask2 {project_params.resolve_path(parameters['relion_refine_general_solvent_mask2'])}"
    if parameters.get('relion_refine_general_lowpass_mask',False):
        options += f" --lowpass_mask {project_params.resolve_path(parameters['relion_refine_general_lowpass_mask'])}"
    if parameters.get('relion_refine_general_tau',False):
        options += f" --tau {project_params.resolve_path(parameters['relion_refine_general_tau'])}"
    if parameters.get('relion_refine_general_local_symmetry',False):
        options += f" --local_symmetry {project_params.resolve_path(parameters['relion_refine_general_local_symmetry'])}"

    #====== Initialisation =====

    # === Scalars ===
    options += f" --offset {parameters['relion_refine_init_offset']}"
    if parameters['relion_refine_init_ini_high'] > 0:
        options += f" --ini_high {parameters['relion_refine_init_ini_high']}"
    
    # === Booleans ===
    if parameters.get('relion_refine_init_denovo_3dref',False):
        options += " --denovo_3dref"
    if parameters.get('relion_refine_init_firstiter_cc',False):
        options += " --firstiter_cc"

    #====== Orientations =====
    
    # === Scalars ===
    options += f" --oversampling {parameters['relion_refine_orientations_oversampling']}"
    options += f" --healpix_order {parameters['relion_refine_orientations_healpix_order']}"
    if parameters['relion_refine_orientations_psi_step'] > 0:
        options += f" --psi_step {parameters['relion_refine_orientations_psi_step']}"
    if parameters['relion_refine_orientations_limit_tilt'] > 0:
        options += f" --limit_tilt {parameters['relion_refine_orientations_limit_tilt']}"
    options += f" --offset_range {parameters['relion_refine_orientations_offset_range']}"
    options += f" --offset_step {parameters['relion_refine_orientations_offset_step']}"
    if parameters['relion_refine_orientations_offset_range_x'] > 0:
        options += f" --offset_range_x {parameters['relion_refine_orientations_offset_range_x']}"
    options += f" --auto_local_healpix_order {parameters['relion_refine_orientations_auto_local_healpix_order']}"
    if parameters['relion_refine_orientations_offset_range_y'] > 0:
        options += f" --offset_range_y {parameters['relion_refine_orientations_offset_range_y']}"
    if parameters['relion_refine_orientations_offset_range_z'] > 0:
        options += f" --offset_range_z {parameters['relion_refine_orientations_offset_range_z']}"
    if parameters['relion_refine_orientations_helical_offset_step'] > 0:
        options += f" --helical_offset_step {parameters['relion_refine_orientations_helical_offset_step']}"
    options += f" --perturb {parameters['relion_refine_orientations_perturb']}"
    if parameters['relion_refine_orientations_sigma_ang'] > 0:
        options += f" --sigma_ang {parameters['relion_refine_orientations_sigma_ang']}"
    if parameters['relion_refine_orientations_sigma_rot'] > 0:
        options += f" --sigma_rot {parameters['relion_refine_orientations_sigma_rot']}"
    if parameters['relion_refine_orientations_sigma_tilt'] > 0:
        options += f" --sigma_tilt {parameters['relion_refine_orientations_sigma_tilt']}"
    if parameters['relion_refine_orientations_sigma_psi'] > 0:
        options += f" --sigma_psi {parameters['relion_refine_orientations_sigma_psi']}"

    # === Strings ===
    if parameters['relion_refine_orientations_relax_sym']:
        options += f" --relax_sym {parameters['relion_refine_orientations_relax_sym']}"

    # === Booleans ===
    if parameters.get('relion_refine_orientations_auto_refine',False):
        options += " --auto_refine"
    if parameters.get('relion_refine_orientations_auto_sampling',False):
        options += " --auto_sampling"
    if parameters.get('relion_refine_orientations_skip_align',False):
        options += " --skip_align"
    if parameters.get('relion_refine_orientations_skip_rotate',False):
        options += " --skip_rotate"
    if parameters.get('relion_refine_orientations_bimodal_psi',False):
        options += " --bimodal_psi"

    #====== Corrections =====

    # === Booleans ===
    if parameters.get('relion_refine_corrections_ctf',False):
        options += " --ctf"
    if parameters.get('relion_refine_corrections_pad_ctf',False):
        options += " --pad_ctf"
    if parameters.get('relion_refine_corrections_ctf_intact_first_peak',False):
        options += " --ctf_intact_first_peak"
    if parameters.get('relion_refine_corrections_ctf_uncorrected_ref',False):
        options += " --ctf_uncorrected_ref"
    if parameters.get('relion_refine_corrections_ctf_phase_flipped',False):
        options += " --ctf_phase_flipped"
    if parameters.get('relion_refine_corrections_only_flip_phases',False):
        options += " --only_flip_phases"
    if parameters.get('relion_refine_corrections_norm',False):
        options += " --norm"
    if parameters.get('relion_refine_corrections_scale',False):
        options += " --scale"
    if parameters.get('relion_refine_corrections_no_norm',False):
        options += " --no_norm"
    if parameters.get('relion_refine_corrections_no_scale',False):
        options += " --no_scale"

    #====== Stochastic Gradient Descent =====

    # === Booleans ===
    if parameters['relion_refine_stochastic_grad_desc_grad']:
        options += " --grad"
    if parameters['relion_refine_stochastic_grad_desc_no_init_blobs']:
        options += " --no_init_blobs"
    if parameters['relion_refine_stochastic_grad_desc_som']:
        options += " --som"

    # === Scalars ===
    if parameters['relion_refine_stochastic_grad_desc_grad_em_iters'] > 0:
        options += f" --grad_em_iters {parameters['relion_refine_stochastic_grad_desc_grad_em_iters']}"
    if parameters['relion_refine_stochastic_grad_desc_grad_ini_subset'] > 0:
        options += f" --grad_ini_subset {parameters['relion_refine_stochastic_grad_desc_grad_ini_subset']}"
    if parameters['relion_refine_stochastic_grad_desc_grad_fin_subset'] > 0:
        options += f" --grad_fin_subset {parameters['relion_refine_stochastic_grad_desc_grad_fin_subset']}"
    options += f" --grad_write_iter {parameters['relion_refine_stochastic_grad_desc_grad_write_iter']}"
    if parameters['relion_refine_stochastic_grad_desc_maxsig'] > 0:
        options += f" --maxsig {parameters['relion_refine_stochastic_grad_desc_maxsig']}"
    options += f" --som_ini_nodes {parameters['relion_refine_stochastic_grad_desc_som_ini_nodes']}"
    options += f" --grad_ini_frac {parameters['relion_refine_stochastic_grad_desc_grad_ini_frac']}"
    options += f" --grad_fin_frac {parameters['relion_refine_stochastic_grad_desc_grad_fin_frac']}"
    options += f" --grad_min_resol {parameters['relion_refine_stochastic_grad_desc_grad_min_resol']}"
    if parameters['relion_refine_stochastic_grad_desc_grad_ini_resol'] > 0:
        options += f" --grad_ini_resol {parameters['relion_refine_stochastic_grad_desc_grad_ini_resol']}"
    if parameters['relion_refine_stochastic_grad_desc_grad_fin_resol'] > 0:
        options += f" --grad_fin_resol {parameters['relion_refine_stochastic_grad_desc_grad_fin_resol']}"
    options += f" --mu {parameters['relion_refine_stochastic_grad_desc_mu']}"
    if parameters['relion_refine_stochastic_grad_desc_grad_stepsize'] > 0:
        options += f" --grad_stepsize {parameters['relion_refine_stochastic_grad_desc_grad_stepsize']}"
    options += f" --som_connectivity {parameters['relion_refine_stochastic_grad_desc_som_connectivity']}"
    options += f" --som_inactivity_threshold {parameters['relion_refine_stochastic_grad_desc_som_inactivity_threshold']}"
    options += f" --som_neighbour_pull {parameters['relion_refine_stochastic_grad_desc_som_neighbour_pull']}"
    options += f" --class_inactivity_threshold {parameters['relion_refine_stochastic_grad_desc_class_inactivity_threshold']}"

    # === Strings ===
    if parameters.get('relion_refine_stochastic_grad_desc_grad_stepsize_scheme',False):
        options += f" --grad_stepsize_scheme {parameters['relion_refine_stochastic_grad_desc_grad_stepsize_scheme']}"

    #====== Subtomogram averaging =====
    
    # === Booleans ===
    if parameters.get('relion_refine_subtomogram_averaging_normalised_subtomo',False):
        options += " --normalised_subtomo"
    if parameters.get('relion_refine_subtomogram_averaging_skip_subtomo_multi',False):
        options += " --skip_subtomo_multi"
    if parameters.get('relion_refine_subtomogram_averaging_ctf3d_not_squared',False):
        options += " --ctf3d_not_squared"

    # === Scalars ===
    options += f" --subtomo_multi_thr {parameters['relion_refine_subtomogram_averaging_subtomo_multi_thr']}"

    #====== Computation =====
    
    # === Scalars ===
    options += f" --pool {parameters['relion_refine_computation_pool']}"
    if parameters['relion_refine_computation_j'] > 0:
        options += f" --j {parameters['relion_refine_computation_j']}"
    options += f" --keep_free_scratch {parameters['relion_refine_computation_keep_free_scratch']}"

    # === Booleans ===
    if parameters.get('relion_refine_computation_dont_combine_weights_via_disc',False):
        options += " --dont_combine_weights_via_disc"
    if parameters.get('relion_refine_computation_onthefly_shifts',False):
        options += " --onthefly_shifts"
    if parameters.get('relion_refine_computation_no_parallel_disc_io',False):
        options += " --no_parallel_disc_io"
    if parameters.get('relion_refine_computation_preread_images',False):
        options += " --preread_images"
    if parameters.get('relion_refine_computation_reuse_scratch',False):
        options += " --reuse_scratch"
    if parameters.get('relion_refine_computation_keep_scratch',False):
        options += " --keep_scratch"
    if parameters.get('relion_refine_computation_fast_subsets',False):
        options += " --fast_subsets"
    if parameters.get('relion_refine_computation_gpu',False):
        options += " --gpu"
        if parameters['relion_refine_computation_free_gpu_memory'] > 0:
            options += f" --free_gpu_memory {parameters['relion_refine_computation_free_gpu_memory']}"

    # === String options ===
    if parameters['relion_refine_computation_scratch_dir']:
        options += f" --scratch_dir {parameters['relion_refine_computation_scratch_dir']}"

    #====== Expert options =====

    # === Integers ===
    options += f" --pad {parameters['relion_refine_expert_pad']}"
    options += f" --r_min_nn {parameters['relion_refine_expert_r_min_nn']}"
    if parameters['relion_refine_expert_random_seed'] > 0:
        options += f" --random_seed {parameters['relion_refine_expert_random_seed']}"
    if parameters['relion_refine_expert_coarse_size'] > 0:
        options += f" --coarse_size {parameters['relion_refine_expert_coarse_size']}"
    options += f" --maskedge {parameters['relion_refine_expert_maskedge']}"
    options += f" --incr_size {parameters['relion_refine_expert_incr_size']}"
    options += f" --failsafe_threshold {parameters['relion_refine_expert_failsafe_threshold']}"
    options += f" --auto_iter_max {parameters['relion_refine_expert_auto_iter_max']}"
    if parameters['relion_refine_expert_nr_parts_sigma2noise'] > 0:
        options += f" --nr_parts_sigma2noise {parameters['relion_refine_expert_nr_parts_sigma2noise']}"

    # === Floats ===
    if parameters['relion_refine_expert_ref_angpix'] > 0:
        options += f" --ref_angpix {parameters['relion_refine_expert_ref_angpix']}"
    options += f" --adaptive_fraction {parameters['relion_refine_expert_adaptive_fraction']}"
    if parameters['relion_refine_expert_strict_highres_exp'] > 0:
        options += f" --strict_highres_exp {parameters['relion_refine_expert_strict_highres_exp']}"
    if parameters['relion_refine_expert_strict_lowres_exp'] > 0:
        options += f" --strict_lowres_exp {parameters['relion_refine_expert_strict_lowres_exp']}"

    # === Booleans ===
    if parameters.get('relion_refine_expert_nn', False):
        options += " --nn"
    if parameters.get('relion_refine_expert_fix_sigma_noise', False):
        options += " --fix_sigma_noise"
    if parameters.get('relion_refine_expert_fix_sigma_offset', False):
        options += " --fix_sigma_offset"
    if parameters.get('relion_refine_expert_print_metadata_labels', False):
        options += " --print_metadata_labels"
    if parameters.get('relion_refine_expert_print_symmetry_ops', False):
        options += " --print_symmetry_ops"
    if parameters.get("relion_refine_expert_dont_check_norm", False):
        options += " --dont_check_norm"
    if parameters.get("relion_refine_expert_always_cc", False):
        options += " --always_cc"
    if parameters.get("relion_refine_expert_solvent_correct_fsc", False):
        options += " --solvent_correct_fsc"
    if parameters.get("relion_refine_expert_skip_maximize", False):
        options += " --skip_maximize"
    if parameters.get("relion_refine_expert_blush", False):
        options += " --blush"
        if parameters.get("relion_refine_expert_blush_skip_spectral_trailing", False):
            options += " --blush_skip_spectral_trailing"
    if parameters.get("relion_refine_expert_external_reconstruct", False):
        options += " --external_reconstruct"
    if parameters.get("relion_refine_expert_auto_ignore_angles", False):
        options += " --auto_ignore_angles"
    if parameters.get("relion_refine_expert_auto_resol_angles", False):
        options += " --auto_resol_angles"
    if parameters.get("relion_refine_expert_allow_coarser_sampling", False):
        options += " --allow_coarser_sampling"
    if parameters.get("relion_refine_expert_trust_ref_size", False):
        options += " --trust_ref_size"
    if parameters.get("relion_refine_expert_dont_skip_gridding", False):
        options += " --dont_skip_gridding"

    mpi_processes = math.floor(parameters['slurm_tasks'] / parameters['relion_refine_computation_j'])
    
    command = f"mpirun --oversubscribe -n {parameters['slurm_tasks']} {get_relion_path()}/relion_refine_mpi --o Refine3D/job001/run --auto_refine --split_random_halves --ios matching_optimisation_set.star --trust_ref_size --ini_high 40 --dont_combine_weights_via_disc --pool 10 --pad 2  --ctf --particle_diameter 130 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym {parameters['particle_sym']} --low_resol_join_halves 40 --norm --scale  --j 2 --gpu "" --pipeline_control Refine3D/job001"
    logger.warning(f"LEGACY command: {command}")
    
    command = f"mpirun --oversubscribe -n {mpi_processes} {get_relion_path()}/relion_refine_mpi --o Refine3D/job001/run --ios matching_optimisation_set.star --ref {parameters["refine_model"]} --particle_diameter {2*parameters['particle_rad']} --sym {parameters['particle_sym']} {options} --pipeline_control Refine3D/job001"
    logger.warning(f"ACTUAL command: {command}")

    # define observer to intersect relion standard output
    iteration = 0
    def obs(line):
        global iteration
        if "Auto-refine: Iteration=" in line:
            iteration = int(line.split("Iteration=")[-1])
        elif "Auto-refine: Resolution=" in line and iteration > 1:            
            resolution = line.split("Resolution=")[-1].split()[0]

            class_num = 1
            half_map_file_name = f"Refine3D/job001/run_it{iteration-1:03d}_half1_class{class_num:03d}.mrc"
            
            name = os.path.split(os.getcwd())[-1] + f"_r{class_num:02d}_{iteration+1:02d}"
            current_map = Path.cwd().parent.stem + f"_r{class_num:02d}_{iteration+1:02d}.mrc"
            
            # produce map snapshots
            output_png = os.path.join( "..", "frealign", "maps", Path(current_map).stem + "_map.png" )
            lim = frealign.build_map_montage( half_map_file_name, radius, output_png )

            img2webp(output_png,output_png.replace(".png",".webp"),"-resize 1024x")
            os.remove(output_png)

            # produce cropped version of map
            rec = mrc.read(half_map_file_name)
            cropped_volume = rec[ lim:-lim, lim:-lim, lim:-lim ]
            cropped_volume_filename = f"../frealign/maps/{Path(current_map).stem}_crop.mrc"
            mrc.write(cropped_volume, cropped_volume_filename)

            # link to map file
            target_map = f"../frealign/maps/{Path(current_map).stem}.mrc"
            if os.path.exists(target_map):
                os.remove(target_map)
            symlink_relative(half_map_file_name, target_map)

            # link to bld file
            source_bld = half_map_file_name.replace(".mrc","_angdist.bild")
            target_bld = f"../frealign/maps/{Path(current_map).stem}.bild"
            if os.path.exists(target_bld):
                os.remove(target_bld)
            symlink_relative(source_bld, target_bld)

            parameters = project_params.load_parameters("..")
            with open(f"../{parameters['data_set']}.films") as f:
                imagelist = f.read().splitlines()
                
            refine_star_file = f"Refine3D/job001/run_it{iteration-1:03d}_data.star"
            tomo_star_file = f"{parameters['data_set']}_tomograms.star"
            particles_star_file = f"stacks/{parameters['data_set']}_particles.star"
            
            plot.generate_plots_relion_tomo(refine_star_file, particles_star_file, Path(current_map).stem)

            # combine 2D plots from used particles and global statistics for histograms
            # read saved pickle files
            dataset_name = Path(current_map).stem
            with open(f"{dataset_name}_temp.pkl", 'rb') as f1:
                plot_outputs = pickle.load(f1)
                plot_outputs_used = plot_outputs
            with open(f"{dataset_name}_meta_temp.pkl", 'rb') as f2:
                metadata = pickle.load(f2)
                metadata_used = metadata

            consolidated_plot_outputs = plot_outputs.copy()
            consolidated_plot_outputs["def_rot_histogram"] = plot_outputs_used["def_rot_histogram"]
            consolidated_plot_outputs["def_rot_scores"] = plot_outputs_used["def_rot_scores"]

            consolidated_metadata = metadata.copy()
            consolidated_metadata["particles_used"] = metadata_used["particles_used"]
            consolidated_metadata["phase_residual"] = metadata_used["phase_residual"]

            temp_par_obj = Parameters()
            occ_col = temp_par_obj.get_index_of_column(OCCUPANCY)
            score_col = temp_par_obj.get_index_of_column(SCORE)
            
            consolidated_metadata["phase_residual"] = 0
            
            current_dir = os.getcwd()
            os.chdir(os.environ["PYP_SCRATCH"])
            command = f"/usr/local/bin/relion_postprocess --i {current_dir}/Refine3D/job001/run_it{iteration-1:03d}_half1_class001.mrc --angpix {parameters['scope_pixel']*parameters['data_bin']*parameters['extract_bin']}"
            local_run.run_shell_command(command,log_level=logging.NOTSET)
            
            # calculate FSC curves
            fsc_star_file = "postprocess.star"
            fsc_star_metadata = pyp_metadata.parse_star_tables(fsc_star_file)
            current_fsc = fsc_star_metadata["data_fsc"].to_numpy()[:,-2:]
            for file in glob.glob("postprocess*"):
                try:
                    os.remove(file)
                except:
                    pass
            os.chdir(current_dir)

            fsc_file = os.path.join("../frealign/maps", parameters['data_set'] + "_r01_fsc.txt" )
            if os.path.isfile(fsc_file):
                oldFSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)
                if current_fsc.shape[0] == oldFSCs.shape[0]:
                    if oldFSCs.shape[1] < iteration + 1:
                        FSCs = np.zeros([oldFSCs.shape[0], iteration + 1])
                        FSCs[:, : oldFSCs.shape[1]] = oldFSCs
                    else:
                        FSCs = oldFSCs
                else:
                    logger.warning(
                        "Size of FSC curve has changed from {0} to {1}. Not plotting past results.".format(
                            current_fsc.shape[0], oldFSCs.shape[0]
                        )
                    )
                    FSCs = np.zeros([current_fsc.shape[0], iteration + 1])
                    # set new x-axis
                    FSCs[:, 0] = current_fsc[:, 0]
                    # try to recover old FSC values
                    FSCs[:,1:iteration] = oldFSCs[:current_fsc.shape[0],1:iteration]
                FSCs[:, iteration] = current_fsc[:, 1]
            else:
                FSCs = current_fsc
            np.savetxt(fsc_file, FSCs, fmt="%10.5f")

            # send to website
            save_reconstruction_to_website( name=Path(current_map).stem, fsc=FSCs, plots=consolidated_plot_outputs, metadata=consolidated_metadata )
    local_run.stream_shell_command(command, observer=obs)


def relion_mask_create():
    
    command = f"relion_mask_create --i relion/Refine3D/job002/run_class001.mrc --o m/mask_4apx.mrc --ini_threshold 0.04"
    local_run.stream_shell_command(command)
