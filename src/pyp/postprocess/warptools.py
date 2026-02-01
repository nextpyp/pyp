import glob
import os
import numpy as np
from pathlib import Path

from pyp.inout.image import mrc
from pyp.system.logging import logger
from pyp.system import local_run
from pyp.system import project_params
from pyp.system.utils import get_imod_path

def get_warptools_path():
    warptools_path = '/opt/conda/envs/warp'
    return f"export LD_LIBRARY_PATH=/opt/conda/envs/warp/lib:$LD_LIBRARY_PATH; micromamba run -n warp {warptools_path}/bin/"

def warptools_noise2map(half1, parameters, tomogram=False):
    """
    Noise2Map training
    Will take all the *half1.rec from mrc folder as list to train and run denoise
    """

    assert os.path.exists(half1), "Cannot proceed without a valid half map"
            
    """
    Noise2Map 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 Noise2Map

    -a, --observation1         Relative path to a folder containing files with the first observation of the objects (e.g. first half-maps).
    -b, --observation2         Relative path to a folder containing files with the second observation of the objects (e.g. second half-maps). Names of the files must match those of the first observation.
    --observation_combined     (Default: ) Relative path to a folder containing maps that combine first and second observations in a way that is more complex than simple averaging. This is especially
                                relevant for raw tomograms. Names of the files must match those of the first observation.
    -h, --half1                Relative path to the first single half-map (use this when you have only one set of half-maps, use --observation1/2 otherwise).
    -f, --half2                Relative path to the second single half-map (use this when you have only one set of half-maps, use --observation1/2 otherwise).
    --3dctf                    Relative path to a folder containing 3D CTFs for tomograms.
    --denoise_separately       (Default: false) If true, both observations will be denoised separately in the end. If false, their average will be denoised.
    --mini_model               (Default: false) Use a really shallow and slim model to avoid overfitting with very little data.
    --start_model              (Default: ) Name of the file with the initial (pre-trained) model.
    --old_model                (Default: ) Name of the folder with the pre-trained model. Leave empty to train a new one.
    --learningrate_start       (Default: 0.0001) Initial learning rate that will be decreased exponentially to reach the final learning rate.
    --learningrate_finish      (Default: 1E-06) Final learning rate, after exponential decrease from the initial rate.
    --window                   (Default: 64) Size of the cubic window used during training and denoising. Should be a multiple of 16. Bigger = needs more memory.
    --dont_flatten_spectrum    (Default: false) Don't flatten the spectrum of the maps beyond 10 Angstrom to sharpen them. Pixel size must be specified for flattening.
    --dont_augment             (Default: false) Don't augment data through random rotations. Only rotations by multiples of 180 degrees will be used.
    --overflatten_factor       (Default: 1) Overflattening (oversharpening) factor in case a flat spectrum isn't enough. 1.0 = flat
    --angpix                   (Default: -1) Pixel size used for spectrum flattening.
    --mask                     (Default: ) Relative path to a common mask for all maps. It can be used for spectrum flattening and map trimming.
    --lowpass                  (Default: -1) Low-pass filter to be applied to denoised maps (in Angstroms).
    --crop_map                 (Default: false) If true, the denoised result will be cropped to only contain the masked area.
    --mask_output              (Default: false) Masks the denoised maps with the supplied mask. Requires keep_dimensions to be enabled.
    --iterations               (Default: 1500) Number of iterations. 600–1200 for SPA half-maps, 10 000+ for raw tomograms.
    --batchsize                (Default: 4) Batch size for model training. Decrease if you run out of memory. The number of iterations will be adjusted automatically. Should be a multiple of the number of
                                GPUs used in training.
    --gpuid_network            (Default: 0) Comma-separated GPU IDs used for network training.
    --gpuid_preprocess         (Default: 1) GPU ID used for data preprocessing. Ideally not the GPU used for training
    --help                     Display this help screen.
    --version                  Display version information.
    """

    output = []
    def obs(line):
        output.append(line)
    
    options = ""

    if tomogram:
        prefix = "tomo_denoise"
        options += " --dont_flatten_spectrum"
    else:
        prefix = "reconstruct_denoise"
        if not parameters.get(f"reconstruct_denoise_flatten_spectrum"):
            options += f" --dont_flatten_spectrum"
        else:
            options += f" --angpix {parameters.get('scope_pixel')*parameters.get('extract_bin')}"
            options += f" --overflatten_factor {parameters.get(f'reconstruct_denoise_overflatten_factor')}"

    if parameters.get(f"{prefix}_lowpass"):
        options += f" --lowpass {parameters.get(f'{prefix}_lowpass')}"

    if parameters.get(f"{prefix}_denoise_separately",False):
        options += " --denoise_separately"

    if parameters.get(f"{prefix}_mini_model",False):
        options += " --mini_model"

    if parameters.get(f"{prefix}_start_model",False):
        options += f" --start_model {project_params.resolve_path(parameters[f'{prefix}_start_model'])}"

    if parameters.get(f"{prefix}_old_model",False):
        options += f" --old_model {project_params.resolve_path(parameters[f'{prefix}_old_model'])}"

    if parameters.get(f"{prefix}_learningrate_start",False):
        options += f" --learningrate_start {parameters[f'{prefix}_learningrate_start']}"

    if parameters.get(f"{prefix}_learningrate_finish",False):
        options += f" --learningrate_finish {parameters[f'{prefix}_learningrate_finish']}"

    if parameters.get(f"{prefix}_window",False):
        options += f" --window {parameters[f'{prefix}_window']}"

    if parameters.get(f"{prefix}_dont_augment",False):
        options += " --dont_augment"

    if parameters.get(f"{prefix}_lowpass",False) and parameters.get(f'{prefix}_lowpass') > 0:
        options += f" --lowpass {parameters[f'{prefix}_lowpass']}"

    if parameters.get(f"{prefix}_iterations",False):
        options += f" --iterations {parameters[f'{prefix}_iterations']}"

    if parameters.get(f"{prefix}_batchsize",False):
        options += f" --batchsize {parameters[f'{prefix}_batchsize']}"

    command = f"{get_warptools_path()}Noise2Map --half1 {half1} --half2 {half1.replace('half1','half2')} {options}"
    local_run.stream_shell_command(command)

    result = f"./denoised/{Path(half1).stem}.mrc"
    
    # convert to 32 bits for compatibility
    command = f"{get_imod_path()}/bin/newstack -quiet -mode 2 {result} {result}~ && mv {result}~ {result}"
    local_run.run_shell_command(command)
    
    return result


def create_settings(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command create_settings:
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    -o, --output           REQUIRED Path to the new settings file
    --folder_processing    Processing folder location
    --folder_data          REQUIRED Raw data folder location
    --recursive            Recursively search for files in sub-folders (only when processing and raw data locations are different)
    --extension            Import file search term: Use e.g. *.mrc to process all MRC files, or something more specific like FoilHole1_*.mrc
    --angpix               REQUIRED Unbinned pixel size in Angstrom. Alternatively specify the path to an image or MDOC file to read the value from. If a wildcard pattern is specified, the first file will be used
    --bin                  2^x pre-binning factor, applied in Fourier space when loading raw data. 0 = no binning, 1 = 2x2 binning, 2 = 4x4 binning, supports non-integer values
    --bin_angpix           Choose the binning exponent automatically to match this target pixel size in Angstrom
    --gain_path            Path to gain file, relative to import folder
    --defects_path         Path to defects file, relative to import folder
    --gain_flip_x          Flip X axis of the gain image
    --gain_flip_y          Flip Y axis of the gain image
    --gain_transpose       Transpose gain image (i.e. swap X and Y axes)
    --exposure             Default: 1. Overall exposure per Angstrom^2; use negative value to specify exposure/frame instead
    --eer_ngroups          Default: 40. Number of groups to combine raw EER frames into, i.e. number of 'virtual' frames in resulting stack; use negative value to specify the number of frames per virtual frame instead
    --eer_groupexposure    As an alternative to --eer_ngroups, fractionate the frames so that a group will have this exposure in e-/A^2; this overrides --eer_ngroups
    --tomo_dimensions      X, Y, and Z dimensions of the full tomogram in unbinned pixels, separated by 'x', e.g. 4096x4096x1000
    """

    extra_options = ""
    if parameters.get("gain_fliph"):
        extra_options += " --gain_flip_x"

    if parameters.get("gain_flipv"):
        extra_options += " --gain_flip_y"        

    if not parameters.get("gain_rotation") % 2:
        extra_options += " --gain_transpose"
        
    # create frame series settings file
    command = f"{get_warptools_path()}WarpTools create_settings --folder_data {str(Path(project_params.resolve_path(parameters.get('data_path'))).parents[0])} --folder_processing warp_frameseries --output warp_frameseries.settings --extension {str(Path(project_params.resolve_path(parameters.get('data_path'))).suffix)} --angpix {parameters.get('scope_pixel')} --gain_path {parameters.get('gain_reference')} {extra_options} --exposure {parameters.get('scope_dose_rate')}"
    local_run.stream_shell_command(command)
    
    # create tilt series settings file
    command = f"{get_warptools_path()}WarpTools create_settings --output warp_tiltseries.settings --folder_processing warp_tiltseries --folder_data tomostar --extension '*.tomostar' --angpix {parameters.get('scope_pixel')} --gain_path {parameters.get('gain_reference')} {extra_options} --exposure {parameters.get('scope_dose_rate')} --tomo_dimensions 4400x6000x1000"
    local_run.stream_shell_command(command)

def fs_motion_and_ctf(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command fs_motion_and_ctf:                                                                                                        [23/2032]
    -------------------------------------------------------------------------------------------Data import settings--------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --m_range_min             Default: 500. Minimum resolution in Angstrom to consider in fit
    --m_range_max             Default: 10. Maximum resolution in Angstrom to consider in fit
    --m_bfac                  Default: -500. Downweight higher spatial frequencies using a B-factor, in Angstrom^2
    --m_grid                  Resolution of the motion model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto
    --c_window                Default: 512. Patch size for CTF estimation in binned pixels
    --c_range_min             Default: 30. Minimum resolution in Angstrom to consider in fit
    --c_range_max             Default: 4. Maximum resolution in Angstrom to consider in fit
    --c_defocus_min           Default: 0.5. Minimum defocus value in um to explore during fitting
    --c_defocus_max           Default: 5. Maximum defocus value in um to explore during fitting
    --c_voltage               Default: 300. Acceleration voltage of the microscope in kV
    --c_cs                    Default: 2.7. Spherical aberration of the microscope in mm
    --c_amplitude             Default: 0.07. Amplitude contrast of the sample, usually 0.07-0.10 for cryo
    --c_fit_phase             Fit the phase shift of a phase plate
    --c_use_sum               Use the movie average spectrum instead of the average of individual frames' spectra. Can help in the absence of an energy filter, or when signal is low.
    --c_grid                  Resolution of the defocus model grid in X, Y, and temporal dimensions, separated by 'x': e.g. 5x5x40; empty = auto; Z > 1 is purely experimental
    --out_averages            Export aligned averages
    --out_average_halves      Export aligned averages of odd and even frames separately, e.g. for denoiser training
    --out_thumbnails          Export thumbnails, scaled so that the long edge has this length in pixels
    --out_skip_first          Default: 0. Skip first N frames when exporting averages
    --out_skip_last           Default: 0. Skip last N frames when exporting averages
    --------------------------------------------------------------------------------------------Work distribution----------------------------------------------------------------
    --device_list             Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice               Default: 1. Number of processes per GPU
    -----------------------------------------------------------------------------------Advanced data import & flow options------------------------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    ------------------------------------------------------------------------------------Advanced remote work distribution-------------------------------------------------------------------------------------
    --workers                 List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces
    """

    # Frame Series Motion and CTF Estimation
    command = f"{get_warptools_path()}WarpTools fs_motion_and_ctf --settings warp_frameseries.settings --m_grid 1x1x3 --c_grid 2x2x1 --c_range_max 7 --c_defocus_max 8 --c_use_sum --out_averages --out_average_halves"
    local_run.stream_shell_command(command)
    
def ts_import(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_import:
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --mdocs              REQUIRED Path to the folder containing MDOC files
    --pattern            Default: *.mdoc. File name pattern to search for in the MDOC folder
    --exclude_pattern    Default: . Don't import MDOC files that contain this pattern in their name, e.g. 'unsorted'
    --frameseries        REQUIRED Path to a folder containing frame series processing results and their aligned averages
    --tilt_exposure      REQUIRED Per-tilt exposure in e-/A^2
    --dont_invert        Don't invert tilt angles compared to IMOD's convention (inversion is usually needed to match IMOD's geometric handedness). This will flip the geometric handedness
    --override_axis      Override the tilt axis angle with this value
    --auto_zero          Adjust tilt angles so that the tilt with the highest average intensity becomes the 0-tilt
    --tilt_offset        Subtract this value from all tilt angle values to compensate pre-tilt
    --max_tilt           Default: 90. Exclude all tilts above this (absolute) tilt angle
    --min_intensity      Default: 0. Exclude tilts if their average intensity is below MinIntensity * cos(angle) * 0-tilt intensity; set to 0 to not exclude anything
    --max_mask           Default: 1. Exclude tilts if more than this fraction of their pixels is masked; needs frame series with BoxNet masking results
    --min_ntilts         Default: 1. Only import tilt series that have at least this many tilts after all the other filters have been applied
    -o, --output         REQUIRED Path to a folder where the created .tomostar files will be saved
    --strict             Ensures that progress report formatting stays the same across all tools.
    """

    command = f"{get_warptools_path()}WarpTools ts_import --mdocs mdoc --frameseries warp_frameseries --tilt_exposure {parameters.get('scope_dose_rate')} --min_intensity 0.3 --dont_invert --output tomostar"
    local_run.stream_shell_command(command)


def ts_etomo_patches(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_etomo_patches:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --angpix                  Rescale tilt images to this pixel size; normally 10–15 for cryo data; leave out to keep the original pixel size
    --mask                    Apply mask to each image if available; masked areas will be filled with Gaussian noise
    --min_fov                 Default: 0. Disable tilts that contain less than this fraction of the tomogram's field of view due to excessive shifts
    --initial_axis            Override initial tilt axis angle with this value
    --do_axis_search          Fit a new tilt axis angle for the whole dataset
    --patch_size              Default: 500. patch size for patch tracking in Angstroms
    --delete_intermediate     Delete tilt series stacks generated for Etomo
    --thumbnails              Create thumbnails for each tilt image using the same pixel size as the stack; only makes sense without delete_intermediate
    ------------------------------------------------------------------------------Work distribution------------------------------------------------------------------------------
    --device_list             Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice               Default: 1. Number of processes per GPU
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    ----------------------------------------------------------------------Advanced remote work distribution----------------------------------------------------------------------
    --workers                 List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces
    """
    
    binned_pixel_size = parameters.get('scope_pixel_size') * parameters.get('tomo_rec_binning')
    
    # produce single patch size (in A)
    patch_size_in_A = ( parameters.get('tomo_ali_patches_size_x') + parameters.get('tomo_ali_patches_size_y') ) / 2.0 / parameters.get('scope_pixel_size')
    
    command = f"{get_warptools_path()}WarpTools ts_etomo_patches --settings warp_tiltseries.settings --angpix {binned_pixel_size} --patch_size {patch_size_in_A} --initial_axis {parameters.get('scope_tilt_axis')}"
    local_run.stream_shell_command(command)
    
def ts_defocus_hand(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_defocus_hand:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --check                   Only check the defocus handedness, but don't set anything
    --set_auto                Check the defocus handedness and set the determined value for all tilt series
    --set_flip                Set handedness to 'flip' for all tilt series
    --set_noflip              Set handedness to 'no flip' for all tilt series
    --set_switch              Switch whatever handedness value each tilt series has to the opposite value
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    """
    command = f"{get_warptools_path()}WarpTools ts_defocus_hand --settings warp_tiltseries.settings --check"
    local_run.stream_shell_command(command)

def ts_ctf(parameters):
    
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_ctf:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --window                  Default: 512. Patch size for CTF estimation in binned pixels
    --range_low               Default: 30. Lowest (worst) resolution in Angstrom to consider in fit
    --range_high              Default: 4. Highest (best) resolution in Angstrom to consider in fit
    --defocus_min             Default: 0.5. Minimum defocus value in um to explore during fitting (positive = underfocus)
    --defocus_max             Default: 5. Maximum defocus value in um to explore during fitting (positive = underfocus)
    --voltage                 Default: 300. Acceleration voltage of the microscope in kV
    --cs                      Default: 2.7. Spherical aberration of the microscope in mm
    --amplitude               Default: 0.07. Amplitude contrast of the sample, usually 0.07-0.10 for cryo
    --fit_phase               Fit the phase shift of a phase plate
    --auto_hand               Run defocus handedness estimation based on this many tilt series (e.g. 10), then estimate CTF with the correct handedness
    ------------------------------------------------------------------------------Work distribution------------------------------------------------------------------------------
    --device_list             Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice               Default: 1. Number of processes per GPU
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    ----------------------------------------------------------------------Advanced remote work distribution----------------------------------------------------------------------
    --workers                 List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces
    """
    command = f"{get_warptools_path()}WarpTools ts_ctf --settings warp_tiltseries.settings --range_high {parameters.get('ctf_max_res')} --defocus_max {parameters.get('ctf_max_def')/10000}"
    local_run.stream_shell_command(command)
    
def ts_reconstruct(parameters):
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_reconstruct:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --angpix                  REQUIRED Pixel size of the reconstructed tomograms in Angstrom
    --halfmap_frames          Also produce two half-tomograms, each reconstructed from half of the frames (requires running align_frameseries with --average_halves previously)
    --halfmap_tilts           Also produce two half-tomograms, each reconstructed from half of the tilts (doesn't work quite as well as --halfmap_frames)
    --deconv                  Also produce a deconvolved version; all half-tomograms, if requested, will also be deconvolved
    --deconv_strength         Default: 1. Strength of the deconvolution filter, if requested
    --deconv_falloff          Default: 1. Fall-off of the deconvolution filter, if requested
    --deconv_highpass         Default: 300. High-pass value (in Angstrom) of the deconvolution filter, if requested
    --keep_full_voxels        Mask out voxels that aren't contained in some of the tilt images (due to excessive sample shifts); don't use if you intend to run template matching
    --dont_invert             Don't invert the contrast; contrast inversion is needed for template matching on cryo data, i.e. when the density is dark in original images
    --dont_normalize          Don't normalize the tilt images
    --dont_mask               Don't apply a mask to each tilt image if available; otherwise, masked areas will be filled with Gaussian noise
    --dont_overwrite          Don't overwrite existing tomograms in output directory
    --subvolume_size          Default: 64. Reconstruction is performed locally using sub-volumes of this size in pixel
    --subvolume_padding       Default: 3. Padding factor for the reconstruction sub-volumes (helps with aliasing effects at sub-volume borders)
    ------------------------------------------------------------------------------Work distribution------------------------------------------------------------------------------
    --device_list             Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice               Default: 1. Number of processes per GPU
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    ----------------------------------------------------------------------Advanced remote work distribution----------------------------------------------------------------------
    --workers                 List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces
    """
    
    command = f"{get_warptools_path()}WarpTools ts_reconstruct --settings warp_tiltseries.settings --angpix 10"
    local_run.stream_shell_command(command)

# optional
def ts_import_alignments(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_import_alignments:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --alignments              REQUIRED Path to a folder containing one sub-folder per tilt series with alignment results from IMOD or AreTomo
    --alignment_angpix        REQUIRED Pixel size (in Angstrom) of the images used to create the alignments (used to convert the alignment shifts from pixels to Angstrom)
    --min_fov                 Default: 0. Disable tilts that contain less than this fraction of the tomogram's field of view due to excessive shifts
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    """
    
    command = f"{get_warptools_path()}WarpTools ts_import_alignments --settings warp_tiltseries.settings --alignments warp_tiltseries/tiltstack/TS_1 --alignment_angpix {parameters.get('tomo_ali_binning')}"
    local_run.stream_shell_command(command)


# Particle picking

def ts_template_match(parameters):
    
    """    
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_template_match:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --tomo_angpix             REQUIRED Pixel size of the reconstructed tomograms in Angstrom
    --template_path           Path to the template file
    --template_emdb           Instead of providing a local map, download the EMDB entry with this ID and use its main map
    --template_angpix         Pixel size of the template; leave empty to use value from map header
    --template_diameter       REQUIRED Template diameter in Angstrom
    --template_flip           Mirror the template along the X axis to flip the handedness; '_flipx' will be added to the template's name
    --symmetry                Default: C1. Symmetry of the template, e.g. C1, D7, O
    --subdivisions            Default: 3. Number of subdivisions defining the angular search step: 2 = 15° step, 3 = 7.5°, 4 = 3.75° and so on
    --tilt_range              Limit the range of angles between the reference's Z axis and the tomogram's XY plane to plus/minus this value, in °; useful for matching filamentslying flat in the XY plane
    --batch_angles            Default: 32. How many orientations to evaluate at once; memory consumption scales linearly with this; higher than 32 probably won't lead to speed-ups
    --peak_distance           Minimum distance (in Angstrom) between peaks; leave empty to use template diameter
    --npeaks                  Default: 2000. Maximum number of peak positions to save
    --dont_normalize          Don't set score distribution to median = 0, stddev = 1
    --whiten                  Perform spectral whitening to give higher-resolution information more weight; this can help when the alignments are already good and you need moreselective matching
    --lowpass                 Default: 1. Gaussian low-pass filter to be applied to template and tomogram, in fractions of Nyquist; 1.0 = no low-pass, <1.0 = low-pass
    --lowpass_sigma           Default: 0.1. Sigma (i.e. fall-off) of the Gaussian low-pass filter, in fractions of Nyquist; larger value = slower fall-off
    --max_missing_tilts       Default: 2. Dismiss positions not covered by at least this many tilts; set to -1 to disable position culling
    --reuse_results           Reuse correlation volumes from a previous run if available, only extract peak positions
    --check_hand              Default: 0. Also try a flipped version of the template on this many tomograms to see what geometric hand they have
    --subvolume_size          Default: 192. Matching is performed locally using sub-volumes of this size in pixel
    --override_suffix         Override the default STAR file suffix derived from the template name; must include the leading underscore if you want to have it
    ------------------------------------------------------------------------------Work distribution------------------------------------------------------------------------------
    --device_list             Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice               Default: 1. Number of processes per GPU
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    ----------------------------------------------------------------------Advanced remote work distribution----------------------------------------------------------------------
    --workers                 List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces
    """
    
    command = f"{get_warptools_path()}WarpTools ts_template_match --settings warp_tiltseries.settings --tomo_angpix 10 --subdivisions 3 --template_emdb 15854 --template_diameter 130 --symmetry O --whiten --check_hand 2"
    local_run.stream_shell_command(command)

def threshold_picks(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command threshold_picks:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --in_suffix               REQUIRED Suffix for the names of the input STAR files (file names will be assumed to match {item name}_{--in_suffix}.star pattern)
    --out_suffix              REQUIRED Suffix for the names of the output STAR files (file names will be {item name}_{--in_suffix}_{--outsuffix}.star)
    --out_combined            Path to a single STAR file into which all results will be combined; internal paths will be made relative to this location
    --minimum                 Remove all particles below this threshold
    --maximum                 Remove all particles above this threshold
    --top_series              Keep this many top-scoring series
    --top_picks               Keep this many top-scoring particles for each series
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data              Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files with one file name per line.
    --input_data_recursive    Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing        Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing       Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata         Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                  Ensures that progress report formatting stays the same across all tools.
    """
    
    command = f"{get_warptools_path()}WarpTools threshold_picks --settings warp_tiltseries.settings --in_suffix 15854 --out_suffix clean --minimum 3"
    local_run.stream_shell_command(command)

def ts_export_particles(parameters):
    
    """
    WarpTools - a collection of tools for EM data pre-processing
    Version 2.0.0
    Showing all available options for command ts_export_particles:
    ----------------------------------------------------------------------------Data import settings-----------------------------------------------------------------------------
    --settings                 REQUIRED Path to Warp's .settings file, typically located in the processing folder. Default file name is 'previous.settings'.
    --------------------------------------------------------------------STAR files with particle coordinates---------------------------------------------------------------------
    --input_star               Single STAR file containing particle poses to be exported
    --input_directory          Directory containing multiple STAR files each with particle poses to be exported
    --input_pattern            Default: *.star. Wildcard pattern to search for from the input directory
    -----------------------------------------------------------------------------Coordinate scaling------------------------------------------------------------------------------
    --coords_angpix            Pixel size for particles coordinates in input star file(s)
    --normalized_coords        Are coordinates normalised to the range [0, 1] (e.g. from Warp's template matching)
    -----------------------------------------------------------------------------------Output------------------------------------------------------------------------------------
    --output_star              REQUIRED STAR file for exported particles
    --output_angpix            REQUIRED Pixel size at which to export particles
    --box                      REQUIRED Output has this many pixels/voxels on each side
    --diameter                 REQUIRED Particle diameter in angstroms
    --relative_output_paths    Make paths in output STAR file relative to the location of the STAR file. They will be relative to the working directory otherwise.
    -----------------------------------------------------------------Export type (REQUIRED, mutually exclusive)------------------------------------------------------------------
    --2d                       Output particles as 2d image series centered on the particle (particle series)
    --3d                       Output particles as 3d images (subtomograms)
    -------------------------------------------------------------------------------Expert options--------------------------------------------------------------------------------
    --dont_normalize_input     Don't normalize the entire field of view in input 2D images after high-pass filtering
    --dont_normalize_3d        Don't normalize output particle volumes (only works with --3d)
    --n_tilts                  Number of tilt images to include in the output, images with the lowest overall exposure will be included first
    --max_missing_tilts        Default: 5. Particles not visible in more than this number of tilts will be excluded (only works with --2d)
    ------------------------------------------------------------------------------Work distribution------------------------------------------------------------------------------
    --device_list              Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice                Default: 1. Number of processes per GPU
    ---------------------------------------------------------------------Advanced data import & flow options---------------------------------------------------------------------
    --input_data               Overrides the list of input files specified in the .settings file. Accepts a space-separated list of files, wildcard patterns, or .txt files withone file name per line.
    --input_data_recursive     Enables recursive search for files matching the wildcard pattern specified in --input_data. Only applicable when processing and directories are separate. All file names must be unique.
    --input_processing         Specifies an alternative directory containing pre-processed results. Overrides the processing directory in the .settings file.
    --output_processing        Specifies an alternative directory to save processing results. Overrides the processing directory in the .settings file.
    --input_norawdata          Ignore the existence of raw data and look for XML metadata in the processing directory instead.
    --strict                   Ensures that progress report formatting stays the same across all tools.
    ----------------------------------------------------------------------Advanced remote work distribution----------------------------------------------------------------------
    --workers                  List of remote workers to be used instead of locally spawned processes. Formatted as hostname:port, separated by spaces
    """
    
    command = f"{get_warptools_path()}WarpTools ts_export_particles --settings warp_tiltseries.settings --input_directory warp_tiltseries/matching --input_pattern '*15854_clean.star' --normalized_coords --output_star relion/matching.star --output_angpix 4 --box 64 --diameter 130 --relative_output_paths --2d"
    local_run.stream_shell_command(command)
    
# 3D refinement

def create_population(parameters):

    """
    MTools 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 MTools
    -d, --directory    Required. Path to the directory where the new population will be located. All future species will also go there, so make sure there is enough space.
    -n, --name         Required. Name of the new population.
    --help             Display this help screen.
    --version          Display version information.
    """
        
    command = f"{get_warptools_path()}MTools create_population --directory m --name {parameters.get('data_set')}"
    local_run.stream_shell_command(command)

def create_source(parameters):
    
    """
    MTools 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 MTools
    -p, --population             Required. Path to the .population file to which to add the new data source.
    -s, --processing_settings    Required. Path to a .settings file used to pre-process the frame or tilt series this source should include; desktop Warp will usually generate a previous.settings file
    -n, --name                   Required. Name of the new data source.
    --nframes                    Maximum number of tilts or frames to use in refinements. Leave empty or set to 0 to use the maximum number available.
    --files                      Optional STAR file with a list of files to intersect with the full list of frame or tilt series referenced by the settings.
    -o, --output                 Optionally, override the default path where the .source file will be saved.
    --dont_version               If set, the source will not be versioned.
    --help                       Display this help screen.
    --version                    Display version information.
    """
    
    command = f"{get_warptools_path()}MTools create_source --name {parameters.get('data_set')} --population m/{parameters.get('data_set')}.population --processing_settings warp_tiltseries.settings"
    local_run.stream_shell_command(command)

def create_species(parameters):
    
    """
    MTools 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 MTools
    -p, --population          Required. Path to the .population file to which to add the new data source.
    -n, --name                Required. Name of the new species.
    -d, --diameter            Required. Molecule diameter in Angstrom.
    -s, --sym                 (Default: C1) Point symmetry, e.g. C1, D7, O.
    --helical_units           (Default: 1) Number of helical asymmetric units (only relevant for helical symmetry).
    --helical_twist           Helical twist in degrees, positive = right-handed (only relevant for helical symmetry).
    --helical_rise            Helical rise in Angstrom (only relevant for helical symmetry).
    --helical_height          Height of the helical segment along the Z axis in Angstrom (only relevant for helical symmetry).
    -t, --temporal_samples    (Default: 1) Number of temporal samples in each particle pose's trajectory.
    --half1                   Required. Path to first half-map file.
    --half2                   Required. Path to second half-map file.
    -m, --mask                Required. Path to a tight binary mask file. M will automatically expand and smooth it based on current resolution
    --angpix                  Override pixel size value found in half-maps.
    --angpix_resample         Resample half-maps and masks to this pixel size.
    --lowpass                 Optional low-pass filter (in Angstrom), applied to both half-maps.
    --particles_relion        Path to _data.star-like particle metadata from RELION.
    --particles_m             Path to particle metadata from M.
    --angpix_coords           Override pixel size for RELION particle coordinates.
    --angpix_shifts           Override pixel size for RELION particle shifts.
    --ignore_unmatched        Don't fail if there are particles that don't match any data sources.
    --dont_use_denoiser       Use low-pass filtering for regularization instead of a denoiser.
    -o, --output              Optionally, override default path where the .species file and all data will be saved.
    --dont_version            If set, the source will not be versioned.
    --help                    Display this help screen.
    --version                 Display version information.
    """

    command = f"{get_warptools_path()}MTools create_species --population m/{parameters.get('data_set')}.population --name apoferritin --diameter 130 --sym O --temporal_samples 1 --half1 relion/Refine3D/job002/run_half1_class001_unfil.mrc --half2 relion/Refine3D/job002/run_half2_class001_unfil.mrc --mask m/mask_4apx.mrc --particles_relion relion/Refine3D/job002/run_data.star --angpix_resample 0.7894 --lowpass 10"
    local_run.stream_shell_command(command)

def mcore_iterate(parameters):

    """
    MCore 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 MCore
    --port                        (Default: 14300) Port to use for REST API calls, set to -1 to disable
    --devicelist                  Space-separated list of GPU IDs to use for processing. Default: all GPUs in the system
    --perdevice_preprocess        Number of processes per GPU used for map pre-processing; leave blank = default to --perdevice_refine value
    --perdevice_refine            (Default: 1) Number of processes per GPU used for refinement; set to >1 to improve utilization if your GPUs have enough memory
    --perdevice_postprocess       Number of processes per GPU used for map pre-processing; leave blank = default to --perdevice_refine value
    --workers_preprocess          List of remote workers to be used instead of locally spawned processes for map pre-processing. Formatted as hostname:port, separated by spaces
    --workers_refine              List of remote workers to be used instead of locally spawned processes for refinement. Formatted as hostname:port, separated by spaces
    --workers_postprocess         List of remote workers to be used instead of locally spawned processes for map post-processing. Formatted as hostname:port, separated by spaces
    --population                  Required. Path to the .population file containing descriptions of data sources and species
    --iter                        (Default: 3) Number of refinement sub-iterations
    --first_iteration_fraction    (Default: 1) Use this fraction of available resolution for alignment in first sub-iteration, increase linearly to 1.0 towards last sub-iterations
    --min_particles               (Default: 1) Only use series with at least N particles in the field of view
    --cpu_memory                  Use CPU memory to store particle images during refinement (GPU by default)
    --weight_threshold            (Default: 0.05) Refine each tilt/frame up to the resolution at which the exposure weighting function (B-factor) reaches this value
    --refine_imagewarp            Refine image warp with a grid of XxY dimensions. Examples: leave blank = don't refine, '1x1', '6x4'
    --refine_particles            Refine particle poses
    --refine_mag                  Refine anisotropic magnification
    --refine_doming               Refine doming (frame series only)
    --refine_stageangles          Refine stage angles (tilt series only)
    --refine_volumewarp           Refine volume warp with a grid of XxYxZxT dimensions (tilt series only). Examples: leave blank = don't refine, '1x1x1x20', '4x6x1x41'
    --refine_tiltmovies           Refine tilt movie alignments (tilt series only)
    --ctf_batch                   (Default: 32) Batch size for CTF refinements. Lower = less memory, higher = faster
    --ctf_minresolution           (Default: 8) Use only species with at least this resolution (in Angstrom) for CTF refinement
    --ctf_defocus                 Refine defocus using a local search
    --ctf_defocusexhaustive       Refine defocus using a more exhaustive grid search in the first sub-iteration; only works in combination with ctf_defocus
    --ctf_phase                   Refine phase shift (phase plate data only)
    --ctf_cs                      Refine spherical aberration, which is also a proxy for pixel size
    --ctf_zernike3                Refine Zernike polynomials of 3rd order (beam tilt, trefoil - fast)
    --ctf_zernike5                Refine Zernike polynomials of 5th order (fast)
    --ctf_zernike2                Refine Zernike polynomials of 2nd order (slow)
    --ctf_zernike4                Refine Zernike polynomials of 4th order (slow)
    --help                        Display this help screen.
    --version                     Display version information.
    """
    
    binary_options = ""
    if parameters.get('mcore_refine_imagewarp'):
        binary_options += " --refine_imagewarp"
    if parameters.get('mcore_refine_particles'):
        binary_options += " --refine_particles"
    if parameters.get('mcore_refine_mag'):
        binary_options += " --refine_mag"
    if parameters.get('mcore_refine_doming'):
        binary_options += " --refine_doming"
    if parameters.get('mcore_refine_stageangles'):
        binary_options += " --refine_stageangles"
    if parameters.get('mcore_refine_volumewarp'):
        binary_options += " --refine_volumewarp"
    if parameters.get('mcore_refine_tiltmovies'):
        binary_options += " --refine_tiltmovies"   

    if parameters.get('mcore_ctf_defocus'):
        binary_options = " --ctf_defocus" 
        if parameters.get('mcore_ctf_defocusexhaustive'):
            binary_options += " --ctf_defocusexhaustive"    
    if parameters.get('mcore_ctfphase'):
        binary_options += " --ctf_phase"
    if parameters.get('mcore_ctf_cs'):
        binary_options += "  --ctf_cs"
    if parameters.get('mcore_ctf_zernike3'):
        binary_options += " --ctf_zernike3"
    if parameters.get('mcore_ctf_zernike5'):
        binary_options += " --ctf_zernike5"
    if parameters.get('mcore_ctf_zernike2'):
        binary_options += " --ctf_zernike2"
    if parameters.get('mcore_ctf_zernike4'):
        binary_options += " --ctf_zernike4"
        
    command = f"{get_warptools_path()}MCore --population m/{parameters.get('data_set')}.population --port {parameters.get('mcore_resources_port')} --devicelist {parameters.get('mcore_resources_devicelist')} --perdevice_preprocess {parameters.get('mcore_resources_perdevice_preprocess')} --perdevice_refine {parameters.get('mcore_resources_perdevice_refine')} --perdevice_postprocess {parameters.get('mcore_resources_perdevice_postprocess')} --workers_preprocess {parameters.get('mcore_resources_workers_preprocess')} --workers_refine {parameters.get('mcore_resources_workers_refine')}  --workers_postprocess {parameters.get('mcore_resources_workers_postprocess')} --iter {parameters.get('mcore_refine_iter')} --first_iteration_fraction {parameters.get('mcore_refine_iteration_fraction')} --min_particles {parameters.get('mcore_refine_min_particles')} --cpu_memory {parameters.get('mcore_refine_cpu_memory')} --weight_threshold {parameters.get('mcore_refine_weight_threshold')} --ctf_batch {parameters.get('mcore_ctf_batch')} --ctf_minresolution {parameters.get('mcore_ctf_minresolution')} {binary_options}"
    local_run.stream_shell_command(command)

def check_setup():
        
    command = f"{get_warptools_path()}MCore --population m/10491.population --iter 0"
    local_run.stream_shell_command(command)

def first_refinement(parameters):
    
    command = f"{get_warptools_path()}MCore --population m/10491.population --refine_imagewarp 6x4 --refine_particles --ctf_defocus --ctf_defocusexhaustive --perdevice_refine 4"
    local_run.stream_shell_command(command)

# Benefits from a Higher Resolution Reference
def second_refinement(parameters):
    
    command = f"{get_warptools_path()}MCore --population m/10491.population --refine_imagewarp 6x4 --refine_particles --ctf_defocus"
    local_run.stream_shell_command(command)

#  Stage Angle Refinement
def third_refinement(parameters):
    
    command = f"{get_warptools_path()}MCore --population m/10491.population --refine_imagewarp 6x4 --refine_particles --refine_stageangles"
    local_run.stream_shell_command(command)

# Magnification/Cs/Zernike3
def fourth_refinement(parameters):
    
    command = f"{get_warptools_path()}MCore --population m/10491.population --refine_imagewarp 6x4 --refine_particles --refine_mag --ctf_cs --ctf_defocus --ctf_zernike3"
    local_run.stream_shell_command(command)

# Weights (Per-Tilt Series)
def fifth_refinement(parameters):
    
    """
    EstimateWeights 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 EstimateWeights
    --population          Required. Path to the .population file.
    --source              Required. Name of the data source.
    --resolve_frames      Estimate weights per frame or tilt. Can be combined with 'resolve_items', resulting in Nitems*Nframes weights.
    --resolve_items       Estimate weights per item, i.e. frame series or tilt series. Can be combined with 'resolve_frames', resulting in Nitems*Nframes weights.
    --resolve_location    Estimate weights that depend on particle position within the image or volume. Cannot be combined with others.
    --resolve_sources     Estimate weights per item, i.e. frame series or tilt series. Can be combined with 'resolve_frames', resulting in Nitems*Nframes weights.
    --fit_anisotropy      Fit anisotropic B-factors. Only makes sense when fitting per-item, per-frame/tilt (except maybe in tilted data where BIM is perpendicular to tilt axis?).
    --do_tiltframes       Estimate weights for tilt movies frames. Only works for tilt series where the original movies are available.
    --grid_width          Width of the parameter grid when fitting spatially resolved weights.
    --grid_height         Height of the parameter grid when fitting spatially resolved weights.
    --min_resolution      (Default: 20) Minimum resolution to consider for estimation.
    --reset               Resets all previously fitted weights for all items; not compatible with any other options.
    --help                Display this help screen.
    --version             Display version information.
    """
    
    binary_options = ""
    if parameters.get('mcore_weights_resolve_frames'):
        binary_options += " --resolve_frames" 
    if parameters.get('mcore_weights_resolve_items'):
        binary_options += " --resolve_items" 
    if parameters.get('mcore_weights_resolve_location'):
        binary_options += " --resolve_location" 
    if parameters.get('mcore_weights_resolve_sources'):
        binary_options += " --resolve_sources" 
    if parameters.get('mcore_weights_fit_anisotropy'):
        binary_options += " --fit_anisotropy" 
    if parameters.get('mcore_weights_do_tiltframes'):
        binary_options += " --do_tiltframes" 
    if parameters.get('mcore_weights_resolve_frames'):
        binary_options += " --resolve_frames" 
    if parameters.get('mcore_weights_reset'):
        binary_options += " --reset"
        
    command = f"{get_warptools_path()}EstimateWeights --population m/10491.population --source 10491 {binary_options} --grid_width {parameters.get('mcore_weights_grid_width')} --grid_height {parameters.get('mcore_weights_grid_height')} --min_resolution {parameters.get('mcore_weights_min_resolution')}"
    local_run.stream_shell_command(command)
    
    command = f"{get_warptools_path()}MCore --population m/10491.population"
    local_run.stream_shell_command(command)

# Weights (Per-Tilt, Averaged over all Tilt Series)
def sixth_refinement(parameters):
    
    command = f"{get_warptools_path()}EstimateWeights --population m/10491.population --source 10491 --resolve_frames"
    local_run.stream_shell_command(command)

    command = f"{get_warptools_path()}MCore --population m/10491.population --perdevice_refine 4 --refine_particles"
    local_run.stream_shell_command(command)

# temporal pose resolution
def sixth_refinement(parameters):
    
    """
    MTools 2.0.0+844343b6327c7775ae56e82e409e82da715c646d
    Copyright (C) 2025 MTools
    -p, --population    Required. Path to the .population file.
    -s, --species       Required. Path to the .species file, or its GUID.
    --samples           Required. The new number of samples, usually between 1 (small particles) and 3 (very large particles).
    --help              Display this help screen.
    --version           Display version information.
    """
    
    command = f"{get_warptools_path()}MTools resample_trajectories --population m/{parameters.get('data_set')}.population --species m/species/apoferritin_797f75c2/apoferritin.species --samples 2"
    local_run.stream_shell_command(command)
    
    command = f"{get_warptools_path()}MCore --population m/{parameters.get('data_set')}.population --refine_imagewarp 6x4 --refine_particles --refine_stageangles --refine_mag --ctf_cs --ctf_defocus --ctf_zernike3"
    local_run.stream_shell_command(command)

# tilt-series pre-processing
def pre_processing(parameters):
    
    settings_file = "warp_frameseries.settings"
    if not os.path.exists(settings_file):
        create_settings(parameters)
    
    # movie frame alignment
    fs_motion_and_ctf(parameters)
    
    # import frame averages
    ts_import(parameters)
    
    # tilt-series alignment
    ts_etomo_patches(parameters)
    
    # tilt-series CTF estimation
    ts_ctf(parameters)
    
    # tomogram reconstruction
    ts_reconstruct(parameters)

# particle picking using template matching
def particle_picking(parameters):
        
    ts_template_match(parameters)
    
    threshold_picks(parameters)
    
    ts_export_particles(parameters)
    
# 3D refinement
def refinement(parameters):

    settings_file = parameters.get('data_set') + ".settings"
    if not os.path.exits(settings_file):

        create_population(parameters)

        create_source(parameters)

        create_species(parameters)

    # actual refinement
    check_setup(parameters)