import os
import subprocess

from pyp.system import local_run
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path, get_tomoctf_path
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def run_slurm_command(command):
    [output, error] = local_run.run_shell_command(command,verbose=False)
    return output, error


def cistem_resize(input_filename, output_filename, new_size, normalize=False):
    """Resize mrc volume using cistem's resize

    Parameters
    ----------
    input_filename : str
        Input mrc file
    output_filename : str
        Output mrc file
    new_size : int
        Size of output in voxels
    normalize : bool, optional
        Normalize and zero-float, by default False

    Returns
    -------
    str, str
        Output and error streams
    """

    """
    **   Welcome to Resize   **

                Version : 1.00
            Compiled : Mar 27 2020
                Mode : Interactive

    Input image file name [input.mrc]                  : 
    Output image file name [output.mrc]                : 
    Is the input a volume [NO]                         : 
    New X-Size [100]                                   :
    New Y-Size [100]                                   :
    New Z-Size [100]                                   :
    Normalize and Zero-float Input? [NO]               :

    Resizing Volume...
    """

    if normalize:
        normalize = "Y"
    else:
        normalize = "N"

    command = (
        f"{os.environ['PYP_DIR']}/external/cistem2/resize << EOF\n"
        f"{input_filename}\n"
        f"{output_filename}\n"
        f"Yes\n"
        f"{new_size}\n"
        f"{new_size}\n"
        f"{new_size}\n"
        f"{normalize}\n"
        "EOF\n"
    )

    output, error = run_slurm_command(command)

    return output, error


def cistem_rescale(input_filename, output_filename, new_size):
    """Rescale mrc volume using cistem's rescale

    Parameters
    ----------
    input_filename : str
        Input mrc file
    output_filename : str
        Output mrc file
    new_size : int
        Size of output in voxels

    Returns
    -------
    str, str
        Output and error streams
    """

    """
            **   Welcome to Resample   **

             Version : 1.00
            Compiled : Mar 27 2020
                Mode : Interactive

    Input image file name [input.mrc]                  : 
    Output image file name [output.mrc]                :
    Is the input a volume [NO]                         : 
    New X-Size [100]                                   :
    New Y-Size [100]                                   :
    New Z-Size [100]                                   :

    Resampling Volume...
    """

    command = (
        f"{os.environ['PYP_DIR']}/external/cistem2/resample << EOF\n"
        f"{input_filename}\n"
        f"{output_filename}\n"
        f"Yes\n"
        f"{new_size}\n"
        f"{new_size}\n"
        f"{new_size}\n"
        "EOF\n"
    )

    output, error = run_slurm_command(command)

    return output, error


def cistem_calculate_fsc(half1, half2, apix, mask, mw):

    """
        **   Welcome to CalculateFSC   **

               Version : 1.00
              Compiled : Mar 27 2020
                  Mode : Interactive

Input reconstruction 1 [my_reconstruction_1.mrc]   :
Input reconstruction 2 [my_reconstruction_2.mrc]   :
Input mask file name [mask.mrc]                    :
Output resolution statistics [my_statistics.txt]   :
Pixel size (A) [1.0]                               :
Inner mask radius (A) [0.0]                        :
Outer mask radius (A) [100.0]                      :
Molecular mass of particle (kDa) [1000.0]          :
Use 3D mask [No]                                   :
    """

    command = (
        f"{os.path.join(get_frealign_paths()['cistem2'],'calculate_fsc')} << EOF\n"
        f"{half1}\n"
        f"{half2}\n"
        f"{mask}\n"
        f"{statustics.txt}\n"
        f"{apix}\n"
        f"{inner_mask_radius}\n"
        f"{outer_mask_radius}\n"
        f"{mw}\n"
        f"{use_mask}\n"
        "EOF\n"
    )

    output, error = run_slurm_command(command)

    return output, error


def avgstack(input_filename, output_filename, start_end_section="/"):
    """Run IMOD binary avgstack.

    From `avgstack`_:
        Avgstack will read sections from an image file and average them.  The
        input file may have any mode.  The mode of the output file will be 2
        (real numbers).  The inputs to the program are:

            Input file name
            Output file name
            Starting and ending section numbers to average (/ for the default,
            all sections in the file, else use '%d,%d,%d...')

    Parameters
    ----------
    input_filename : string
        Filename of the input image file (e.g., .mrc)
    ref_file : string
        Filename of the output image file (e.g., .avg)
    start_end_section : string, optional
        Command separated list of sections to include (default: '/', for all
        sections)

    Returns
    ----------
    output : string
        Output of avgstack run
    error : string or None
        Error of avgstack run, or None on success

    References
    ----------
    .. _avgstack: https://bio3d.colorado.edu/imod/doc/man/avgstack.html
    """
    command = (
        f"{get_imod_path()}/bin/avgstack << EOF\n"
        f"{input_filename}\n"
        f"{output_filename}\n"
        f"{start_end_section}\n"
        "EOF\n"
    )

    output, error = run_slurm_command(command)

    return output, error


def newstack(input_filename, output_filename, threads, **kwargs):
    """Run IMOD binary newstack.

    From `newstack`_:
        Newstack is a general stack editor to move images into, out of, or
        between stacks.  It can float the images to a common range or mean of
        density. It can bin images and apply a general linear transformations
        as well as a specified rotation or expansion. It can put the output
        into a smaller or larger array and independently recenter each image
        separately from the transformation.  Images can be taken from multiple
        input files and placed into multiple output files.

        ...

        INPUT AND OUTPUT FILE OPTIONS
            These options are involved in specifying input and output files.

        -input (-in) OR -InputFile     File name
                Input image file.  Input files may also be entered after all
                arguments on the command line, as long as an output file is the
                last name on the command line.  Files entered with this option
                will be processed before files at the end of the command line,
                if any.  (Successive entries accumulate)

        -output (-ou) OR -OutputFile   File name
                Output image file.  The last filename on the command line will
                also be treated as an output file, following any specified by
                this option.  (Successive entries accumulate)

    Parameters
    ----------
    input_filename : string
        Filename of the input image file (e.g., .mrc).
    ref_file : string
        Filename of the output image file (e.g., .mrc).
    **kwargs
        Optional arguments that conform to documentation in newstack_.

    Returns
    ----------
    output : string
        Output of newstack run.
    error : string or None
        Error of newstack run, or None on success.

    References
    ----------
    .. _newstack: https://bio3d.colorado.edu/imod/doc/man/newstack.html
    """
    # add env for parallel run
    env = "export OMP_NUM_THREADS={0}; export NCPUS={0}; IMOD_FORCE_OMP_THREADS={0}; ".format(threads)
    command = f"{env}{get_imod_path()}/bin/newstack {input_filename} {output_filename}"
    for key, val in kwargs.items():
        # logger.info(f"key {key} val {val}")
        command += f" -{key} {val}"

    output, error = run_slurm_command(command)

    return output, error


def replace_sections(new_stack, original_stack, sections):
    """Replaces specified sections in original_stack with new_stack.

    Parameters
    ----------
    new_stack : str
        Path to stack file that will be used to replace
    original_stack : str
        Path to original stack file that will have sections replaced
    sections : str
        List of section numbers
    """
    command = "{0}/bin/newstack {1} {2} -replace {3}".format(
        get_imod_path(), new_stack, original_stack, sections,
    )
    local_run.run_shell_command(command)


def write_current_particle(
    parameters, scratch_stackfile, film, particle, sections, dryrun=False
):
    """Write out current particle to file."""

    particle_stack = (
        "/scratch/"
        + parameters["data_set"]
        + "_frames_T%02d" % (film)
        + "_P%04d" % (particle)
        + "_stack.mrc"
    )
    command = "{0}/bin/newstack {1} {2} -secs {3}".format(
        get_imod_path(), scratch_stackfile, particle_stack, sections,
    )

    if dryrun:
        return command

    logger.info("Saving " + particle_stack)
    local_run.run_shell_command(command)
    return particle_stack


def tomo_ctf_grad(
    tilt_series_filename,
    tilt_angles_filename,
    output_diag_filename,
    tile_size,
    scope_cs,
    scope_voltage,
    scope_wgh,
    scope_mag,
    dstep,
    ctf_min,
    ctf_min_res,
    ctf_max_res,
    ctf_min_def,
    ctf_max_def,
    fstep,
):
    """Determine tomo defocus gradient direction.

    DETERMINATION OF DEFOCUS GRADIENT DIRECTION, V1.O (Aug-2009)

    Input tilt-series file name:

    Input tilt-angles file name:

    Output diagnostic file name:

    Tile size:

    Input CTF parameters:
    CS[mm],  HT[kV],  AmpCnst,  XMAG,  DStep[um]:

    Input Fitting parameters:
    CTFMin[0,1], ResMin[A], ResMax[A]:
    dFMin[A],   dFMax[A],    FStep:

    Parameters
    ----------
    tilt_series_filename : string
    tilt_angles_filename : string
    output_diag_filename : string
    tile_size : string
    scope_cs : string
    scope_voltage : string
    scope_wgh : string
    scope_mag : string
    dstep : string
    ctf_min : string
    ctf_min_res : string
    ctf_max_res : string
    ctf_min_def : string
    ctf_max_def : string
    fstep : string

    Returns
    ----------
    output : string
        Output of tomo_ctf_grad run.
    error : string or None
        Error of tomo_ctf_grad run, or None on success.
    """

    command = (
        f"{get_tomoctf_path()}/bin/tomoctfgrad.exe << EOF\n"
        f"{tilt_series_filename}\n"
        f"{tilt_angles_filename}\n"
        f"{output_diag_filename}\n"
        f"{tile_size}\n"
        f"{scope_cs},{scope_voltage},{scope_wgh},{scope_mag},{dstep}\n"
        f"{ctf_min},{ctf_min_res},{ctf_max_res}\n"
        f"{ctf_min_def},{ctf_max_def},{fstep}\n"
        "EOF\n"
    )

    output, error = run_slurm_command(command)

    return output, error


'''
    command="""
%s > %s 2>&1 << EOF
%s.mrc
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
No
No
No
No
Yes
Yes
%s
%s
No
EOF
"""  % (timeout_command(ctffind_command, 600), logfile, image_file, output_spectra, float(parameters['data_bin']) * float(parameters['scope_pixel']), parameters['scope_voltage'], parameters['scope_cs'], parameters['scope_wgh'], parameters['ctf_tile'], parameters['ctf_min_res'], parameters['ctf_max_res'], mindefocus, maxdefocus, parameters['ctf_fstep'], tilt_axis, tilt_angle)


    # Input image file name [TS_06_29.mrc]               :
    # Output diagnostic image file name
    # [diagnostic_output.mrc]                            :
    # Pixel size [0.675]                                 :
    # Acceleration voltage [300.0]                       :
    # Spherical aberration [2.70]                        :
    # Amplitude contrast [0.07]                          :
    # Size of amplitude spectrum to compute [512]        :
    # Minimum resolution [50]                            :
    # Maximum resolution [10]                            :
    # Minimum defocus [20000]                            :
    # Maximum defocus [24000]                            :
    # Defocus search step [100.0]                        :
    # Do you know what astigmatism is present? [No]      :
    # Slower, more exhaustive search? [No]               :
    # Use a restraint on astigmatism? [No]               :
    # Find additional phase shift? [No]                  :
    # Determine sample tilt? [y]                         :
    # Do you want to set expert options? [y]             :
    # Resample micrograph if pixel size too small? [Yes] :
    # Do you already know the defocus? [No]              :
    # Do you already know the tilt-geometry? [yes]       :
    # Known tilt angle [45]                              :
    # Known tilt axis [85.3]                             :
    # Desired number of parallel threads [10]            :


    # print command
    [ output, error ] = local_run.run_shell_command(command)
    # print output
    # print error




    # TODO: write the following out to wrapper function
    spectra = 'ctffind4.mrc'
    mrc.write( C, spectra ) 
    
    logfile = 'ctffind4.log'

    ctffind_command = 'ctffind-4.1.5/bin/ctffind --amplitude-spectrum-input'

    command="""
%s > %s 2>&1 << EOF
%s
power.mrc
%s
%s
%s
%s
%s
%s
%s
%s
%s
%s
no
no
yes
%s
no
no
EOF
"""  % ( timeout_command(ctffind_command, 600), logfile, spectra, float(parameters['extract_bin']) * float(parameters['scope_pixel']), parameters['scope_voltage'], parameters['scope_cs'], parameters['scope_wgh'], parameters['ctf_tile'], parameters['ctf_min_res'], parameters['ctf_max_res'], mindefocus, maxdefocus, parameters['ctf_fstep'], parameters['ctf_dast'] )
        
    # Input image file name
    # Input is a movie (stack of frames)           [yes]
    # Number of frames to average together           [4]
    # Output diagnostic filename [diagnostic_output.mrc]
    # Pixel size                                [0.3185]
    # Acceleration voltage                       [300.0]
    # Spherical aberration                         [2.7]
    # Amplitude contrast                          [0.07]
    # Size of power spectrum to compute            [512]
    # Minimum resolution                          [30.0]
    # Maximum resolution                             [3]
    # Minimum defocus                           [5000.0]
    # Maximum defocus                          [50000.0]
    # Defocus search step                        [500.0]
    # Do you know what astigmatism is present?      [no]
    # Slower, more exhaustive search?              [yes]
    # Use a restraint on astigmatism?              [yes]
    # Expected (tolerated) astigmatism           [200.0]
    # Find additional phase shift?                  [no]
    # Do you want to set expert options?             [y]

    # print command
    [ output, error ] = subprocess.Popen( command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
'''


"""
frealign_v9.11/bin/refine3d << eot >>spr_frames_00_04_r01_08_msearch_n.log_0000001_0000027 2>&1
../spr_frames_00_04_stack.mrc
spr_frames_00_04_r01_08.par
spr_frames_00_04_r01_07.mrc
statistics_r01.txt
yes
spr_frames_00_04_r01_08_match.mrc_0000001_0000027
spr_frames_00_04_r01_08.par_0000001_0000027
/dev/null
O
1
27
1.08
300.0
2.7
0.07
400.0
65
100.0
3.0
30.0
8
97.5
3.0
200
20
0
0
0
0
0
0
500.0
50.0
1
no
yes
yes
yes
yes
yes
yes
no
no
no
no
eot
"""
