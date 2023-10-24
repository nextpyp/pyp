"""PYP tests.

Includes regression test suites for SPR, TOMO, and CSP.

SPR tests can take up to 10 minutes. Please be patient.
"""

# TODO: (1) Try to put these files to compare in a textfile.
# (2) Further modularize each of the tests in TestSPR; will also be useful for TOMO and CSP.

import datetime
import itertools
import os
import sys
import re
import glob
import shutil
import subprocess
import tempfile
import math
import time
from distutils import dir_util, file_util
from pathlib import Path
from pprint import pprint as pp

import numpy as np
import pytest
from numpy.ma import array
from numpy.testing import assert_allclose
from scipy.signal.signaltools import decimate


sys.path.append('../src')
# from pyp.system.utils import get_slurm_path
from pyp.inout.image import mrc
from pyp.inout.metadata import frealign_parfile


# initialize arguments
@pytest.fixture(scope="module", autouse=True)
def set_options(get_options):
    """Set the global variables for this test sequence."""
    global SAVE_RESULTS
    global SLURM_MODE
    global FILESYS_PREFIX

    SAVE_RESULTS = get_options["save_results"]
    SLURM_MODE = get_options["slurm_mode"]
    FILESYS_PREFIX = get_options["filesys_prefix"]


flatten = itertools.chain.from_iterable


def cat(input):
    """Join an iterable of text after stringifying."""
    inputt = map(str, input)
    return "".join(inputt)


def parse_commands(config_file):
    """Parse a newline separated file of commands into a list."""
    with open(config_file) as f:
        commands = f.read().splitlines()
        return commands


def assert_compare_volumes(vol, vol2):

    arr1 = mrc.read(vol)
    arr2 = mrc.read(vol2)

    assert(arr1.shape == arr2.shape)
    assert((arr1 == arr2).all())

    print("SUCCESS: {} matches {}".format(vol, vol2))


def run_spr_jobs(command, slurm=True):
    """Run and complete the SPR test jobs.
    
    Parameters
    ----------
    command : string
        Shell command to run job
    slurm : bool
        Whether to use slurm
    """
    SPR_SWARM_PATTERN = r"Submitting 2 job\(s\) to batch system \((\d+)\)"
    SPR_MERGE_PATTERN = (
        r"Submitting 1 job\(s\) to batch system \(Submitted batch job (\d+)\)"
    )

    patterns = (SPR_SWARM_PATTERN, SPR_MERGE_PATTERN)
    if slurm:
        run_slurm_job(command, patterns)
    else:
        run_shell_command(command)


def run_fyp_jobs(command, slurm=True):
    """Run and complete the FYP test jobs.
    
    Parameters
    ----------
    command : string
        Shell command to run job
    slurm : bool
        Whether to use slurm
    """
    FYP_SWARM_PATTERN = r"Submitting 307 jobs to batch system \((\d+)\)"
    FYP_MERGE_PATTERN = (
        r"Submitting 1 jobs to batch system \(Submitted batch job (\d+)\)"
    )

    patterns = (FYP_SWARM_PATTERN, FYP_MERGE_PATTERN)
    if slurm:
        run_slurm_job(command, patterns)
    else:
        run_shell_command(command)


def run_tomo_jobs(command, slurm=True):
    """Run and complete the TOMO test jobs.
    
    Parameters
    ----------
    command : string
        Shell command to run job
    slurm : bool
        Whether to use slurm
    """
    TOMO_SWARM_PATTERN = r"Submitting 1 job\(s\) to batch system \((\d+)\)"
    TOMO_MERGE_PATTERN = (
        r"Submitting 1 job\(s\) to batch system \(Submitted batch job (\d+)\)"
    )

    patterns = (TOMO_SWARM_PATTERN, TOMO_MERGE_PATTERN)
    if slurm:
        run_slurm_job(command, patterns)
    else:
        run_shell_command(command)


def run_csp_spr_jobs(command, slurm=True):
    """Run and complete the CSP SPR test jobs.
    
    Parameters
    ----------
    command : string
        Shell command to run job
    slurm : bool
        Whether to use slurm
    """
    CSP_SPR_SWARM_PATTERN = r"Submitting 3 jobs to batch system  \((\d+)\)"
    CSP_SPR_MERGE_PATTERN = (
        r"Submitting 1 jobs to batch system \(Submitted batch job (\d+)\)"
    )

    patterns = (CSP_SPR_SWARM_PATTERN, CSP_SPR_MERGE_PATTERN)
    if slurm:
        run_slurm_job(command, patterns)
    else:
        run_shell_command(command)


def run_slurm_job(command, patterns):
    """Run SLURM job and wait until completion.

    Parameters
    ----------
    command : string
        Shell command to run SLURM job
    patterns : List[str]
        Patterns to capture SLURM jobids from stdout
    """
    out = run_shell_command(command)
    job_ids = list(
        flatten(re.findall(pattern, out, re.MULTILINE) for pattern in patterns)
    )

    print("job ids obtained", job_ids)
    wait_till_complete(job_ids)


def run_shell_command(command):
    """Run shell command and return stdout."""
    # print("Running command", command)
    out = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        text=True,
    )
    # print("Completed with output:")
    print(out.stdout)
    return out.stdout


def wait_till_complete(job_ids):
    """Wait for the completion of given jobs if not throw exception when timeout."""
    #PRINT_JOB_IDS_COMMAND = "squeue -u $USER | awk '{print $1}'"
    SLEEP_TIME = 15
    TIMEOUT_TIME = 60 * 15  # 15 minutes

    merge_job = max(list(map(lambda x: int(x.split("_")[0]), job_ids)))
    start_time = time.time()
    curr_point = 0

    slurm_output = Path(".") / "swarm" / f"slurm-{merge_job}.out"

    while True:
        wait_time = time.time() - start_time
        if wait_time > TIMEOUT_TIME:
            raise RuntimeError("Slurm job {} TIMEOUT !!".format(merge_job))

        if slurm_output.exists():
            if "PYP terminated successfully" in slurm_output.read_text():
                print(f"Merge job {merge_job} completed!")
                break
        
        if math.floor(wait_time/SLEEP_TIME) >= curr_point: 
            print(f"Waiting slurm job {merge_job} to complete {math.floor(wait_time/SLEEP_TIME) * '.'}")
            curr_point += 1 
        
        time.sleep(SLEEP_TIME)



def assert_files_match(
    output_file, ref_file, comments=["C", "PTLIDX"], rtol=1e-5, atol=1e-8, mask_nan=False
):
    """Check that text files match.

    Only compares non NaNs.

    Parameters
    ----------
    output_file : string
        Path of the output file produced by the test
    ref_file : string
        Path of the output file produced by the test
    comments : string
        Comment lines starts with this symbol
    rtol : float
        Relative error tolerance which is a fraction of the ref_file values
    atol : float
        Absolute error tolerance independent of ref_file values
    mask_nan : bool
        Whether to ignore NaN elements in either file
    """

    if "mrc" in str(output_file):
        assert_compare_volumes(output_file, ref_file)
    else: 
        cols = [] if not str(output_file).endswith("_boxes3d.txt") else [0, 1, 2, 3, 4]
        output, ref = [np.loadtxt(f, comments=comments, usecols=cols) for f in (output_file, ref_file)]
        
        if mask_nan:
            mask = ~(np.isnan(output) | np.isnan(ref))
            assert_allclose(output[mask], ref[mask], rtol, atol)
        else:
            assert_allclose(output, ref, rtol, atol)

        print("SUCCESS: {} matches {}".format(output_file, ref_file))




@pytest.fixture(scope="module")
def create_dirs():
    """Copy relevant files from the respective project folder (SPR, TOMO or CSP) into a temp dir.

    The test PYP/FYP commands will be run in this temp dir before the dir is erased.
    """

    # (SAVE_RESULTS, SLURM_MODE, FILESYS_PREFIX) = set_options

    old_dir = Path().cwd()

    os.makedirs(FILESYS_PREFIX, exist_ok=True)
    new_dir = Path(tempfile.mkdtemp(prefix=FILESYS_PREFIX))

    print("old dir", old_dir)
    print("new dir", new_dir)

    def _create_dirs(mode, folders, files):
        for d in folders:
            os.makedirs(new_dir / mode / d, exist_ok=True)

        for f in files:
            for _f in glob.glob(str(old_dir / mode /f)):
                file = Path(_f).relative_to(old_dir / mode)
                file_util.copy_file(old_dir / mode / file, new_dir / mode / file)

        os.chdir(new_dir / mode)
        print("Copied files", list(os.walk(Path.cwd())))

        # write mpirun.mynodes
        if not SLURM_MODE:
            import contextlib

            nodes = run_shell_command(
                "scontrol show hostname $SLURM_JOB_NODELIST"
            ).splitlines()
            with open(new_dir / mode / "frealign/mpirun.mynodes", "w") as f:
                with contextlib.redirect_stdout(f):
                    for n in nodes:
                        # double write hack
                        print(n)
                        print(n)
            # unset
            if os.environ.get("SLURM_SUBMIT_DIR"):
                print("deleting SLURM_SUBMIT_DIR", os.environ.get("SLURM_SUBMIT_DIR"))
                del os.environ["SLURM_SUBMIT_DIR"]

        return (old_dir / mode, new_dir / mode)

    yield _create_dirs

    if not SAVE_RESULTS:
        os.chdir(old_dir)
        print("removing dir", new_dir)
        shutil.rmtree(new_dir)


@pytest.fixture(scope="class")
def use_spr_temp_dirs(request, create_dirs):
    """Create SPR folders and populate with files."""
    mode = "spr"
    folders = ["raw"]
    files = [
        ".pyp_history",
        ".csp_history",
        ".pyp_config.toml",
        "raw/K3_gain.mrc",
        "apoferrintin.mrc",
        "raw/20201009_1138_NBtest_A007_G001_H079_D007.tif",
        "raw/20201009_1138_NBtest_A007_G009_H097_D007.tif",
    ]

    old_dir, new_dir = create_dirs(mode, folders, files)
    request.cls.old_dir = old_dir
    request.cls.new_dir = new_dir

@pytest.fixture(scope="class")
def use_csp_temp_dirs(request, create_dirs):
    """Create csp folders and populate with files."""
    mode = "spr"
    folders = ["csp", "frealign/maps"]
    files = [
        "frealign/spr_01.par",
        "frealign/maps/spr_r01_02.mrc",
        "frealign/maps/spr_r01_02.par",
        "csp/20201009_1138_NBtest_A007_G009_H097_D007.allparxs",
        "csp/20201009_1138_NBtest_A007_G001_H079_D007.allparxs",
        "csp/20201009_1138_NBtest_A007_G009_H097_D007.allboxes",
        "csp/20201009_1138_NBtest_A007_G001_H079_D007.allboxes",
    ]

    old_dir, new_dir = create_dirs(mode, folders, files)
    request.cls.old_dir = old_dir
    request.cls.new_dir = new_dir

@pytest.mark.usefixtures("use_spr_temp_dirs")
@pytest.mark.skip(reason="")
class TestSPR:
    """Suite of tests for complete standard SPR pipeline."""

    @pytest.fixture(scope="class")
    def spr_commands(self):
        command_files = [".pyp_history", ".csp_history"]
        commands = list(flatten(parse_commands(f) for f in command_files))
        print("All commands are")
        print(commands)
        return iter(commands)

    def test_spr_preprocess_to_pick(self, spr_commands):

        """Test simple SPR pipeline from preprocess to picking.

        Directory:
        .

        Command:
        pyp -scope_pixel 1.08 -detect_method all -detect_rad 60 -gain_flipv -particle_mw 400 -particle_sym O 
        -gain_reference `pwd`/raw/K3_gain.mrc 
        -slurm_tasks 2 -data_mode spr -extract_bin 1 -extract_box 192 -extract_bnd 192  -particle_rad 70

        """
        print("In first SPR test")


        print("original dir", self.old_dir)
        print("current dir", self.new_dir)

        command = next(spr_commands)


        # func to call pyp
        run_spr_jobs(command, slurm=SLURM_MODE)

        # compile/create list of all the .xf, .boxx, .ctf files
        files_to_compare = [
            "ali/20201009_1138_NBtest_A007_G001_H079_D007.xf",
            "ali/20201009_1138_NBtest_A007_G009_H097_D007.xf",
            "box/20201009_1138_NBtest_A007_G001_H079_D007.box",
            "box/20201009_1138_NBtest_A007_G009_H097_D007.box",
            "box/20201009_1138_NBtest_A007_G001_H079_D007.boxx",
            "box/20201009_1138_NBtest_A007_G009_H097_D007.boxx",
            "ctf/20201009_1138_NBtest_A007_G001_H079_D007.ctf",
            "ctf/20201009_1138_NBtest_A007_G009_H097_D007.ctf",
        ]

        # compare all of these files with reference
        [
            assert_files_match(self.new_dir / f, self.old_dir / f)
            for f in files_to_compare
        ]

    def test_spr_refinement(self, spr_commands):
        """Test simple SPR refinement and produce a visual output for comparison.

        Using `-mode 1` refinement
        
        Directory:
        frealign/

        Command:
        fyp -refine_dataset spr_frames_00_04 -refine_model `pwd`/apoferritin.mrc \
        -particle_sym O -refine_rhref 8:7:6:4:3 -cutoff 1 -refine_metric new \
        -refine_mode 1 -refine_iter 8 -refine_maxiter 8
        """
        print("In 2nd SPR test: global intial refinement")

        command = next(spr_commands)


        # func to call fyp
        run_spr_jobs(command, slurm=SLURM_MODE)

        # check the fsc, res, and par files
        files_to_compare = [
            "frealign/maps/spr_r01_02.par",
            "frealign/maps/spr_r01_02.mrc",
        ]

        # compare all of these files with reference
        [
            assert_files_match(self.new_dir / f, self.old_dir / f)
            for f in files_to_compare
        ]

    @pytest.mark.usefixtures("use_csp_temp_dirs")
    def test_spr_reconstruction(self, spr_commands):
        """Test simple SPR reconstruction with given par file.
        
        Run `-mode 0` reconstruction. 

        Directory:
        frealign/

        Command:
        fyp -dataset spr_frames_00_04 -model `pwd`/apoferritin.mrc
        -symmetry O -rhref 8:7:6:4:3 -cutoff 1 -metric new
        -mode 0 -iter 9 -maxiter 9
        """

        print("In fourth SPR test: mode 0 reconstruction")

        command = next(spr_commands)


        # func to call fyp
        run_spr_jobs(command, slurm=SLURM_MODE)

        # check the fsc, res, and par files
        files_to_compare = [
            "frealign/maps/spr_r01_03.par",
            "frealign/maps/spr_r01_03.mrc",
            "frealign/maps/spr_r01_fsc.txt",
            "frealign/maps/spr_r01_res.txt",
        ]

                # compare all of these files with reference
        [
            assert_files_match(self.new_dir / f, self.old_dir / f)
            for f in files_to_compare
        ]


@pytest.fixture(scope="class")
def use_tomo_temp_dirs(request, create_dirs):
    """Create TOMO folders and populate with files."""
    mode = "tomo"
    folders = ["raw", "frealign", "mod"]
    files = [
        ".pyp_history",
        "raw/gain.mrc",
        "raw/*.tif", 
        "mod/*.spk",
        "tomo_csp_ref.mrc",
        "pyp_update_volumes.txt"
    ]

    old_dir, new_dir = create_dirs(mode, folders, files)
    request.cls.old_dir = old_dir
    request.cls.new_dir = new_dir


@pytest.mark.usefixtures("use_tomo_temp_dirs")
class TestTOMO:
    """Suite of tests for complete standard TOMO pipeline."""

    @pytest.fixture(scope="class")
    def tomo_commands(self):
        command_files = [
            ".pyp_history",
        ]
        commands = list(flatten(parse_commands(f) for f in command_files))
        print("All commands are")
        print(commands)
        return iter(commands)

    @pytest.fixture()
    def use_frealign_dir(self):
        """Temporarily change to frealign/ dir."""
        main_dir = Path().cwd()
        os.chdir("frealign/")
        yield
        os.chdir(main_dir)

    def test_tomo_preprocess_to_pick(self, tomo_commands):
        """Test simple TOMO pipeline from preprocess to picking.
        """
        TILTSERIES = "TS_01"

        print("TOMO preprocessing test")

        print("original dir", self.old_dir)
        print("current dir", self.new_dir)

        command = next(tomo_commands)

        # func to call pyp
        run_tomo_jobs(command, slurm=SLURM_MODE)

        # compile/create list of all the .xf files
        frame_xf = [str(Path(f).relative_to(self.old_dir)) for f in glob.glob(str(self.old_dir/f"ali/{TILTSERIES}_*.xf"))] 
        files_to_compare = [f"raw/{TILTSERIES}.rawtlt", 
                            f"raw/{TILTSERIES}.order", 
                            f"ali/{TILTSERIES}.prexg", 
                            f"ali/{TILTSERIES}.xf", 
                            f"ali/{TILTSERIES}.tlt"] + frame_xf 

        # compare all of these files with reference
        [
            assert_files_match(self.new_dir / f, self.old_dir / f)
            for f in files_to_compare
        ]

        # compare ctf files ignoring NaNs
        ctf_files_to_compare = [f"ctf/{TILTSERIES}.ctf", f"ctf/{TILTSERIES}.def"]
        [
            assert_files_match(self.new_dir / f, self.old_dir / f, mask_nan=True)
            for f in ctf_files_to_compare
        ]

        # check extracted sub-volumes
        new_subvolumes = glob.glob(str(self.new_dir/f"sva/{TILTSERIES}_spk????.rec"))
        old_subvolumes = glob.glob(str(self.old_dir/f"sva/{TILTSERIES}_spk????.rec"))

        assert( len(new_subvolumes) == len(old_subvolumes) )

        for vol, vol2 in zip(old_subvolumes, new_subvolumes):
            assert_compare_volumes(vol, vol2)

        
    


    def test_tomo_csp(self, tomo_commands):
        """ Test TOMO CSP

        Args:
            tomo_commands (iterator): commands issued for tomo csp
        """
        TILTSERIES = "TS_01"

        command = next(tomo_commands)

        # func to call pyp
        run_tomo_jobs(command, slurm=SLURM_MODE)

        files_to_compare = [
            "frealign/maps/tomo_r01_02.par", 
            "frealign/maps/tomo_r01_fsc.txt", 
            f"csp/{TILTSERIES}_boxes3d.txt",
            f"csp/{TILTSERIES}.allboxes"
        ]

        # first decompress parfile
        new_par_zip = self.new_dir/f"frealign/maps/tomo_r01_02.par.bz2"
        os.chdir("frealign/maps")
        frealign_parfile.Parameters.decompress_file(new_par_zip , threads=1)
        os.chdir("../..")

        # compare all of these files with reference
        [
            assert_files_match(self.new_dir / f, self.old_dir / f)
            for f in files_to_compare
        ]

        



@pytest.fixture(scope="class")
def use_csp_spr_temp_dirs(request, create_dirs):
    """Create CSP SPR folders and populate with files."""
    mode = "csp_spr"
    folders = ["raw", "ali", "box", "ctf", "frealign/maps"]
    files = [
        ".pyp_history",
        ".pyp_config.toml",
        "csp_spr.micrographs",
        "csp_spr.films",
        "raw/Gain.mrc",
        "raw/movie_01.tif",
        "raw/movie_02.tif",
        "ali/movie_01.blr",
        "ali/movie_01.xf",
        "ali/movie_01_xray.mod",
        "ali/movie_02.blr",
        "ali/movie_02.xf",
        "ali/movie_02_xray.mod",
        "box/movie_01.box",
        "box/movie_01.boxx",
        "box/movie_02.box",
        "box/movie_02.boxx",
        "ctf/movie_01_avgrot.txt",
        "ctf/movie_01.ctf",
        "ctf/movie_02_avgrot.txt",
        "ctf/movie_02.ctf",
        "frealign/frealign.config",
        "frealign/maps/cls_1_spr_frames_00_04_r01_08.par",
        "frealign/maps/cls_1_spr_frames_00_04_r01_08.mrc",
    ]

    old_dir, new_dir = create_dirs(mode, folders, files)
    request.cls.old_dir = old_dir
    request.cls.new_dir = new_dir


@pytest.mark.usefixtures("use_csp_spr_temp_dirs")
@pytest.mark.skip(reason="")
class TestCSP_SPR:
    """Suite of tests for complete CSP SPR pipeline."""

    @pytest.fixture(scope="class")
    def csp_spr_commands(self):
        command_files = [
            ".pyp_history",
        ]
        commands = list(flatten(parse_commands(f) for f in command_files))
        print("All commands are")
        print(commands)
        return iter(commands)

    @pytest.fixture()
    def use_frealign_dir(self):
        """Temporarily change to frealign/ dir."""
        main_dir = Path().cwd()
        os.chdir("frealign/")
        yield
        os.chdir(main_dir)

    def test_csp_spr_preprocess_to_pick(self, csp_spr_commands):
        """Test CSP SPR pipeline.

        Directory:
        .

        Command:
        pyp -data_set csp_spr -extract_cls 1 -movie_ali frealign_spline -extract_fmt frealign_local 
        -csp_no_stacks True -tasks_per_arr 2 
        -class_par `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.par 
        -class_ref `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.mrc 
        -csp_refine `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.par 
        -refine_model `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.mrc 
        -data_gain `pwd`/raw/Gain.mrc
        
        Compares the resulting files in the frealign/maps folder:
        ./frealign/maps:
        
        """
        print("In first CSP SPR test")

        print("original dir", self.old_dir)
        print("current dir", self.new_dir)

        command = next(csp_spr_commands)

        assert command == (
            "pyp -data_set csp_spr -extract_cls 1 -movie_ali frealign_spline "
            "-extract_fmt frealign_local -csp_no_stacks True -tasks_per_arr 2 "
            "-class_par `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.par "
            "-class_ref `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.mrc "
            "-csp_refine `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.par "
            "-refine_model `pwd`/frealign/maps/cls_1_spr_frames_00_04_r01_08.mrc "
            "-data_gain `pwd`/raw/Gain.mrc"
        )

        # func to call pyp
        run_csp_spr_jobs(command, slurm=SLURM_MODE)

        # rename files
        INITIAL_PATTERN = r"csp_(\d+)_(\d+)_csp_spr_frames_00_04"
        NEW_PATTERN = r"cls_1_csp_spr_frames_00_04"

        main_dir = Path().cwd()
        os.chdir("frealign/maps")
        for f in os.listdir(Path()):
            print("Trying to match", f)
            if re.match(INITIAL_PATTERN, f):
                new_fname = re.sub(INITIAL_PATTERN, NEW_PATTERN, f)
                try:
                    print("Renaming {} to {}".format(f, new_fname))
                    os.rename(f, new_fname)
                except:
                    pass
                else:
                    print("Rename successful!")

        os.chdir(main_dir)

        files_to_compare = ["frealign/maps/cls_1_csp_spr_frames_00_04_r01_02.par"]

        # compare all of these files with reference
        [
            assert_files_match(self.new_dir / f, self.old_dir / f)
            for f in files_to_compare
        ]
