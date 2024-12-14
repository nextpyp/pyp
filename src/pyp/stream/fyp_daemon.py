import datetime
import glob
import math
import multiprocessing
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path, PosixPath
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from pyp.inout.image import mrc
from pyp.inout.metadata import frealign_parfile, pyp_metadata, generateRelionParFileNew
from pyp.inout.metadata.core import spa_extract_coordinates_legacy, get_max_resolution
from pyp.refine.frealign import frealign
from pyp.streampyp.logging import TQDMLogger
from pyp.system import mpi, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_frealign_paths
from pyp.utils import get_relative_path, timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def get_existing_films(parameters: dict) -> List[str]:
    try:
        with open(Path.cwd().parent / f"{parameters['data_set']}.films") as f:
            existing_films = [line.strip() for line in f.readlines()]
    except:
        existing_films = []
    return existing_films


def get_box_files(box_dir: PosixPath) -> List[PosixPath]:
    box_files = [file for file in box_dir.iterdir() if file.suffix == ".box"]
    return sorted(box_files)

def get_pkl_files(meta_dir: PosixPath) -> List[PosixPath]:
    # only look at the pkl files that have particles picked 
    pkl_files = [file for file in meta_dir.iterdir() if file.suffix == ".pkl" and "box" in pyp_metadata.LocalMetadata(file).data]
    return sorted(pkl_files)

def get_allboxes_and_allparxs_from_box(
    pkl_file: PosixPath, parameters: dict
) -> Tuple[List[List[float]], List[List[str]]]:

    assert pkl_file.suffix == ".pkl", f"Not a pkl file: {pkl_file}"
    meta_dir = pkl_file.parent

    project_folder = meta_dir.parent
    current_directory = Path.cwd()
    os.chdir(project_folder)
    try:
        [allboxes, allparxs] = spa_extract_coordinates_legacy(
            filename=pkl_file.stem,
            parameters=parameters,
            only_inside=False,
            use_frames=False,
            use_existing_frame_alignments=False,
            path=meta_dir
        )
    except:
        type, value, traceback = sys.exc_info()
        sys.__excepthook__(type, value, traceback)
        logger.warning("Particle coordinates extraction failed, maybe a restart or cleaning is happening?")
        os.chdir(current_directory)
        return [], []

    os.chdir(current_directory)

    return allboxes, allparxs


def write_allparxs_to_file(
    output_path: PosixPath, allparxs: List[List[str]]
) -> None:
    with open(output_path, "w") as f:
        # Use only the columns that are useful for 2D classification
        for item in allparxs[0]:
            item = item.split()[:15] # Only the first 15 are useful for 2D classification
            item = tuple([float(x) for x in item])
            # Set the width for the columns (see frealign_parfile.NEW_PAR_STRING_TEMPLATE_WO_NO)
            # Only 15 columns fit into this template
            f.writelines(f"{frealign_parfile.NEW_PAR_STRING_TEMPLATE_WO_NO}\n" % item)


def get_positions_and_new_particle_count_from_box_files(
    prev_name: str,
    boxes_lists: dict,
    threshold_type: str,
    meta_dir: PosixPath,
    parameters: dict,
    allparxs_dir: PosixPath
) -> int:

    new_particles = 0
    old_boxes_lists = boxes_lists.copy()

    project_directory = Path().cwd().parent
    mrc_dir = project_directory / "mrc"
    flag = {}
    # Loop until we have enough particles
    while new_particles < parameters[threshold_type]:

        number_of_particles_changed = False

        check_list = get_file_indicator(mrc_dir=mrc_dir)
        pkl_files = [
            pkl_file
            for pkl_file in get_pkl_files(meta_dir=meta_dir)
            if pkl_file.stem not in boxes_lists and pkl_file.stem in check_list
            and get_max_resolution(pkl_file.stem,path=meta_dir) < parameters['class2d_ctf_min_res']
        ]

        if len(pkl_files) > 0:
            logger.info(f"Generating metadata for {len(pkl_files):,} new images with CTF resolution better than {parameters['class2d_ctf_min_res']} A")

            # generate allparx files from new micrographs
            with tqdm(desc="Progress", total=len(pkl_files), file=TQDMLogger()) as pbar:
                for file in pkl_files:
                    allboxes, allparxs = get_allboxes_and_allparxs_from_box(file, parameters)
                    film_name = file.stem
                    allparxs_fpath = allparxs_dir / f"{film_name}.allparxs"
                    write_allparxs_to_file(output_path=allparxs_fpath, allparxs=allparxs)

                    # Validate the file we just created
                    with open(allparxs_fpath, "r") as fh:
                        lines = fh.readlines()

                        if "nan" in lines:
                            logger.warning(
                                f"Frealign parameter file contains NANs. Removing {allparxs_fpath}"
                            )
                            os.remove(allparxs_fpath)
                        else:
                            if parameters["slurm_verbose"]:
                                logger.info(f"Number of particles from {film_name}: {len(lines)}")
                            if len(lines) != len(allboxes):
                                logger.error(
                                    "Number of particles does not match number of coordinates to extract"
                                    f"Removing {allparxs_fpath}"
                                )
                                os.remove(allparxs_fpath)
                            else:
                                boxes_lists[film_name] = allboxes
                                new_particles += len(lines)

                                number_of_particles_changed = True
                    pbar.update(1)

        if len(boxes_lists) and (len(boxes_lists)-len(old_boxes_lists)) > 0 and number_of_particles_changed:
            logger.info(f"{len(boxes_lists)-len(old_boxes_lists):,} micrographs, {new_particles:,} particles detected so far")

        flag = detect_flags(existing_unique_name=prev_name, project_directory=project_directory, existing_boxes_lists=old_boxes_lists)
        if not "None" in flag.values():
            return new_particles, flag

        time.sleep(30)

    return new_particles, flag


def load_ctf_data_into_parameters(ctf_file: str, parameters: dict) -> dict:
    # TODO: Confirm if deprecated
    # wait until file is created
    while not os.path.exists(ctf_file):
        time.sleep(10)

    logger.info(f"CTF file exists and is located in {ctf_file}")

    ctf = np.loadtxt(ctf_file)
    parameters["scope_pixel"] = str(ctf[9])
    parameters["scope_voltage"] = str(ctf[10])
    parameters["scope_mag"] = str(ctf[11])
    project_params.save_parameters(parameters, "..")

    logger.info("Done loading CTF data into parameters.")

    return parameters


def get_new_boxes(boxes_before: dict, boxes_after) -> dict:

    assert len(boxes_after.keys()) >= len(boxes_before.keys()), f"Boxes do not monotonically increase"

    old_keys = set(boxes_before.keys())
    new_keys = set(boxes_after.keys())

    new_boxes = {}
    for key in (new_keys - old_keys):
        new_boxes[key] = boxes_after[key]

    return new_boxes


def generate_unique_name(parameters: dict) -> str:
    return (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + "_"
        + parameters["data_set"]
    )


def write_par_to_file(
    new_par_filename: str,
    previous_par_filename: Optional[str],
    new_films: List[str],
    allparxs_dir: PosixPath,
    parameters: dict
) -> int:

    # This modified version of the `csp_merge_parameters` function from
    # `src/pyp/inout/metadata/frealign_parfile.py` is written here because the
    # `csp_merge_parameters` function is too inflexible for our use

    with open(new_par_filename, "w") as f:

        if previous_par_filename is None:
            if "frealignx" in parameters.get("refine_metric", "").lower():
                f.writelines(frealign_parfile.FREALIGNX_PAR_HEADER)
            else:
                f.writelines(frealign_parfile.NEW_PAR_HEADER)

            counter = 1
            film = 1

        else:
            logger.info(
                f"Existing alignments from {previous_par_filename} will be copied to "
                f"{new_par_filename} before new alignments are written"
            )

            with open(previous_par_filename) as prev_f:
                previous_alignments_with_header = prev_f.readlines()
                latest_counter = int(previous_alignments_with_header[-1].strip().split()[0])
                latest_film = int(previous_alignments_with_header[-1][:65].split()[-1])

            for line in previous_alignments_with_header:
                f.write(line)

            counter = latest_counter + 1
            film = latest_film + 1

        new_alignments = sorted(
            [
                i
                for i in allparxs_dir.iterdir()
                if i.suffix == ".allparxs" and i.stem in new_films
            ]
        )

        lcolumn = 52
        hcolumn = 58

        # concatenate all files
        for parx in new_alignments:
            with open(parx, "r") as infile:
                for line in infile:
                    # update film number
                    f.write(
                        "%7d" % (counter)
                        + line[:lcolumn]
                        + "%6d" % (film - 1)
                        + line[hcolumn:]
                    )
                    counter += 1
            film += 1
            os.remove(parx)

    logger.info(f"Done merging files into {new_par_filename}")

    return counter


def write_stacks_to_file(
    new_name: str,
    previous_name: Optional[str],
    new_films: List[str],
    boxes_lists: dict,
    ali_dir: PosixPath,
    stack_dir: PosixPath,
    parameters: dict
):
    from pyp.extract import extract_particles

    mpi_funcs, mpi_args = [], []

    for film_name in new_films:
        if not os.path.exists(stack_dir / f"one-micrograph-stack-{film_name}.mrc"):
            mpi_funcs.append(extract_particles)
            mpi_args.append([(
                str(ali_dir / film_name)+".mrc",
                str(stack_dir / f"one-micrograph-stack-{film_name}.mrc"),
                boxes_lists[film_name],
                parameters["detect_rad"],   # radius
                parameters["class2d_box"],  # box size
                parameters["class2d_bin"],  # binning
                parameters["scope_pixel"],
                1, # cpus
                parameters,
                True,
                True,
                "imod",
                False,
                False
            )])
    try:
        t = timer.Timer(text="Extract particles took: {}", logger=logger.info)
        t.start()

        if len(mpi_funcs) > 0:

            mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=parameters["slurm_verbose"], silent=True)

            # remove micrographs from local scratch
            [os.remove(stack_dir / f"{film}.mrc") for film in new_films if os.path.exists(stack_dir / f"{film}.mrc")]

        t.stop()

        new_stacks = [
            str(stack_dir / f"one-micrograph-stack-{film_name}.mrc")
            for film_name in new_films
        ]
        new_stack = str(stack_dir / f"{new_name}_stack.mrc")
        previous_stack = str(stack_dir / f"{previous_name}_stack.mrc")

        logger.info(f"Merging particle stacks into {new_stack}")

        if previous_name is None or not os.path.exists(previous_stack):
            mrc.merge_fast(new_stacks, new_stack, remove=True)
        else:
            mrc.merge_fast([previous_stack] + new_stacks, new_stack, remove=True)

        logger.info(f"Finished merging particle stacks into {new_stack}")

    except:
        type, value, traceback = sys.exc_info()
        sys.__excepthook__(type, value, traceback)
        logger.warning("Something happened during extraction, could be a restart or clear is happening")


def run_refinement(  # rename to daemon2D after testing
    classification_status: dict,  # ab initio | seeded startup | refinement TODO: add to config file
    new_name: str,
    previous_name: Optional[str],
    boxes_lists: List[PosixPath],
    ali_dir: PosixPath,
    allparxs_dir: PosixPath,
    stack_dir: PosixPath,
    parameters: dict
):
    # addtional iteration using high-resolution cutoff for refinement (used by ab-initio and seeded-startup)
    ITER = 1
    flag = {}

    current_directory = Path().cwd().parent
    frealign_directory = Path().cwd()
    os.chdir(stack_dir)

    # sort the new films to make sure allparxs and particle stacks are in the same order
    new_films = sorted(list(boxes_lists.keys()))

    if "ab-initio" in classification_status.keys() and classification_status["ab-initio"] <= parameters['class2d_iters_init']:
        classification_type = "ab initio"
        start_iteration = classification_status["ab-initio"]
        max_capacity = parameters['class2d_max_ab_initio']
    elif "seeded_startup" in classification_status.keys() and classification_status["seeded_startup"] <= parameters['class2d_iters_seed']:
        classification_type = "seeded-startup"
        start_iteration = classification_status["seeded_startup"]
        max_capacity = parameters['class2d_max_seeded']
    elif "refinement" in classification_status.keys() and classification_status["refinement"] <= parameters["class2d_iters_refine"]:
        classification_type = "refinement"
        start_iteration = classification_status["refinement"]
        max_capacity = parameters['class2d_max_refinement']

    logger.info(f"Beginning {classification_type} classification")

    if classification_type == "ab initio":
        previous_name = None
        if parameters.get("slurm_verbose"):
            logger.info(
                "Forcing previous_name to be None since classification_type is 'ab initio'"
            )
    else:
        assert previous_name is not None

    # TODO: par file changes with mode. Use old par for next classification run
    new_par_filename = f"{new_name}_01.par"  # "_01.par" suffix is needed to work with daemon_2D
    output_par_filename = new_par_filename
    previous_par_filename = f"{previous_name}_01.par" if previous_name else None

    if not Path(new_par_filename).exists() or not (stack_dir / f"{new_name}_stack.mrc").exists():

        particle_num = write_par_to_file(
            new_par_filename, previous_par_filename, new_films, allparxs_dir, parameters
        )

        # Extract particles for each micrograph into one .mrc (src/pyp/extract/core.py extract_particles)
        # Single micrograph stack file name is something like one-micrograph-stack-<film_name>.mrc
        # Merged stack file name is <new_name>_stack.mrc
        write_stacks_to_file(
            new_name, previous_name, new_films, boxes_lists, ali_dir, stack_dir, parameters
        )

    else:
        with open(new_par_filename, 'r') as f:
            particle_num = len(f.readlines())

    high_res_initial = parameters['class2d_rhini']

    if particle_num >= max_capacity:
        class_fraction = round(max_capacity / particle_num, 2)
    else:
        class_fraction = 1.0

    # detect flag before run anything, skip the extraction is for safty with simple restart
    flag = detect_flags(existing_unique_name=new_name, project_directory=current_directory.parent, existing_boxes_lists=boxes_lists)
    if not "None" in flag.values(): return flag, classification_status

    if classification_type == "ab initio":
        logger.info("Preparing initial 2D classes")
        frealign.refine2d(
            input_particle_stack=str(stack_dir / f"{new_name}_stack.mrc"),
            input_frealign_par=new_par_filename,
            input_reconstruction="my_input_classes.mrc",
            parameters=parameters,
            pngfile=Path(frealign_directory, f"{new_name}_classes.png"),
            new_name=new_name,
            output_frealign_par="my_refined_parameters.par",
            output_reconstruction="my_initial_classes.mrc",
            logfile="/dev/null",
            dump_file=f"{new_name}_dump_file.dat",
            classes=parameters['class2d_num'],
            class_fraction=class_fraction,
            low_res_limit=parameters['class2d_rlref'],
            high_res_limit=high_res_initial
        )

        resolution_cycle_count = max(parameters['class2d_iters_init'], 2)
        for cycle_number in range(start_iteration, resolution_cycle_count+ITER):  # Move "if" blocks for modes inside here
            high_res_limit = (
                high_res_initial - (
                    (
                        (high_res_initial - parameters['class2d_rhref']) / (resolution_cycle_count - 1)
                    ) * min(cycle_number, resolution_cycle_count-1)
                )
            )

            # for large particle stack, always using smaller fraction
            if particle_num >= max_capacity:
                class_fraction = round(max_capacity / particle_num, 2)
            else:
                class_fraction = parameters['class2d_fraction'] if cycle_number < resolution_cycle_count else 1.0

            if "slurm_verbose" in parameters and parameters["slurm_verbose"]:
                logger.info(f"Using fraction {class_fraction} from total particles number {particle_num:,} for ab initio")

            flag = detect_flags(existing_unique_name=new_name, project_directory=current_directory.parent, existing_boxes_lists=boxes_lists)
            if not "None" in flag.values(): return flag, classification_status

            logger.info(f"Ab-initio mode : iteration{cycle_number+1:3}/{resolution_cycle_count+ITER:2}, High Res Limit: {high_res_limit:5.2f}, Fraction of particles: {class_fraction:4.2}")

            # use either reconstruction from previous iteration or the one generated using random seeding
            reconstruction_iter = Path(f"cycle_{cycle_number}.mrc")
            input_reconstruction = str(reconstruction_iter) if reconstruction_iter.exists() else "my_initial_classes.mrc"
            # use either the parfile from previous iteration or the one compiled using allparxs (initial parfile)
            parfile_iter = Path(f"cycle_{cycle_number}.par")
            new_par_filename = str(parfile_iter) if parfile_iter.exists() else new_par_filename

            output_reconstruction = f"cycle_{cycle_number+1}.mrc"
            output_parfile = f"cycle_{cycle_number+1}.par"

            refine2d_and_merge2d(new_name,
                                 str(stack_dir / f"{new_name}_stack.mrc"),
                                 new_par_filename,
                                 input_reconstruction,
                                 parameters,
                                 Path(frealign_directory, f"{new_name}_{cycle_number+1}_classes.png"),
                                 output_parfile,
                                 output_reconstruction,
                                 stack_dir,
                                 cycle_number,
                                 0,
                                 class_fraction,
                                 parameters['class2d_rlref'],
                                 high_res_limit
                                 )

            if cycle_number == resolution_cycle_count+ITER-1:
                shutil.copy2(output_reconstruction, "my_initial_classes.mrc")
                shutil.copy2(output_parfile, output_par_filename)
                [os.remove(f) for f in os.listdir(".") if f.startswith("cycle")]

                classification_status["seeded_startup"] = 0

            classification_status["ab-initio"] += 1

    elif classification_type == "seeded-startup":
        resolution_cycle_count = max(parameters['class2d_iters_seed'], 2)
        for cycle_number in range(start_iteration, resolution_cycle_count+ITER):
            high_res_limit = (
                high_res_initial - (
                    (
                        (high_res_initial - parameters['class2d_rhref']) / (resolution_cycle_count - 1)
                    ) * min(cycle_number, resolution_cycle_count-1)
                )
            )

            if particle_num >= max_capacity:
                class_fraction = round(max_capacity / particle_num, 2)
            else:
                class_fraction = parameters['class2d_fraction'] if cycle_number < resolution_cycle_count else 1.0

            if "slurm_verbose" in parameters and parameters["slurm_verbose"]:
                logger.info(f"Using fraction {class_fraction} from total particles number {particle_num:,} for seeded_startup")

            flag = detect_flags(existing_unique_name=new_name, project_directory=current_directory.parent, existing_boxes_lists=boxes_lists)
            if not "None" in flag.values(): return flag, classification_status
            logger.info(f"Seeded startup : iteration{cycle_number+1:3}/{resolution_cycle_count+ITER:2}, High Res Limit: {high_res_limit:5.2f}, Fraction of particles: {class_fraction:4.2}")

            # use either reconstruction from previous iteration or the one generated using random seeding
            reconstruction_iter = Path(f"cycle_{cycle_number}.mrc")
            input_reconstruction = str(reconstruction_iter) if reconstruction_iter.exists() else "my_initial_classes.mrc"
            # use either the parfile from previous iteration or the one compiled using allparxs (initial parfile)
            parfile_iter = Path(f"cycle_{cycle_number}.par")
            new_par_filename = str(parfile_iter) if parfile_iter.exists() else new_par_filename

            output_reconstruction = f"cycle_{cycle_number+1}.mrc"
            output_parfile = f"cycle_{cycle_number+1}.par"
            output_png = Path(frealign_directory, f"{new_name}_{cycle_number+1}_classes.png") \
                        if cycle_number == resolution_cycle_count+ITER-1 \
                        else None

            refine2d_and_merge2d(new_name,
                                 str(stack_dir / f"{new_name}_stack.mrc"),
                                 new_par_filename,
                                 input_reconstruction,
                                 parameters,
                                 output_png,
                                 output_parfile,
                                 output_reconstruction,
                                 stack_dir,
                                 cycle_number,
                                 0,
                                 class_fraction,
                                 parameters['class2d_rlref'],
                                 high_res_limit
                                 )

            if cycle_number == resolution_cycle_count+ITER-1:
                shutil.copy2(output_reconstruction, "my_initial_classes.mrc")
                shutil.copy2(output_parfile, output_par_filename)
                [os.remove(f) for f in os.listdir(".") if f.startswith("cycle")]

                classification_status["refinement"] = 0

            classification_status["seeded_startup"] += 1

    elif classification_type == "refinement":
        refinement_cycle_count = max(parameters['class2d_iters_refine'], 2)

        if particle_num >= max_capacity:
            class_fraction = round(max_capacity / particle_num, 2)
        else:
            class_fraction = 1.0

        if "slurm_verbose" in parameters and parameters["slurm_verbose"]:
            logger.info(f"Using fraction {class_fraction} from total particles number {particle_num:,} for refinement")

        high_res_limit = parameters['class2d_rhref']

        for cycle_number in range(start_iteration, refinement_cycle_count):

            flag = detect_flags(existing_unique_name=new_name, project_directory=current_directory.parent, existing_boxes_lists=boxes_lists)
            if not "None" in flag.values(): return flag, classification_status

            logger.info(f"Refinement mode: iteration{cycle_number:3}/{refinement_cycle_count:2}, High Res Limit: {high_res_limit:5.2f}, Fraction of particles: {class_fraction:4.2f}")

            # use either reconstruction from previous iteration or the one generated using random seeding
            reconstruction_iter = Path(f"cycle_{cycle_number}.mrc")
            input_reconstruction = str(reconstruction_iter) if reconstruction_iter.exists() else "my_initial_classes.mrc"
            # use either the parfile from previous iteration or the one compiled using allparxs (initial parfile)
            parfile_iter = Path(f"cycle_{cycle_number}.par")
            new_par_filename = str(parfile_iter) if parfile_iter.exists() else new_par_filename

            output_reconstruction = f"cycle_{cycle_number+1}.mrc"
            output_parfile = f"cycle_{cycle_number+1}.par"
            output_png = Path(frealign_directory, f"{new_name}_{cycle_number+1}_classes.png") \
                        if cycle_number == refinement_cycle_count-1 \
                        else None

            refine2d_and_merge2d(new_name,
                        str(stack_dir / f"{new_name}_stack.mrc"),
                        new_par_filename,
                        input_reconstruction,
                        parameters,
                        output_png,
                        output_parfile,
                        output_reconstruction,
                        stack_dir,
                        cycle_number,
                        0,
                        class_fraction,
                        parameters['class2d_rlref'],
                        high_res_limit
                        )

            if cycle_number == refinement_cycle_count-1:
                shutil.copy2(output_reconstruction, "my_initial_classes.mrc")
                shutil.copy2(output_parfile, output_par_filename)
                [os.remove(f) for f in os.listdir(".") if f.startswith("cycle")]
                # reset refinement status for new comming particles
                classification_status["refinement"] = 0

            classification_status["refinement"] += 1

    os.chdir(current_directory)

    return flag, classification_status


def refine2d_and_merge2d(name: str,
                         input_stack: str,
                         input_parfile: str,
                         input_reconstruction: str,
                         parameters: dict,
                         png_classes: PosixPath,
                         output_parfile: str,
                         output_reconstruction: str,
                         working_directory: PosixPath,
                         cycle: int,
                         classes: int,
                         class_fraction: float,
                         low_res_limit: float,
                         high_res_limit: float):

    # split refine2d into several processes and run via MPI
    splitted_parfiles, dumpfiles = frealign.refine2d_mpi(
                                    input_stack,
                                    input_parfile,
                                    input_reconstruction,
                                    name,
                                    parameters,
                                    classes,
                                    class_fraction,
                                    low_res_limit,
                                    high_res_limit,
                                    )

    # merge parfiles (using MPI as well) (looks like class2D use film column to assign classes)
    frealign_parfile.Parameters.merge_parameters(splitted_parfiles,
                                                output_parfile,
                                                "new",
                                                frealignx=False,
                                                update_film=False)

    # remove splitted parfiles (since we're not doing so in merge_parameters)
    [os.remove(f) for f in splitted_parfiles]

    # merge2d to merge all the intermediate reconstruction .dat
    occ_classes = frealign.merge2d(cycle, working_directory, dumpfiles, output_reconstruction, parameters)

    occ_classes = get_num_particles_classes(output_parfile, parameters["class2d_num"])

    # save png
    if png_classes is not None:
        frealign.plot_refine2d_reconstructions(output_reconstruction, Path(png_classes).stem.replace("_classes",""), png_classes, parameters, occ_classes)


def get_num_particles_classes(output_parfile: str, num_classes: int):

    data = frealign_parfile.Parameters.from_file(output_parfile).data
    num_particles_classes = {}

    for class_ind in range(num_classes):
        particles_class = data[data[:, 7] == class_ind+1]
        num_particles = particles_class.shape[0]
        total_scores = np.sum(particles_class[:, 15])
        mean_score = total_scores / num_particles

        num_particles_classes[class_ind+1] = num_particles

    return num_particles_classes

def fyp_daemon(existing_unique_name=None, existing_boxes_lists=dict()):

    # define flags to keep track of daemon state
    start_flag = os.path.join(Path(os.getcwd()).parent, "fypd.start")
    stop_flag = os.path.join(Path(os.getcwd()).parent, "fypd.stop")
    restart_flag = os.path.join(Path(os.getcwd()).parent, "fypd.restart")
    clear_flag = os.path.join(Path(os.getcwd()).parent, "fypd.clear")

    flag = {}
    new_status = {}

    status = {"ab-initio":0} # used for tracking classification type and iterations

    # clean the flag in the beginning
    [os.remove(f) for f in [stop_flag, restart_flag, clear_flag] if os.path.exists(f)]

    # raise start flag
    Path(start_flag).touch()

    # Set up paths for the steps that follow
    frealign_dir = Path.cwd()

    stream_session_dir = frealign_dir.parent
    ali_dir = stream_session_dir / "mrc"
    meta_dir = stream_session_dir / "pkl"
    local_scratch_dir = Path(os.environ["PYP_SCRATCH"])

    # Set up parameters for the steps that follow
    mparameters = project_params.load_parameters("..")
    mparameters["extract_box"] = mparameters["class2d_box"]
    mparameters["extract_bin"] = mparameters["class2d_bin"]

    incremental_threshold_2D = mparameters["class2d_inc"]

    global_start = True

    # main loop to perform refinement mode if new_particles reach the threshold (class2d_inc)
    # new_particles is reset to 0 after running a refinement

    # first time reach the threshold -> seeded-startup
    # after it, every time it reach the threshold -> refinement
    # seeded_startup = True

    daemon_start_time = time.time()
    while time.time() - daemon_start_time < datetime.timedelta(
        days=mparameters["stream_session_timeout"]
    ).total_seconds() and not os.path.exists(stop_flag):

        if global_start:
            boxes_lists = dict() if existing_boxes_lists is None else existing_boxes_lists

            new_particles = 0
            if len(existing_boxes_lists.keys()) == 0:
                new_particles, flag = \
                    get_positions_and_new_particle_count_from_box_files(
                        prev_name=None,
                        boxes_lists=boxes_lists,
                        threshold_type="class2d_min",
                        meta_dir=meta_dir,
                        parameters=mparameters,
                        allparxs_dir=local_scratch_dir
                    )
                if "stop" in flag.values(): return flag

            try:
                # Run initial 2D classification
                new_name = generate_unique_name(mparameters) if existing_unique_name is None else existing_unique_name
                flag, new_status = run_refinement(
                        classification_status=status,
                        previous_name=None,
                        boxes_lists=boxes_lists,
                        parameters=mparameters,
                        new_name=new_name,
                        allparxs_dir=local_scratch_dir,
                        ali_dir=ali_dir,
                        stack_dir=local_scratch_dir
                        )
                logger.info(f"Class2D status is {new_status}")
                if "stop" in flag.values(): return flag

                global_start = False
                new_status["seeded_startup"] = 0
            except:
                type, value, traceback = sys.exc_info()
                sys.__excepthook__(type, value, traceback)
                logger.warning("Inconsistencies detected during processing, waiting 30 seconds before resuming")
                time.sleep(30)
                pass
        try:
            if os.path.exists(restart_flag):
                logger.info("Restart flag detected")

                # in case run_refinement() exit under local_scratch_dir
                os.chdir(frealign_dir)

                previous_parameters = mparameters
                nextpyp_saved = "fypd.restart"
                if not os.path.getsize(os.path.join(stream_session_dir, nextpyp_saved)) == 0:
                    new_parameters = project_params.load_parameters(path=stream_session_dir, param_file_name=nextpyp_saved)
                    reload_default = True
                else:
                    logger.warning("Empty restart flag, will read the configure file instead")
                    new_parameters = project_params.load_parameters(path=stream_session_dir)
                    reload_default = False

                if reload_default:
                    # read specification file to get missing defaut parameters
                    import toml
                    specifications = toml.load("/opt/pyp/config/pyp_config.toml")
                    # figure out which parameters need to be added and set as default values
                    for t in specifications["tabs"].keys():

                        if not t.startswith("_"):

                            for p in specifications["tabs"][t].keys():

                                if f"{t}_{p}" not in new_parameters:

                                    if "default" in specifications["tabs"][t][p]:
                                        new_parameters[f"{t}_{p}"] = specifications["tabs"][t][p]["default"]

                    # restart won't consider data path and data mode parameters which should reset whole session
                    new_parameters["data_path"] = previous_parameters["data_path"]
                    if "data_mode" not in new_parameters.keys():
                        new_parameters["data_mode"] = previous_parameters["data_mode"]
                    if "data_set" not in new_parameters.keys():
                        new_parameters["data_set"] = previous_parameters["data_set"]
                    
                    force_preprocessing = False
                else:
                    force_preprocessing = any(["_force" in key and new_parameters[key]==True for key in new_parameters ])
                                    
                    # disable the _force parameters in the configure file
                    if force_preprocessing:
                        for key in new_parameters:
                            if "_force" in key and new_parameters[key]==True and not "_force_integer" in key:
                                new_parameters[key] = False
                            else:
                                pass
                        project_params.save_parameters(new_parameters, path=stream_session_dir)

                # detect differences at class2d parameters
                different_values = {k for k in previous_parameters.keys() & new_parameters.keys() if previous_parameters[k] != new_parameters[k]}
                
                #  path type re-evaluation
                if "gain_reference" in different_values:
                    if project_params.resolve_path(mparameters["gain_reference"]) == project_params.resolve_path(new_parameters["gain_reference"]):
                        different_values.remove("gain_reference")

                logger.info(f"Parameters changed: {different_values}")
                # update parameters from new parameters
                for key in different_values:
                    mparameters[key] = new_parameters[key]

                # remove flag before running any refinement
                try:
                    os.remove(restart_flag)
                except:
                    logger.info("Cannot remove restart file")

                if any(
                    ["class2d_box" in different_values,
                    "class2d_bin" in different_values,
                    "class2d_num" in different_values,
                    "class2d_fraction" in different_values,
                    mparameters["class2d_min"] > previous_parameters["class2d_min"],
                    any("class2d_r" in key for key in different_values),
                    force_preprocessing,
                    ]
                ):
                    logger.warning("Main parameters changed, will re-extract particles and restart refinement")
                    # re-extract particles
                    # remove stacks
                    [ os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir,"*stack*.mrc") ) ]

                    # remove existing 2D averages and parfile
                    # Keep old classes in place
                    # [os.remove(f) for f in glob.glob( os.path.join(frealign_dir, "maps", "*.webp") )]
                    [os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir, "*.par") )]
                    [os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir, "*classes*") ) ]
                    [os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir, "cycle_*") )]

                    # re-initialize ab-initio
                    boxes_lists = dict()
                    new_status["ab-initio"] = 0
                    try:
                        del new_status["seeded_startup"]
                        del new_status["refinement"]
                    except:
                        logger.info("Classification status is ab initio only, nothing to change")
                    # get boxes again
                    boxes_lists_before = boxes_lists.copy()

                    new_particles, flag = \
                        get_positions_and_new_particle_count_from_box_files(
                            prev_name=new_name,
                            boxes_lists=boxes_lists,
                            threshold_type="class2d_min",
                            meta_dir=meta_dir,
                            parameters=mparameters,
                            allparxs_dir=local_scratch_dir
                        )

                    if "stop" in flag.values():
                        break

                    new_name = generate_unique_name(mparameters) if existing_unique_name is None else existing_unique_name
                    flag, new_status= run_refinement(
                            classification_status=new_status,
                            previous_name=None,
                            boxes_lists=boxes_lists,
                            parameters=mparameters,
                            new_name=new_name,
                            allparxs_dir=local_scratch_dir,
                            ali_dir=ali_dir,
                            stack_dir=local_scratch_dir
                            )
                    logger.info(f"Class2D status is {new_status}")
                    if "stop" in flag.values():
                        break

                    new_status["seeded_startup"] = 0

                else:
                    logger.warning("Simple restart, will use most recent set of classes as references for alignment")

                    # in case run_refinement() exit under local_scratch_dir
                    os.chdir(frealign_dir)
                    # count number of new particles (there's an while loop inside until getting enough particles)
                    boxes_lists = dict()
                    boxes_lists_before = boxes_lists.copy()
                    new_particles, flag = \
                        get_positions_and_new_particle_count_from_box_files(
                            prev_name=new_name,
                            boxes_lists=boxes_lists,
                            threshold_type="class2d_inc",
                            meta_dir=meta_dir,
                            parameters=mparameters,
                            allparxs_dir=local_scratch_dir
                        )
                    if "stop" in flag.values():
                        break

                    new_boxes_lists = get_new_boxes(boxes_lists_before, boxes_lists)
                    # only run refinement mode if we get additional particles
                    logger.info(f"{new_particles} new particles detected, exceeds {incremental_threshold_2D} threshold")

                    previous_name = new_name
                    new_name = generate_unique_name(mparameters)

                    flag, new_status = run_refinement(
                            classification_status=new_status,
                            new_name=new_name,
                            previous_name=previous_name,
                            boxes_lists=new_boxes_lists,
                            ali_dir=ali_dir,
                            allparxs_dir=local_scratch_dir,
                            stack_dir=local_scratch_dir,
                            parameters=mparameters
                        )
                    logger.info(f"Class2D status is {new_status}")
                    if "stop" in flag.values():
                        break

            if os.path.exists(clear_flag):
                logger.info("Clear flag detected")
                logger.warning("Will do a deep clean of previous refinement results")

                # remove existing parfile and webp files
                [os.remove(f) for f in glob.glob( os.path.join(frealign_dir, "*.webp") )]

                # in case run_refinement() exit under local_scratch_dir
                os.chdir(frealign_dir)

                previous_parameters = mparameters
                nextpyp_saved = "fypd.clear"
                if not os.path.getsize(os.path.join(stream_session_dir, nextpyp_saved)) == 0:
                    new_parameters = project_params.load_parameters(path=stream_session_dir, param_file_name=nextpyp_saved)
                    reload_default = True
                else:
                    logger.warning("Empty clear flag, will read the configure file instead")
                    new_parameters = project_params.load_parameters(path=stream_session_dir)
                    reload_default = False

                if reload_default:
                    # read specification file to get missing defaut parameters
                    import toml
                    specifications = toml.load("/opt/pyp/config/pyp_config.toml")
                    # figure out which parameters need to be added and set as default values
                    for t in specifications["tabs"].keys():

                        if not t.startswith("_"):

                            for p in specifications["tabs"][t].keys():

                                if f"{t}_{p}" not in new_parameters:

                                    if "default" in specifications["tabs"][t][p]:
                                        new_parameters[f"{t}_{p}"] = specifications["tabs"][t][p]["default"]

                    # restart won't consider data path and data mode parameters which should reset whole session
                    new_parameters["data_path"] = previous_parameters["data_path"]
                    if "data_mode" not in new_parameters.keys():
                        new_parameters["data_mode"] = previous_parameters["data_mode"]
                    if "data_set" not in new_parameters.keys():
                        new_parameters["data_set"] = previous_parameters["data_set"]

                # detect differences at class2d parameters
                different_values = {k for k in previous_parameters.keys() & new_parameters.keys() if previous_parameters[k] != new_parameters[k]}
                #  path type re-evaluation
                if "gain_reference" in different_values:
                    if project_params.resolve_path(mparameters["gain_reference"]) == project_params.resolve_path(new_parameters["gain_reference"]):
                        different_values.remove("gain_reference")

                logger.info(f"Parameters changed: {different_values}")
                # update paramters from new parameters
                for key in different_values:
                    mparameters[key] = new_parameters[key]

                # clear everything related
                [os.remove(f) for f in glob.glob( os.path.join(frealign_dir, "scratch", "*" ) )]
                [os.remove(f) for f in glob.glob( os.path.join(frealign_dir, "maps", "*" ) ) if Path(f).suffix != ".webp"]
                [os.remove(f) for f in glob.glob( os.path.join(frealign_dir, "log", "*" ) )]
                [os.remove(f) for f in glob.glob( os.path.join(frealign_dir, "swarm", "*" ) )]
                [os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir, "*" ) )]
                [os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir, "maps", "*" ) )]
                [os.remove(f) for f in glob.glob( os.path.join(local_scratch_dir, "scratch", "*" ) )]

                try:
                    os.remove(clear_flag)
                except:
                    logger.info("Can't remove clear flag")

                boxes_lists = dict()
                new_status["ab-initio"] = 0
                try:
                    del new_status["seeded_startup"]
                    del new_status["refinement"]
                except:
                    logger.info("Classification status is ab initio only, nothing to change")
                # get boxes again
                boxes_lists_before = boxes_lists.copy()

                new_particles, flag = \
                    get_positions_and_new_particle_count_from_box_files(
                        prev_name=new_name,
                        boxes_lists=boxes_lists,
                        threshold_type="class2d_min",
                        meta_dir=meta_dir,
                        parameters=mparameters,
                        allparxs_dir=local_scratch_dir
                    )

                if "stop" in flag.values():
                    break

                new_name = generate_unique_name(mparameters) if existing_unique_name is None else existing_unique_name
                flag, new_status= run_refinement(
                        classification_status=new_status,
                        previous_name=None,
                        boxes_lists=boxes_lists,
                        parameters=mparameters,
                        new_name=new_name,
                        allparxs_dir=local_scratch_dir,
                        ali_dir=ali_dir,
                        stack_dir=local_scratch_dir
                        )
                logger.info(f"Class2D status is {new_status}")
                if "stop" in flag.values():
                    break

                new_status["seeded_startup"] = 0

            os.chdir(frealign_dir)
            # count number of new particles (there's an while loop inside until getting enough particles)
            boxes_lists_before = boxes_lists.copy()
            new_particles, flag = \
                get_positions_and_new_particle_count_from_box_files(
                    prev_name=new_name,
                    boxes_lists=boxes_lists,
                    threshold_type="class2d_inc",
                    meta_dir=meta_dir,
                    parameters=mparameters,
                    allparxs_dir=local_scratch_dir
                )
            if "stop" in flag.values():
                break
            elif not "None" in flag.values():
                continue

            new_boxes_lists = get_new_boxes(boxes_lists_before, boxes_lists)

            # only run refinement mode if we get additional particles
            logger.info(f"{new_particles:,} new particles detected, exceeds {incremental_threshold_2D:,} threshold")

            previous_name = new_name
            new_name = generate_unique_name(mparameters)

            flag, new_status = run_refinement(
                    classification_status=new_status,
                    new_name=new_name,
                    previous_name=previous_name,
                    boxes_lists=new_boxes_lists,
                    ali_dir=ali_dir,
                    allparxs_dir=local_scratch_dir,
                    stack_dir=local_scratch_dir,
                    parameters=mparameters
                )
            logger.info(f"Class2D status is {new_status}")
            if "stop" in flag.values():
                break

        except:
            type, value, traceback = sys.exc_info()
            sys.__excepthook__(type, value, traceback)
            logger.warning("Inconsistencies detected during processing, waiting 30 seconds before resuming")
            time.sleep(30)
            pass

    if os.path.exists(stop_flag):
        try:
            os.remove(stop_flag)
            os.remove(start_flag)
        except:
            logger.info("Can't remove stop flag")


def detect_flags(existing_unique_name=None, existing_boxes_lists=dict(), project_directory=Path().cwd()):

    local_scratch = os.environ["PYP_SCRATCH"]

    # define flags to keep track of daemon state
    start_flag = "fypd.start"
    stop_flag = "fypd.stop"
    restart_flag = "fypd.restart"
    clear_flag = "fypd.clear"

    if (project_directory / clear_flag).exists():
        # os.remove(project_directory / clear_flag)
        return {"type": "clear"}

    elif (project_directory / restart_flag).exists():
        # os.remove(project_directory / restart_flag)
        return {"type": "restart"}

    elif (project_directory / stop_flag).exists():
        # os.remove(project_directory / stop_flag)
        # os.remove(project_directory / start_flag) # stop will clean start flag
        return {"type": "stop"}

    else:
        return {"type": "None"}


def daemon_2D(mparameters, initial_threshold, incremental_threshold):

    start_flag = os.path.join(Path(os.getcwd()).parent, "fyp2Dd.start")
    stop_flag = os.path.join(Path(os.getcwd()).parent, "fyp2Dd.stop")
    restart_flag = os.path.join(Path(os.getcwd()).parent, "fyp2Dd.restart")
    clear_flag = os.path.join(Path(os.getcwd()).parent, "fyp2Dd.clear")

    # raise start flag
    Path(start_flag).touch()

    films = "../eman/{}.films".format(mparameters["data_set"])
    if os.path.exists(films):
        os.remove(films)

    new_particles = 0

    existing_films = []

    # wait until enough particles accumulate
    while new_particles < initial_threshold:
        new_films = [
            i.replace("_01.par", "")
            for i in sorted(glob.glob("*_01.par"))
            if not "_" + mparameters["data_set"] in i
        ]
        logger.info(f"new_films: {new_films}")
        new_particles = 0
        for f in new_films:
            with open(f + "_01.par", "r") as fh:
                new_particles += int(fh.readlines()[-1].split()[0])
        time.sleep(10)

    first_pass = True

    while True:

        if os.path.exists(films):
            existing_films = [line.strip() for line in open(films, "r")]
            new_films = [
                i.replace("_01.par", "")
                for i in sorted(glob.glob("*_01.par"))
                if not i.replace("_01.par", "") in existing_films
                and not "_" + mparameters["data_set"] in i
            ]

            # count number of new particles added
            new_particles = 0
            for f in new_films:
                with open(f + "_01.par", "r") as fh:
                    par_particles = int(fh.readlines()[-1].split()[0])
                    # Produce stack .mrc files -> local scratch (only in box for initial tests)
                    stack_particles = int(
                        mrc.readHeaderFromFile(f + "_stack.mrc")["nz"]
                    )
                    if par_particles == stack_particles:
                        new_particles += par_particles
                    else:
                        logger.warning(
                            "par file and stack file sizes differ {0}, {1}".format(
                                par_particles, stack_particles
                            )
                        )
                        new_films.pop(new_films.index(f))

            first_pass = False

        # re-start refinement if enough new particles are detected
        if first_pass or (new_particles > incremental_threshold and not first_pass):

            logger.info(f"{new_particles:,} new particles detected.")

            # save updated film list
            f = open(films, "w")
            f.write("\n".join(existing_films + new_films))
            f.close()

            # new refinement run
            # Need stack that is concatenation of individual stacks before running this
            new_name = generate_unique_name(mparameters)

            stacks = [i + "_stack.mrc" for i in existing_films + new_films]
            new_stack = "../eman/{0}_stack.mrc".format(new_name) # concatenation of individual stacks

            # Ensure that stack .mrc (patches from micrograph) (look for existing functions) and stack .par (metadata: alignment, ctf, etc.) (concatenated; one for each micrograph) (look for existing functions) are available; save in local /scratch

            # launch relion 2D classification
            current_dir = os.getcwd()
            os.chdir("..")
            # Replace `generateRelionParFileNew` with refine2D
            # os.chdir("relion")
            frealign_paths = get_frealign_paths()
            refine2d_binary_path = frealign_paths["frealignx"]
            # Call refine2d like `frealign.refine2d()``
            new_rstacks = [
                "Particles/Micrographs/" + i + "_particles.mrcs" for i in new_films
            ]
            new_rstack = "{0}_stack.mrcs".format(new_name)
            if len(existing_films) > 0:
                rparameters = project_params.load_relion_parameters(".")
                previous_rstack = rparameters["dataset"] + "_stack.mrcs"
                mrc.merge_fast([previous_rstack] + new_rstacks, new_rstack,remove=False)
                # get rid of files from previous run
                os.remove(previous_rstack)
                # os.remove( previous_rstack.replace('_stack.mrcs','.star') )
                # os.remove( previous_rstack.replace('_stack.mrcs','_particles.star') )
            else:
                mrc.merge_fast(new_rstacks, new_rstack,remove=False)

            com = '{2}/pyp/pyp_relion.py -mode Class2D -dataset {0} -classes {1} -tau 2 -queue="{3}"'.format(
                new_name, 50, os.environ["PYP_DIR"], mparameters["slurm_queue"]
            )
            logger.info(com)
            logger.info(
                subprocess.check_output(
                    com, stderr=subprocess.STDOUT, shell=True, text=True
                )
            )
            os.chdir(current_dir)

            # wait until relion classification is done
            while os.path.exists("..//relion/" + new_name + "_particles.star"):
                time.sleep(60)

        else:

            time.sleep(60)

        if os.path.exists(restart_flag):
            # TODO: execute restart operations
            time.sleep()

        if os.path.exists(clear_flag):
            # TODO: execute clear operations
            time.sleep()

        if os.path.exists(stop_flag):
            try:
                os.remove(stop_flag)
            except:
                raise Exception("Cannot remove " + stop_flag)
            break


def get_file_indicator(mrc_dir):

    # get the names indicating preprocessing finished.

    finished_list = [file.stem for file in mrc_dir.iterdir() if file.suffix == ".mrc"]

    return sorted(finished_list)