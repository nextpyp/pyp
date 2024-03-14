import argparse
import collections
import csv
import glob
import math
import multiprocessing
import os
import random
import re
import shutil
import socket
import sys
from pathlib import Path, PosixPath

import numpy as np
import toml

from pyp import utils
from pyp.analysis import statistics
from pyp.system import project_params, slurm, user_comm
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import clear_scratch
from pyp.utils import get_relative_path, movie2regex
from pyp.system.db_comm import save_parameters_to_website

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def spr_is_done(name):
    extensions = ["avg", "xf"]
    return utils.has_files(name, extensions)


def csp_is_done(name, use_frames=False):

    frame_identification = ""
    if use_frames:
        frame_identification = "_local"

    extensions = [".allboxes"]

    return utils.has_files(name + frame_identification, extensions, suffix = True)


def tiltseries_align_is_done(metadata):
    return metadata != None and "ali" in metadata and "tlt" in metadata

def get_particle_increment_in_rec(
    fparameters, iteration, particles, classes, dataset, machinefile, machinerecfile
):

    if "cc" in project_params.param(fparameters["metric"], iteration):
        nodes = len(open(machinefile, "r").read().split())
        if "MYCORES" in os.environ:
            cores = int(os.environ["MYCORES"])
        else:
            cores = multiprocessing.cpu_count() * nodes
        procs = cores / 17
        # increment = int( math.ceil( particles / float(cores) ) )
        increment = max(int(math.ceil(particles / float(nodes))), 1000)
        increment = max(int(math.ceil(particles / float(procs))), 1000)

        machinerecfile = machinefile

    else:

        nodes = 1
        cores = multiprocessing.cpu_count() * nodes
        increment = max(int(math.ceil(particles / float(cores))), 500)
        if not os.path.exists(machinerecfile):
            open(machinerecfile, "w").write(socket.gethostname())

        procs = cores

        # clear scratch
        clear_scratch()

        for ref in range(classes):
            # TODO: copy files
            shutil.copy(
                "scratch/%s_r%02d_%02d.par" % (dataset, ref + 1, iteration),
                os.environ["PYP_SCRATCH"],
            )

            # copy reference to local scratch
            shutil.copy(
                "scratch/%s_r%02d_%02d.mrc" % (dataset, ref + 1, iteration - 1),
                os.environ["PYP_SCRATCH"],
            )

            star_file = "scratch/%s_r%02d_%02d.star" % (dataset, ref + 1, iteration,)
            if os.path.exists(star_file):
                shutil.copy(star_file, os.environ["PYP_SCRATCH"])

    return machinerecfile, increment, procs


def get_particle_increment(fp, iteration, particles):

    increment = 50

    if int(project_params.param(fp["mode"], iteration)) == 4:
        increment = min(25, int(particles / 100))

    elif int(project_params.param(fp["mode"], iteration)) == 1:
        increment = min(1000, int(particles / 5000))

    return increment


def get_film_order(parameters, name):
    # find film number for this micrograph in the dataset (starts from 1)
    try:
        with open(parameters["data_set"] + ".films") as x:
            series = [
                num
                for num, line in enumerate(x, 1)
                if "{}".format(name) == line.strip()
            ][0]
    except:
        raise Exception("ERROR - Cannot find film number for " + name)

    return series


def get_film_name(parameters, series):
    # find film name given series number (starts from 1)
    f = open(parameters["data_set"] + ".films").read().splitlines()
    try:
        return f[series - 1]
    except IndexError as e:
        logger.info(e)


def get_relevant_films(parameters, array_job_num):
    # obtain the film names of the given array job
    """
    START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $TASKS_PER_ARR + 1 ))
    MAX_TASK=$(wc -l < %s)
    END_NUM=$(( $SLURM_ARRAY_TASK_ID * $TASKS_PER_ARR ))
    """
    tasks_per_arr = int(parameters["slurm_bundle_size"])
    start_job_num = (array_job_num - 1) * tasks_per_arr + 1
    end_job_num = array_job_num * tasks_per_arr

    return [
        get_film_name(parameters, num) for num in range(start_job_num, end_job_num + 1)
    ]


def get_missing_files(parameters, inputlist, verbose=True):
    missing_files = []
    for sname in inputlist:
        try:
            command = "cat swarm/pre_process.swarm | grep %s.log | awk '{print $NF}'" % sname
            [output, error] = run_shell_command(command, verbose=False)
            logfile = output.replace("../", "").strip()
            if (
                not os.path.exists(logfile)
                or not "finished successfully" in open(logfile).read()
            ):
                if not os.path.exists(logfile):
                    logger.info(f"{logfile} does not exist")
                else:
                    logger.info(f"{sname} did not terminate successfully")

                if "csp_no_stacks" in parameters.keys() and parameters["csp_no_stacks"]:
                    # find all movies that were part of the array job and add to missing files
                    series = project_params.get_film_order(parameters, sname)
                    array_job_num = slurm.get_array_job(parameters, series)
                    all_files = project_params.get_relevant_films(parameters, array_job_num)
                    [missing_files.append(f) for f in all_files if f not in missing_files]
                else:
                    missing_files.append(sname)
        except:
            logger.error("File swarm/pre_process.swarm not found")
            pass

    return missing_files


def find_last_frame(parameters, inputlist):
    if int(parameters["movie_last"]) == -1:
        ctffile = "ctf/" + inputlist[0] + ".ctf"
        if os.path.isfile(ctffile):
            ctf = np.loadtxt(ctffile)
        last = int(ctf[8])
    else:
        last = int(parameters["movie_last"])

    return last


def check_parameter_consistency(parameters):
    if (
        parameters["data_mode"] == "spr"
        and "extract_box" in parameters.keys() and parameters["extract_box"] > 0
        and "frealign" not in parameters["extract_fmt"].lower()
        and "relion" not in parameters["extract_fmt"].lower()
        and "eman" not in parameters["extract_fmt"].lower()
        and not int(parameters["class_num"]) > 0
    ):
        raise Exception(
            "Attempting to extract particles without specifying a valid format.\n\tPlease specify a valid value for -extract_fmt"
        )


def resolve_path(parameter):
    if isinstance(parameter, PosixPath) or not "PosixPath" in parameter:
        return str(parameter)
    else:
        return str(eval(parameter))


def create_micrographs_list(parameters):
    micrographs = "{}.micrographs".format(parameters["data_set"])
    if os.path.exists(micrographs + "_missing"):
        micrographs += "_missing"
    if not os.path.isfile(micrographs):
        files = []
        types = [
            "raw/" + s
            for s in (
                "*.dm?",
                "*.mrc",
                "*.tgz",
                "*.bz2",
                "*.tbz",
                "*.tif",
                "*.tiff",
                "*.bz2",
                "*.eer",
            )
        ]
        for t in types:
            files.extend(
                [
                    os.path.splitext(os.path.basename(s))[0].replace(".tar", "")
                    for s in glob.glob(t)
                ]
            )

        files = [file for file in files if not "gain" in file.lower()]

        # remake a list if tilt movies (*.tif) are all separated
        movie_extension = Path(parameters["movie_pattern"]).suffix

        # look for mdoc files in mdoc folder
        mdocs = list()
        if "data_path_mdoc" in parameters and parameters["data_path_mdoc"] != None and Path(resolve_path(parameters["data_path_mdoc"])).exists:
            mdoc_folder = Path(resolve_path(parameters["data_path_mdoc"])).parent
            mdocs = list(mdoc_folder.glob("*.mdoc"))
        # if none found, look in raw data folder
        if len(mdocs) == 0:
            data_path = Path(resolve_path(parameters["data_path"]))
            data_folder = data_path.parent
            mdocs = list(data_folder.glob("*.mdoc"))

        if parameters["data_mode"] == "tomo": 
            if not parameters["movie_mdoc"] and len(parameters["movie_pattern"]) > 0 and len(glob.glob("raw/*" + movie_extension)) > 0:
                regex = movie2regex(parameters["movie_pattern"], filename="*")
                r = re.compile(regex)
                match_files = [
                    re.match(r, f)
                    for f in [f.replace("raw/", "") for f in glob.glob("raw/*" + movie_extension)]
                ]
                files = [m.group(1) for m in match_files if m != None]
                logger.info("Create micrograph list using movie patterns")

            elif parameters["movie_mdoc"] and len(mdocs) > 0:
                files = [str(f.name).replace(".mdoc", "").replace(".mrc", "") for f in mdocs]
                logger.info("Create micrograph list using mdocs files")
                # NOTE: one mdoc for one tilt-series (rather than one tilt)
            else:
                logger.info("Create micrograph list using detected files (one mrc per tilt-series)")

        files = sorted(list(set(files)))
        logger.info("Found {} unique file(s) for processing".format(len(files)))
        f = open(micrographs, "w")
        f.write("\n".join([s for s in files if len(s) > 0]))
        f.close()
    else:
        f = open(micrographs, "r")
        files = f.read().split("\n")
        f.close()

    if len(files) == 0:
        os.remove(micrographs)
        raise Exception(
            "Empty image list. No valid files found in raw data directory. Stop."
        )

    return micrographs, files


def get_align_option(
    fp, iteration, keep_previous_alignments, dataset, need_initialize_classification
):
    alignment_option = 1
    if iteration == 2 and not keep_previous_alignments:
        parameter_file = np.array(
            [
                line.split()
                for line in open("%s_%02d.par" % (dataset, iteration - 1))
                if not line.startswith("C")
            ],
            dtype=float,
        )
        if parameter_file.shape[1] < 16 and parameter_file[:, 1:4].sum() == 0:
            alignment_option = 2
            alignment_option = 1
    if (
        need_initialize_classification
        or int(project_params.param(fp["mode"], iteration)) == 0
    ):
        alignment_option = 0

    return alignment_option


def select_clean_micrographs(parameters, micrographs, inputlist):

    if not os.path.exists(micrographs + "_clean") and len(inputlist) > 100:
        dbase = np.loadtxt(parameters["data_set"] + "_dbase.txt", skiprows=1, dtype=str)
        # th = statistics.optimal_threshold( dbase[:,19].astype('f'), 'optimal', parameters['data_set'] + '.png' )
        th = statistics.optimal_threshold(
            dbase[:, 19].astype("f"), "optimal", parameters["data_set"] + ".pdf"
        )
        if parameters["slurm_email"]:
            user_comm.notify(parameters["data_set"], parameters["data_set"] + ".png")
        logger.info("Filtering out micrographs below threshold = %f", th)
        names = dbase[:, 0].squeeze()
        clean_list = names[(dbase[:, 19].astype("f") >= th).squeeze()].tolist()
        # clean_list = dbase[ dbase[:,19].astype('f') >= th ][:,0].tolist()
        shutil.move(micrographs, micrographs + "_all")
        np.savetxt(micrographs + "_clean", clean_list, fmt="%s")
        os.symlink(micrographs + "_all", micrographs)
        shutil.copy(
            parameters["data_set"] + "_dbase.txt",
            parameters["data_set"] + "_all_dbase.txt",
        )
    else:
        clean_list = inputlist

    return clean_list


def param(parameter, iteration):
    if isinstance(parameter, str):
        listed = parameter.split(":")
        value = listed[min(iteration - 2, len(listed) - 1)]
        if "random" in value:
            # refine 3 parameters at random
            items = ["1", "1", "0", "0", "0"]
            random.shuffle(items)
            value = ",".join(items)
        return value
    else:
        return parameter


def parse_from_groups(groups, my_parameters=0):

    description = groups["_help"]
    parser = argparse.ArgumentParser(description=description)

    for group in groups:

        if not group.startswith("_"):

            if "_name" not in groups[group]:
                raise Exception(
                    "Configuration file entry {} is missing entry {}".format(
                        group, "_name"
                    )
                )
            name = groups[group]["_name"]
            if "_description" not in groups[group]:
                raise Exception(
                    "Configuration file entry {} is missing entry {}".format(
                        group, "_description"
                    )
                )
            description = groups[group]["_description"]
            args_group = parser.add_argument_group(name, description)

            for parameter in groups[group]:

                if not parameter.startswith("_"):

                    option = "-" + group + "_" + parameter
                    long_option = "-" + option

                    if "description" not in groups[group][parameter]:
                        raise Exception(
                            "Configuration file entry {} is missing entry {}".format(
                                groups[group][parameter], "description"
                            )
                        )
                    description = groups[group][parameter]["description"]

                    if "type" not in groups[group][parameter]:
                        raise Exception(
                            "Configuration file entry {} is missing entry {}".format(
                                groups[group][parameter], "type"
                            )
                        )
                    partype = groups[group][parameter]["type"]

                    # required?
                    required = False
                    if "required" in groups[group][parameter].keys():
                        required = groups[group][parameter]["required"]
                        # check if value was already provided
                        if (
                            my_parameters != 0
                            and group + "_" + parameter in my_parameters.keys()
                        ):
                            required = False

                    # default value
                    default = None
                    if "default" in groups[group][parameter].keys():
                        default = groups[group][parameter]["default"]

                    if "bool" in partype:
                        # add true option
                        new_group = parser.add_mutually_exclusive_group(required=required)
                        new_group.add_argument(
                            option,
                            long_option,
                            help=description,
                            action="store_true",
                            dest=option[1:],
                            default=default,
                        )
                        # add false or "-no" option
                        new_group.add_argument(
                            "-no" + option,
                            "--no" + option,
                            help=description,
                            action="store_false",
                            dest=option[1:],
                        )
                    else:
                        if partype == "enum":
                            enum = groups[group][parameter]["enum"]
                            args_group.add_argument(
                                option,
                                long_option,
                                help=description,
                                choices=enum,
                                required=required,
                                default=default,
                            )
                        elif partype == "path":
                            args_group.add_argument(
                                option,
                                long_option,
                                help=description,
                                type=Path,
                                required=required,
                                default=default,
                            )
                        else:
                            args_group.add_argument(
                                option,
                                long_option,
                                help=description,
                                type=eval(partype),
                                required=required,
                                default=default,
                            )
    return parser


def parse_parameters(my_parameters,block,mode):
    """Parse PYP parameters

    Parameters
    ----------
    my_parameters : dict
        Existing parameters

    Returns
    -------
    dict
        Parsed PYP parameters

    Raises
    ------
    Exception
        [description]
    """
    # read specification file
    specification_file = "/opt/pyp/config/pyp_config.toml"

    specifications = toml.load(specification_file)

    # only check tabs included in current block and mode
    blocks = [ specifications["blocks"][b]["tabs"] for b in specifications["blocks"].keys() if mode in b and ( ( block in b and block != "import" ) or b.endswith("_import_raw") ) ]
    blocks = list(np.unique(np.concatenate(blocks).flat))
    tabs = {}
    tabs["_name"] = specifications["tabs"]["_name"]
    tabs["_help"] = specifications["tabs"]["_help"]
    tabs["_description"] = specifications["tabs"]["_description"]
    for t in specifications["tabs"].keys():
        if t in blocks:
            tabs[t] = specifications["tabs"][t]

    # create parser
    parser = parse_from_groups(tabs, my_parameters)

    # parse input
    if my_parameters != 0:
        parameters = parser.parse_args(namespace=argparse.Namespace(**my_parameters))
    else:
        parameters = parser.parse_args()
    return vars(parameters)


def load_parameters(path=".", param_file_name=".pyp_config.toml"):
    configuration_file = Path(path) / param_file_name
    if os.path.exists(configuration_file):
        my_namespace = toml.load(configuration_file)

        # add defaults to the parameters, to prevent KeyErrors
        def set_default(key, value):
            if key not in my_namespace:
                my_namespace[key] = value

        set_default('slurm_queue', '')

    else:
        my_namespace = 0
    return my_namespace

# remove entries that are no longer defined in the configuration file
def sanitize_parameters(parameters):

    # read specification file
    import toml
    specifications = toml.load("/opt/pyp/config/pyp_config.toml")

    clean_parameters = {}

    # figure out which parameters need to be reset to their default values
    for t in specifications["tabs"].keys():
        if not t.startswith("_"):
            for p in specifications["tabs"][t].keys():
                if not p.startswith("_"):
                    if parameters.get(f"{t}_{p}"):
                        clean_parameters[f"{t}_{p}"] = parameters[f"{t}_{p}"]
    return clean_parameters

def save_parameters(parameters, path=".", param_file_name=".pyp_config.toml"):
    # WARNING - toml.dump does not support saving entries that are None, so those will not be saved
    parameter_file = Path(path) / param_file_name
    with open(parameter_file, "w") as f:
        toml.dump(parameters, f)

    # save parameters to website
    try:
        save_parameters_to_website(sanitize_parameters(parameters))
    except:
        logger.warning("Detected inconsistencies in pyp configuration file")
        type, value, traceback = sys.exc_info()
        sys.__excepthook__(type, value, traceback)

def load_pyp_parameters(path="."):
    return load_parameters(path)


def load_3davg_parameters(path="."):
    return load_parameters(path, param_file_name="3davg.config")


def load_relion_parameters(path="."):
    return load_parameters(path, param_file_name="relion.config")


def save_pyp_parameters(parameters, path="."):
    save_parameters(parameters, path)


def save_3davg_parameters(parameters, path="."):
    save_parameters(parameters, path, param_file_name="3davg.config")


def save_relion_parameters(parameters, path="."):
    save_parameters(parameters, path, param_file_name="relion.config")


def parse_pyp_arguments():
    """Parse argument format from toml configuration file

    Returns
    -------
     parser
        Argument parser
    """

    parser = argparse.ArgumentParser(
        description="PYP", formatter_class=argparse.RawTextHelpFormatter
    )

    configuration = toml.load("/opt/pyp/config/pyp_args.toml")["tabs"]

    # for each tab type
    for key in configuration.keys():

        for parameter in key.keys():

            option = "-" + key + "_" + parameter
            long_option = "-" + option
            description = configuration[key][parameter].description
            partype = configuration[key][parameter].type

            # if is required
            if not "default" in configuration[key][parameter].keys():
                parser.add_argument(
                    option, long_option, help=description, type=partype, required=True
                )
            else:
                default = configuration[key][parameter][default]
                parser.add_argument(
                    option, long_option, help=description, type=partype, default=default
                )

    return parser.parse_args()


def parse_arguments(mode):
    """Parse argument format from toml configuration file

    Parameters
    ----------
    mode : str
        Name of execution mode

    Returns
    -------
    parser
        Argument parser
    """

    configuration = toml.load("/opt/pyp/config/pyp_modes.toml")[mode]

    # switch to full format if in streampyp
    if mode == "streampyp":

        # build arguments using pyp's format
        parser = parse_from_groups(configuration)

    else:

        description = configuration["_help"]
        parser = argparse.ArgumentParser(description=description)

        for key in configuration.keys():

            if not key.startswith("_"):
                option = "-" + key
                long_option = "-" + option
                description = configuration[key]["help"]
                partype = configuration[key]["type"]

                # if is required
                default = None
                required = True
                if "default" in configuration[key].keys():
                    default = configuration[key]["default"]
                    required = False

                if "bool" in partype:
                    parser.add_mutually_exclusive_group(required=required)
                    parser.add_argument(
                        option,
                        long_option,
                        help=description,
                        action="store_true",
                        dest=key,
                        default=default,
                    )
                    parser.add_argument(
                        "-no" + option,
                        "--no" + option,
                        help=description,
                        action="store_false",
                        dest=key,
                    )
                else:

                    if partype == "enum":
                        enum = configuration[key]["enum"]
                        parser.add_argument(
                            option,
                            long_option,
                            help=description,
                            choices=enum,
                            required=required,
                            default=default,
                        )
                    elif partype == "path":
                        parser.add_argument(
                            option,
                            long_option,
                            help=description,
                            type=Path,
                            required=required,
                            default=default,
                        )
                    else:
                        parser.add_argument(
                            option,
                            long_option,
                            help=description,
                            type=eval(partype),
                            required=required,
                            default=default,
                        )

    return parser.parse_args()

def get_mask_from_projects() -> str:
    """get_mask_from_projects Get the latest mask from existing projects

    _extended_summary_

    Returns
    -------
    str
        Path to shape mask
    """
    # be sure you're in a project where you can look for masking project
    masking_blocks = list(Path().cwd().parent.glob("*masking*"))
    masking_blocks.sort(key=os.path.getctime, reverse=True)

    for block in masking_blocks:
        mask = Path(block) / "frealign" / "maps" / "mask.mrc"
        if mask.exists():
            return str(mask)

    return "none"


def get_weight_from_projects(weight_folder: Path, parameters: dict) -> str:
    """get_weight_from_projects 

        Find weight file from current project or its parent, prioritize to use the file as follows:
        1. Global weight in the current project
        2. Global weight in the parent project

    Returns
    -------
    str
        _description_
    """
    weight_current = Path(weight_folder) / "global_weight.txt"
    weight_parent = Path(parameters["data_parent"]) / "frealign" / "weights" / "global_weight.txt"

    if weight_current.exists():
        return str(weight_current)

    elif weight_parent.exists():
        return str(weight_parent)

    return None

def parameter_force_check(previous_parameters, new_parameters, project_dir="."):

    all_differences = {k for k in previous_parameters.keys() & new_parameters.keys() if previous_parameters[k] != new_parameters[k] and 'force' not in k}

    differences = {d for d in all_differences if not ( ( isinstance(previous_parameters[d],PosixPath) or isinstance(previous_parameters[d],str) ) and project_params.resolve_path(previous_parameters[d]) == project_params.resolve_path(new_parameters[d]) ) }

    if previous_parameters.get("slurm_verbose"):
        if len(differences):
            logger.info(f"Parameters changed: {differences}")
        else:
            logger.info("No parameter changes detected")

    try:
        data_set = previous_parameters["data_set"]
    except KeyError:
        data_set = None
    micrographs = os.path.join( project_dir, "{}.micrographs".format(data_set) )
    if os.path.exists(micrographs):
        with open(micrographs) as f:
            inputlist = [line.strip() for line in f]

        # initialize all _force parameters if previous run was successful
        if len(get_missing_files(previous_parameters, inputlist, verbose=False)) == 0:
            new_parameters["movie_force"] = False
            new_parameters["ctf_force"] = False
            new_parameters["detect_force"] = False
            if "tomo" in previous_parameters["data_mode"]:
                new_parameters["tomo_ali_force"] = False
                new_parameters["tomo_vir_force"] = False
                new_parameters["tomo_rec_force"] = False

    if len(differences) > 0:

        for k in differences:

            # cases could change all the workflow
            if any(
                ["scope_" in k and not "scope_tilt_axis" in k, "movie_" in k, "gain_" in k ]
                ):

                # for some cases movie alignment won't change
                if (
                    k == "movie_first"
                    or k == "movie_last"
                    or k == "movie_pbc"
                    or k == "movie_boff"
                ):
                    # assume we are recomputing frame averages
                    logger.info(
                        f"Frame averages will be re-computed to reflect change in parameter {k}"
                    )
                    [os.remove(f) for f in glob.glob("mrc/*.mrc")]
                    [os.remove(f) for f in glob.glob("mrc/*.tif")]
                    new_parameters["movie_force"] = True

                else:
                    # assume we are recomputing frame alignment
                    logger.info(
                        f"Frame alignment parameters will be re-computed to reflect change in parameter {k}"
                    )
                    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "ali" "*.xf") ) ]

                    new_parameters["movie_force"] = True
                    # Triggering all following recalculations
                    new_parameters["ctf_force"] = True
                    clean_ctf_files(project_dir)
                    new_parameters["detect_force"] = True
                    clean_picking_files(project_dir)

                    if "tomo" in previous_parameters["data_mode"]:
                        new_parameters["tomo_ali_force"] = True
                        new_parameters["tomo_vir_force"] = True
                        new_parameters["tomo_rec_force"] = True
                        clean_tomo_vir_particles(project_dir)

            # specific ones
            else:
                if "ctf_" in k and not "_ctf" in k and not "use" in k:
                    # assume we are recomputing CTF
                    logger.info(
                        f"CTF parameters will be re-computed to reflect change in parameter {k}"
                    )
                    new_parameters["ctf_force"] = True
                    clean_ctf_files(project_dir)

                if "detect_" in k or "tomo_spk_" in k:
                    # assume we are re-picking particles
                    logger.info(
                        f"Particle positions will be re-computed to reflect change in parameter {k}"
                    )
                    clean_picking_files(project_dir)
                    new_parameters["detect_force"] = True

                if "extract_cls" == k:
                    # clear up previous coordinates
                    logger.info(
                        f"ALLBOXS and ALLPARXS parameters will be re-computed to reflect change in parameter {k}"
                    )
                    clean_picking_files(project_dir)

                # tomo reconstruction cascade
                if "tomo_ali" in k or "scope_tilt_axis" in k:
                    logger.info(
                        f"Tilt-series will be re-aligned to reflect change in parameter {k}"
                    )
                    new_parameters["tomo_ali_force"] = True
                    new_parameters["tomo_rec_force"] = True
                    new_parameters["tomo_vir_force"] = True
                    clean_tomo_vir_particles(project_dir)

                elif "tomo_rec" in k:
                    logger.info(
                        f"Tomograms will be re-computed to reflect change in parameter {k}"
                    )
                    new_parameters["tomo_rec_force"] = True
                    if not "tomo_rec_erase_fiducials" in k:
                        new_parameters["tomo_vir_force"] = True
                        clean_tomo_vir_particles(project_dir)

                elif "tomo_vir_" in k and "tomo_vir_detect_" not in k:
                    logger.info(
                        f"Virions will be re-computed to reflect change in parameter {k}"
                    )
                    new_parameters["tomo_vir_force"] = True
                    clean_tomo_vir_particles(project_dir)

                    thresholds_file = os.path.join( project_dir, "next", "virion_thresholds.next")
                    if os.path.exists(thresholds_file):
                        os.remove(thresholds_file)

    else:
        # rerun a failed job without changing parameters 
        if len(glob.glob("mrc/*mrc")) == 0:
            logger.info("No processed results detected in the mrc/ folder, will force movie alignment, ctf estimation, and particle detection")
            new_parameters["movie_force"] = True
            # Triggering all following recalculations
            new_parameters["ctf_force"] = True
            clean_ctf_files(project_dir)
            new_parameters["detect_force"] = True
            clean_picking_files(project_dir)

            if "tomo" in previous_parameters["data_mode"]:
                new_parameters["tomo_ali_force"] = True
                new_parameters["tomo_vir_force"] = True
                new_parameters["tomo_rec_force"] = True
                clean_tomo_vir_particles(project_dir)

        elif "tomo" in previous_parameters["data_mode"] and len(glob.glob("mrc/*rec")) == 0:
            new_parameters["tomo_vir_force"] = True
            new_parameters["tomo_rec_force"] = True
            clean_tomo_vir_particles(project_dir)

    return new_parameters


def clean_ctf_files(project_dir):

    [
        os.remove(f)
        for f in glob.glob( os.path.join(project_dir, "ctf" "*.ctf") ) + glob.glob( os.path.join(project_dir, "ctf" "*.def") )
    ]


def clean_picking_files(project_dir):

    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "csp", "*.*") )]
    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "sva", "*.*") )]


def clean_tomo_vir_particles(project_dir):

    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "mrc", "*_vir????_binned_nad.*") )]
    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "sva", "*_vir*.*") )]
    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "next", "virion_thresholds.next") )]
    [os.remove(f) for f in glob.glob( os.path.join(project_dir, "csp", "*.*") )]


def get_latest_refinement_reference(parent_path: str):
    """get_latest_refinement_reference Get the latest parfile/reference from parent block

    _extended_summary_

    Parameters
    ----------
    parent_path : str
        Path to the parent block

    Returns
    -------
        Path to parfile and reference
        
    """

    parent_refinement_path = Path(parent_path) / "frealign" / "maps"
    
    if not parent_refinement_path.exists():
        return None, None
    
    parfiles = sorted(list(parent_refinement_path.glob("*_r01_??.par*")), key=lambda x: str(x))
    references = sorted(list(parent_refinement_path.glob("*_r01_??.mrc")), key=lambda x: str(x))

    latest_parfile = parfiles[-1] if len(parfiles) > 0 else None
    latest_reference = references[-1] if len(references) > 0 else None

    return latest_parfile, latest_reference
