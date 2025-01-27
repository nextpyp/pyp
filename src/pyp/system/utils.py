import GPUtil
import os
import socket
from pwd import getpwnam
from pyp.system.singularity import get_pyp_configuration

def timeout_command(command, time, full_path=False):
    if full_path:
        timeout_command = "timeout {1}s {0}".format(command, time)
    else:
        timeout_command = "timeout {2}s {0}/{1}".format(
            os.environ["PYP_DIR"], command, time
        )

    return timeout_command


def ctime(path):
    """Returns the number of milliseconds since path was last modified."""
    seconds = os.path.getctime(path)
    return int(seconds * 1000)


def clear_scratch():
    return


def eman_load_command():
    load_eman_cmd = "export PYTHONPATH=/opt/eman2/pkgs"
    return load_eman_cmd


def imod_load_command():
    load_imod_cmd = "export IMOD_DIR={0};".format(get_imod_path())
    return load_imod_cmd

def legacy_imod_load_command():
    load_imod_cmd = "export IMOD_DIR={0};".format(get_legacy_imod_path())
    return load_imod_cmd

def phenix_load_command():
    phenix = "  ; /programs/phenix-1.18.2-3874/phenix-1.18.2-3874/build/bin/"
    return phenix


def get_slurm_path():
    return "/opt/slurm/bin/"

def get_imod_path():
    return "/opt/IMOD"
    # return "export LD_LIBRARY_PATH=/opt/IMOD/qtlib:/opt/IMOD/lib:$LD_LIBRARY_PATH /opt/IMOD"

def get_legacy_imod_path():
    return "/opt/IMOD_4.11.24"

def cuda_path_prefix(command):
    config = get_pyp_configuration()
    if 'cudaLibs' in config["pyp"]:
        command = f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{':'.join(path for path in config['pyp']['cudaLibs'])}; " + command
    else:
        command = f"export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.5/targets/x86_64-linux/lib/stubs:/opt/imod_4.11.24/qtlib; " + command
    return command

def get_aretomo_path():
    config = get_pyp_configuration()
    if 'areTomo2' in config["pyp"]:
        command = config["pyp"]["areTomo2"]
    else:
        command = "/opt/pyp/external/AreTomo2/AreTomo2"
    command = cuda_path_prefix(command)
    return command

def get_motioncor3_path():
    config = get_pyp_configuration()
    if 'motionCor3' in config["pyp"]:
        command = config["pyp"]["motionCor3"]
    else:
        command = "/opt/pyp/external/MotionCor3/MotionCor3"
    command = cuda_path_prefix(command)
    return command

def get_gpu_id():
    return 0

def get_gpu_ids(parameters,separator=","):
    # return list of GPU devices based on gres variable
    gres = parameters["slurm_gres"]
    if "gpu" in gres:
        for g in gres.split(","):
            if "gpu" in g:
                for i in g.split(":"):
                    if i.isdigit():
                        gpus = int(i)
                        break
        return separator.join(str(x) for x in range(gpus))
    else:
        return 0

def get_gpu_file():
    return os.path.join(os.environ["PYP_SCRATCH"],"gpu_device.id")

def needs_gpu(parameters):
    # enable Nvidia GPU?
    
    gpu_for_movies = parameters.get("movie_force") and parameters.get("movie_ali") == 'motioncor'

    gpu_for_alignment = parameters.get("tomo_ali_force") and parameters.get("tomo_ali_method") == "aretomo"
    
    gpu_for_reconstruction = parameters.get("tomo_rec_force") and parameters.get("tomo_rec_method") == "aretomo"
    
    gpu_for_denoising = (
        parameters.get("micromon_block") == "tomo-denoising-train"
        or parameters.get("micromon_block") == "tomo-denoising-eval" and not ( parameters.get("tomo_denoise_method") == "topaz" and not parameters.get("tomo_denoise_topaz_use_gpu") )
    )

    gpu_for_segmentation = parameters.get("micromon_block") == "tomo-segmentation-open" and parameters.get("tomo_mem_use_gpu")

    gpu_for_mining = "tomo-milo" in parameters.get("micromon_block")

    gpu_for_picking = ( parameters.get("micromon_block") == "tomo-particles-train" 
                       or parameters["data_mode"] == "spr" and "train" in parameters["detect_method"]
                       or parameters["data_mode"] == "tomo" and "train" in parameters["tomo_vir_method"] 
                       or parameters["data_mode"] == "tomo" and "train" in parameters["tomo_spk_method"] and parameters["tomo_vir_method"] == "none" )
    
    gpu_for_heterogeneity = "drgn" in parameters.get("micromon_block")

    return (
        gpu_for_movies
        or gpu_for_alignment
        or gpu_for_reconstruction
        or gpu_for_denoising
        or gpu_for_segmentation
        or gpu_for_mining
        or gpu_for_picking
        or gpu_for_heterogeneity
    ) 

def get_gpu_queue(parameters):
    # try to get the gpu partition
    queue = ""
    return queue
    config = get_pyp_configuration()
    if "slurm" in config:
        if ( "slurm_queue_gpu" not in parameters or parameters["slurm_queue_gpu"] == None ):
            try:
                parameters["slurm_queue_gpu"] = config["slurm"]["gpuQueues"][0]
                queue = parameters["slurm_queue_gpu"]
            except:
                logger.warning("No GPU partitions configured for this instance?")
                pass
        elif "slurm_queue_gpu" in parameters and not parameters["slurm_queue_gpu"] == None:
            queue = parameters["slurm_queue_gpu"]
        else:
            logger.warning("No GPU partitions configured for this instance?")
    return queue

def get_gpu_devices():
    try:
        devices = GPUtil.getAvailable(order = 'load', limit = 64, maxLoad = 0.1, maxMemory = 0.1, includeNan=False, excludeID=[], excludeUUID=[])
    except:
        devices = []

def get_relion_path():
    return "{0}/external/postproc".format(os.environ["PYP_DIR"])


def get_multirun_path():
    return "{0}/external/multirun".format(os.environ["PYP_DIR"])


def get_tomo_path():
    return "{0}/external/TOMO".format(os.environ["PYP_DIR"])


def get_bsoft_path():
    return "{0}/external/bsoft".format(os.environ["PYP_DIR"])

def get_topaz_path():
    return "/opt/conda/envs/pyp/bin"

def get_embfactor_path():
    return "{0}/external/embfactor".format(os.environ["PYP_DIR"])


def get_frealign_paths():
    frealign_paths = {
        "cc3m": "{0}/external/frealign_v9.10".format(os.environ["PYP_DIR"]),
        "cclin": "{0}/external/frealign_v9.10_dev".format(os.environ["PYP_DIR"]),
        "new": "{0}/external/frealign_v9.11".format(os.environ["PYP_DIR"]),
        "frealignx": "{0}/external/frealignx".format(os.environ["PYP_DIR"]),
        "cistem2": "{0}/external/cistem2".format(os.environ["PYP_DIR"]),
    }
    return frealign_paths

def get_parameter_files_path():
    return "{0}/src/pyp/refine/3DAVG".format(os.environ["PYP_DIR"])


def get_summovie_path():
    return "{0}/external/summovie_1.0.2".format(os.environ["PYP_DIR"])


def get_unblur_path():
    return "{0}/external/unblur_1.0.2".format(os.environ["PYP_DIR"])


def get_unblur2_path():
    return "{0}/external/cistem2".format(os.environ["PYP_DIR"])

def get_tomoctf_path():
    return "{0}/external/tomoctf_src_June2014".format(os.environ["PYP_DIR"])

def get_csp_path():
    return "{0}/external/CSP".format(os.environ["PYP_DIR"])


def get_bm4d_path():
    return "{0}/external/bm4d".format(os.environ["PYP_DIR"])


def get_bfactor_path():
    return "{0}/external/bfactor_v1.04".format(os.environ["PYP_DIR"])


def get_ctffind4_path():
    return "{0}/external/ctffind4".format(os.environ["PYP_DIR"])


def get_ctffind_tilt_path():
    return "{0}/external/cistem2".format(os.environ["PYP_DIR"])


def get_shell_multirun_path():
    return "{0}/external/shell".format(os.environ["PYP_DIR"])

def check_env():
     # set environment to avoid potential lib conflicts
    if os.environ.get("LD_LIBRARY_PATH") and  "/.singularity.d/libs" in os.environ["LD_LIBRARY_PATH"]:
        current_env = os.environ["LD_LIBRARY_PATH"]
        os.environ["LD_LIBRARY_PATH"] = current_env.replace("/.singularity.d/libs", "")

def is_dcc():
    # kept for compatibility
    return True


# quality of service
def qos(partition):
    if "ccr" in partition and (
        getpwnam(os.environ["USER"]).pw_uid == 32194
        or getpwnam(os.environ["USER"]).pw_uid == 27129
        or getpwnam(os.environ["USER"]).pw_uid == 35302
    ):
        return "--qos ccrprio"
    else:
        return ""
