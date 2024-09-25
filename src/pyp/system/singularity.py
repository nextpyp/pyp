#!/usr/bin/env python

import os
import sys

import toml

from pyp.streampyp.web import Web


def get_pyp_configuration():

    # first try environment variable
    if "PYP_CONFIG" in os.environ:
        configuration_file = os.environ["PYP_CONFIG"]
    else:
        configuration_file = ""
    # then try default location
    if not os.path.exists(configuration_file):
        configuration_file = os.path.join(os.environ["HOME"], ".pyp/config.toml")
    if not os.path.exists(configuration_file):
        print("ERROR: Cannot find configuration file " + configuration_file)
    config = toml.load(configuration_file)

    return config


def get_singularity_command(command, parameters, gpu=False):

    config = get_pyp_configuration()

    binds = "-B " + ",".join(config["pyp"]["binds"])

    if os.path.exists(config["pyp"]["scratch"]):
        binds = binds + "," + config["pyp"]["scratch"]
    elif os.path.exists(config["pyp"]["scratch"].split("$")[0]):
        binds = binds + "," + config["pyp"]["scratch"].split("$")[0]

    if "SINGULARITY_CONTAINER" in os.environ:
        binds += " --no-home -B {0}/.ssh".format(os.environ["HOME"])

    if "sources" in config["pyp"].keys():
        binds += " -B {0}:/opt/pyp".format(config["pyp"]["sources"])

    container = config["pyp"]["container"]

    if gpu:
        gpu_enable = "--nv"
    else:
        gpu_enable = ""

    command = (
        f"mkdir -p {os.environ['PYP_SCRATCH']}; singularity --quiet --silent exec {gpu_enable} {binds} {container} {command} {parameters}"
    )

    return command


# command to run slurm
def run_ssh(command):

    config = get_pyp_configuration()

    server = config["slurm"]["host"]

    command = f"ssh {server} \"bash --login -c '{command}'\""

    return command


# command to run slurm
def run_slurm(command, path="", env="", quick=False):
    """Container-aware launching of slurm commands

    Parameters
    ----------
    command : str
        Slurm command to run [sbatch, scontrol, squeue, etc.]
    path : str, optional
        Location to run, by default ''
    env : str, optional
        Environment variables to define, by default ''

    Returns
    -------
    [type] : str
        Command to run
    """

    config = get_pyp_configuration()

    if "path" in config["slurm"]:
        slurm_path = config["slurm"]["path"] + " > /dev/null 2>&1; "
    else:
        slurm_path = ""
    slurm_path = slurm_path + f"{command}"

    # command = os.path.join( slurm_path, command )
    command = slurm_path

    # add paths if specified
    if len(path) > 0:

        command += " -D {0}".format(path)

    # add exports if specified
    if len(env) > 0:

        command += " --export=ALL,{0}={0}".format(env)

    if quick and "quickQueue" in config["slurm"]:
        command = command + " " + config["slurm"]["quickQueue"]

    elif "queue" in config["slurm"]:
        command = command + " " + config["slurm"]["queue"]

    return command


# command to run pyp
def run_pyp(command, script=False, cpus=1, gpu=False):

    # we always want to execute pyp inside the container
    command = "/opt/pyp/bin/run/" + command

    # if this pyp instance was launched by the website, don't wrap the command with another containerization step
    # the website will handle the re-containerization, so just return the raw command
    if Web.exists:
        return command

    # if script or not os.environ['SINGULARITY_CONTAINER']:
    if script:

        command = get_singularity_command(command=command, parameters="", gpu=gpu)

        singularity_path = ""
        if "singularity" in get_pyp_configuration()["pyp"].keys():
            singularity_path = get_pyp_configuration()["pyp"]["singularity"]
        elif "singularity" in get_pyp_configuration()["slurm"].keys():
            singularity_path = get_pyp_configuration()["slurm"]["singularity"]
        if len(singularity_path) > 0:
            command = singularity_path + "; " + command

    return command


def get_mpirun_command(cpus=1):
    mpirun = get_pyp_configuration()["slurm"]["mpirun"] + "; mpirun"
    mpirun = mpirun + " --oversubscribe -n {}".format(cpus)

    return mpirun
