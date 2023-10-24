import glob
import os
import subprocess

import numpy as np

from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def need_reporting():
    """Returns whether should send results of run to user"""
    return False


def notify(
    subject,
    attach="",
    body="*** This is an automatically generated email ***",
    recipient="",
):
    """Attaches output image and notify the user"""
    if need_reporting():
        if not "projects_cmm" in attach:
            pwuid = "espydaemon@gmail.com"
            pwuid = "r6u1p6l5j5y8m1g8@spycryoem.slack.com"
        else:
            pwuid = "v2m4s1y8y7a6b4q2@cryo-em-group.slack.com"
        netid = subprocess.check_output("whoami", shell=True, text=True).strip()
        pwuid = "{}@duke.edu".format(netid)
        if os.path.exists(attach):
            com = """echo "{0}" | mutt -a "{1}" -s "{2}" -e "set realname=\\"PYP Daemon\\"  " {3} -- {4}""".format(
                body, attach, subject, recipient, pwuid
            )
            com = """echo "{0}" | mail -S "from=PYP Daemon <DoNotReply>" -s "{2}" -a "{1}" {3}""".format(
                body, attach, subject, pwuid
            )
        else:
            com = """echo "{0}" | mutt -s "{1}" -e "set realname=\\"PYP Daemon\\"  " {2} -- {3}""".format(
                body, subject, recipient, pwuid
            )
            com = """echo "{0}" | mail -S "from=PYP Daemon <DoNotReply>" -s "{1}" {2}""".format(
                body, subject, pwuid
            )
        logger.info("Notifying recipient with the following message")
        run_shell_command(com)


def pyp_swarm_notify(parameters, name, current_path):
    if (
        len(glob.glob(str(current_path / "ctf/*.png"))) == 1
        and "projects_mmc" in str(current_path)
        and "t" in parameters["email"].lower()
    ):

        if os.path.exists("%s_boxed.webp" % name):
            attach = os.getcwd() + "/%s_boxed.webp" % name
        else:
            attach = os.getcwd() + "/%s.png" % name
        notify(parameters["data_set"] + " (" + name + ")", attach)


def frealign_rec_merge_notify(mparameters, fparameters, dataset, ranking, iteration):
    if (
        need_reporting()
        and "t" in mparameters["email"].lower()
        and (iteration % 10 == 0)
    ):
        png_plot = "../maps/%s_classes.png" % dataset
        attach = os.getcwd() + "/" + png_plot
        notify(dataset + " (Classes)", attach)

        # figure out highest-resolution class
        best_class = np.argmax(ranking) + 1
        subject = "%s_r%02d_%02d (3D)" % (
            fparameters["dataset"],
            best_class,
            iteration,
        )
        png_plot = "../maps/%s_r%02d_%02d.png" % (
            fparameters["dataset"],
            best_class,
            iteration,
        )
        attach = os.getcwd() + "/" + png_plot
        notify(subject, attach)
