"""Logging module. Adapted from `AMSET`_ and `Real Python`_


.. _AMSET:
    https://github.com/hackingmaterials/amset
.. _Real Python:
    https://realpython.com/python-logging/
"""

import datetime
import logging
import sys
from pathlib import Path
from typing import Union

from pyp.streampyp.logging import WebLogHandler
from pyp.streampyp.web import Web
from pyp.system.mongo_handler import MongoHandler

__author__ = "Xiaochen Du"
__maintainer__ = "Alberto Bartesaghi"
__email__ = "alberto@cs.duke.edu"

logger = logging.getLogger(__name__)


def initialize_pyp_logger(
    log_name: str = "pyp",
    directory: Union[str, Path] = ".",
    filename: Union[str, Path, bool] = False,
    level: int = logging.INFO,
    print_log: bool = True,
) -> logging.Logger:
    """Initialize the default logger with stdout and file handlers.

    Parameters
    ----------
    log_name : str, optional
        Name of the logging object, by default "pyp"
    directory : Union[str, Path], optional
        Path to the folder where the log file will be written, by default "."
    filename : Union[str, Path, bool], optional
        The log filename. If False, no log will be written, by default False
    level : int, optional
        The log level, by default logging.INFO
    print_log : bool, optional
        Whether to print the log to the screen, by default True

    Returns
    -------
    logging.Logger
        A logging instance with customized formatter and handlers.
    """

    DATE_TIME_FMT = "%Y-%m-%d %H:%M:%S"

    log = logging.getLogger(log_name)
    log.setLevel(level)
    log.handlers = []  # reset logging handlers if they already exist

    screen_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d | %(message)s"
    file_format = screen_format

    # screen_handler is the regular stream (bash i/o) handler
    # in PYP, the stream is redirected to a text file
    # file_handler writes output to a file, which is unnecessary in the current context
    # but in the future, we might want to configure a separate file_handler

    if print_log:
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setLevel(logging.INFO)
        screen_formatter = logging.Formatter(screen_format)
        screen_formatter.datefmt = DATE_TIME_FMT

        screen_handler.setFormatter(screen_formatter)
        log.addHandler(screen_handler)

        def decorate_emit(fn):

            reset = "\x1b[0m"
            bright = "\x1b[1m"
            dim = "\x1b[2m"
            underscore = "\x1b[4m"
            blink = "\x1b[5m"
            reverse = "\x1b[7m"
            hidden = "\x1b[8m"

            black = "\x1b[30m"
            red = "\x1b[31m"
            green = "\x1b[32m"
            yellow = "\x1b[33m"
            blue = "\x1b[34m"
            magenta = "\x1b[35m"
            cyan = "\x1b[36m"
            white = "\x1b[37m"

            BGblack = "\x1b[40m"
            BGred = "\x1b[41m"
            BGgreen = "\x1b[42m"
            BGyellow = "\x1b[43m"
            BGblue = "\x1b[44m"
            BGmagenta = "\x1b[45m"
            BGcyan = "\x1b[46m"
            BGwhite = "\x1b[47m"

            # add methods we need to the class
            def new(*args):
                levelno = args[0].levelno
                if levelno >= logging.CRITICAL:
                    color = "\x1b[31;1m"
                    color = red
                elif levelno >= logging.ERROR:
                    color = "\x1b[31;1m"
                    color = red
                elif levelno >= logging.WARNING:
                    color = "\x1b[33;1m"
                    color = yellow
                elif levelno >= logging.INFO:
                    color = "\x1b[32;1m"
                    color = green
                elif levelno >= logging.DEBUG:
                    color = "\x1b[35;1m"
                    color = green
                else:
                    color = "\x1b[0m"
                    color = white

                # print message in color:
                # args[0].name
                # args[0].levelname
                # args[0].msg

                args[0].name = "{0}{1}{2}".format(
                    bright, "%s" % args[0].name, reset
                )
                args[0].levelname = "{0}{1}\x1b[0m".format(color, args[0].levelname)

                # new feature i like: bolder each args of message
                # args[0].args = tuple('\x1b[1m' + arg + '\x1b[0m' for arg in args[0].args)
                return fn(*args)

            return new

        screen_handler.emit = decorate_emit(screen_handler.emit)
        log.addHandler(screen_handler)

    if filename:
        file_handler = logging.FileHandler(Path(directory) / filename, mode="w")
        file_handler.setLevel(logging.ERROR)
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        # log.addHandler(file_handler)

    # route log entries to the website, if it exists
    if Web.exists:
        log.addHandler(WebLogHandler())

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        now = datetime.datetime.now()
        exit_msg = "pyp exiting at {}".format(now.strftime(DATE_TIME_FMT))

        log.error(
            "\n  ERROR: {}".format(exit_msg),
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception

    return log
