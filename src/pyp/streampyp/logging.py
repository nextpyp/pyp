
import logging
import os
import time

from pyp.streampyp.web import Web


def get_slurm_array_id():
    """
    get the SLURM array id, or None
    """
    try:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        return None


def is_logging_allowed():
    """
    logging is only allowed from non-SLURM jobs,
    non-array SLURM jobs, or the first element of a SLURM array job
    """
    array_id = get_slurm_array_id()
    return array_id is None or array_id == 1


class WebLogHandler(logging.Handler):

    def __init__(self):
        logging.Handler.__init__(self)
        self.enabled = is_logging_allowed()

    def emit(self, record):

        # skip logging if we're disabled
        if not self.enabled:
            return

        timestamp = int(record.created * 1000)
        level = record.levelno
        path = record.name  # usually a path to a python script
        line = record.lineno
        msg = record.message

        Web().log(timestamp, level, path, line, msg)


class TQDMLogger:

    def __init__(self):
        self.buffer = ''
        self.web = None
        if Web.exists and is_logging_allowed():
            self.web = Web()

    def write(self, val):
        self.buffer += val

    def flush(self):
        print(self.buffer)
        if self.web is not None:

            timestamp = int(time.time() * 1000)

            # special constant the website associates with progress messages
            level = -10

            # there's no real file/line info for progress messages,
            # since they don't originate from a eg log.info() call
            path = ""
            line = 0

            msg = self.buffer

            self.web.log(timestamp, level, path, line, msg)

        self.buffer = ''
