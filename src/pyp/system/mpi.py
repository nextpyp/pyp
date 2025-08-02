import os
import sys
import subprocess
import numpy as np
from collections import Callable
from joblib import Parallel, delayed
import contextlib
import joblib
from tqdm import tqdm

from pyp.system.local_run import run_shell_command
from pyp.streampyp.logging import TQDMLogger
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path, timer

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)
parallel = None

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()

def get_process_information():
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    elif "SLURM_NTASKS" in os.environ:
        num_cpus = int(os.environ["SLURM_NTASKS"])
    else:
        num_cpus = 1
    return num_cpus


def initialize_worker_pool():
    global parallel
    num_cpus = get_process_information()
    parallel = Parallel(n_jobs=max(num_cpus, 1))
    return num_cpus


def terminate_worker_pool():
    # TODO: terminate workers manually 
    return 


def submit_jobs_file_to_workers(commands_file, working_path=os.getcwd()):
    """Run commands from a file in parallel using MPI.

    Parameters
    ----------
    commands_file : str
        File name containing the commands to run
    working_path : str, optional
        Directory where commands should be executed, by default os.getcwd()
    """
    # get list of tasks
    with open(commands_file) as f:
        commands = f.read().split("\n")

    new_commands = []
    for command in commands:
        new_commands.append("source " + command)

    submit_jobs_to_workers(new_commands, working_path)


def submit_jobs_to_workers(commands, working_path=os.getcwd(), verbose=False, silent = False):
    """Run shell command in parallel using MPI.

    Parameters
    ----------
    commands : list
        List of commands to execute
    working_path : str, optional
        Directory where commands should be executed, by default os.getcwd()
    """
    # first, detect if we are using MPI
    num_cpus = get_process_information()

    if num_cpus > 1 and len(commands) > 1:

        first_command = commands[0].split('\n')[0]
        if "/opt/" in first_command:
            first_command = first_command.split("/opt/")[1]
        if not silent:
            logger.info(f"Running {len(commands):,} command(s) ({first_command})")

        # NOTE: be aware of the current working directory for all the workers, as they might be initiated in a different place
        current_directory = os.getcwd()

        if silent:
            parallel(delayed(run_shell_command)(f"cd '{current_directory}' && " + i) for i in commands)
        else:
            with tqdm_joblib(tqdm(desc="Progress", total=len(commands), miniters=1, file=TQDMLogger())) as progress_bar:
                parallel(delayed(run_shell_command)(f"cd '{current_directory}' && " + i) for i in commands)

        if not silent:
            logger.info(f"{len(commands):,} command(s) finished")

    else:

        # execute all commands serially
        first_time = True
        for command in commands:
            try:
                [output, error] = run_shell_command(command, verbose=False)
                if first_time:
                    if verbose:
                        logger.info("Now executing {}".format(command))
                    first_time = False
            except:
                if not "frealign" in command:
                    logger.info("Ignoring exception caused by {}".format(command))
                pass
    # all done
    return


def submit_function_to_workers(function, arguments, verbose=False, silent=False):
    """Run python function in parallel using MPI.

    Parameters
    ----------
    function : function
        Python function object
    arguments : list
        List of arguments
    """

    if len(arguments) == 0:
        logger.warning("No arguments provided to function %s, skipping execution." % function.__name__)
        return

    funcs = []
    args = []
    num_processes = 0

    if isinstance(function, Callable):
        assert(type(arguments) == list), f"Arguments have to be a list but is {type(arguments)}"
        assert(type(arguments[0]) == tuple), f"Single argument passed to MPI needs to be tuple but is {type(arguments[0])}"
        funcs.append(function)
        args.append(arguments)
        num_processes += len(arguments)

    elif type(function) == list:
        assert(len(function) == len(arguments)), f"Number of functions ({len(function)}) and argument list ({len(arguments)}) should be the same"
        for f, g in zip(function, arguments):
            assert(type(g) == list), f"Arguments have to be a list but is {type(g)}"
            assert(len(g) > 0), f"Function {f.__name__} has 0 argument"
            assert(type(g[0]) == tuple), f"Single argument passed to MPI needs to be tuple but is {type(g[0])}"
            funcs.append(f)
            args.append(g)
            num_processes += len(g)
    else:
        raise Exception("MPI does not recognize this function.\n %s "%(type(function)))


    # first, detect if we are using MPI
    num_cpus = get_process_information()

    if num_cpus > 1:
        # NOTE: be aware of the current working directory for all the workers, as they might be initiated in a different place
        def wrapper(func, *arg, current_directory=os.getcwd()):
           os.chdir(current_directory)
           func(*arg)

        current_directory = os.getcwd()
        if not silent:
            logger.info(f"Running {num_processes:,} function(s) ({', '.join([f.__name__ for f in funcs])})")
        with tqdm_joblib(tqdm(desc="Progress", total=num_processes, miniters=1, file=TQDMLogger(), disable=silent)) as progress_bar:
            parallel(delayed(wrapper)(func, *arg, current_directory=current_directory) for idx, func in enumerate(funcs) for arg in args[idx])
        if not silent:
            logger.info(f"{num_processes:,} functions(s) finished")

    else:
        # execute all commands serially
        if not silent:
            logger.info(f"Running {num_processes:,} function(s) ({', '.join([f.__name__ for f in funcs])})")
        with tqdm(desc="Progress", total=num_processes, file=TQDMLogger(), disable=silent) as pbar:
           for function, arguments in zip(funcs, args):
                for argument in arguments:
                    function(*argument)
                    pbar.update(1)
    # all done
    return