import filecmp
import glob
import os
import shutil
from pathlib import Path

from pyp.system.logging import logger

def load_files(current_path,working_path,f):
    if (current_path / f).is_file():
        logger.info(f"Retrieving {Path(f).name}")
        shutil.copy2(current_path / f, working_path)
    elif Path(f).exists():
        logger.info(f"Retrieving {Path(f).name}")
        shutil.copy2(f, working_path)

def load_results(file_list, files_path, working_path):
    """Load existing results from files."""

    arguments = []
    current_path = Path(files_path)
    for f in file_list:
        arguments.append(
            (
                current_path,
                working_path,
                f, 
            )
        )
    
    from pyp.system import mpi
    mpi.submit_function_to_workers(
        load_files, arguments, silent=True
    )

def transfer_files(project_path,d,file):
    target = Path(project_path) / d / os.path.split(file)[-1]
    # make all intermediate directories if not exist
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(file):
        if os.path.exists(target):
            if not filecmp.cmp(file, target):
                logger.info("Updating %s" % Path(target).name)
                os.remove(target)
                shutil.copy2(file, target)
            else:
                logger.info("Keeping existing %s" % Path(target).name)
        else:
            logger.info("Saving %s" % Path(target).name)
            shutil.copy2(file, target)

def save_results(files, project_path):
    """Save processing results"""

    arguments = []
    for d in files.keys():
        for f in files[d].split():
            for file in glob.glob(f):
                arguments.append(
                    (
                        project_path,
                        d, 
                        file,
                    )
                )
    from pyp.system import mpi
    
    if len(arguments) > 0:
        mpi.submit_function_to_workers(
            transfer_files, arguments, silent = True
        )
