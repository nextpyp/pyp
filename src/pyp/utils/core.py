import errno
import os
from itertools import chain
from pathlib import Path
from typing import Union

cat = "".join
flatten = chain.from_iterable


def has_files(basename, trailers, suffix=False):
    if suffix:
        return all(os.path.exists(f"{basename}{sufx}") for sufx in trailers)
    return all(os.path.exists(f"{basename}.{ext}") for ext in trailers)


def symlink_force(source, target):
    try:
        symlink_relative(source, target)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(target)
            symlink_relative(source, target)
        else:
            raise e

def symlink_relative(target: Union[Path, str], destination: Union[Path, str]):
    """Create a symlink pointing to ``target`` from ``location``.
    Args:
        target: The target of the symlink (the file/directory that is pointed to)
        destination: The location of the symlink itself.
    """
    target = Path(target)
    destination = Path(destination)
    target_dir = destination.parent
    target_dir.mkdir(exist_ok=True, parents=True)
    relative_source = os.path.relpath(target, target_dir)
    dir_fd = os.open(str(target_dir.absolute()), os.O_RDONLY)
    try:
        os.symlink(relative_source, destination.name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)

def makedirs_list(paths):
    [os.makedirs(f) for f in paths if not os.path.exists(f) and not os.path.islink(f)]


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.resolve()


def get_src_path() -> Path:
    """Returns Path of the src/ folder."""
    return get_project_root().joinpath("src")


def get_relative_path(path: Union[str, Path]) -> Path:
    """Returns Path relative to src/ folder."""
    # return Path(path).resolve().relative_to(get_src_path())
    try:
        return Path(path).resolve().relative_to(get_src_path())
    except:
        return path


def movie2regex(pattern, filename="*"):
    """Translate the pattern in movie_pattern to RegEx, used to search corresponding frame files

        The following tags in the pattern will be replaced by regular expression
        ---------------------------------------------------------
        TILTSERIES -> name of the tiltseries
        SCANORD -> scanning order (i.e. 000, 001, 002, ...)
        ANGLE -> tilt angles (i.e. -0.0, 10, 0.5, ...)

    Parameters
    ----------
    pattern : str
        Pattern of filename
    filename : str
        Name of the tilt-series

    Returns
    -------
    str
        RegEx
    """
    DELIMITERS = ["[", "]", "<", ">", "{", "}", "_", "-", ".", "(", ")", "|"]

    STAR = "[\w,\W]+"
    STAR_NUM_LETTER_ONLY = "[0-9a-zA-Z]+"

    # starts with
    regex = "^" + pattern

    # add escape to these special characters
    for d in DELIMITERS:
        regex = regex.replace(d, f"\{d}")
        filename = filename.replace(d, f"\{d}")

    if filename == "*":
        regex = regex.replace("TILTSERIES", "(%s)" % (STAR))
    else:
        regex = regex.replace("TILTSERIES", "(" + filename + ")")

    regex = regex.replace("SCANORD", "(\d+)")
    regex = regex.replace("ANGLE", "([+-]?[0-9]+(\.[0-9]+)?)")

    regex = regex.replace("*", STAR_NUM_LETTER_ONLY)

    # ends with
    regex += "$"

    return regex
