from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def update_pyp_params_using_ctf(parameters, ctf, save=False):
    """Update frealign_parfile based on ctf file.

    Parameters
    ----------
    parameters : dict
        Parameters loaded from a .pyp_config file
    ctf : list
        CTF parameters in the same format as in a .ctf file
    save : bool
        Whether to save updated parameters to .pyp_config

    Returns
    ----------
    parameters : dict
        Updated parameters
    """

    if parameters["scope_pixel"] == 0:
        parameters["scope_pixel"] = str(ctf[9])
    if parameters["scope_voltage"] == 0:
        parameters["scope_voltage"] = str(ctf[10])
    if parameters["scope_mag"] == 0:
        parameters["scope_mag"] = str(ctf[11])
    if save:
        project_params.save_pyp_parameters(parameters)
    return parameters


def update_pyp_scope_params(
    parameters, scope_pixel, scope_voltage, scope_mag, save=False
):
    """Update ctf parameters in frealign_parfile.

    Parameters
    ----------
    parameters : dict
        Parameters loaded from a .pyp_config file
    scope_pixel : str
    scope_voltage : str
    scope_mag : str
    save : bool
        Whether to save updated parameters to .pyp_config

    Returns
    ----------
    parameters : dict
        Updated parameters
    """

    if parameters["scope_pixel"] == 0:
        parameters["scope_pixel"] = str(scope_pixel)
    if parameters["scope_voltage"] == 0:
        parameters["scope_voltage"] = str(scope_voltage)
    if parameters["scope_mag"] == 0:
        parameters["scope_mag"] = str(scope_mag)
    if save:
        project_params.save_pyp_parameters(parameters)
    return parameters
