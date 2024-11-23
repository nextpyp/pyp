
from __future__ import annotations

import sys
import toml

from typing import Optional, List, Dict, Any

from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path


relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def make_full_id(tab_id: str, arg_id: str) -> str:
    return f'{tab_id}_{arg_id}'


class ParamsArg:

    def __init__(self, config: ParamsConfig, tab_id: str, arg_id: str):
        self._config = config
        self.tab_id = tab_id
        self.arg_id = arg_id

    def _arg_config(self) -> Dict[str, Any]:
        return self._config._config['tabs'][self.tab_id][self.arg_id]

    def full_id(self) -> str:
        return make_full_id(self.tab_id, self.arg_id)

    def type(self) -> str:
        return self._arg_config()['type']

    def default_arg(self) -> Optional[ParamsArg]:

        try:
            d = self._arg_config()['default']
        except KeyError:
            return None

        # look for a ref default
        try:
            ref = d['ref']
        except (KeyError, TypeError):
            return None

        # found one: resolve the source
        return self._config.arg(make_full_id(self.tab_id, ref))

    def default_value(self) -> Optional[Any]:

        try:
            d = self._arg_config()['default']
        except KeyError:
            return None

        # look for a ref default
        try:
            _ = d['ref']
        except (KeyError, TypeError):
            # default is value, not a ref: return the value
            return self.check(d)

        # default is a ref, not a value: return nothing
        return None

    def check(self, value: Any) -> Any:

        arg_type = self.type()

        if arg_type == 'bool':
            if isinstance(value, bool):
                return value
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected bool')

        elif arg_type == 'int':
            if isinstance(value, int):
                return value
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected int')

        elif arg_type == 'float':
            if isinstance(value, float):
                return value
            elif isinstance(value, int):
                # "promote" to float
                return float(value)
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected float')

        elif arg_type == 'float2':
            if isinstance(value, list):
                # check the types for the x,y coords
                if isinstance(value[0], float):
                    x = value[0]
                elif isinstance(value[0], int):
                    x = float(value[0])
                else:
                    raise Exception(f'unexpected type {type(value[0])} for x coordinate in arg {self.full_id}, expected float')
                if isinstance(value[1], float):
                    y = value[1]
                elif isinstance(value[1], int):
                    y = float(value[1])
                else:
                    raise Exception(f'unexpected type {type(value[1])} for x coordinate in arg {self.full_id}, expected float')
                return [x, y]
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected [float]')

        elif arg_type == 'str':
            if isinstance(value, str):
                return value
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected str')

        elif arg_type == 'enum':
            if isinstance(value, str):
                # check that the value matches the enum values
                enum_values = self._arg_config()['enum'].keys()
                if value in enum_values:
                    return value
                else:
                    raise Exception(f'unexpected value {value} for arg {self.full_id}, expected one of {list(enum_values)}')
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected str')

        elif arg_type == 'path':
            if isinstance(value, str):
                return value
            else:
                raise Exception(f'unexpected type {type(value)} for arg {self.full_id}, expected str')

        else:
            raise Exception(f'unrecognized arg type: {arg_type}')


DEFAULT_CONFIG_PATH = "/opt/pyp/config/pyp_config.toml"


class ParamsConfig:

    @classmethod
    def from_str(cls, content: str) -> ParamsConfig:
        return cls(toml.loads(content))

    @classmethod
    def from_file(cls, path: str = DEFAULT_CONFIG_PATH) -> ParamsConfig:
        """
        Parameters
        ----------
        path:
            Path to the TOML config file

        Returns
        -------
        A class to make querying the parameters config easier
        """
        with open(path, 'r') as f:
            return cls.from_str(f.read())

    def __init__(self, config: Dict[str, Any]):

        self._config = config

        # make an index for the args by full_id
        self._full_ids = {}
        tabs = None
        try:
            tabs = self._config['tabs']
        except KeyError:
            logger.warn("Skipping indexing of params: No tabs defined in config")
        if tabs is not None:
            for tab_id, tab in [(k, v) for (k, v) in tabs.items() if not k.startswith('_')]:
                for arg_id, arg in [(k, v) for k, v in tab.items() if not k.startswith('_')]:
                    self._full_ids[f'{tab_id}_{arg_id}'] = (tab_id, arg_id)

    def args(self) -> List[ParamsArg]:
        return [ParamsArg(self, tab_id, arg_id) for tab_id, arg_id in self._full_ids.values()]

    def arg(self, full_id: str) -> Optional[ParamsArg]:
        try:
            tab_id, arg_id = self._full_ids[full_id]
        except KeyError:
            return None
        return ParamsArg(self, tab_id, arg_id)


def get_params_file_path(args: Optional[List[str]] = None) -> Optional[str]:
    """
    Parameters
    ----------
    args:
        The CLI args to process, defaults to `sys.argv`

    Returns
    -------
    If there's exactly one CLI argument of the form `-params_file=<path>` or `--params_file=<path>`,
    then returns <path>. Otherwise, returns None.
    """

    if args is None:
        args = sys.argv

    arg = args[-1]

    arg_name = 'params_file'
    if arg.startswith(f'-{arg_name}='):
        return arg[len(arg_name) + 2:]
    elif arg.startswith(f'--{arg_name}='):
        return arg[len(arg_name) + 3:]
    else:
        return None


def parse_params_from_file(config: ParamsConfig, path: str) -> Dict[str, Any]:
    """
    Parameters
    ----------
    config:
        the parameters configuration
    path:
        the path to the parameter file

    Returns
    -------
    The parameters read from the file, with any missing values filled in by default values
    """
    with open(path, 'r') as f:
        return parse_params_from_str(config, f.read())


def parse_params_from_str(config: ParamsConfig, content: str) -> Dict[str, Any]:
    """
    Parameters
    ----------
    config:
        the parameters configuration
    content:
        The content of the TOML file containing the parameters

    Returns
    -------
    The parameters read from the content string, with any missing values filled in by default values
    """

    # parse the parameters file
    params_toml = toml.loads(content)

    # convert the params from TOML format to a dict of the expected value types
    params = {}
    for full_id, value in params_toml.items():

        # lookup the param config, if any
        arg = config.arg(full_id)
        if arg is not None:

            # have a configured param, check the type
            params[full_id] = arg.check(value)

        else:

            # no configured param, just pass the value through and hope the existing type makes sense
            params[full_id] = value

    # insert default values for missing params
    for arg in config.args():

        # if arg has a value already, skip it
        if arg.full_id() in params:
            continue

        while True:

            # look for a default value
            d = arg.default_value()
            if d is not None:
                params[arg.full_id()] = d
                break

            # look for a default reference
            default_arg = arg.default_arg()
            if default_arg is not None:
                if default_arg.full_id() in params:
                    # source arg has value: use that
                    params[arg.full_id()] = params[default_arg.full_id()]
                    break
                else:
                    # source arg has no value: recurse
                    arg = default_arg
                    continue

            # no default: use a None value to avoid KeyErrors
            params[arg.full_id()] = None
            break

    return params
