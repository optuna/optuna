import os
from typing import NamedTuple
from typing import Optional
import yaml


PFNOptConfig = NamedTuple('_BasePFNOptConfig', [('default_storage', Optional[str])])

BASE_PFNOPT_CONFIG = PFNOptConfig(default_storage=None)
DEFAULT_CONFIG_PATH = os.path.expanduser('~/.pfnopt.yml')


def load_pfnopt_config(path=None):
    # type: (Optional[str]) -> PFNOptConfig

    config_path = path or DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        if path is not None:
            # Config file was specified, but not exists.
            raise IOError('Config file {} not found.'.format(config_path))
        else:
            return BASE_PFNOPT_CONFIG

    with open(config_path, 'r') as fw:
        config_str = fw.read()
    config = yaml.load(config_str)

    if config is None:
        return BASE_PFNOPT_CONFIG

    if not isinstance(config, dict):
        raise ValueError('Format error found in the config file.')

    for key in config.keys():
        if key not in PFNOptConfig._fields:
            raise ValueError('Unknown key found in the config file: {}'.format(key))

    return BASE_PFNOPT_CONFIG._replace(**config)
