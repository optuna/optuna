import os
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import NamedTuple
from typing import Optional
import yaml


PFNOptConfig = NamedTuple('_BasePFNOptConfig', [('default_storage', Optional[str])])

BASE_PFNOPT_CONFIG = PFNOptConfig(default_storage=None)


def load_pfnopt_config(path=None):
    # type: (str) -> PFNOptConfig

    path = path or os.path.expanduser('~/.pfnopt.yml')

    with open(path, 'r') as fw:
        stream = fw.read()
    config = yaml.load(stream)

    replace_dict = {}  # type: Dict[str, Any]
    if isinstance(config, dict):
        replace_dict = {k: v for k, v in config.items() if k in PFNOptConfig._fields}

    return BASE_PFNOPT_CONFIG._replace(**replace_dict)
