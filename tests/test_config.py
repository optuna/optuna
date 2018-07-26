from mock import patch
import os
import shutil
import tempfile
from typing import Optional  # NOQA

import pfnopt


_dummy_home = None  # type: Optional[str]


def setup_module():
    # type: () -> None

    global _dummy_home
    _dummy_home = tempfile.mkdtemp()


def teardown_module():
    # type: () -> None

    if _dummy_home:
        shutil.rmtree(_dummy_home)


def test_load_pfnopt_config():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('default_storage: some_storage\n')

        config = pfnopt.config.load_pfnopt_config(tf.name)
        assert config.default_storage == 'some_storage'


def test_load_pfnopt_config_default_config_path():
    # type: () -> None

    assert _dummy_home is not None

    with patch.dict(os.environ, {'HOME': _dummy_home}):
        config_path = os.path.join(_dummy_home, '.pfnopt.yml')
        with open(config_path, 'w') as fw:
            fw.write('default_storage: some_storage\n')

        config = pfnopt.config.load_pfnopt_config()
        assert config.default_storage == 'some_storage'


def test_load_pfnopt_config_base_values():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('dummy_key: dummy_value\n')

        config = pfnopt.config.load_pfnopt_config(tf.name)
        assert config == pfnopt.config.BASE_PFNOPT_CONFIG


def test_load_pfnopt_config_empty_file():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('')

        config = pfnopt.config.load_pfnopt_config(tf.name)
        assert config == pfnopt.config.BASE_PFNOPT_CONFIG


def test_load_pfnopt_config_non_dict():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('some_str')

        config = pfnopt.config.load_pfnopt_config(tf.name)
        assert config == pfnopt.config.BASE_PFNOPT_CONFIG
