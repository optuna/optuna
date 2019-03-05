from mock import patch
import os
import pytest
import shutil
import tempfile

import optuna
from optuna import types

if types.TYPE_CHECKING:
    from typing import Optional  # NOQA

_dummy_home = None  # type: Optional[str]


def setup_module():
    # type: () -> None

    global _dummy_home
    _dummy_home = tempfile.mkdtemp()


def teardown_module():
    # type: () -> None

    if _dummy_home:
        shutil.rmtree(_dummy_home)


def test_load_optuna_config():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('default_storage: some_storage\n')

        config = optuna.config.load_optuna_config(tf.name)
        assert config.default_storage == 'some_storage'


def test_load_optuna_config_default_config_path():
    # type: () -> None

    assert _dummy_home is not None

    config_path = os.path.join(_dummy_home, '.optuna.yml')
    with patch.object(optuna.config, 'DEFAULT_CONFIG_PATH', config_path):
        with open(config_path, 'w') as fw:
            fw.write('default_storage: some_storage\n')

        config = optuna.config.load_optuna_config()
        assert config.default_storage == 'some_storage'


def test_load_optuna_config_base_values():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('dummy_key: dummy_value\n')

        with pytest.raises(ValueError):
            optuna.config.load_optuna_config(tf.name)


def test_load_optuna_config_empty_file():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('')

        config = optuna.config.load_optuna_config(tf.name)
        assert config == optuna.config.BASE_OPTUNA_CONFIG


def test_load_optuna_config_not_found():
    # type: () -> None

    assert _dummy_home is not None

    config_path = os.path.join(_dummy_home, 'dummy.yml')
    with pytest.raises(IOError):
        optuna.config.load_optuna_config(config_path)


def test_load_optuna_config_default_config_not_found():
    # type: () -> None

    assert _dummy_home is not None

    config_path = os.path.join(_dummy_home, 'dummy.yml')
    with patch.object(optuna.config, 'DEFAULT_CONFIG_PATH', config_path):
        config = optuna.config.load_optuna_config()
        assert config.default_storage is None


def test_load_optuna_config_non_dict():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, 'w') as fw:
            fw.write('some_str')

        with pytest.raises(ValueError):
            optuna.config.load_optuna_config(tf.name)
