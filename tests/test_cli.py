import pytest
import re
import subprocess
import tempfile
from types import TracebackType  # NOQA
from typing import Any  # NOQA
from typing import IO  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA

import pfnopt
from pfnopt.cli import Studies
from pfnopt.storages import RDBStorage
from pfnopt.trial import Trial  # NOQA


TEST_CONFIG_TEMPLATE = 'default_storage: sqlite:///{default_storage}\n'


class StorageConfigSupplier(object):

    def __init__(self, config_template):
        # type: (str) -> None

        self.tempfile_storage = None  # type: Optional[IO[Any]]
        self.tempfile_config = None  # type: Optional[IO[Any]]
        self.config_template = config_template

    def __enter__(self):
        # type: () -> Tuple[str, str]

        self.tempfile_storage = tempfile.NamedTemporaryFile()
        self.tempfile_config = tempfile.NamedTemporaryFile()

        with open(self.tempfile_config.name, 'w') as fw:
            fw.write(self.config_template.format(default_storage=self.tempfile_storage.name))

        return 'sqlite:///{}'.format(self.tempfile_storage.name), self.tempfile_config.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (Type[BaseException], BaseException, TracebackType) -> None

        if self.tempfile_storage:
            self.tempfile_storage.close()
        if self.tempfile_config:
            self.tempfile_config.close()


def _add_option(base_command, key, value, condition):
    # type: (List[str], str, str, bool) -> List[str]

    if condition:
        return base_command + [key, value]
    else:
        return base_command


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_create_study_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        # Create study.
        command = ['pfnopt', 'create-study']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)
        subprocess.check_call(command)

        # Command output should be in uuid string format.
        study_uuid = str(subprocess.check_output(command).decode().strip())
        uuid_re = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        assert re.match(uuid_re, study_uuid) is not None

        # study_uuid should be stored in storage.
        study_id = storage.get_study_id_from_uuid(study_uuid)
        assert study_id == 2


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_study_set_user_attr_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        # Create study.
        study_uuid = storage.get_study_uuid_from_id(storage.create_new_study_id())

        base_command = ['pfnopt', 'study', 'set-user-attr', '--study', study_uuid]
        base_command = _add_option(base_command, '--storage', storage_url, 'storage' in options)
        base_command = _add_option(base_command, '--config', config_path, 'config' in options)

        example_attrs = {'architecture': 'ResNet', 'baselen_score': '0.002'}
        for key, value in example_attrs.items():
            subprocess.check_call(base_command + ['--key', key, '--value', value])

        # Attrs should be stored in storage.
        study_id = storage.get_study_id_from_uuid(study_uuid)
        study_user_attrs = storage.get_study_user_attrs(study_id)
        assert len(study_user_attrs) == 3  # Including the system attribute key.
        assert all([study_user_attrs[k] == v for k, v in example_attrs.items()])


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_studies_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        # First study.
        study_uuid_1 = storage.get_study_uuid_from_id(storage.create_new_study_id())

        # Second study.
        study_uuid_2 = storage.get_study_uuid_from_id(
            storage.create_new_study_id(study_name='study_2'))
        pfnopt.minimize(objective_func, n_trials=10, storage=storage, study=study_uuid_2)

        # Run command.
        command = ['pfnopt', 'studies']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)

        output = str(subprocess.check_output(command).decode().strip())
        rows = output.split('\n')

        def get_row_elements(row_index):
            # type: (int) -> List[str]

            return [r.strip() for r in rows[row_index].split('|')[1: -1]]

        assert len(rows) == 6
        assert tuple(get_row_elements(1)) == Studies._study_list_header

        # Check study_uuid and n_trials for the first study.
        elms = get_row_elements(3)
        assert elms[0] == study_uuid_1
        assert len(elms[1]) == 0
        assert elms[3] == '0'

        # Check study_uuid and n_trials for the second study.
        elms = get_row_elements(4)
        assert elms[0] == study_uuid_2
        assert elms[1] == 'study_2'
        assert elms[3] == '10'


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_dashboard_command(options):
    # type: (List[str]) -> None

    with \
            StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path), \
            tempfile.NamedTemporaryFile('r') as tf_report:

            storage = RDBStorage(storage_url)
            study_uuid = storage.get_study_uuid_from_id(storage.create_new_study_id())

            command = ['pfnopt', 'dashboard', '--study', study_uuid, '--out', tf_report.name]
            command = _add_option(command, '--storage', storage_url, 'storage' in options)
            command = _add_option(command, '--config', config_path, 'config' in options)
            subprocess.check_call(command)

            html = tf_report.read()
            assert '<body>' in html
            assert 'bokeh' in html


# An example of objective functions for testing minimize command
def objective_func(trial):
    # type: (Trial) -> float

    x = trial.suggest_uniform('x', -10, 10)
    return (x + 5) ** 2


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_minimize_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        command = ['pfnopt', 'minimize', '--n-trials', '10', '--create-study',
                   __file__, 'objective_func']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)
        subprocess.check_call(command)

        study_uuid = storage.get_study_uuid_from_id(storage.create_new_study_id())
        command = ['pfnopt', 'minimize', '--study', study_uuid, '--n-trials', '10',
                   __file__, 'objective_func']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)
        subprocess.check_call(command)

        study = pfnopt.Study(storage=storage_url, study_uuid=study_uuid)
        assert len(study.trials) == 10
        assert 'x' in study.best_params


def test_minimize_command_inconsistent_args():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)

        # Feeding neither --create-study nor --study
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['pfnopt', 'minimize', '--storage', db_url, '--n-trials', '10',
                                   __file__, 'objective_func'])

        # Feeding both --create-study and --study
        study_uuid = str(subprocess.check_output(
            ['pfnopt', 'create-study', '--storage', db_url]).decode().strip())
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['pfnopt', 'minimize', '--storage', db_url, '--n-trials', '10',
                                   __file__, 'objective_func',
                                   '--create-study', '--study', study_uuid])


def test_empty_argv():
    # type: () -> None

    command_empty = ['pfnopt']
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ['pfnopt', 'help']
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output
