import os
import py  # NOQA
import pytest
import re
import shutil
import subprocess
from subprocess import CalledProcessError
import tempfile

import optuna
from optuna.cli import Studies
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages import RDBStorage
from optuna.structs import CLIUsageError
from optuna.trial import Trial  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from types import TracebackType  # NOQA
    from typing import Any  # NOQA
    from typing import IO  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA
    from typing import Type  # NOQA

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
        command = ['optuna', 'create-study']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)
        subprocess.check_call(command)

        # Command output should be in name string format (no-name + UUID).
        study_name = str(subprocess.check_output(command).decode().strip())
        name_re = r'^no-name-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        assert re.match(name_re, study_name) is not None

        # study_name should be stored in storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert study_id == 2


def test_create_study_command_with_study_name():
    # type: () -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)
        study_name = 'test_study'

        # Create study with name.
        command = ['optuna', 'create-study', '--storage', storage_url, '--study-name', study_name]
        study_name = str(subprocess.check_output(command).decode().strip())

        # Check if study_name is stored in the storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_name_from_id(study_id) == study_name


def test_create_study_command_without_storage_url():
    # type: () -> None

    dummy_home = tempfile.mkdtemp()

    env = os.environ
    env['HOME'] = dummy_home
    with pytest.raises(subprocess.CalledProcessError) as err:
        subprocess.check_output(['optuna', 'create-study'], env=env)
    usage = err.value.output.decode()
    assert usage.startswith('usage:')

    shutil.rmtree(dummy_home)


def test_create_study_command_with_direction():
    # type: () -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        command = ['optuna', 'create-study', '--storage', storage_url, '--direction', 'minimize']
        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_direction(study_id) == optuna.structs.StudyDirection.MINIMIZE

        command = ['optuna', 'create-study', '--storage', storage_url, '--direction', 'maximize']
        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_direction(study_id) == optuna.structs.StudyDirection.MAXIMIZE

        command = ['optuna', 'create-study', '--storage', storage_url, '--direction', 'test']

        # --direction should be either 'minimize' or 'maximize'.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(command)


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_study_set_user_attr_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        # Create study.
        study_name = storage.get_study_name_from_id(storage.create_new_study_id())

        base_command = ['optuna', 'study', 'set-user-attr', '--study', study_name]
        base_command = _add_option(base_command, '--storage', storage_url, 'storage' in options)
        base_command = _add_option(base_command, '--config', config_path, 'config' in options)

        example_attrs = {'architecture': 'ResNet', 'baselen_score': '0.002'}
        for key, value in example_attrs.items():
            subprocess.check_call(base_command + ['--key', key, '--value', value])

        # Attrs should be stored in storage.
        study_id = storage.get_study_id_from_name(study_name)
        study_user_attrs = storage.get_study_user_attrs(study_id)
        assert len(study_user_attrs) == 2
        assert all([study_user_attrs[k] == v for k, v in example_attrs.items()])


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_studies_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        # First study.
        study_1 = optuna.create_study(storage)

        # Second study.
        study_2 = optuna.create_study(storage, study_name='study_2')
        study_2.optimize(objective_func, n_trials=10)

        # Run command.
        command = ['optuna', 'studies']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)

        output = str(subprocess.check_output(command).decode().strip())
        rows = output.split('\n')

        def get_row_elements(row_index):
            # type: (int) -> List[str]

            return [r.strip() for r in rows[row_index].split('|')[1:-1]]

        assert len(rows) == 6
        assert tuple(get_row_elements(1)) == Studies._study_list_header

        # Check study_name and n_trials for the first study.
        elms = get_row_elements(3)
        assert elms[0] == study_1.study_name
        assert elms[2] == '0'

        # Check study_name and n_trials for the second study.
        elms = get_row_elements(4)
        assert elms[0] == study_2.study_name
        assert elms[2] == '10'


def test_create_study_command_with_skip_if_exists():
    # type: () -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)
        study_name = 'test_study'

        # Create study with name.
        command = ['optuna', 'create-study', '--storage', storage_url, '--study-name', study_name]
        study_name = str(subprocess.check_output(command).decode().strip())

        # Check if study_name is stored in the storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_name_from_id(study_id) == study_name

        # Try to create the same name study without `--skip-if-exists` flag (error).
        command = ['optuna', 'create-study', '--storage', storage_url, '--study-name', study_name]
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_output(command)

        # Try to create the same name study with `--skip-if-exists` flag (OK).
        command = [
            'optuna', 'create-study', '--storage', storage_url, '--study-name', study_name,
            '--skip-if-exists'
        ]
        study_name = str(subprocess.check_output(command).decode().strip())
        new_study_id = storage.get_study_id_from_name(study_name)
        assert study_id == new_study_id  # The existing study instance is reused.


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_dashboard_command(options):
    # type: (List[str]) -> None

    with \
            StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path), \
            tempfile.NamedTemporaryFile('r') as tf_report:

        storage = RDBStorage(storage_url)
        study_name = storage.get_study_name_from_id(storage.create_new_study_id())

        command = ['optuna', 'dashboard', '--study', study_name, '--out', tf_report.name]
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)
        subprocess.check_call(command)

        html = tf_report.read()
        assert '<body>' in html
        assert 'bokeh' in html


@pytest.mark.parametrize('origins',
                         [['192.168.111.1:5006'], ['192.168.111.1:5006', '192.168.111.2:5006']])
def test_dashboard_command_with_allow_websocket_origin(origins):
    # type: (List[str]) -> None

    with \
            StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path), \
            tempfile.NamedTemporaryFile('r') as tf_report:

        storage = RDBStorage(storage_url)
        study_name = storage.get_study_name_from_id(storage.create_new_study_id())
        command = [
            'optuna', 'dashboard', '--study', study_name, '--out', tf_report.name, '--storage',
            storage_url
        ]
        for origin in origins:
            command.extend(['--allow-websocket-origin', origin])
        subprocess.check_call(command)

        html = tf_report.read()
        assert '<body>' in html
        assert 'bokeh' in html


# An example of objective functions for testing study optimize command
def objective_func(trial):
    # type: (Trial) -> float

    x = trial.suggest_uniform('x', -10, 10)
    return (x + 5)**2


@pytest.mark.parametrize('options', [['storage'], ['config'], ['storage', 'config']])
def test_study_optimize_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        storage = RDBStorage(storage_url)

        study_name = storage.get_study_name_from_id(storage.create_new_study_id())
        command = [
            'optuna', 'study', 'optimize', '--study', study_name, '--n-trials', '10', __file__,
            'objective_func'
        ]
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)
        subprocess.check_call(command)

        study = optuna.load_study(storage=storage_url, study_name=study_name)
        assert len(study.trials) == 10
        assert 'x' in study.best_params

        # Check if a default value of study_name is stored in the storage.
        assert storage.get_study_name_from_id(study.study_id).startswith(DEFAULT_STUDY_NAME_PREFIX)


def test_study_optimize_command_inconsistent_args():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)

        # --study argument is missing.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call([
                'optuna', 'study', 'optimize', '--storage', db_url, '--n-trials', '10', __file__,
                'objective_func'
            ])


def test_empty_argv():
    # type: () -> None

    command_empty = ['optuna']
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ['optuna', 'help']
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output


def test_get_storage_url(tmpdir):
    # type: (py.path.local) -> None

    storage_in_args = 'sqlite:///args.db'
    storage_in_config = 'sqlite:///config.db'
    sample_config_file = tmpdir.join('optuna.yml')
    sample_config_file.write('default_storage: {}'.format(storage_in_config))

    sample_config = optuna.config.load_optuna_config(str(sample_config_file))
    default_config = optuna.config.load_optuna_config(None)

    # storage_url has priority over config_path.
    assert storage_in_args == optuna.cli.get_storage_url(storage_in_args, sample_config)
    assert storage_in_args == optuna.cli.get_storage_url(storage_in_args, default_config)
    assert storage_in_config == optuna.cli.get_storage_url(None, sample_config)

    # Config file does not have default_storage key.
    empty_config_file = tmpdir.join('empty.yml')
    empty_config_file.write('')
    empty_config = optuna.config.load_optuna_config(str(empty_config_file))
    with pytest.raises(CLIUsageError):
        optuna.cli.get_storage_url(None, empty_config)


@pytest.mark.parametrize('options', [[], ['storage'], ['config'], ['storage', 'config']])
def test_storage_upgrade_command(options):
    # type: (List[str]) -> None

    with StorageConfigSupplier(TEST_CONFIG_TEMPLATE) as (storage_url, config_path):
        command = ['optuna', 'storage', 'upgrade']
        command = _add_option(command, '--storage', storage_url, 'storage' in options)
        command = _add_option(command, '--config', config_path, 'config' in options)

        if len(options) == 0:
            with pytest.raises(CalledProcessError):
                subprocess.check_call(command)
        else:
            subprocess.check_call(command)
