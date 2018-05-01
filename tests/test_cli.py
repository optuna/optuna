import pytest
import re
import subprocess
import tempfile

import pfnopt
from pfnopt.storages import RDBStorage
from pfnopt.trial import Trial  # NOQA


def test_create_study_command():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)
        command = ['pfnopt', 'create-study', '--url', db_url]

        subprocess.check_call(command)

        # command output should be in uuid string format
        study_uuid = str(subprocess.check_output(command).decode().strip())
        uuid_re = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        assert re.match(uuid_re, study_uuid) is not None

        # uuid should be stored in storage
        storage = RDBStorage(db_url)
        study_id = storage.get_study_id_from_uuid(study_uuid)
        assert study_id == 2


def test_study_set_user_attr():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)

        # make study
        command = ['pfnopt', 'create-study', '--url', db_url]
        study_uuid = str(subprocess.check_output(command).decode().strip())

        example_attrs = {'architecture': 'ResNet', 'baselen_score': '0.002'}
        base_command = [
            'pfnopt', 'study', 'set-user-attr', '--url', db_url, '--study-uuid', study_uuid]
        for key, value in example_attrs.items():
            subprocess.check_call(base_command + ['--key', key, '--value', value])

        # attrs should be stored in storage
        storage = RDBStorage(db_url)
        study_id = storage.get_study_id_from_uuid(study_uuid)
        study_user_attrs = storage.get_study_user_attrs(study_id)
        assert len(study_user_attrs) == 3  # Including the system attribute key.
        assert all([study_user_attrs[k] == v for k, v in example_attrs.items()])


def test_dashboard_command():
    # type: () -> None

    with \
            tempfile.NamedTemporaryFile() as tf_db, \
            tempfile.NamedTemporaryFile('r') as tf_report:

            db_url = 'sqlite:///{}'.format(tf_db.name)
            command_mkstudy = ['pfnopt', 'create-study', '--url', db_url]
            study_uuid = subprocess.check_output(command_mkstudy).strip()

            command_report = [
                'pfnopt', 'dashboard', '--url', db_url, '--study-uuid', study_uuid,
                '--out', tf_report.name]
            subprocess.check_call(command_report)

            html = tf_report.read()
            assert '<body>' in html
            assert 'bokeh' in html


# An example of objective functions for testing minimize command
def objective_func(trial):
    # type: (Trial) -> float

    x = trial.suggest_uniform('x', -10, 10)
    return (x + 5) ** 2


def test_minimize_command_in_memory():
    # type: () -> None

    subprocess.check_call(
        ['pfnopt', 'minimize', '--n-trials', '10', '--create-study', __file__, 'objective_func'])


def test_minimize_command_rdb():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)
        subprocess.check_call(['pfnopt', 'minimize', '--url', db_url, '--n-trials',
                               '10', '--create-study', __file__, 'objective_func'])

        study_uuid = str(subprocess.check_output(
            ['pfnopt', 'create-study', '--url', db_url]).decode().strip())
        subprocess.check_call(['pfnopt', 'minimize', '--url', db_url, '--study-uuid', study_uuid,
                               '--n-trials', '10', __file__, 'objective_func'])

        study = pfnopt.Study(storage=db_url, study_uuid=study_uuid)
        assert len(study.trials) == 10
        assert 'x' in study.best_params


def test_minimize_command_inconsistent_args():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)

        # Feeding neither --create-study nor --study-uuid
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['pfnopt', 'minimize', '--url', db_url, '--n-trials', '10',
                                   __file__, 'objective_func'])

        # Feeding both --create-study and --study-uuid
        study_uuid = str(subprocess.check_output(
            ['pfnopt', 'create-study', '--url', db_url]).decode().strip())
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(['pfnopt', 'minimize', '--url', db_url, '--n-trials', '10',
                                   __file__, 'objective_func',
                                   '--create-study', '--study-uuid', study_uuid])


def test_empty_argv():
    # type: () -> None

    command_empty = ['pfnopt']
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ['pfnopt', 'help']
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output
