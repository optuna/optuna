import re
import subprocess
import tempfile

from pfnopt.storages import RDBStorage


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
            'pfnopt', 'study', 'set-user-attr', '--url', db_url, '--study_uuid', study_uuid]
        for key, value in example_attrs.items():
            subprocess.check_call(base_command + ['--key', key, '--value', value])

        # attrs should be stored in storage
        storage = RDBStorage(db_url)
        study_id = storage.get_study_id_from_uuid(study_uuid)
        assert storage.get_study_user_attrs(study_id) == example_attrs


def test_dashboard_command():
    # type: () -> None

    with \
            tempfile.NamedTemporaryFile() as tf_db, \
            tempfile.NamedTemporaryFile('r') as tf_report:

            db_url = 'sqlite:///{}'.format(tf_db.name)
            command_mkstudy = ['pfnopt', 'create-study', '--url', db_url]
            study_uuid = subprocess.check_output(command_mkstudy).strip()

            command_report = [
                'pfnopt', 'dashboard', '--url', db_url, '--study_uuid', study_uuid,
                '--out', tf_report.name]
            subprocess.check_call(command_report)

            html = tf_report.read()
            assert '<body>' in html
            assert 'bokeh' in html


def test_empty_argv():
    # type: () -> None

    command_empty = ['pfnopt']
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ['pfnopt', 'help']
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output
