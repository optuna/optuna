import re
import subprocess
import tempfile
import unittest

from pfnopt.storage import RDBStorage


class TestMakeStudy(unittest.TestCase):

    def test_command(self):
        # type: () -> None

        tf = tempfile.NamedTemporaryFile()
        db_url = 'sqlite:///{}'.format(tf.name)
        command = ['pfnopt', 'mkstudy', '--url', db_url]

        # command exit code should be 0
        assert subprocess.check_call(command) == 0

        # command output should be in uuid string format
        study_uuid = subprocess.check_output(command).decode().strip()
        uuid_re = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
        assert re.match(uuid_re, study_uuid) is not None

        # uuid should be stored in storage
        storage = RDBStorage(db_url)
        study_id = storage.get_study_id_from_uuid(study_uuid)
        assert study_id == 2

        tf.close()
