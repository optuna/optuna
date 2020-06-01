import re
import subprocess
from subprocess import CalledProcessError
import tempfile

import pytest

import optuna
from optuna.cli import _Studies
from optuna.exceptions import CLIUsageError
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages import RDBStorage
from optuna.study import StudyDirection
from optuna.testing.storage import StorageSupplier
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

    from optuna.trial import Trial  # NOQA


def test_create_study_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        # Create study.
        command = ["optuna", "create-study", "--storage", storage_url]
        subprocess.check_call(command)

        # Command output should be in name string format (no-name + UUID).
        study_name = str(subprocess.check_output(command).decode().strip())
        name_re = r"^no-name-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
        assert re.match(name_re, study_name) is not None

        # study_name should be stored in storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert study_id == 2


def test_create_study_command_with_study_name():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
        study_name = "test_study"

        # Create study with name.
        command = ["optuna", "create-study", "--storage", storage_url, "--study-name", study_name]
        study_name = str(subprocess.check_output(command).decode().strip())

        # Check if study_name is stored in the storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_name_from_id(study_id) == study_name


def test_create_study_command_without_storage_url():
    # type: () -> None

    with pytest.raises(subprocess.CalledProcessError) as err:
        subprocess.check_output(["optuna", "create-study"])
    usage = err.value.output.decode()
    assert usage.startswith("usage:")


def test_create_study_command_with_direction():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        command = ["optuna", "create-study", "--storage", storage_url, "--direction", "minimize"]
        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_direction(study_id) == StudyDirection.MINIMIZE

        command = ["optuna", "create-study", "--storage", storage_url, "--direction", "maximize"]
        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_direction(study_id) == StudyDirection.MAXIMIZE

        command = ["optuna", "create-study", "--storage", storage_url, "--direction", "test"]

        # --direction should be either 'minimize' or 'maximize'.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(command)


def test_delete_study_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
        study_name = "delete-study-test"

        # Create study.
        command = ["optuna", "create-study", "--storage", storage_url, "--study-name", study_name]
        subprocess.check_call(command)
        assert study_name in {s.study_name: s for s in storage.get_all_study_summaries()}

        # Delete study.
        command = ["optuna", "delete-study", "--storage", storage_url, "--study-name", study_name]
        subprocess.check_call(command)
        assert study_name not in {s.study_name: s for s in storage.get_all_study_summaries()}


def test_delete_study_command_without_storage_url():
    # type: () -> None

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(["optuna", "delete-study", "--study-name", "dummy_study"])


def test_study_set_user_attr_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        # Create study.
        study_name = storage.get_study_name_from_id(storage.create_new_study())

        base_command = [
            "optuna",
            "study",
            "set-user-attr",
            "--study-name",
            study_name,
            "--storage",
            storage_url,
        ]

        example_attrs = {"architecture": "ResNet", "baselen_score": "0.002"}
        for key, value in example_attrs.items():
            subprocess.check_call(base_command + ["--key", key, "--value", value])

        # Attrs should be stored in storage.
        study_id = storage.get_study_id_from_name(study_name)
        study_user_attrs = storage.get_study_user_attrs(study_id)
        assert len(study_user_attrs) == 2
        assert all([study_user_attrs[k] == v for k, v in example_attrs.items()])


def test_studies_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        # First study.
        study_1 = optuna.create_study(storage)

        # Second study.
        study_2 = optuna.create_study(storage, study_name="study_2")
        study_2.optimize(objective_func, n_trials=10)

        # Run command.
        command = ["optuna", "studies", "--storage", storage_url]

        output = str(subprocess.check_output(command).decode().strip())
        rows = output.split("\n")

        def get_row_elements(row_index):
            # type: (int) -> List[str]

            return [r.strip() for r in rows[row_index].split("|")[1:-1]]

        assert len(rows) == 6
        assert tuple(get_row_elements(1)) == _Studies._study_list_header

        # Check study_name and n_trials for the first study.
        elms = get_row_elements(3)
        assert elms[0] == study_1.study_name
        assert elms[2] == "0"

        # Check study_name and n_trials for the second study.
        elms = get_row_elements(4)
        assert elms[0] == study_2.study_name
        assert elms[2] == "10"


def test_create_study_command_with_skip_if_exists():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
        study_name = "test_study"

        # Create study with name.
        command = ["optuna", "create-study", "--storage", storage_url, "--study-name", study_name]
        study_name = str(subprocess.check_output(command).decode().strip())

        # Check if study_name is stored in the storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_name_from_id(study_id) == study_name

        # Try to create the same name study without `--skip-if-exists` flag (error).
        command = ["optuna", "create-study", "--storage", storage_url, "--study-name", study_name]
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_output(command)

        # Try to create the same name study with `--skip-if-exists` flag (OK).
        command = [
            "optuna",
            "create-study",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--skip-if-exists",
        ]
        study_name = str(subprocess.check_output(command).decode().strip())
        new_study_id = storage.get_study_id_from_name(study_name)
        assert study_id == new_study_id  # The existing study instance is reused.


def test_dashboard_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage, tempfile.NamedTemporaryFile("r") as tf_report:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        study_name = storage.get_study_name_from_id(storage.create_new_study())

        command = [
            "optuna",
            "dashboard",
            "--study-name",
            study_name,
            "--out",
            tf_report.name,
            "--storage",
            storage_url,
        ]
        subprocess.check_call(command)

        html = tf_report.read()
        assert "<body>" in html
        assert "bokeh" in html


@pytest.mark.parametrize(
    "origins", [["192.168.111.1:5006"], ["192.168.111.1:5006", "192.168.111.2:5006"]]
)
def test_dashboard_command_with_allow_websocket_origin(origins):
    # type: (List[str]) -> None

    with StorageSupplier("sqlite") as storage, tempfile.NamedTemporaryFile("r") as tf_report:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        study_name = storage.get_study_name_from_id(storage.create_new_study())
        command = [
            "optuna",
            "dashboard",
            "--study-name",
            study_name,
            "--out",
            tf_report.name,
            "--storage",
            storage_url,
        ]
        for origin in origins:
            command.extend(["--allow-websocket-origin", origin])
        subprocess.check_call(command)

        html = tf_report.read()
        assert "<body>" in html
        assert "bokeh" in html


# An example of objective functions for testing study optimize command
def objective_func(trial):
    # type: (Trial) -> float

    x = trial.suggest_uniform("x", -10, 10)
    return (x + 5) ** 2


def test_study_optimize_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        study_name = storage.get_study_name_from_id(storage.create_new_study())
        command = [
            "optuna",
            "study",
            "optimize",
            "--study-name",
            study_name,
            "--n-trials",
            "10",
            __file__,
            "objective_func",
            "--storage",
            storage_url,
        ]
        subprocess.check_call(command)

        study = optuna.load_study(storage=storage_url, study_name=study_name)
        assert len(study.trials) == 10
        assert "x" in study.best_params

        # Check if a default value of study_name is stored in the storage.
        assert storage.get_study_name_from_id(study._study_id).startswith(
            DEFAULT_STUDY_NAME_PREFIX
        )


def test_study_optimize_command_inconsistent_args():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = "sqlite:///{}".format(tf.name)

        # --study-name argument is missing.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(
                [
                    "optuna",
                    "study",
                    "optimize",
                    "--storage",
                    db_url,
                    "--n-trials",
                    "10",
                    __file__,
                    "objective_func",
                ]
            )


def test_empty_argv():
    # type: () -> None

    command_empty = ["optuna"]
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ["optuna", "help"]
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output


def test_check_storage_url():
    # type: () -> None

    storage_in_args = "sqlite:///args.db"
    assert storage_in_args == optuna.cli._check_storage_url(storage_in_args)

    with pytest.raises(CLIUsageError):
        optuna.cli._check_storage_url(None)


def test_storage_upgrade_command():
    # type: () -> None

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        command = ["optuna", "storage", "upgrade"]
        with pytest.raises(CalledProcessError):
            subprocess.check_call(command)

        command.extend(["--storage", storage_url])
        subprocess.check_call(command)
