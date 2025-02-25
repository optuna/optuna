from __future__ import annotations

from collections.abc import Callable
import copy
import json
import os
import platform
import re
import subprocess
from subprocess import CalledProcessError
import tempfile
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import fakeredis
import numpy as np
from pandas import Timedelta
from pandas import Timestamp
import pytest
import yaml

import optuna
import optuna.cli
from optuna.exceptions import CLIUsageError
from optuna.exceptions import ExperimentalWarning
from optuna.storages import JournalStorage
from optuna.storages import RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalRedisBackend
from optuna.study import StudyDirection
from optuna.testing.storages import StorageSupplier
from optuna.testing.tempfile_pool import NamedTemporaryFilePool
from optuna.trial import Trial
from optuna.trial import TrialState


output_formats = pytest.mark.parametrize("output_format", (None, "value", "table", "json", "yaml"))


# An example of objective functions
def objective_func(trial: Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x + 5) ** 2


# An example of objective functions for branched search spaces
def objective_func_branched_search_space(trial: Trial) -> float:
    c = trial.suggest_categorical("c", ("A", "B"))
    if c == "A":
        x = trial.suggest_float("x", -10, 10)
        return (x + 5) ** 2
    else:
        y = trial.suggest_float("y", -10, 10)
        return (y + 5) ** 2


# An example of objective functions for multi-objective optimization
def objective_func_multi_objective(trial: Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -10, 10)
    return (x + 5) ** 2, (x - 5) ** 2


def _parse_output(output: str, output_format: str) -> Any:
    """Parse CLI output.

    Args:
        output:
            The output of command.
        output_format:
            The format of output specified by command.

    Returns:
        For table format, a list of dict formatted rows.
        For JSON or YAML format, a list or a dict corresponding to ``output``.
    """
    if output_format == "value":
        return [values.split(" ") for values in output.split(os.linesep)]
    elif output_format == "table":
        rows = output.split(os.linesep)
        assert all(len(rows[0]) == len(row) for row in rows)
        # Check ruled lines.
        assert rows[0] == rows[2] == rows[-1]

        keys = [r.strip() for r in rows[1].split("|")[1:-1]]
        ret = []
        for record in rows[3:-1]:
            attrs = {}
            for key, attr in zip(keys, record.split("|")[1:-1]):
                attrs[key] = attr.strip()
            ret.append(attrs)
        return ret
    elif output_format == "json":
        return json.loads(output)
    elif output_format == "yaml":
        return yaml.safe_load(output)
    else:
        assert False


def _get_output(command: list[str], output_format: str) -> Any:
    output = str(subprocess.check_output(command).decode().strip())
    ret = _parse_output(output, output_format)

    # Since keys are not given in value format, it checks matching with the output in table format.
    if output_format == "value":
        # NOTE(nabenabe): We cannot use this function for `test_ask_XXX` because this part executes
        # the provided command, creating another trial for `ask` and making the output different.
        table_command = copy.copy(command)
        table_command += ["--format", "table"]
        table_output = str(subprocess.check_output(table_command).decode().strip())
        table_ret = _parse_output(table_output, "table")
        assert len(ret) == len(table_ret)
        for record1, record2 in zip(ret, table_ret):
            assert " ".join(record1).strip() == " ".join(record2.values()).strip()
        return table_ret

    return ret


@pytest.mark.skip_coverage
def test_create_study_command() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        # Create study.
        command = ["optuna", "create-study", "--storage", storage_url]
        subprocess.check_call(command)

        # Command output should be in name string format (no-name + UUID).
        study_name = str(subprocess.check_output(command).decode().strip())
        name_re = r"^no-name-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
        assert re.match(name_re, study_name) is not None

        # study_name should be stored in storage.
        storage.get_study_id_from_name(study_name)


@pytest.mark.skip_coverage
def test_create_study_command_with_study_name() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"

        # Create study with name.
        command = ["optuna", "create-study", "--storage", storage_url, "--study-name", study_name]
        study_name = str(subprocess.check_output(command).decode().strip())

        # Check if study_name is stored in the storage.
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_name_from_id(study_id) == study_name


@pytest.mark.skip_coverage
def test_create_study_command_without_storage_url() -> None:
    with pytest.raises(subprocess.CalledProcessError) as err:
        subprocess.check_output(
            ["optuna", "create-study"],
            env={k: v for k, v in os.environ.items() if k != "OPTUNA_STORAGE"},
        )
    usage = err.value.output.decode()
    assert usage.startswith("usage:")


@pytest.mark.skip_coverage
def test_create_study_command_with_storage_env() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        # Create study.
        command = ["optuna", "create-study"]
        env = {**os.environ, "OPTUNA_STORAGE": storage_url}
        subprocess.check_call(command, env=env)

        # Command output should be in name string format (no-name + UUID).
        study_name = str(subprocess.check_output(command, env=env).decode().strip())
        name_re = r"^no-name-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
        assert re.match(name_re, study_name) is not None

        # study_name should be stored in storage.
        storage.get_study_id_from_name(study_name)


@pytest.mark.skip_coverage
def test_create_study_command_with_direction() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        command = ["optuna", "create-study", "--storage", storage_url, "--direction", "minimize"]
        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_directions(study_id) == [StudyDirection.MINIMIZE]

        command = ["optuna", "create-study", "--storage", storage_url, "--direction", "maximize"]
        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        assert storage.get_study_directions(study_id) == [StudyDirection.MAXIMIZE]

        command = ["optuna", "create-study", "--storage", storage_url, "--direction", "test"]

        # --direction should be either 'minimize' or 'maximize'.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(command)


@pytest.mark.skip_coverage
def test_create_study_command_with_multiple_directions() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        command = [
            "optuna",
            "create-study",
            "--storage",
            storage_url,
            "--directions",
            "minimize",
            "maximize",
        ]

        study_name = str(subprocess.check_output(command).decode().strip())
        study_id = storage.get_study_id_from_name(study_name)
        expected_directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
        assert storage.get_study_directions(study_id) == expected_directions

        command = [
            "optuna",
            "create-study",
            "--storage",
            storage_url,
            "--directions",
            "minimize",
            "maximize",
            "test",
        ]

        # Each direction in --directions should be either `minimize` or `maximize`.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(command)

        command = [
            "optuna",
            "create-study",
            "--storage",
            storage_url,
            "--direction",
            "minimize",
            "--directions",
            "minimize",
            "maximize",
            "test",
        ]

        # It can't specify both --direction and --directions
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_call(command)


@pytest.mark.skip_coverage
def test_delete_study_command() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "delete-study-test"

        # Create study.
        command = ["optuna", "create-study", "--storage", storage_url, "--study-name", study_name]
        subprocess.check_call(command)
        assert study_name in {s.study_name: s for s in storage.get_all_studies()}

        # Delete study.
        command = ["optuna", "delete-study", "--storage", storage_url, "--study-name", study_name]
        subprocess.check_call(command)
        assert study_name not in {s.study_name: s for s in storage.get_all_studies()}


@pytest.mark.skip_coverage
def test_delete_study_command_without_storage_url() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(
            ["optuna", "delete-study", "--study-name", "dummy_study"],
            env={k: v for k, v in os.environ.items() if k != "OPTUNA_STORAGE"},
        )


@pytest.mark.skip_coverage
def test_study_set_user_attr_command() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        # Create study.
        study_name = storage.get_study_name_from_id(
            storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        )

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
        assert all(study_user_attrs[k] == v for k, v in example_attrs.items())


@pytest.mark.skip_coverage
@output_formats
def test_study_names_command(output_format: str | None) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        expected_study_names = ["study-names-test1", "study-names-test2"]
        expected_column_name = "name"

        # Create a study.
        command = [
            "optuna",
            "create-study",
            "--storage",
            storage_url,
            "--study-name",
            expected_study_names[0],
        ]
        subprocess.check_output(command)

        # Get study names.
        command = ["optuna", "study-names", "--storage", storage_url]
        if output_format is not None:
            command += ["--format", output_format]
        study_names = _get_output(command, output_format or "value")

        # Check user_attrs are not printed.
        assert len(study_names) == 1
        assert study_names[0]["name"] == expected_study_names[0]

        # Create another study.
        command = [
            "optuna",
            "create-study",
            "--storage",
            storage_url,
            "--study-name",
            expected_study_names[1],
        ]
        subprocess.check_output(command)

        # Get study names.
        command = ["optuna", "study-names", "--storage", storage_url]
        if output_format is not None:
            command += ["--format", output_format]
        study_names = _get_output(command, output_format or "value")

        assert len(study_names) == 2
        for i, study_name in enumerate(study_names):
            assert list(study_name.keys()) == [expected_column_name]
            assert study_name["name"] == expected_study_names[i]


@pytest.mark.skip_coverage
def test_study_names_command_without_storage_url() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(
            ["optuna", "study-names", "--study-name", "dummy_study"],
            env={k: v for k, v in os.environ.items() if k != "OPTUNA_STORAGE"},
        )


@pytest.mark.skip_coverage
@output_formats
def test_studies_command(output_format: str | None) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        # First study.
        study_1 = optuna.create_study(storage=storage)

        # Run command.
        command = ["optuna", "studies", "--storage", storage_url]
        if output_format is not None:
            command += ["--format", output_format]

        studies = _get_output(command, output_format or "table")

        expected_keys = ["name", "direction", "n_trials", "datetime_start"]

        # Check user_attrs are not printed.
        if output_format in (None, "table", "value"):
            assert list(studies[0].keys()) == expected_keys
        else:
            assert set(studies[0].keys()) == set(expected_keys)

        # Add a second study.
        study_2 = optuna.create_study(
            storage=storage, study_name="study_2", directions=["minimize", "maximize"]
        )
        study_2.optimize(objective_func_multi_objective, n_trials=10)
        study_2.set_user_attr("key_1", "value_1")
        study_2.set_user_attr("key_2", "value_2")

        # Run command again to include second study.
        studies = _get_output(command, output_format or "table")

        expected_keys = ["name", "direction", "n_trials", "datetime_start", "user_attrs"]

        assert len(studies) == 2
        for study in studies:
            if output_format in (None, "table", "value"):
                assert list(study.keys()) == expected_keys
            else:
                assert set(study.keys()) == set(expected_keys)

        # Check study_name, direction, n_trials and user_attrs for the first study.
        assert studies[0]["name"] == study_1.study_name
        if output_format in (None, "table", "value"):
            assert studies[0]["n_trials"] == "0"
            assert eval(studies[0]["direction"]) == ("MINIMIZE",)
            assert eval(studies[0]["user_attrs"]) == {}
        else:
            assert studies[0]["n_trials"] == 0
            assert studies[0]["direction"] == ["MINIMIZE"]
            assert studies[0]["user_attrs"] == {}

        # Check study_name, direction, n_trials and user_attrs for the second study.
        assert studies[1]["name"] == study_2.study_name
        if output_format in (None, "table", "value"):
            assert studies[1]["n_trials"] == "10"
            assert eval(studies[1]["direction"]) == ("MINIMIZE", "MAXIMIZE")
            assert eval(studies[1]["user_attrs"]) == {"key_1": "value_1", "key_2": "value_2"}
        else:
            assert studies[1]["n_trials"] == 10
            assert studies[1]["direction"] == ["MINIMIZE", "MAXIMIZE"]
            assert studies[1]["user_attrs"] == {"key_1": "value_1", "key_2": "value_2"}


@pytest.mark.skip_coverage
@output_formats
def test_studies_command_flatten(output_format: str | None) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        # First study.
        study_1 = optuna.create_study(storage=storage)

        # Run command.
        command = ["optuna", "studies", "--storage", storage_url, "--flatten"]
        if output_format is not None:
            command += ["--format", output_format]

        studies = _get_output(command, output_format or "table")

        expected_keys_1 = [
            "name",
            "direction_0",
            "n_trials",
            "datetime_start",
        ]

        # Check user_attrs are not printed.
        if output_format in (None, "table", "value"):
            assert list(studies[0].keys()) == expected_keys_1
        else:
            assert set(studies[0].keys()) == set(expected_keys_1)

        # Add a second study.
        study_2 = optuna.create_study(
            storage=storage, study_name="study_2", directions=["minimize", "maximize"]
        )
        study_2.optimize(objective_func_multi_objective, n_trials=10)
        study_2.set_user_attr("key_1", "value_1")
        study_2.set_user_attr("key_2", "value_2")

        # Run command again to include second study.
        studies = _get_output(command, output_format or "table")

        if output_format in (None, "table", "value"):
            expected_keys_1 = expected_keys_2 = [
                "name",
                "direction_0",
                "direction_1",
                "n_trials",
                "datetime_start",
                "user_attrs",
            ]
        else:
            expected_keys_1 = ["name", "direction_0", "n_trials", "datetime_start", "user_attrs"]
            expected_keys_2 = [
                "name",
                "direction_0",
                "direction_1",
                "n_trials",
                "datetime_start",
                "user_attrs",
            ]

        assert len(studies) == 2
        if output_format in (None, "table", "value"):
            assert list(studies[0].keys()) == expected_keys_1
            assert list(studies[1].keys()) == expected_keys_2
        else:
            assert set(studies[0].keys()) == set(expected_keys_1)
            assert set(studies[1].keys()) == set(expected_keys_2)

        # Check study_name, direction, n_trials and user_attrs for the first study.
        assert studies[0]["name"] == study_1.study_name
        if output_format in (None, "table", "value"):
            assert studies[0]["n_trials"] == "0"
            assert studies[0]["user_attrs"] == "{}"
        else:
            assert studies[0]["n_trials"] == 0
            assert studies[0]["user_attrs"] == {}
        assert studies[0]["direction_0"] == "MINIMIZE"

        # Check study_name, direction, n_trials and user_attrs for the second study.
        assert studies[1]["name"] == study_2.study_name
        if output_format in (None, "table", "value"):
            assert studies[1]["n_trials"] == "10"
            assert studies[1]["user_attrs"] == "{'key_1': 'value_1', 'key_2': 'value_2'}"
        else:
            assert studies[1]["n_trials"] == 10
            assert studies[1]["user_attrs"] == {"key_1": "value_1", "key_2": "value_2"}
        assert studies[1]["direction_0"] == "MINIMIZE"
        assert studies[1]["direction_1"] == "MAXIMIZE"


@pytest.mark.skip_coverage
@pytest.mark.parametrize("objective", (objective_func, objective_func_branched_search_space))
@output_formats
def test_trials_command(objective: Callable[[Trial], float], output_format: str | None) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(storage=storage, study_name=study_name)
        study.optimize(objective, n_trials=n_trials)
        attrs = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        # Run command.
        command = [
            "optuna",
            "trials",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
        ]

        if output_format is not None:
            command += ["--format", output_format]

        trials = _get_output(command, output_format or "table")

        assert len(trials) == n_trials

        df = study.trials_dataframe(attrs, multi_index=True)

        for i, trial in enumerate(trials):
            for key in df.columns:
                expected_value = df.loc[i][key]

                # The param may be NaN when the objective function has branched search space.
                if (
                    key[0] == "params"
                    and isinstance(expected_value, float)
                    and np.isnan(expected_value)
                ):
                    if output_format in (None, "table", "value"):
                        assert key[1] not in eval(trial["params"])
                    else:
                        assert key[1] not in trial["params"]
                    continue

                if key[1] == "":
                    value = trial[key[0]]
                else:
                    if output_format in (None, "table", "value"):
                        value = eval(trial[key[0]])[key[1]]
                    else:
                        value = trial[key[0]][key[1]]

                if isinstance(value, (int, float)):
                    if np.isnan(expected_value):
                        assert np.isnan(value)
                    else:
                        assert value == expected_value
                elif isinstance(expected_value, Timestamp):
                    assert value == expected_value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(expected_value, Timedelta):
                    assert value == str(expected_value.to_pytimedelta())
                else:
                    assert value == str(expected_value)


@pytest.mark.skip_coverage
@pytest.mark.parametrize("objective", (objective_func, objective_func_branched_search_space))
@output_formats
def test_trials_command_flatten(
    objective: Callable[[Trial], float], output_format: str | None
) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(storage=storage, study_name=study_name)
        study.optimize(objective, n_trials=n_trials)
        attrs = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        # Run command.
        command = [
            "optuna",
            "trials",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--flatten",
        ]

        if output_format is not None:
            command += ["--format", output_format]

        trials = _get_output(command, output_format or "table")

        assert len(trials) == n_trials

        df = study.trials_dataframe(attrs)

        for i, trial in enumerate(trials):
            assert set(trial.keys()) <= set(df.columns)
            for key in df.columns:
                expected_value = df.loc[i][key]

                # The param may be NaN when the objective function has branched search space.
                if (
                    key.startswith("params_")
                    and isinstance(expected_value, float)
                    and np.isnan(expected_value)
                ):
                    if output_format in (None, "table", "value"):
                        assert trial[key] == ""
                    else:
                        assert key not in trial
                    continue

                value = trial[key]

                if isinstance(value, (int, float)):
                    if np.isnan(expected_value):
                        assert np.isnan(value)
                    else:
                        assert value == expected_value
                elif isinstance(expected_value, Timestamp):
                    assert value == expected_value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(expected_value, Timedelta):
                    assert value == str(expected_value.to_pytimedelta())
                else:
                    assert value == str(expected_value)


@pytest.mark.skip_coverage
@pytest.mark.parametrize("objective", (objective_func, objective_func_branched_search_space))
@output_formats
def test_best_trial_command(
    objective: Callable[[Trial], float], output_format: str | None
) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(storage=storage, study_name=study_name)
        study.optimize(objective, n_trials=n_trials)
        attrs = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        # Run command.
        command = [
            "optuna",
            "best-trial",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
        ]

        if output_format is not None:
            command += ["--format", output_format]

        best_trial = _get_output(command, output_format or "table")

        if output_format in (None, "table", "value"):
            assert len(best_trial) == 1
            best_trial = best_trial[0]

        df = study.trials_dataframe(attrs, multi_index=True)

        for key in df.columns:
            expected_value = df.loc[study.best_trial.number][key]

            # The param may be NaN when the objective function has branched search space.
            if (
                key[0] == "params"
                and isinstance(expected_value, float)
                and np.isnan(expected_value)
            ):
                if output_format in (None, "table", "value"):
                    assert key[1] not in eval(best_trial["params"])
                else:
                    assert key[1] not in best_trial["params"]
                continue

            if key[1] == "":
                value = best_trial[key[0]]
            else:
                if output_format in (None, "table", "value"):
                    value = eval(best_trial[key[0]])[key[1]]
                else:
                    value = best_trial[key[0]][key[1]]

            if isinstance(value, (int, float)):
                if np.isnan(expected_value):
                    assert np.isnan(value)
                else:
                    assert value == expected_value
            elif isinstance(expected_value, Timestamp):
                assert value == expected_value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(expected_value, Timedelta):
                assert value == str(expected_value.to_pytimedelta())
            else:
                assert value == str(expected_value)


@pytest.mark.skip_coverage
@pytest.mark.parametrize("objective", (objective_func, objective_func_branched_search_space))
@output_formats
def test_best_trial_command_flatten(
    objective: Callable[[Trial], float], output_format: str | None
) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(storage=storage, study_name=study_name)
        study.optimize(objective, n_trials=n_trials)
        attrs = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        # Run command.
        command = [
            "optuna",
            "best-trial",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--flatten",
        ]

        if output_format is not None:
            command += ["--format", output_format]

        best_trial = _get_output(command, output_format or "table")

        if output_format in (None, "table", "value"):
            assert len(best_trial) == 1
            best_trial = best_trial[0]

        df = study.trials_dataframe(attrs)

        assert set(best_trial.keys()) <= set(df.columns)
        for key in df.columns:
            expected_value = df.loc[study.best_trial.number][key]

            # The param may be NaN when the objective function has branched search space.
            if (
                key.startswith("params_")
                and isinstance(expected_value, float)
                and np.isnan(expected_value)
            ):
                if output_format in (None, "table", "value"):
                    assert best_trial[key] == ""
                else:
                    assert key not in best_trial
                continue

            value = best_trial[key]
            if isinstance(value, (int, float)):
                if np.isnan(expected_value):
                    assert np.isnan(value)
                else:
                    assert value == expected_value
            elif isinstance(expected_value, Timestamp):
                assert value == expected_value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(expected_value, Timedelta):
                assert value == str(expected_value.to_pytimedelta())
            else:
                assert value == str(expected_value)


@pytest.mark.skip_coverage
@output_formats
def test_best_trials_command(output_format: str | None) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(
            storage=storage, study_name=study_name, directions=("minimize", "minimize")
        )
        study.optimize(objective_func_multi_objective, n_trials=n_trials)
        attrs = (
            "number",
            "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        # Run command.
        command = [
            "optuna",
            "best-trials",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
        ]

        if output_format is not None:
            command += ["--format", output_format]

        trials = _get_output(command, output_format or "table")
        best_trials = [trial.number for trial in study.best_trials]

        assert len(trials) == len(best_trials)

        df = study.trials_dataframe(attrs, multi_index=True)

        for trial in trials:
            if output_format in (None, "table", "value"):
                number = int(trial["number"])
            else:
                number = trial["number"]
            assert number in best_trials
            for key in df.columns:
                expected_value = df.loc[number][key]

                # The param may be NaN when the objective function has branched search space.
                if (
                    key[0] == "params"
                    and isinstance(expected_value, float)
                    and np.isnan(expected_value)
                ):
                    if output_format in (None, "table", "value"):
                        assert key[1] not in eval(trial["params"])
                    else:
                        assert key[1] not in trial["params"]
                    continue

                if key[1] == "":
                    value = trial[key[0]]
                else:
                    if output_format in (None, "table", "value"):
                        value = eval(trial[key[0]])[key[1]]
                    else:
                        value = trial[key[0]][key[1]]

                if isinstance(value, (int, float)):
                    if np.isnan(expected_value):
                        assert np.isnan(value)
                    else:
                        assert value == expected_value
                elif isinstance(expected_value, Timestamp):
                    assert value == expected_value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(expected_value, Timedelta):
                    assert value == str(expected_value.to_pytimedelta())
                else:
                    assert value == str(expected_value)


@pytest.mark.skip_coverage
@output_formats
def test_best_trials_command_flatten(output_format: str | None) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(
            storage=storage, study_name=study_name, directions=("minimize", "minimize")
        )
        study.optimize(objective_func_multi_objective, n_trials=n_trials)
        attrs = (
            "number",
            "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        # Run command.
        command = [
            "optuna",
            "best-trials",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--flatten",
        ]

        if output_format is not None:
            command += ["--format", output_format]

        trials = _get_output(command, output_format or "table")
        best_trials = [trial.number for trial in study.best_trials]

        assert len(trials) == len(best_trials)

        df = study.trials_dataframe(attrs)

        for trial in trials:
            assert set(trial.keys()) <= set(df.columns)
            if output_format in (None, "table", "value"):
                number = int(trial["number"])
            else:
                number = trial["number"]
            for key in df.columns:
                expected_value = df.loc[number][key]

                # The param may be NaN when the objective function has branched search space.
                if (
                    key.startswith("params_")
                    and isinstance(expected_value, float)
                    and np.isnan(expected_value)
                ):
                    if output_format in (None, "table", "value"):
                        assert trial[key] == ""
                    else:
                        assert key not in trial
                    continue

                value = trial[key]
                if isinstance(value, (int, float)):
                    if np.isnan(expected_value):
                        assert np.isnan(value)
                    else:
                        assert value == expected_value
                elif isinstance(expected_value, Timestamp):
                    assert value == expected_value.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(expected_value, Timedelta):
                    assert value == str(expected_value.to_pytimedelta())
                else:
                    assert value == str(expected_value)


@pytest.mark.skip_coverage
def test_create_study_command_with_skip_if_exists() -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name
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


@pytest.mark.skip_coverage
def test_empty_argv() -> None:
    command_empty = ["optuna"]
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ["optuna", "help"]
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output


def test_check_storage_url() -> None:
    storage_in_args = "sqlite:///args.db"
    assert storage_in_args == optuna.cli._check_storage_url(storage_in_args)

    with pytest.warns(ExperimentalWarning):
        with patch.dict("optuna.cli.os.environ", {"OPTUNA_STORAGE": "sqlite:///args.db"}):
            optuna.cli._check_storage_url(None)

    with pytest.raises(CLIUsageError):
        optuna.cli._check_storage_url(None)


@pytest.mark.skipif(platform.system() == "Windows", reason="Skip on Windows")
@patch("optuna.storages.journal._redis.redis")
def test_get_storage_without_storage_class(mock_redis: MagicMock) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db") as fp:
        storage = optuna.cli._get_storage(f"sqlite:///{fp.name}", storage_class=None)
        assert isinstance(storage, RDBStorage)

    with tempfile.NamedTemporaryFile(suffix=".log") as fp:
        storage = optuna.cli._get_storage(fp.name, storage_class=None)
        assert isinstance(storage, JournalStorage)
        assert isinstance(storage._backend, JournalFileBackend)

    mock_redis.Redis = fakeredis.FakeRedis
    storage = optuna.cli._get_storage("redis://localhost:6379", storage_class=None)
    assert isinstance(storage, JournalStorage)
    assert isinstance(storage._backend, JournalRedisBackend)

    with pytest.raises(CLIUsageError):
        optuna.cli._get_storage("./file-not-found.log", storage_class=None)


@pytest.mark.skipif(platform.system() == "Windows", reason="Skip on Windows")
@patch("optuna.storages.journal._redis.redis")
def test_get_storage_with_storage_class(mock_redis: MagicMock) -> None:
    with tempfile.NamedTemporaryFile(suffix=".db") as fp:
        storage = optuna.cli._get_storage(f"sqlite:///{fp.name}", storage_class=None)
        assert isinstance(storage, RDBStorage)

    with tempfile.NamedTemporaryFile(suffix=".log") as fp:
        storage = optuna.cli._get_storage(fp.name, storage_class="JournalFileBackend")
        assert isinstance(storage, JournalStorage)
        assert isinstance(storage._backend, JournalFileBackend)

    mock_redis.Redis = fakeredis.FakeRedis
    storage = optuna.cli._get_storage(
        "redis:///localhost:6379", storage_class="JournalRedisBackend"
    )
    assert isinstance(storage, JournalStorage)
    assert isinstance(storage._backend, JournalRedisBackend)

    with pytest.raises(CLIUsageError):
        with tempfile.NamedTemporaryFile(suffix=".db") as fp:
            optuna.cli._get_storage(f"sqlite:///{fp.name}", storage_class="InMemoryStorage")


@pytest.mark.skip_coverage
def test_storage_upgrade_command() -> None:
    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        command = ["optuna", "storage", "upgrade"]
        with pytest.raises(CalledProcessError):
            subprocess.check_call(
                command,
                env={k: v for k, v in os.environ.items() if k != "OPTUNA_STORAGE"},
            )

        command.extend(["--storage", storage_url])
        subprocess.check_call(command)


@pytest.mark.skip_coverage
def test_storage_upgrade_command_with_invalid_url() -> None:
    command = ["optuna", "storage", "upgrade", "--storage", "invalid-storage-url"]
    with pytest.raises(CalledProcessError):
        subprocess.check_call(command)


parametrize_for_ask = pytest.mark.parametrize(
    "sampler,sampler_kwargs,output_format",
    [
        (None, None, None),
        ("RandomSampler", None, None),
        ("TPESampler", '{"multivariate": true}', None),
        (None, None, "value"),
        (None, None, "table"),
        (None, None, "json"),
        (None, None, "yaml"),
    ],
)


@pytest.mark.skip_coverage
@parametrize_for_ask
def test_ask(
    sampler: str | None,
    sampler_kwargs: str | None,
    output_format: str | None,
) -> None:
    study_name = "test_study"
    search_space = (
        '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}}, '
        '"y": {"name": "CategoricalDistribution", "attributes": {"choices": ["foo"]}}}'
    )

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
            "--search-space",
            search_space,
        ]

        if sampler is not None:
            args += ["--sampler", sampler]
        if sampler_kwargs is not None:
            args += ["--sampler-kwargs", sampler_kwargs]
        if output_format is not None:
            args += ["--format", output_format]

        if output_format != "value":
            trial = _get_output(args, output_format or "json")
        else:
            output = str(subprocess.check_output(args).decode().strip())
            ret = output.split(maxsplit=1)
            assert len(ret) == 2
            trial = [{"number": ret[0], "params": ret[1]}]

        if output_format in ("table", "value"):
            assert len(trial) == 1
            trial = trial[0]
            assert trial["number"] == "0"
            params = eval(trial["params"])
            assert len(params) == 2
            assert 0 <= params["x"] <= 1
            assert params["y"] == "foo"
        else:
            assert trial["number"] == 0
            assert 0 <= trial["params"]["x"] <= 1
            assert trial["params"]["y"] == "foo"


@pytest.mark.skip_coverage
@parametrize_for_ask
def test_ask_flatten(
    sampler: str | None,
    sampler_kwargs: str | None,
    output_format: str | None,
) -> None:
    study_name = "test_study"
    search_space = (
        '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}}, '
        '"y": {"name": "CategoricalDistribution", "attributes": {"choices": ["foo"]}}}'
    )

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
            "--search-space",
            search_space,
            "--flatten",
        ]

        if sampler is not None:
            args += ["--sampler", sampler]
        if sampler_kwargs is not None:
            args += ["--sampler-kwargs", sampler_kwargs]
        if output_format is not None:
            args += ["--format", output_format]

        if output_format != "value":
            trial = _get_output(args, output_format or "json")
        else:
            output = str(subprocess.check_output(args).decode().strip())
            ret = output.split(maxsplit=2)
            assert len(ret) == 3
            trial = [{"number": ret[0], "params_x": ret[1], "params_y": ret[2]}]

        if output_format in ("table", "value"):
            assert len(trial) == 1
            trial = trial[0]
            assert trial["number"] == "0"
            assert 0 <= float(trial["params_x"]) <= 1
            assert trial["params_y"] == "foo"
        else:
            assert trial["number"] == 0
            assert 0 <= trial["params_x"] <= 1
            assert trial["params_y"] == "foo"


@pytest.mark.skip_coverage
@output_formats
def test_ask_empty_search_space(output_format: str) -> None:
    study_name = "test_study"

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
        ]

        if output_format is not None:
            args += ["--format", output_format]

        if output_format != "value":
            trial = _get_output(args, output_format or "json")
        else:
            output = str(subprocess.check_output(args).decode().strip())
            ret = output.split(maxsplit=1)
            assert len(ret) == 2
            trial = [{"number": ret[0], "params": ret[1]}]

        if output_format in ("table", "value"):
            assert len(trial) == 1
            trial = trial[0]
            assert trial["number"] == "0"
            assert trial["params"] == "{}"
        else:
            assert trial["number"] == 0
            assert trial["params"] == {}


@pytest.mark.skip_coverage
@output_formats
def test_ask_empty_search_space_flatten(output_format: str) -> None:
    study_name = "test_study"

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
            "--flatten",
        ]

        if output_format is not None:
            args += ["--format", output_format]

        if output_format != "value":
            trial = _get_output(args, output_format or "json")
        else:
            output = str(subprocess.check_output(args).decode().strip())
            trial = [{"number": output}]

        if output_format in ("table", "value"):
            assert len(trial) == 1
            trial = trial[0]
            assert trial["number"] == "0"
            assert "params" not in trial
        else:
            assert trial["number"] == 0
            assert "params" not in trial


@pytest.mark.skip_coverage
def test_ask_sampler_kwargs_without_sampler() -> None:
    study_name = "test_study"
    search_space = (
        '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}}, '
        '"y": {"name": "CategoricalDistribution", "attributes": {"choices": ["foo"]}}}'
    )

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
            "--search-space",
            search_space,
            "--sampler-kwargs",
            '{"multivariate": true}',
        ]

        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        error_message = result.stderr.decode()
        assert "`--sampler_kwargs` is set without `--sampler`." in error_message


@pytest.mark.skip_coverage
def test_ask_without_create_study_beforehand() -> None:
    study_name = "test_study"
    search_space = (
        '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}}, '
        '"y": {"name": "CategoricalDistribution", "attributes": {"choices": ["foo"]}}}'
    )

    with NamedTemporaryFilePool() as fp:
        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
            "--search-space",
            search_space,
        ]

        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        error_message = result.stderr.decode()
        assert (
            "Implicit study creation within the 'ask' command was dropped in Optuna v4.0.0."
            in error_message
        )


@pytest.mark.skip_coverage
@pytest.mark.parametrize(
    "direction,directions,sampler,sampler_kwargs",
    [
        (None, None, None, None),
        ("minimize", None, None, None),
        (None, "minimize maximize", None, None),
        (None, None, "RandomSampler", None),
        (None, None, "TPESampler", '{"multivariate": true}'),
    ],
)
def test_create_study_and_ask(
    direction: str | None,
    directions: str | None,
    sampler: str | None,
    sampler_kwargs: str | None,
) -> None:
    study_name = "test_study"
    search_space = (
        '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}}, '
        '"y": {"name": "CategoricalDistribution", "attributes": {"choices": ["foo"]}}}'
    )

    with NamedTemporaryFilePool() as fp:
        create_study_args = [
            "optuna",
            "create-study",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
        ]

        if direction is not None:
            create_study_args += ["--direction", direction]
        if directions is not None:
            create_study_args += ["--directions"] + directions.split()
        subprocess.check_call(create_study_args)

        args = [
            "optuna",
            "ask",
            "--storage",
            fp.name,
            "--study-name",
            study_name,
            "--search-space",
            search_space,
        ]

        if sampler is not None:
            args += ["--sampler", sampler]
        if sampler_kwargs is not None:
            args += ["--sampler-kwargs", sampler_kwargs]

        trial = _get_output(args, "json")

        assert trial["number"] == 0
        assert 0 <= trial["params"]["x"] <= 1
        assert trial["params"]["y"] == "foo"


@pytest.mark.skip_coverage
def test_tell() -> None:
    study_name = "test_study"

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output: Any = subprocess.check_output(
            [
                "optuna",
                "ask",
                "--storage",
                fp.name,
                "--study-name",
                study_name,
                "--format",
                "json",
            ]
        )
        output = output.decode("utf-8")
        output = json.loads(output)
        trial_number = output["number"]

        subprocess.check_output(
            [
                "optuna",
                "tell",
                "--storage",
                fp.name,
                "--trial-number",
                str(trial_number),
                "--values",
                "1.2",
            ]
        )

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(fp.name)
        )
        study = optuna.load_study(storage=storage, study_name=study_name)
        assert len(study.trials) == 1
        assert study.trials[0].state == TrialState.COMPLETE
        assert study.trials[0].values == [1.2]

        # Error when updating a finished trial.
        ret = subprocess.run(
            [
                "optuna",
                "tell",
                "--storage",
                fp.name,
                "--trial-number",
                str(trial_number),
                "--values",
                "1.2",
            ]
        )
        assert ret.returncode != 0

        # Passing `--skip-if-finished` to a finished trial for a noop.
        subprocess.check_output(
            [
                "optuna",
                "tell",
                "--storage",
                fp.name,
                "--trial-number",
                str(trial_number),
                "--values",
                "1.3",  # Setting a different value and make sure it's not persisted.
                "--skip-if-finished",
            ]
        )

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(fp.name)
        )
        study = optuna.load_study(storage=storage, study_name=study_name)
        assert len(study.trials) == 1
        assert study.trials[0].state == TrialState.COMPLETE
        assert study.trials[0].values == [1.2]


@pytest.mark.skip_coverage
def test_tell_with_nan() -> None:
    study_name = "test_study"

    with NamedTemporaryFilePool() as fp:
        args = ["optuna", "create-study", "--storage", fp.name, "--study-name", study_name]
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output: Any = subprocess.check_output(
            [
                "optuna",
                "ask",
                "--storage",
                fp.name,
                "--study-name",
                study_name,
                "--format",
                "json",
            ]
        )
        output = output.decode("utf-8")
        output = json.loads(output)
        trial_number = output["number"]

        subprocess.check_output(
            [
                "optuna",
                "tell",
                "--storage",
                fp.name,
                "--trial-number",
                str(trial_number),
                "--values",
                "nan",
            ]
        )

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(fp.name)
        )
        study = optuna.load_study(storage=storage, study_name=study_name)
        assert len(study.trials) == 1
        assert study.trials[0].state == TrialState.FAIL
        assert study.trials[0].values is None


@pytest.mark.skip_coverage
@pytest.mark.parametrize(
    "verbosity, expected",
    [
        ("--verbose", True),
        ("--quiet", False),
    ],
)
def test_configure_logging_verbosity(verbosity: str, expected: bool) -> None:
    with NamedTemporaryFilePool() as fp, StorageSupplier("journal", file=fp) as storage:
        assert isinstance(storage, JournalStorage)
        storage_url = fp.name

        # Create study.
        args = ["optuna", "create-study", "--storage", storage_url, verbosity]
        # `--verbose` makes the log level DEBUG.
        # `--quiet` makes the log level WARNING.
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        error_message = result.stderr.decode()
        assert ("A new study created in Journal with name" in error_message) == expected
