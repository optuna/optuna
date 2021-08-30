import json
import re
import subprocess
from subprocess import CalledProcessError
import tempfile
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from pandas import Timedelta
from pandas import Timestamp
import pytest
import yaml

import optuna
from optuna.cli import _Studies
from optuna.exceptions import CLIUsageError
from optuna.storages import RDBStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study import StudyDirection
from optuna.testing.storage import StorageSupplier
from optuna.trial import Trial
from optuna.trial import TrialState


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
def objective_func_multi_objective(trial: Trial) -> Tuple[float, float]:

    x = trial.suggest_float("x", -10, 10)
    return (x + 5) ** 2, (x - 5) ** 2


def test_create_study_command() -> None:

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


def test_create_study_command_with_study_name() -> None:

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


def test_create_study_command_without_storage_url() -> None:

    with pytest.raises(subprocess.CalledProcessError) as err:
        subprocess.check_output(["optuna", "create-study"])
    usage = err.value.output.decode()
    assert usage.startswith("usage:")


def test_create_study_command_with_direction() -> None:

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

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


def test_create_study_command_with_multiple_directions() -> None:

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
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


def test_delete_study_command() -> None:

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


def test_delete_study_command_without_storage_url() -> None:

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(["optuna", "delete-study", "--study-name", "dummy_study"])


def test_study_set_user_attr_command() -> None:

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
        assert all(study_user_attrs[k] == v for k, v in example_attrs.items())


def test_studies_command() -> None:

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

        def get_row_elements(row_index: int) -> List[str]:

            return [r.strip() for r in rows[row_index].split("|")[1:-1]]

        assert len(rows) == 6
        for i, element in enumerate(get_row_elements(1)):
            assert element == _Studies._study_list_header[i][0]

        # Check study_name and n_trials for the first study.
        elms = get_row_elements(3)
        assert elms[0] == study_1.study_name
        assert elms[2] == "0"

        # Check study_name and n_trials for the second study.
        elms = get_row_elements(4)
        assert elms[0] == study_2.study_name
        assert elms[2] == "10"


@pytest.mark.parametrize("objective", (objective_func, objective_func_branched_search_space))
def test_trials_command(objective: Callable[[Trial], float]) -> None:

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(storage, study_name=study_name)
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
        df = study.trials_dataframe(attrs)

        # Run command.
        command = [
            "optuna",
            "trials",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--format",
            "json",
            "--flatten",
        ]

        output = str(subprocess.check_output(command).decode().strip())
        trials = json.loads(output)

        assert len(trials) == n_trials

        for i, trial in enumerate(trials):
            assert set(trial.keys()) <= set(df.columns)
            for key in df.columns:
                expected_value = df.loc[i][key]
                if (
                    key.startswith("params_")
                    and isinstance(expected_value, float)
                    and np.isnan(expected_value)
                ):
                    assert key not in trial
                    continue
                value = trial[key]
                if isinstance(value, int) or isinstance(value, float):
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


@pytest.mark.parametrize("objective", (objective_func, objective_func_branched_search_space))
def test_best_trial_command(objective: Callable[[Trial], float]) -> None:

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(storage, study_name=study_name)
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
        df = study.trials_dataframe(attrs)

        # Run command.
        command = [
            "optuna",
            "best-trial",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--format",
            "json",
            "--flatten",
        ]

        output = str(subprocess.check_output(command).decode().strip())
        best_trial = json.loads(output)

        assert set(best_trial.keys()) <= set(df.columns)
        for key in df.columns:
            expected_value = df.loc[study.best_trial.number][key]
            if (
                key.startswith("params_")
                and isinstance(expected_value, float)
                and np.isnan(expected_value)
            ):
                assert key not in best_trial
                continue
            value = best_trial[key]
            if isinstance(value, int) or isinstance(value, float):
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


def test_best_trials_command() -> None:

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)
        study_name = "test_study"
        n_trials = 10

        study = optuna.create_study(
            storage, study_name=study_name, directions=("minimize", "minimize")
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
        df = study.trials_dataframe(attrs)

        # Run command.
        command = [
            "optuna",
            "best-trials",
            "--storage",
            storage_url,
            "--study-name",
            study_name,
            "--format",
            "json",
            "--flatten",
        ]

        output = str(subprocess.check_output(command).decode().strip())
        trials = json.loads(output)
        best_trials = [trial.number for trial in study.best_trials]

        assert len(trials) == len(best_trials)

        for trial in trials:
            assert set(trial.keys()) <= set(df.columns)
            assert trial["number"] in best_trials
            for key in df.columns:
                expected_value = df.loc[trial["number"]][key]
                if (
                    key.startswith("params_")
                    and isinstance(expected_value, float)
                    and np.isnan(expected_value)
                ):
                    assert key not in trial
                    continue
                value = trial[key]
                if isinstance(value, int) or isinstance(value, float):
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


def test_create_study_command_with_skip_if_exists() -> None:

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


def test_dashboard_command() -> None:

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
def test_dashboard_command_with_allow_websocket_origin(origins: List[str]) -> None:

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


def test_study_optimize_command() -> None:

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


def test_study_optimize_command_inconsistent_args() -> None:

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


def test_empty_argv() -> None:

    command_empty = ["optuna"]
    command_empty_output = str(subprocess.check_output(command_empty))

    command_help = ["optuna", "help"]
    command_help_output = str(subprocess.check_output(command_help))

    assert command_empty_output == command_help_output


def test_check_storage_url() -> None:

    storage_in_args = "sqlite:///args.db"
    assert storage_in_args == optuna.cli._check_storage_url(storage_in_args)

    with pytest.raises(CLIUsageError):
        optuna.cli._check_storage_url(None)


def test_storage_upgrade_command() -> None:

    with StorageSupplier("sqlite") as storage:
        assert isinstance(storage, RDBStorage)
        storage_url = str(storage.engine.url)

        command = ["optuna", "storage", "upgrade"]
        with pytest.raises(CalledProcessError):
            subprocess.check_call(command)

        command.extend(["--storage", storage_url])
        subprocess.check_call(command)


@pytest.mark.parametrize(
    "direction,directions,sampler,sampler_kwargs,output_format",
    [
        (None, None, None, None, None),
        ("minimize", None, None, None, None),
        (None, "minimize maximize", None, None, None),
        (None, None, "RandomSampler", None, None),
        (None, None, "TPESampler", '{"multivariate": true}', None),
        (None, None, None, None, "json"),
        (None, None, None, None, "yaml"),
    ],
)
def test_ask(
    direction: Optional[str],
    directions: Optional[str],
    sampler: Optional[str],
    sampler_kwargs: Optional[str],
    output_format: Optional[str],
) -> None:

    study_name = "test_study"
    search_space = (
        '{"x": {"name": "UniformDistribution", "attributes": {"low": 0.0, "high": 1.0}}, '
        '"y": {"name": "CategoricalDistribution", "attributes": {"choices": ["foo"]}}}'
    )

    with tempfile.NamedTemporaryFile() as tf:
        db_url = "sqlite:///{}".format(tf.name)

        args = [
            "optuna",
            "ask",
            "--storage",
            db_url,
            "--study-name",
            study_name,
            "--search-space",
            search_space,
        ]

        if direction is not None:
            args += ["--direction", direction]
        if directions is not None:
            args += ["--directions"] + directions.split()
        if sampler is not None:
            args += ["--sampler", sampler]
        if sampler_kwargs is not None:
            args += ["--sampler-kwargs", sampler_kwargs]
        if output_format is None:
            args += ["--format", "json"]
        else:
            args += ["--format", output_format]

        output: Any = subprocess.check_output(args)
        output = output.decode("utf-8")

        if output_format is None or output_format == "json":
            output = json.loads(output)
        else:  # "yaml".
            output = yaml.load(output)

        assert output["number"] == 0
        assert len(output["params"]) == 2
        assert 0 <= output["params"]["x"] <= 1
        assert output["params"]["y"] == "foo"


def test_ask_empty_search_space() -> None:
    study_name = "test_study"

    with tempfile.NamedTemporaryFile() as tf:
        db_url = "sqlite:///{}".format(tf.name)

        args = [
            "optuna",
            "ask",
            "--storage",
            db_url,
            "--study-name",
            study_name,
            "--format",
            "json",
        ]

        output: Any = subprocess.check_output(args)
        output = output.decode("utf-8")
        output = json.loads(output)

        assert output["number"] == 0
        assert len(output["params"]) == 0


def test_tell() -> None:
    study_name = "test_study"

    with tempfile.NamedTemporaryFile() as tf:
        db_url = "sqlite:///{}".format(tf.name)

        output: Any = subprocess.check_output(
            [
                "optuna",
                "ask",
                "--storage",
                db_url,
                "--study-name",
                study_name,
                "--format",
                "json",
            ]
        )
        output = output.decode("utf-8")
        output = json.loads(output)
        trial_number = output["number"]

        output = subprocess.check_output(
            [
                "optuna",
                "tell",
                "--storage",
                db_url,
                "--trial-number",
                str(trial_number),
                "--values",
                "1.2",
            ]
        )

        study = optuna.load_study(storage=db_url, study_name=study_name)
        assert len(study.trials) == 1
        assert study.trials[0].state == TrialState.COMPLETE
        assert study.trials[0].values == [1.2]
