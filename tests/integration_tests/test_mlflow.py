from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import py
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.mlflow import MLflowCallback


with try_import():
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

pytestmark = pytest.mark.integration


def _objective_func(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", 20, 30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    trial.set_user_attr("my_user_attr", "my_user_attr_value")
    return (x - 2) ** 2 + (y - 25) ** 2 + z


def _multiobjective_func(trial: optuna.trial.Trial) -> Tuple[float, float]:

    x = trial.suggest_float("x", low=-1.0, high=1.0)
    y = trial.suggest_float("y", low=20, high=30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    first_objective = (x - 2) ** 2 + (y - 25) ** 2 + z
    second_objective = (x - 2) ** 3 + (y - 25) ** 3 - z

    return first_objective, second_objective


# This is tool function for a temporary fix on Optuna side. It avoids an error with user
# attributes that are too long. It should be fixed on MLflow side later.
# When it is fixed on MLflow side this test can be removed.
# see https://github.com/optuna/optuna/issues/1340
# see https://github.com/mlflow/mlflow/issues/2931
def _objective_func_long_user_attr(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", 20, 30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    long_str = str(list(range(5000)))
    trial.set_user_attr("my_user_attr", long_str)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


@pytest.mark.parametrize("name,expected", [(None, "Default"), ("foo", "foo")])
def test_use_existing_or_default_experiment(
    tmpdir: py.path.local, name: Optional[str], expected: str
) -> None:

    if name is not None:
        tracking_uri = f"file:{tmpdir}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(name)

    else:
        # Target directory can't exist when initializing first
        # run with default experiment at non-default uri.
        tracking_uri = f"file:{tmpdir}/foo"
        mlflow.set_tracking_uri(tracking_uri)

    mlflc = MLflowCallback(tracking_uri=tracking_uri, create_experiment=False)
    study = optuna.create_study()

    for _ in range(10):
        # Simulate multiple optimization runs under same experiment.
        study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiment = mlfl_client.search_experiments()[0]
    runs = mlfl_client.search_runs(experiment.experiment_id)

    assert experiment.name == expected
    assert len(runs) == 10


def test_study_name(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_uri)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    assert len(mlfl_client.search_experiments()) == 1

    experiment = mlfl_client.search_experiments()[0]
    runs = mlfl_client.search_runs(experiment.experiment_id)

    assert experiment.name == study_name
    assert len(runs) == n_trials


def test_use_existing_experiment_by_id(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("foo")

    mlflow_kwargs = {"experiment_id": experiment_id}
    mlflc = MLflowCallback(
        tracking_uri=tracking_uri, create_experiment=False, mlflow_kwargs=mlflow_kwargs
    )
    study = optuna.create_study()

    for _ in range(10):
        study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiment_list = mlfl_client.search_experiments()
    assert len(experiment_list) == 1

    experiment = experiment_list[0]
    assert experiment.experiment_id == experiment_id
    assert experiment.name == "foo"

    runs = mlfl_client.search_runs(experiment_id)
    assert len(runs) == 10


def test_metric_name(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    metric_name = "my_metric_name"

    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name=metric_name)
    study = optuna.create_study(study_name="my_study")
    study.optimize(_objective_func, n_trials=3, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()

    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    first_run = mlfl_client.search_runs(experiment_id)[0]
    first_run_dict = first_run.to_dictionary()

    assert metric_name in first_run_dict["data"]["metrics"]


@pytest.mark.parametrize(
    "names,expected",
    [
        ("foo", ["foo_0", "foo_1"]),
        (["foo", "bar"], ["foo", "bar"]),
        (("foo", "bar"), ["foo", "bar"]),
    ],
)
def test_metric_name_multiobjective(
    tmpdir: py.path.local, names: Union[str, List[str]], expected: List[str]
) -> None:

    tracking_uri = f"file:{tmpdir}"

    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name=names)
    study = optuna.create_study(study_name="my_study", directions=["minimize", "maximize"])
    study.optimize(_multiobjective_func, n_trials=3, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()

    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    first_run = mlfl_client.search_runs(experiment_id)[0]
    first_run_dict = first_run.to_dictionary()

    assert all([e in first_run_dict["data"]["metrics"] for e in expected])


@pytest.mark.parametrize("run_name,expected", [(None, "0"), ("foo", "foo")])
def test_run_name(tmpdir: py.path.local, run_name: Optional[str], expected: str) -> None:

    tracking_uri = f"file:{tmpdir}"

    mlflow_kwargs = {"run_name": run_name}
    mlflc = MLflowCallback(tracking_uri=tracking_uri, mlflow_kwargs=mlflow_kwargs)
    study = optuna.create_study()
    study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiment = mlfl_client.search_experiments()[0]
    run = mlfl_client.search_runs(experiment.experiment_id)[0]
    tags = run.data.tags
    assert tags["mlflow.runName"] == expected


# This is a test for a temporary fix on Optuna side. It avoids an error with user
# attributes that are too long. It should be fixed on MLflow side later.
# When it is fixed on MLflow side this test can be removed.
# see https://github.com/optuna/optuna/issues/1340
# see https://github.com/mlflow/mlflow/issues/2931
def test_tag_truncation(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_uri)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func_long_user_attr, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    runs = mlfl_client.search_runs(experiment_id)
    assert len(runs) == n_trials

    first_run = runs[0]
    first_run_dict = first_run.to_dictionary()

    my_user_attr = first_run_dict["data"]["tags"]["my_user_attr"]
    assert len(my_user_attr) <= 5000


def test_nest_trials(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(study_name)

    mlflc = MLflowCallback(tracking_uri=tracking_uri, mlflow_kwargs={"nested": True})
    study = optuna.create_study(study_name=study_name)

    n_trials = 3
    with mlflow.start_run() as parent_run:
        study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    experiment_id = experiments[0].experiment_id

    all_runs = mlfl_client.search_runs([experiment_id])
    child_runs = [r for r in all_runs if MLFLOW_PARENT_RUN_ID in r.data.tags]

    assert len(all_runs) == n_trials + 1
    assert len(child_runs) == n_trials
    assert all(r.data.tags[MLFLOW_PARENT_RUN_ID] == parent_run.info.run_id for r in child_runs)
    assert all(set(r.data.params.keys()) == {"x", "y", "z"} for r in child_runs)
    assert all(set(r.data.metrics.keys()) == {"value"} for r in child_runs)


@pytest.mark.parametrize("n_jobs", [2, 4])
def test_multiple_jobs(tmpdir: py.path.local, n_jobs: int) -> None:
    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    # The race-condition usually happens after first trial for each job.
    n_trials = n_jobs * 2

    mlflc = MLflowCallback(tracking_uri=tracking_uri)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc], n_jobs=n_jobs)

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    assert len(experiments) == 1

    experiment_id = experiments[0].experiment_id
    runs = mlfl_client.search_runs([experiment_id])
    assert len(runs) == n_trials


def test_mlflow_callback_fails_when_nest_trials_is_false_and_active_run_exists(
    tmpdir: py.path.local,
) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(study_name)

    mlflc = MLflowCallback(tracking_uri=tracking_uri)
    study = optuna.create_study(study_name=study_name)

    with mlflow.start_run():
        with pytest.raises(Exception, match=r"Run with UUID \w+ is already active."):
            study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])


def test_tag_always_logged(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_uri)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiment = mlfl_client.search_experiments()[0]
    runs = mlfl_client.search_runs([experiment.experiment_id])

    assert all((r.data.tags["direction"] == "MINIMIZE") for r in runs)
    assert all((r.data.tags["state"] == "COMPLETE") for r in runs)


@pytest.mark.parametrize("tag_study_user_attrs", [True, False])
def test_tag_study_user_attrs(tmpdir: py.path.local, tag_study_user_attrs: bool) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_uri, tag_study_user_attrs=tag_study_user_attrs)
    study = optuna.create_study(study_name=study_name)
    study.set_user_attr("my_study_attr", "a")
    study.optimize(_objective_func_long_user_attr, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    runs = mlfl_client.search_runs([experiment_id])
    assert len(runs) == n_trials

    if tag_study_user_attrs:
        assert all((r.data.tags["my_study_attr"] == "a") for r in runs)
    else:
        assert all(("my_study_attr" not in r.data.tags) for r in runs)


@pytest.mark.parametrize("tag_trial_user_attrs", [True, False])
def test_tag_trial_user_attrs(tmpdir: py.path.local, tag_trial_user_attrs: bool) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_uri, tag_trial_user_attrs=tag_trial_user_attrs)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiment = mlfl_client.search_experiments()[0]
    runs = mlfl_client.search_runs([experiment.experiment_id])

    if tag_trial_user_attrs:
        assert all((r.data.tags["my_user_attr"] == "my_user_attr_value") for r in runs)
    else:
        assert all(("my_user_attr" not in r.data.tags) for r in runs)


def test_log_mlflow_tags(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    expected_tags = {"foo": 0, "bar": 1}
    mlflow_kwargs = {"tags": expected_tags}

    mlflc = MLflowCallback(tracking_uri=tracking_uri, mlflow_kwargs=mlflow_kwargs)
    study = optuna.create_study()
    study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiment = mlfl_client.search_experiments()[0]
    run = mlfl_client.search_runs(experiment.experiment_id)[0]
    tags = run.data.tags

    assert all([k in tags.keys() for k in expected_tags.keys()])
    assert all([tags[key] == str(value) for key, value in expected_tags.items()])


@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_track_in_mlflow_decorator(tmpdir: py.path.local, n_jobs: int) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"
    n_trials = n_jobs * 2

    metric_name = "additional_metric"
    metric = 3.14

    mlflc = MLflowCallback(tracking_uri=tracking_uri)

    def _objective_func(trial: optuna.trial.Trial) -> float:
        """Objective function"""

        x = trial.suggest_float("x", -1.0, 1.0)
        y = trial.suggest_float("y", 20, 30, log=True)
        z = trial.suggest_categorical("z", (-1.0, 1.0))
        trial.set_user_attr("my_user_attr", "my_user_attr_value")
        mlflow.log_metric(metric_name, metric)
        return (x - 2) ** 2 + (y - 25) ** 2 + z

    tracked_objective = mlflc.track_in_mlflow()(_objective_func)

    study = optuna.create_study(study_name=study_name)
    study.optimize(tracked_objective, n_trials=n_trials, callbacks=[mlflc], n_jobs=n_jobs)

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    runs = mlfl_client.search_runs(experiment_id)
    assert len(runs) == n_trials

    first_run = runs[0]
    first_run_dict = first_run.to_dictionary()

    assert metric_name in first_run_dict["data"]["metrics"]
    assert first_run_dict["data"]["metrics"][metric_name] == metric

    assert tracked_objective.__name__ == _objective_func.__name__
    assert tracked_objective.__doc__ == _objective_func.__doc__


@pytest.mark.parametrize(
    "func,names,values",
    [
        (_objective_func, ["metric"], [27.0]),
        (_multiobjective_func, ["metric1", "metric2"], [27.0, -127.0]),
    ],
)
def test_log_metric(
    tmpdir: py.path.local, func: Callable, names: List[str], values: List[float]
) -> None:

    tracking_uri = f"file:{tmpdir}"
    study_name = "my_study"

    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name=names)
    study = optuna.create_study(
        study_name=study_name, directions=["minimize" for _ in range(len(values))]
    )
    study.enqueue_trial({"x": 1.0, "y": 20.0, "z": 1.0})
    study.optimize(func, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    runs = mlfl_client.search_runs(experiment_id)
    assert len(runs) == 1

    run = runs[0]
    run_dict = run.to_dictionary()

    assert all(name in run_dict["data"]["metrics"] for name in names)
    assert all([run_dict["data"]["metrics"][name] == val for name, val in zip(names, values)])


def test_log_metric_none(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    metric_name = "metric"
    study_name = "my_study"

    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name=metric_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(lambda _: np.nan, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    runs = mlfl_client.search_runs(experiment_id)
    assert len(runs) == 1

    run = runs[0]
    run_dict = run.to_dictionary()

    # When `values` is `None`, do not save values with metric names.
    assert metric_name not in run_dict["data"]["metrics"]


def test_log_params(tmpdir: py.path.local) -> None:

    tracking_uri = f"file:{tmpdir}"
    metric_name = "metric"
    study_name = "my_study"

    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name=metric_name)
    study = optuna.create_study(study_name=study_name)
    study.enqueue_trial({"x": 1.0, "y": 20.0, "z": 1.0})
    study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_uri)
    experiments = mlfl_client.search_experiments()
    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    runs = mlfl_client.search_runs(experiment_id)
    assert len(runs) == 1

    run = runs[0]
    run_dict = run.to_dictionary()

    for param_name, param_value in study.best_params.items():
        assert param_name in run_dict["data"]["params"]
        assert run_dict["data"]["params"][param_name] == str(param_value)
        assert run_dict["data"]["tags"][f"{param_name}_distribution"] == str(
            study.best_trial.distributions[param_name]
        )


@pytest.mark.parametrize("metrics", [["foo"], ["foo", "bar", "baz"]])
def test_multiobjective_raises_on_name_mismatch(tmpdir: py.path.local, metrics: List[str]) -> None:

    tracking_uri = f"file:{tmpdir}"
    mlflc = MLflowCallback(tracking_uri=tracking_uri, metric_name=metrics)
    study = optuna.create_study(study_name="my_study", directions=["minimize", "maximize"])

    with pytest.raises(ValueError):
        study.optimize(_multiobjective_func, n_trials=1, callbacks=[mlflc])
