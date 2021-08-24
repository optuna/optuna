from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import numpy as np
import py
import pytest

import optuna
from optuna.integration.mlflow import MLflowCallback


def _objective_func(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", 20, 30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    trial.set_user_attr("my_user_attr", "my_user_attr_value")
    return (x - 2) ** 2 + (y - 25) ** 2 + z


def _multiobjective_func(trial: optuna.trial.Trial) -> Tuple[float, float]:

    x = trial.suggest_uniform("x", low=-10, high=10)
    y = trial.suggest_loguniform("y", low=1, high=10)
    first_objective = (x - 2) ** 2 + (y - 25) ** 2
    second_objective = (x - 2) ** 3 + (y - 25) ** 3

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
    assert isinstance(z, float)
    long_str = str(list(range(5000)))
    trial.set_user_attr("my_user_attr", long_str)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


def test_study_name(tmpdir: py.path.local) -> None:

    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_file_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()
    assert "value" in first_run_dict["data"]["metrics"]
    assert "x" in first_run_dict["data"]["params"]
    assert "y" in first_run_dict["data"]["params"]
    assert "z" in first_run_dict["data"]["params"]
    assert first_run_dict["data"]["tags"]["direction"] == "MINIMIZE"
    assert first_run_dict["data"]["tags"]["state"] == "COMPLETE"
    assert (
        first_run_dict["data"]["tags"]["x_distribution"]
        == "UniformDistribution(high=1.0, low=-1.0)"
    )
    assert (
        first_run_dict["data"]["tags"]["y_distribution"]
        == "LogUniformDistribution(high=30.0, low=20.0)"
    )
    assert (
        first_run_dict["data"]["tags"]["z_distribution"]
        == "CategoricalDistribution(choices=(-1.0, 1.0))"
    )
    assert first_run_dict["data"]["tags"]["my_user_attr"] == "my_user_attr_value"


def test_metric_name(tmpdir: py.path.local) -> None:

    tracking_file_name = "file:{}".format(tmpdir)
    metric_name = "my_metric_name"

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=metric_name)
    study = optuna.create_study(study_name="my_study")
    study.optimize(_objective_func, n_trials=3, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()

    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert metric_name in first_run_dict["data"]["metrics"]


@pytest.mark.parametrize(
    "names,expected", [("foo", ["foo_0", "foo_1"]), (["foo", "bar"], ["foo", "bar"])]
)
def test_metric_name_multiobjective(
    tmpdir: py.path.local, names: Union[str, List[str]], expected: List[str]
) -> None:

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=names)
    study = optuna.create_study(study_name="my_study", directions=["minimize", "maximize"])
    study.optimize(_multiobjective_func, n_trials=3, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()

    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert all([e in first_run_dict["data"]["metrics"] for e in expected])


# This is a test for a temporary fix on Optuna side. It avoids an error with user
# attributes that are too long. It should be fixed on MLflow side later.
# When it is fixed on MLflow side this test can be removed.
# see https://github.com/optuna/optuna/issues/1340
# see https://github.com/mlflow/mlflow/issues/2931
def test_tag_truncation(tmpdir: py.path.local) -> None:

    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(tracking_uri=tracking_file_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func_long_user_attr, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    my_user_attr = first_run_dict["data"]["tags"]["my_user_attr"]
    assert len(my_user_attr) <= 5000


def test_nest_trials(tmpdir: py.path.local) -> None:
    tmp_tracking_uri = "file:{}".format(tmpdir)

    study_name = "my_study"
    mlflow.set_tracking_uri(tmp_tracking_uri)
    mlflow.set_experiment(study_name)

    mlflc = MLflowCallback(tracking_uri=tmp_tracking_uri, nest_trials=True)
    study = optuna.create_study(study_name=study_name)

    n_trials = 3
    with mlflow.start_run() as parent_run:
        study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tmp_tracking_uri)
    experiments = mlfl_client.list_experiments()
    experiment_id = experiments[0].experiment_id

    all_runs = mlfl_client.search_runs([experiment_id])
    child_runs = [r for r in all_runs if MLFLOW_PARENT_RUN_ID in r.data.tags]

    assert len(all_runs) == n_trials + 1
    assert len(child_runs) == n_trials
    assert all(r.data.tags[MLFLOW_PARENT_RUN_ID] == parent_run.info.run_id for r in child_runs)
    assert all(set(r.data.params.keys()) == {"x", "y", "z"} for r in child_runs)
    assert all(set(r.data.metrics.keys()) == {"value"} for r in child_runs)


def test_mlflow_callback_fails_when_nest_trials_is_false_and_active_run_exists(
    tmpdir: py.path.local,
) -> None:
    tmp_tracking_uri = "file:{}".format(tmpdir)

    study_name = "my_study"
    mlflow.set_tracking_uri(tmp_tracking_uri)
    mlflow.set_experiment(study_name)

    mlflc = MLflowCallback(tracking_uri=tmp_tracking_uri, nest_trials=False)
    study = optuna.create_study(study_name=study_name)

    with mlflow.start_run():
        with pytest.raises(Exception, match=r"Run with UUID \w+ is already active."):
            study.optimize(_objective_func, n_trials=1, callbacks=[mlflc])


@pytest.mark.parametrize("tag_study_user_attrs", [True, False])
def test_tag_study_user_attrs(tmpdir: py.path.local, tag_study_user_attrs: bool) -> None:
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 3

    mlflc = MLflowCallback(
        tracking_uri=tracking_file_name, tag_study_user_attrs=tag_study_user_attrs
    )
    study = optuna.create_study(study_name=study_name)
    study.set_user_attr("my_study_attr", "a")
    study.optimize(_objective_func_long_user_attr, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
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


def test_track_in_mlflow_decorator(tmpdir: py.path.local) -> None:
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 3

    metric_name = "additional_metric"
    metric = 3.14

    mlflc = MLflowCallback(tracking_uri=tracking_file_name)

    @mlflc.track_in_mlflow()
    def _objective_func(trial: optuna.trial.Trial) -> float:

        x = trial.suggest_float("x", -1.0, 1.0)
        y = trial.suggest_float("y", 20, 30, log=True)
        z = trial.suggest_categorical("z", (-1.0, 1.0))
        assert isinstance(z, float)
        trial.set_user_attr("my_user_attr", "my_user_attr_value")
        mlflow.log_metric(metric_name, metric)
        return (x - 2) ** 2 + (y - 25) ** 2 + z

    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=n_trials, callbacks=[mlflc])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert metric_name in first_run_dict["data"]["metrics"]
    assert first_run_dict["data"]["metrics"][metric_name] == metric


def test_initialize_experiment(tmpdir: py.path.local) -> None:
    tracking_file_name = "file:{}".format(tmpdir)
    metric_name = "my_metric_name"
    study_name = "my_study"

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=metric_name)
    study = optuna.create_study(study_name=study_name)

    mlflc._initialize_experiment(study)

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name


@pytest.mark.parametrize(
    "names,values", [(["metric"], [3.17]), (["metric1", "metric2"], [3.17, 3.18])]
)
def test_log_metric(
    tmpdir: py.path.local, names: List[str], values: List[Optional[float]]
) -> None:

    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=names)
    study = optuna.create_study(study_name=study_name)
    mlflc._initialize_experiment(study)

    with mlflow.start_run():
        mlflc._log_metrics(values)

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == 1

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert all(name in first_run_dict["data"]["metrics"] for name in names)
    assert all(
        [first_run_dict["data"]["metrics"][name] == val for name, val in zip(names, values)]
    )


def test_log_metric_none(tmpdir: py.path.local) -> None:
    tracking_file_name = "file:{}".format(tmpdir)
    metric_name = "my_metric_name"
    study_name = "my_study"
    metric_value = None

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=metric_name)
    study = optuna.create_study(study_name=study_name)
    mlflc._initialize_experiment(study)

    with mlflow.start_run():
        mlflc._log_metrics([metric_value])

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == 1

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert metric_name in first_run_dict["data"]["metrics"]
    assert np.isnan(first_run_dict["data"]["metrics"][metric_name])


def test_log_params(tmpdir: py.path.local) -> None:
    tracking_file_name = "file:{}".format(tmpdir)
    metric_name = "my_metric_name"
    study_name = "my_study"

    param1_name = "my_param1"
    param1_value = "a"
    param2_name = "my_param2"
    param2_value = 5

    params = {param1_name: param1_value, param2_name: param2_value}

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=metric_name)
    study = optuna.create_study(study_name=study_name)
    mlflc._initialize_experiment(study)

    with mlflow.start_run():

        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                param1_name: optuna.distributions.CategoricalDistribution(["a", "b"]),
                param2_name: optuna.distributions.UniformDistribution(0, 10),
            },
            value=5.0,
        )
        mlflc._log_params(trial.params)

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    experiment = experiments[0]
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == 1

    first_run_id = run_infos[0].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert param1_name in first_run_dict["data"]["params"]
    assert first_run_dict["data"]["params"][param1_name] == param1_value

    assert param2_name in first_run_dict["data"]["params"]
    assert first_run_dict["data"]["params"][param2_name] == str(param2_value)


@pytest.mark.parametrize("metrics", [["foo"], ["foo", "bar", "baz"]])
def test_multiobjective_raises_on_name_mismatch(tmpdir: py.path.local, metrics: List[str]) -> None:

    tracking_file_name = "file:{}".format(tmpdir)
    mlflc = MLflowCallback(tracking_uri=tracking_file_name, metric_name=metrics)
    study = optuna.create_study(study_name="my_study", directions=["minimize", "maximize"])

    with pytest.raises(ValueError):
        study.optimize(_multiobjective_func, n_trials=1, callbacks=[mlflc])
