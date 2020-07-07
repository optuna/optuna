from mlflow.tracking import MlflowClient
import py

import optuna
from optuna.integration.mlflow import MLflowCallback


def _objective_func(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_uniform("x", -1.0, 1.0)
    y = trial.suggest_loguniform("y", 20, 30)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    trial.set_user_attr("my_user_attr", "my_user_attr_value")
    return (x - 2) ** 2 + (y - 25) ** 2 + z


# This is tool function for a temporary fix on Optuna side. It avoids an error with user
# attributes that are too long. It should be fixed on MLflow side later.
# When it is fixed on MLflow side this test can be removed.
# see https://github.com/optuna/optuna/issues/1340
# see https://github.com/mlflow/mlflow/issues/2931
def _objective_func_long_user_attr(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_uniform("x", -1.0, 1.0)
    y = trial.suggest_loguniform("y", 20, 30)
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
        == "LogUniformDistribution(high=30, low=20)"
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
