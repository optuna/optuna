import os
import shutil
import tempfile

import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.tensorboard import TensorBoardCallback


with try_import():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

pytestmark = pytest.mark.integration


def _objective_func(trial: optuna.trial.Trial) -> float:
    u = trial.suggest_int("u", 0, 10, step=2)
    v = trial.suggest_int("v", 1, 10, log=True)
    w = trial.suggest_float("w", -1.0, 1.0, step=0.1)
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_float("y", 20.0, 30.0, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    trial.set_user_attr("my_user_attr", "my_user_attr_value")
    return u + v + w + (x - 2) ** 2 + (y - 25) ** 2 + z


def test_study_name() -> None:
    dirname = tempfile.mkdtemp()
    metric_name = "target"
    study_name = "test_tensorboard_integration"

    tbcallback = TensorBoardCallback(dirname, metric_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func, n_trials=1, callbacks=[tbcallback])

    event_acc = EventAccumulator(os.path.join(dirname, "trial-0"))
    event_acc.Reload()

    try:
        assert len(event_acc.Tensors("target")) == 1
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(dirname)


def test_cast_float() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", 1, 2)
        y = trial.suggest_float("y", 1, 2, log=True)
        assert isinstance(x, float)
        assert isinstance(y, float)
        return x + y

    dirname = tempfile.mkdtemp()
    metric_name = "target"
    study_name = "test_tensorboard_integration"

    tbcallback = TensorBoardCallback(dirname, metric_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(objective, n_trials=1, callbacks=[tbcallback])


def test_categorical() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_categorical("x", [1, 2, 3])
        assert isinstance(x, int)
        return x

    dirname = tempfile.mkdtemp()
    metric_name = "target"
    study_name = "test_tensorboard_integration"

    tbcallback = TensorBoardCallback(dirname, metric_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(objective, n_trials=1, callbacks=[tbcallback])


def test_categorical_mixed_types() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_categorical("x", [None, 1, 2, 3.14, True, "foo"])
        assert x is None or isinstance(x, (int, float, bool, str))
        return len(str(x))

    dirname = tempfile.mkdtemp()
    metric_name = "target"
    study_name = "test_tensorboard_integration"

    tbcallback = TensorBoardCallback(dirname, metric_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(objective, n_trials=10, callbacks=[tbcallback])


def test_categorical_unsupported_types() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_categorical("x", [[1, 2], [3, 4, 5], [6]])  # type: ignore[list-item]
        assert isinstance(x, list)
        return len(x)

    dirname = tempfile.mkdtemp()
    metric_name = "target"
    study_name = "test_tensorboard_integration"

    tbcallback = TensorBoardCallback(dirname, metric_name)
    study = optuna.create_study(study_name=study_name)
    study.optimize(objective, n_trials=10, callbacks=[tbcallback])


def test_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        TensorBoardCallback(dirname="", metric_name="")
