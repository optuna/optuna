import os
import pytest
import shutil
import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import optuna
from optuna.integration.tensorboard import TensorBoardCallback


def _objective_func(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_uniform("x", -1.0, 1.0)
    y = trial.suggest_loguniform("y", 20.0, 30.0)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    trial.set_user_attr("my_user_attr", "my_user_attr_value")
    return (x - 2) ** 2 + (y - 25) ** 2 + z


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


def test_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        TensorBoardCallback(dirname="", metric_name="")
