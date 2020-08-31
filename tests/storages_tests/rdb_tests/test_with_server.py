from multiprocessing import Pool
import os
from typing import Sequence
from typing import Tuple

import numpy as np
import pytest

import optuna

_STUDY_NAME = "_test_multiprocess"


def f(x: float, y: float) -> float:
    return (x - 3) ** 2 + y


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    trial.report(x, 0)
    trial.report(y, 1)
    trial.set_user_attr("x", x)
    trial.set_system_attr("y", y)
    return f(x, y)


def run_optimize(args: Tuple[str, str]) -> None:
    study_name = args[0]
    storage_url = args[1]
    # Create a study
    study = optuna.create_study(study_name=study_name, storage=storage_url, load_if_exists=True)
    # Run optimization
    study.optimize(objective, n_trials=20)


@pytest.fixture
def storage_url() -> str:
    if "TEST_DB_URL" not in os.environ:
        pytest.skip("This test requires TEST_DB_URL.")
    storage_url = os.environ["TEST_DB_URL"]
    try:
        optuna.study.delete_study(_STUDY_NAME, storage_url)
    except KeyError:
        pass
    return storage_url


def _check_trials(trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    # Check trial states.
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in trials)

    # Check trial values and params.
    assert all("x" in trial.params for trial in trials)
    assert all("y" in trial.params for trial in trials)
    assert all(
        np.isclose(
            [trial.value for trial in trials],
            [f(trial.params["x"], trial.params["y"]) for trial in trials],
            atol=1e-4,
        )
    )

    # Check intermediate values.
    assert all(len(trial.intermediate_values) == 2 for trial in trials)
    assert all(trial.params["x"] == trial.intermediate_values[0] for trial in trials)
    assert all(trial.params["y"] == trial.intermediate_values[1] for trial in trials)

    # Check attrs.
    assert all(
        np.isclose(
            [trial.user_attrs["x"] for trial in trials],
            [trial.params["x"] for trial in trials],
            atol=1e-4,
        )
    )
    assert all(
        np.isclose(
            [trial.system_attrs["y"] for trial in trials],
            [trial.params["y"] for trial in trials],
            atol=1e-4,
        )
    )


def test_loaded_trials(storage_url: str) -> None:
    # Please create the tables by placing this function before the multi-process tests.

    N_TRIALS = 20
    study = optuna.create_study(study_name=_STUDY_NAME, storage=storage_url)
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    trials = study.trials
    assert len(trials) == N_TRIALS

    _check_trials(trials)

    # Create a new study to confirm the study can load trial properly.
    loaded_study = optuna.load_study(study_name=_STUDY_NAME, storage=storage_url)
    _check_trials(loaded_study.trials)


def test_multiprocess(storage_url: str) -> None:
    n_workers = 8
    study_name = _STUDY_NAME
    with Pool(n_workers) as pool:
        pool.map(run_optimize, [(study_name, storage_url)] * n_workers)

    study = optuna.load_study(study_name=study_name, storage=storage_url)

    trials = study.trials
    assert len(trials) == n_workers * 20

    _check_trials(trials)
