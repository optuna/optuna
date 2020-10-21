from collections import Counter
import itertools
import math
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pytest

import optuna
from optuna import stepwise
from optuna import study
import optuna.distributions
import optuna.study
import optuna.trial


PARAM_KEY = "x"


@pytest.fixture(scope="module", autouse=True)
def set_logging() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_step() -> None:
    seed = 42
    random.seed(seed)
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    trial = optuna.Trial(study, study._storage.create_new_trial(study._study_id))

    low, high = 0, 10
    dists = {PARAM_KEY: optuna.distributions.IntUniformDistribution(low, high)}
    step = stepwise.Step(distributions=dists)

    actual = step.suggest(trial)[PARAM_KEY]
    expected = trial.suggest_int(PARAM_KEY, low, high)

    assert actual == expected


@pytest.mark.parametrize(
    "x, y",
    [
        ([-50, 50], [-99, 0]),
        (["-50", " 50"], ["-99", "0"]),
        ([True, False], [False, True]),
        ([-50.0, 50.0], [-99.0, 0.0]),
    ],
)
def test_grid_step(x: Any, y: Any) -> None:
    search_space = {"x": x, "y": y}

    step = stepwise.GridStep(search_space, n_trials=1)
    assert step.n_trials == 1

    step = stepwise.GridStep(search_space)
    assert step.n_trials == 2 * 2

    def grid_objective(trial: optuna.trial.Trial) -> float:
        values = step.suggest(trial)
        return int(values["x"]) + int(values["y"])

    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(grid_objective, n_trials=step.n_trials)
    actual = {trial.value for trial in study.trials}

    grid = itertools.product(search_space["x"], search_space["y"])
    expected = set(int(values[0]) + int(values[1]) for values in grid)

    assert actual == expected


def make_int_steps(n_steps: int, low: int = 0, high: int = 10) -> List[Tuple[str, stepwise.Step]]:
    """Return  a list of tuples [(0, step_0), ..., (n, step_n)]
    where each step's n_trials argument is set to the index in the list.
    """
    dists = {PARAM_KEY: optuna.distributions.IntUniformDistribution(low, high)}
    return [(str(i), stepwise.Step(dists, n_trials=i)) for i in range(1, n_steps + 1)]


def objective(trial: optuna.Trial, params: Dict[str, Any]) -> float:
    return params.get(PARAM_KEY, -1)


@pytest.mark.parametrize("n_trials", range(-1, 10))
def test_global_n_trials(n_trials: int) -> None:
    n_steps = 3
    tuner = stepwise.StepwiseTuner(objective, steps=make_int_steps(n_steps))
    tuner.optimize(n_trials=n_trials)

    if n_trials < 0:
        assert len(tuner.study.trials) == 0
    else:
        max_n_trials = math.factorial(n_steps)  # by definition of make_int_steps
        assert len(tuner.study.trials) == min(n_trials, max_n_trials) + 1  # add baseline step


@pytest.mark.parametrize("timeout", [-1, 0, 0.5, 1])
def test_global_n_timeout(timeout: float) -> None:
    tuner = stepwise.StepwiseTuner(objective, steps=make_int_steps(1000))
    tuner.optimize(timeout=timeout)

    if timeout > 0:
        duration = sum(
            trial.duration.total_seconds() for trial in tuner.study.trials  # type:ignore
        )
        assert np.isclose(duration, timeout, atol=0.1)
    else:
        assert len(tuner.study.trials) == 0


def check_optimize(tuner: stepwise.StepwiseTuner, expected: float) -> None:
    tuner.optimize(n_trials=99)

    assert tuner.best_params[PARAM_KEY] == expected
    assert tuner.best_value == expected

    step_keys = [trial.system_attrs[tuner.step_name_key] for trial in tuner.study.trials]
    step_counter = Counter(step_keys)
    baseline = step_counter.pop(f"{tuner.step_name_key}:baseline")
    assert baseline == 1

    for step_name, n_trials in step_counter.items():
        step = [step for name, step in tuner.steps if step_name == name][0]  # type:ignore
        if callable(step):
            step = step({})
        assert step.n_trials == n_trials


def test_optimize_default() -> None:
    default_value = 0
    tuner = stepwise.StepwiseTuner(
        objective=objective,
        steps=make_int_steps(n_steps=3, low=1, high=1),
        default_params={PARAM_KEY: default_value},
    )
    check_optimize(tuner, expected=default_value)


def test_optimize() -> None:
    tuner = stepwise.StepwiseTuner(
        objective=objective,
        steps=make_int_steps(n_steps=3, low=1, high=1),
        default_params={PARAM_KEY: 99},
    )
    check_optimize(tuner, expected=1)


def test_callable_step() -> None:
    def make_step(params: Dict[str, Any]) -> stepwise.Step:
        dists = {PARAM_KEY: optuna.distributions.IntUniformDistribution(low=1, high=1)}
        return stepwise.Step(dists, n_trials=1)

    tuner = stepwise.StepwiseTuner(
        objective=objective,
        steps=[("1", make_step)],
        default_params={PARAM_KEY: 99},
    )
    check_optimize(tuner, expected=1)


def test_resume_optimize() -> None:
    steps = make_int_steps(n_steps=1)

    tuner = stepwise.StepwiseTuner(objective=objective, steps=steps)
    tuner.optimize(n_trials=2)
    n_trials = len(tuner.study.trials)

    tuner2 = stepwise.StepwiseTuner(objective=objective, steps=steps, study=tuner.study)
    tuner.optimize(n_trials=1)
    assert len(tuner2.study.trials) == n_trials + 1


def test_all_budgets_none() -> None:
    dists = {PARAM_KEY: optuna.distributions.IntUniformDistribution(0, 1)}
    steps = [("step", stepwise.Step(dists))]
    tuner = stepwise.StepwiseTuner(objective=objective, steps=steps)
    with pytest.raises(ValueError):
        tuner.optimize()
