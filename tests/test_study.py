import itertools
import pytest
import time
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA

import pfnopt
from pfnopt import client
from pfnopt import trial as trial_module


def func(x, y):
    # type: (float, float) -> float
    return (x - 2) ** 2 + (y - 25) ** 2


class Func(object):

    def __init__(self, sleep_sec=None):
        # type: (Optional[float]) -> None
        self.n_calls = 0
        self.sleep_sec = sleep_sec

    def __call__(self, c):
        # type: (client.BaseClient) -> float
        self.n_calls += 1

        # Sleep for testing parallelism
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        x = c.sample_uniform('x', -10, 10)
        y = c.sample_uniform('y', 20, 30)
        return func(x, y)


def check_params(params):
    # type: (Dict[str, Any]) -> None
    assert sorted(params.keys()) == ['x', 'y']


def check_value(value):
    # type: (float) -> None
    assert isinstance(value, float)
    assert 0.0 <= value <= 12.0 ** 2 + 5.0 ** 2


def check_trial(trial):
    # type: (trial_module.Trial) -> None

    if trial.state == trial_module.State.COMPLETE:
        check_params(trial.params)
        check_value(trial.value)


def check_study(study):
    # type: (pfnopt.Study) -> None
    check_params(study.best_params)
    check_value(study.best_value)
    check_trial(study.best_trial)

    for trial in study.trials:
        check_trial(trial)


@pytest.mark.parametrize('n_trials, n_jobs', itertools.product(
    (1, 2, 50),  # n_trials
    (1, 2, 10),  # n_jobs
))
def test_minimize(n_trials, n_jobs):
    # type: (int, int) -> None

    f = Func()
    study = pfnopt.minimize(f, n_trials=n_trials, n_jobs=n_jobs)

    assert f.n_calls == len(study.trials) == n_trials

    check_study(study)


@pytest.mark.parametrize('n_trials, n_jobs', itertools.product(
    (1, 2, 50, None),  # n_trials
    (1, 2, 10),  # n_jobs
))
def test_minimize_timeout(n_trials, n_jobs):
    # type: (int, int) -> None

    sleep_sec = 0.1
    timeout_sec = 1.0

    f = Func(sleep_sec=sleep_sec)
    study = pfnopt.minimize(f, n_trials=n_trials, n_jobs=n_jobs, timeout_seconds=timeout_sec)

    assert f.n_calls == len(study.trials)

    if n_trials is not None:
        assert f.n_calls <= n_trials

    # A thread can process at most (timeout_sec / sleep_sec + 1) trials
    assert f.n_calls <= (timeout_sec / sleep_sec + 1) * n_jobs

    check_study(study)
