import itertools
import multiprocessing
import pytest
import tempfile
import threading
import time
from typing import Any, Tuple  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA

import pfnopt
from pfnopt import client as client_module  # NOQA
from pfnopt.storages import RDBStorage
from pfnopt import trial as trial_module


def func(client):
    # type: (client_module.BaseClient) -> float
    x = client.sample_uniform('x', -10, 10)
    y = client.sample_loguniform('y', 20, 30)
    z = client.sample_categorical('z', (-1.0, 1.0))
    return (x - 2) ** 2 + (y - 25) ** 2 + z


class Func(object):

    def __init__(self, sleep_sec=None):
        # type: (Optional[float]) -> None
        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()

    def __call__(self, client):
        # type: (client_module.BaseClient) -> float
        with self.lock:
            self.n_calls += 1

        # Sleep for testing parallelism
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        return func(client)


def check_params(params):
    # type: (Dict[str, Any]) -> None
    assert sorted(params.keys()) == ['x', 'y', 'z']


def check_value(value):
    # type: (float) -> None
    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0 ** 2 + 5.0 ** 2 + 1.0


def check_trial(trial):
    # type: (trial_module.Trial) -> None

    if trial.state == trial_module.State.COMPLETE:
        check_params(trial.params)
        check_value(trial.value)


def check_study(study):
    # type: (pfnopt.Study) -> None
    for trial in study.trials:
        check_trial(trial)

    complete_trials = [t for t in study.trials if t.state == trial_module.State.COMPLETE]
    if len(complete_trials) == 0:
        with pytest.raises(ValueError):
            study.best_params
        with pytest.raises(ValueError):
            study.best_value
        with pytest.raises(ValueError):
            study.best_trial
    else:
        check_params(study.best_params)
        check_value(study.best_value)
        check_trial(study.best_trial)


def test_minimize_trivial_in_memory_new():
    study = pfnopt.minimize(func, n_trials=10)
    check_study(study)


def test_minimize_trivial_in_memory_resume():
    study = pfnopt.minimize(func, n_trials=10)
    pfnopt.minimize(func, n_trials=10, study=study)
    check_study(study)


def test_minimize_trivial_rdb_new():
    # We prohibit automatic new-study creation when storage is specified.
    with pytest.raises(ValueError):
        pfnopt.minimize(func, n_trials=10, storage='sqlite:///:memory:')


def test_minimize_trivial_rdb_resume_study():
    study = pfnopt.create_study('sqlite:///:memory:')
    pfnopt.minimize(func, n_trials=10, study=study)
    check_study(study)


def test_minimize_trivial_rdb_resume_uuid():
    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)
        study = pfnopt.create_study(db_url)
        study_uuid = study.study_uuid
        study = pfnopt.minimize(func, n_trials=10, storage=db_url, study_uuid=study_uuid)
        check_study(study)


@pytest.mark.parametrize('n_trials, n_jobs, storage', itertools.product(
    (0, 1, 2, 50),  # n_trials
    (1, 2, 10, -1),  # n_jobs
    (None, 'sqlite:///:memory:'),  # storage_class_kwargs
))
def test_minimize_parallel(n_trials, n_jobs, storage):
    # type: (int, int, Optional[str])-> None

    f = Func()
    study = pfnopt.create_study(storage=storage)

    if isinstance(study.storage, RDBStorage) and n_jobs != 1:
        with pytest.raises(TypeError):
            pfnopt.minimize(f, n_trials=n_trials, n_jobs=n_jobs, study=study)
        study.storage.close()
        return

    pfnopt.minimize(f, n_trials=n_trials, n_jobs=n_jobs, study=study)
    assert f.n_calls == len(study.trials) == n_trials
    check_study(study)

    study.storage.close()


@pytest.mark.parametrize('n_trials, n_jobs, storage', itertools.product(
    (0, 1, 2, 50, None),  # n_trials
    (1, 2, 10, -1),  # n_jobs
    (None, 'sqlite:///:memory:'),  # storage_class_kwargs
))
def test_minimize_parallel_timeout(n_trials, n_jobs, storage):
    # type: (int, int, Optional[str]) -> None

    sleep_sec = 0.1
    timeout_sec = 1.0

    f = Func(sleep_sec=sleep_sec)
    study = pfnopt.create_study(storage=storage)

    if isinstance(study.storage, RDBStorage) and n_jobs != 1:
        with pytest.raises(TypeError):
            pfnopt.minimize(
                f, n_trials=n_trials, n_jobs=n_jobs, timeout_seconds=timeout_sec, study=study)
        study.storage.close()
        return

    study = pfnopt.minimize(
        f, n_trials=n_trials, n_jobs=n_jobs, timeout_seconds=timeout_sec, study=study)

    n_jobs_actual = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
    assert len(study.trials) - n_jobs_actual <= f.n_calls <= len(study.trials)

    if n_trials is not None:
        assert f.n_calls <= n_trials

    # A thread can process at most (timeout_sec / sleep_sec + 1) trials
    max_calls = (timeout_sec / sleep_sec + 1) * n_jobs_actual
    assert f.n_calls <= max_calls

    check_study(study)

    study.storage.close()
