import itertools
import multiprocessing
import pickle
import pytest
import tempfile
import threading
import time
from types import TracebackType  # NOQA
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import IO  # NOQA
from typing import Optional  # NOQA
from typing import Type  # NOQA

import pfnopt
import pfnopt.client


STORAGE_MODES = [
    'none',    # We give `None` to storage argument, so InMemoryStorage is used.
    'new',     # We always create a new sqlite DB file for each experiment.
    'common',  # We use a sqlite DB file for the whole experiments.
]

common_tempfile = None  # type: IO[Any]


def setup_module():
    # type: () -> None

    global common_tempfile
    common_tempfile = tempfile.NamedTemporaryFile()


def teardown_module():
    # type: () -> None

    common_tempfile.close()


class StorageSupplier(object):

    def __init__(self, storage_specifier):
        # type: (str) -> None

        self.storage_specifier = storage_specifier
        self.tempfile = None  # type: IO[Any]

    def __enter__(self):
        # type: () -> Optional[pfnopt.storages.BaseStorage]

        if self.storage_specifier == 'none':
            return None
        elif self.storage_specifier == 'new':
            self.tempfile = tempfile.NamedTemporaryFile()
            url = 'sqlite:///{}'.format(self.tempfile.name)
            return pfnopt.storages.RDBStorage(url)
        elif self.storage_specifier == 'common':
            url = 'sqlite:///{}'.format(common_tempfile.name)
            return pfnopt.storages.RDBStorage(url)
        else:
            assert False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (Type[BaseException], BaseException, TracebackType) -> None

        if self.tempfile:
            self.tempfile.close()


def func(client, x_max=1.0):
    # type: (pfnopt.client.BaseClient, float) -> float

    x = client.sample_uniform('x', -x_max, x_max)
    y = client.sample_loguniform('y', 20, 30)
    z = client.sample_categorical('z', (-1.0, 1.0))
    return (x - 2) ** 2 + (y - 25) ** 2 + z


class Func(object):

    def __init__(self, sleep_sec=None):
        # type: (Optional[float]) -> None

        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()
        self.x_max = 10.0

    def __call__(self, client):
        # type: (pfnopt.client.BaseClient) -> float

        with self.lock:
            self.n_calls += 1
            x_max = self.x_max
            self.x_max *= 0.9

        # Sleep for testing parallelism
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        return func(client, x_max)


def check_params(params):
    # type: (Dict[str, Any]) -> None

    assert sorted(params.keys()) == ['x', 'y', 'z']


def check_value(value):
    # type: (float) -> None

    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0 ** 2 + 5.0 ** 2 + 1.0


def check_trial(trial):
    # type: (pfnopt.trial.Trial) -> None

    if trial.state == pfnopt.trial.State.COMPLETE:
        check_params(trial.params)
        check_value(trial.value)


def check_study(study):
    # type: (pfnopt.Study) -> None

    for trial in study.trials:
        check_trial(trial)

    complete_trials = [t for t in study.trials if t.state == pfnopt.trial.State.COMPLETE]
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
    # type: () -> None

    study = pfnopt.minimize(func, n_trials=10)
    check_study(study)


def test_minimize_trivial_in_memory_resume():
    # type: () -> None

    study = pfnopt.minimize(func, n_trials=10)
    pfnopt.minimize(func, n_trials=10, study=study)
    check_study(study)


def test_minimize_trivial_rdb_new():
    # type: () -> None

    # We prohibit automatic new-study creation when storage is specified.
    with pytest.raises(ValueError):
        pfnopt.minimize(func, n_trials=10, storage='sqlite:///:memory:')


def test_minimize_trivial_rdb_resume_study():
    # type: () -> None

    study = pfnopt.create_study('sqlite:///:memory:')
    pfnopt.minimize(func, n_trials=10, study=study)
    check_study(study)


def test_minimize_trivial_rdb_resume_uuid():
    # type: () -> None

    with tempfile.NamedTemporaryFile() as tf:
        db_url = 'sqlite:///{}'.format(tf.name)
        study = pfnopt.create_study(db_url)
        study_uuid = study.study_uuid
        study = pfnopt.minimize(func, n_trials=10, storage=db_url, study_uuid=study_uuid)
        check_study(study)


@pytest.mark.parametrize('n_trials, n_jobs, storage_mode', itertools.product(
    (0, 1, 2, 50),  # n_trials
    (1, 2, 10, -1),  # n_jobs
    STORAGE_MODES,  # storage_mode
))
def test_minimize_parallel(n_trials, n_jobs, storage_mode):
    # type: (int, int, str)-> None

    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = pfnopt.create_study(storage=storage)

        if isinstance(study.storage, pfnopt.storages.RDBStorage) and n_jobs != 1:
            with pytest.raises(TypeError):
                pfnopt.minimize(f, n_trials=n_trials, n_jobs=n_jobs, study=study)
            study.storage.close()
            return

        pfnopt.minimize(f, n_trials=n_trials, n_jobs=n_jobs, study=study)
        assert f.n_calls == len(study.trials) == n_trials
        check_study(study)

        study.storage.close()


@pytest.mark.parametrize('n_trials, n_jobs, storage_mode', itertools.product(
    (0, 1, 2, 50, None),  # n_trials
    (1, 2, 10, -1),  # n_jobs
    STORAGE_MODES,  # storage_mode
))
def test_minimize_parallel_timeout(n_trials, n_jobs, storage_mode):
    # type: (int, int, str) -> None

    sleep_sec = 0.1
    timeout_sec = 1.0
    f = Func(sleep_sec=sleep_sec)

    with StorageSupplier(storage_mode) as storage:
        study = pfnopt.create_study(storage=storage)

        if isinstance(study.storage, pfnopt.storages.RDBStorage) and n_jobs != 1:
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


def test_study_pickle():
    # type: () -> None

    study_1 = pfnopt.minimize(func, n_trials=10)
    check_study(study_1)
    assert len(study_1.trials) == 10
    dumped_bytes = pickle.dumps(study_1)

    study_2 = pickle.loads(dumped_bytes)
    check_study(study_2)
    assert len(study_2.trials) == 10

    pfnopt.minimize(func, n_trials=10, study=study_2)
    check_study(study_2)
    assert len(study_2.trials) == 20
