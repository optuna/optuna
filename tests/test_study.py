import itertools
from mock import Mock  # NOQA
from mock import patch
import multiprocessing
import os
import pandas as pd
import pickle
import pytest
import threading
import time
import uuid

import optuna
from optuna.testing.storage import StorageSupplier
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

STORAGE_MODES = [
    'none',  # We give `None` to storage argument, so InMemoryStorage is used.
    'new',  # We always create a new sqlite DB file for each experiment.
    'common',  # We use a sqlite DB file for the whole experiments.
]

if os.getenv('INCLUDE_SLOW_TESTS') is None:
    MAX_N_TRIALS = 20
    N_JOBS_LIST = [1, 2]
    CACHE_MODES = [True]
else:
    MAX_N_TRIALS = 50
    N_JOBS_LIST = [1, 2, 10, -1]
    CACHE_MODES = [True, False]


def setup_module():
    # type: () -> None

    StorageSupplier.setup_common_tempfile()


def teardown_module():
    # type: () -> None

    StorageSupplier.teardown_common_tempfile()


def func(trial, x_max=1.0):
    # type: (optuna.trial.Trial, float) -> float

    x = trial.suggest_uniform('x', -x_max, x_max)
    y = trial.suggest_loguniform('y', 20, 30)
    z = trial.suggest_categorical('z', (-1.0, 1.0))
    return (x - 2)**2 + (y - 25)**2 + z


class Func(object):
    def __init__(self, sleep_sec=None):
        # type: (Optional[float]) -> None

        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()
        self.x_max = 10.0

    def __call__(self, trial):
        # type: (optuna.trial.Trial) -> float

        with self.lock:
            self.n_calls += 1
            x_max = self.x_max
            self.x_max *= 0.9

        # Sleep for testing parallelism
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        value = func(trial, x_max)
        check_params(trial.params)
        return value


def check_params(params):
    # type: (Dict[str, Any]) -> None

    assert sorted(params.keys()) == ['x', 'y', 'z']


def check_value(value):
    # type: (Optional[float]) -> None

    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0**2 + 5.0**2 + 1.0


def check_frozen_trial(frozen_trial):
    # type: (optuna.structs.FrozenTrial) -> None

    if frozen_trial.state == optuna.structs.TrialState.COMPLETE:
        check_params(frozen_trial.params)
        check_value(frozen_trial.value)


def check_study(study):
    # type: (optuna.Study) -> None

    for trial in study.trials:
        check_frozen_trial(trial)

    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
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
        check_frozen_trial(study.best_trial)


def test_optimize_trivial_in_memory_new():
    # type: () -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_trivial_in_memory_resume():
    # type: () -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10)
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_trivial_rdb_resume_study():
    # type: () -> None

    study = optuna.create_study('sqlite:///:memory:')
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_with_direction():
    # type: () -> None

    study = optuna.create_study(direction='minimize')
    study.optimize(func, n_trials=10)
    assert study.direction == optuna.structs.StudyDirection.MINIMIZE
    check_study(study)

    study = optuna.create_study(direction='maximize')
    study.optimize(func, n_trials=10)
    assert study.direction == optuna.structs.StudyDirection.MAXIMIZE
    check_study(study)

    with pytest.raises(ValueError):
        optuna.create_study(direction='test')


@pytest.mark.parametrize(
    'n_trials, n_jobs, storage_mode, cache_mode',
    itertools.product(
        (0, 1, 2, MAX_N_TRIALS),  # n_trials
        N_JOBS_LIST,  # n_jobs
        STORAGE_MODES,  # storage_mode
        CACHE_MODES,  # cache_mode
    ))
def test_optimize_parallel(n_trials, n_jobs, storage_mode, cache_mode):
    # type: (int, int, str, bool)-> None

    f = Func()

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=n_trials, n_jobs=n_jobs)
        assert f.n_calls == len(study.trials) == n_trials
        check_study(study)


@pytest.mark.parametrize(
    'n_trials, n_jobs, storage_mode, cache_mode',
    itertools.product(
        (0, 1, 2, MAX_N_TRIALS, None),  # n_trials
        N_JOBS_LIST,  # n_jobs
        STORAGE_MODES,  # storage_mode
        CACHE_MODES,  # cache_mode
    ))
def test_optimize_parallel_timeout(n_trials, n_jobs, storage_mode, cache_mode):
    # type: (int, int, str, bool) -> None

    sleep_sec = 0.1
    timeout_sec = 1.0
    f = Func(sleep_sec=sleep_sec)

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout_sec)

        assert f.n_calls == len(study.trials)

        if n_trials is not None:
            assert f.n_calls <= n_trials

        # A thread can process at most (timeout_sec / sleep_sec + 1) trials.
        n_jobs_actual = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        max_calls = (timeout_sec / sleep_sec + 1) * n_jobs_actual
        assert f.n_calls <= max_calls

        check_study(study)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_optimize_with_catch(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)

        def func_value_error(_):
            # type: (optuna.trial.Trial) -> float

            raise ValueError

        # Test default exceptions.
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=20)
        assert len(study.trials) == 1
        assert all(trial.state == optuna.structs.TrialState.FAIL for trial in study.trials)

        # Test acceptable exception.
        study.optimize(func_value_error, n_trials=20, catch=(ValueError, ))
        assert len(study.trials) == 21
        assert all(trial.state == optuna.structs.TrialState.FAIL for trial in study.trials)

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=20, catch=(ArithmeticError, ))
        assert len(study.trials) == 22
        assert all(trial.state == optuna.structs.TrialState.FAIL for trial in study.trials)


@pytest.mark.parametrize('catch', [[], [Exception], None, 1])
def test_optimize_with_catch_invalid_type(catch):
    # type: (Any) -> None

    study = optuna.create_study()

    def func_value_error(_):
        # type: (optuna.trial.Trial) -> float

        raise ValueError

    with pytest.raises(TypeError):
        study.optimize(func_value_error, n_trials=20, catch=catch)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_study_set_and_get_user_attrs(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)

        study.set_user_attr('dataset', 'MNIST')
        assert study.user_attrs['dataset'] == 'MNIST'


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_study_set_and_get_system_attrs(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)

        study.set_system_attr('system_message', 'test')
        assert study.system_attrs['system_message'] == 'test'


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_trial_set_and_get_user_attrs(storage_mode, cache_mode):
    # type: (str, bool) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        trial.set_user_attr('train_accuracy', 1)
        assert trial.user_attrs['train_accuracy'] == 1
        return 0.0

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=1)
        frozen_trial = study.trials[0]
        assert frozen_trial.user_attrs['train_accuracy'] == 1


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_trial_set_and_get_system_attrs(storage_mode, cache_mode):
    # type: (str, bool) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        trial.set_system_attr('system_message', 'test')
        assert trial.system_attrs['system_message'] == 'test'
        return 0.0

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=1)
        frozen_trial = study.trials[0]
        assert frozen_trial.system_attrs['system_message'] == 'test'


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_get_all_study_summaries(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(Func(), n_trials=5)

        summaries = optuna.get_all_study_summaries(study._storage)
        summary = [s for s in summaries if s.study_id == study.study_id][0]

        assert summary.study_name == study.study_name
        assert summary.n_trials == 5


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_get_all_study_summaries_with_no_trials(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)

        summaries = optuna.get_all_study_summaries(study._storage)
        summary = [s for s in summaries if s.study_id == study.study_id][0]

        assert summary.study_name == study.study_name
        assert summary.n_trials == 0
        assert summary.datetime_start is None


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_run_trial(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)

        # Test trial without exception.
        study._run_trial(func, catch=(Exception, ), gc_after_trial=True)
        check_study(study)

        # Test trial with acceptable exception.
        def func_value_error(_):
            # type: (optuna.trial.Trial) -> float

            raise ValueError

        trial = study._run_trial(func_value_error, catch=(ValueError, ), gc_after_trial=True)
        frozen_trial = study._storage.get_trial(trial._trial_id)

        expected_message = 'Setting status of trial#1 as TrialState.FAIL because of the ' \
                           'following error: ValueError()'
        assert frozen_trial.state == optuna.structs.TrialState.FAIL
        assert frozen_trial.system_attrs['fail_reason'] == expected_message

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            study._run_trial(func_value_error, catch=(ArithmeticError, ), gc_after_trial=True)

        # Test trial with invalid objective value: None
        def func_none(_):
            # type: (optuna.trial.Trial) -> float

            return None  # type: ignore

        trial = study._run_trial(func_none, catch=(Exception, ), gc_after_trial=True)
        frozen_trial = study._storage.get_trial(trial._trial_id)

        expected_message = 'Setting status of trial#3 as TrialState.FAIL because the returned ' \
                           'value from the objective function cannot be casted to float. ' \
                           'Returned value is: None'
        assert frozen_trial.state == optuna.structs.TrialState.FAIL
        assert frozen_trial.system_attrs['fail_reason'] == expected_message

        # Test trial with invalid objective value: nan
        def func_nan(_):
            # type: (optuna.trial.Trial) -> float

            return float('nan')

        trial = study._run_trial(func_nan, catch=(Exception, ), gc_after_trial=True)
        frozen_trial = study._storage.get_trial(trial._trial_id)

        expected_message = 'Setting status of trial#4 as TrialState.FAIL because the objective ' \
                           'function returned nan.'
        assert frozen_trial.state == optuna.structs.TrialState.FAIL
        assert frozen_trial.system_attrs['fail_reason'] == expected_message


def test_study_pickle():
    # type: () -> None

    study_1 = optuna.create_study()
    study_1.optimize(func, n_trials=10)
    check_study(study_1)
    assert len(study_1.trials) == 10
    dumped_bytes = pickle.dumps(study_1)

    study_2 = pickle.loads(dumped_bytes)
    check_study(study_2)
    assert len(study_2.trials) == 10

    study_2.optimize(func, n_trials=10)
    check_study(study_2)
    assert len(study_2.trials) == 20


def test_study_trials_dataframe_with_no_trials():
    # type: () -> None

    study_with_no_trials = optuna.create_study()
    trials_df = study_with_no_trials.trials_dataframe()
    assert trials_df.empty


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
@pytest.mark.parametrize('include_internal_fields', [True, False])
def test_trials_dataframe(storage_mode, cache_mode, include_internal_fields):
    # type: (str, bool, bool) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        x = trial.suggest_int('x', 1, 1)
        y = trial.suggest_categorical('y', (2.5, ))
        trial.set_user_attr('train_loss', 3)
        return x + y  # 3.5

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=3)
        df = study.trials_dataframe(include_internal_fields=include_internal_fields)
        # Change index to access rows via trial number.
        df.set_index(('number', ''), inplace=True, drop=False)
        assert len(df) == 3
        # TODO(Yanase): Remove number from system_attrs after adding TrialModel.number.
        # non-nested: 5, params: 2, user_attrs: 1, system_attrs: 1 and 9 in total.
        if include_internal_fields:
            # distributions:2, trial_id: 1
            assert len(df.columns) == 7 + 5
        else:
            assert len(df.columns) == 9

        for i in range(3):
            assert df.number[i] == i
            assert df.state[i] == optuna.structs.TrialState.COMPLETE
            assert df.value[i] == 3.5
            assert isinstance(df.datetime_start[i], pd.Timestamp)
            assert isinstance(df.datetime_complete[i], pd.Timestamp)
            assert df.params.x[i] == 1
            assert df.params.y[i] == 2.5
            assert df.user_attrs.train_loss[i] == 3
            assert df.system_attrs._number[i] == i
            if include_internal_fields:
                assert ('distributions', 'x') in df.columns
                assert ('distributions', 'y') in df.columns
                assert ('trial_id', '') in df.columns  # trial_id depends on other tests.
                assert ('distributions', 'x') in df.columns
                assert ('distributions', 'y') in df.columns


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_trials_dataframe_with_failure(storage_mode, cache_mode):
    # type: (str, bool) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        x = trial.suggest_int('x', 1, 1)
        y = trial.suggest_categorical('y', (2.5, ))
        trial.set_user_attr('train_loss', 3)
        raise ValueError()
        return x + y  # 3.5

    with StorageSupplier(storage_mode, cache_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=3, catch=(ValueError,))
        df = study.trials_dataframe()
        # Change index to access rows via trial number.
        df.set_index(('number', ''), inplace=True, drop=False)
        assert len(df) == 3
        # TODO(Yanase): Remove number from system_attrs after adding TrialModel.number.
        # non-nested: 5, params: 2, user_attrs: 1 system_attrs: 2
        assert len(df.columns) == 10
        for i in range(3):
            assert df.number[i] == i
            assert df.state[i] == optuna.structs.TrialState.FAIL
            assert df.value[i] is None
            assert isinstance(df.datetime_start[i], pd.Timestamp)
            assert isinstance(df.datetime_complete[i], pd.Timestamp)
            assert df.params.x[i] == 1
            assert df.params.y[i] == 2.5
            assert df.user_attrs.train_loss[i] == 3
            assert df.system_attrs._number[i] == i
            assert ('system_attrs', 'fail_reason') in df.columns


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_create_study(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        # Test creating a new study.
        study = optuna.create_study(storage=storage, load_if_exists=False)

        # Test `load_if_exists=True` with existing study.
        optuna.create_study(study_name=study.study_name, storage=storage, load_if_exists=True)

        if isinstance(study._storage, optuna.storages.InMemoryStorage):
            # `InMemoryStorage` does not share study's namespace (i.e., no name conflicts occur).
            optuna.create_study(study_name=study.study_name, storage=storage, load_if_exists=False)
        else:
            # Test `load_if_exists=False` with existing study.
            with pytest.raises(optuna.structs.DuplicatedStudyError):
                optuna.create_study(study_name=study.study_name,
                                    storage=storage,
                                    load_if_exists=False)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_load_study(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        if storage is None:
            # `InMemoryStorage` can not be used with `load_study` function.
            return

        study_name = str(uuid.uuid4())

        with pytest.raises(ValueError):
            # Test loading an unexisting study.
            optuna.study.load_study(study_name=study_name, storage=storage)

        # Create a new study.
        created_study = optuna.study.create_study(study_name=study_name, storage=storage)

        # Test loading an existing study.
        loaded_study = optuna.study.load_study(study_name=study_name, storage=storage)
        assert created_study.study_id == loaded_study.study_id


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_delete_study(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        # Get storage object because delete_study does not accept None.
        storage = optuna.storages.get_storage(storage=storage)
        assert storage is not None

        # Test deleting a non-existing study.
        with pytest.raises(ValueError):
            optuna.delete_study("invalid-study-name", storage)

        # Test deleting an existing study.
        study = optuna.create_study(storage=storage, load_if_exists=False)
        optuna.delete_study(study.study_name, storage)

        # Test failed to delete the study which is already deleted.
        if not isinstance(study._storage, optuna.storages.InMemoryStorage):
            # Skip `InMemoryStorage` because it just internally initializes trials and so on.
            with pytest.raises(ValueError):
                optuna.delete_study(study.study_name, storage)


def test_nested_optimization():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        with pytest.raises(RuntimeError):
            trial.study.optimize(lambda _: 0.0, n_trials=1)

        return 1.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=10, catch=())


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
def test_append_trial(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        assert len(study.trials) == 0

        study._append_trial(value=0.8)
        assert len(study.trials) == 1
        assert study.best_value == 0.8


def test_storage_property():
    # type: () -> None

    study = optuna.create_study()
    assert study.storage == study._storage


@patch('optuna.study.gc.collect')
def test_optimize_with_gc(collect_mock):
    # type: (Mock) -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10, gc_after_trial=True)
    check_study(study)
    assert collect_mock.call_count == 10


@patch('optuna.study.gc.collect')
def test_optimize_without_gc(collect_mock):
    # type: (Mock) -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10, gc_after_trial=False)
    check_study(study)
    assert collect_mock.call_count == 0


@pytest.mark.parametrize('n_jobs', [1, 4])
def test_callbacks(n_jobs):
    # type: (int) -> None

    study = optuna.create_study()

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        return trial.suggest_int('x', 1, 1)

    # Empty callback list.
    study.optimize(objective, callbacks=[], n_trials=10, n_jobs=n_jobs)

    # A callback.
    values = []
    callbacks = [lambda study, trial: values.append(trial.value)]
    study.optimize(objective, callbacks=callbacks, n_trials=10, n_jobs=n_jobs)
    assert values == [1] * 10

    # Two callbacks.
    values = []
    params = []
    callbacks = [
        lambda study, trial: values.append(trial.value),
        lambda study, trial: params.append(trial.params)
    ]
    study.optimize(objective, callbacks=callbacks, n_trials=10, n_jobs=n_jobs)
    assert values == [1] * 10
    assert params == [{'x': 1}] * 10

    # If a trial is failed with an exception and the exception is caught by the study,
    # callbacks are invoked.
    states = []
    callbacks = [lambda study, trial: states.append(trial.state)]
    study.optimize(
        lambda t: 1/0, callbacks=callbacks, n_trials=10, n_jobs=n_jobs,
        catch=(ZeroDivisionError,))
    assert states == [optuna.structs.TrialState.FAIL] * 10

    # NOTE: Because `Study.optimize` blocks forever if `n_jobs` is more than `1` and
    #       an uncaught exception is raised during an optimization,
    #       we test the following scenario only when `n_jobs==1`.
    #       For the details of this problem, please see https://github.com/pfnet/optuna/issues/538.
    #
    # TODO(ohta): Fix `Study.optimize`

    if n_jobs == 1:
        # If a trial is failed with an exception and the exception isn't caught by the study,
        # callbacks aren't invoked.
        states = []
        callbacks = [lambda study, trial: states.append(trial.state)]
        with pytest.raises(ZeroDivisionError):
            study.optimize(lambda t: 1/0, callbacks=callbacks,
                           n_trials=10, n_jobs=n_jobs, catch=())
        assert states == []
