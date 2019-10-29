import collections
import datetime
import gc
import math
import multiprocessing
import multiprocessing.pool

try:
    import pandas as pd  # NOQA
    _pandas_available = True
except ImportError as e:
    _pandas_import_error = e
    # trials_dataframe is disabled because pandas is not available.
    _pandas_available = False

from six.moves import queue
import threading
import time
import warnings

from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import structs
from optuna import trial as trial_module
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from multiprocessing import Queue  # NOQA
    from typing import Any  # NOQA
    from typing import Callable
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Set  # NOQA
    from typing import Tuple  # NOQA
    from typing import Type  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA

    ObjectiveFuncType = Callable[[trial_module.Trial], float]


class BaseStudy(object):
    def __init__(self, study_id, storage):
        # type: (int, storages.BaseStorage) -> None

        self.study_id = study_id
        self._storage = storage

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.params

    @property
    def best_value(self):
        # type: () -> float
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.
        """

        best_value = self.best_trial.value
        assert best_value is not None

        return best_value

    @property
    def best_trial(self):
        # type: () -> structs.FrozenTrial
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.structs.FrozenTrial` object of the best trial.
        """

        return self._storage.get_best_trial(self.study_id)

    @property
    def direction(self):
        # type: () -> structs.StudyDirection
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.structs.StudyDirection` object.
        """

        return self._storage.get_study_direction(self.study_id)

    @property
    def trials(self):
        # type: () -> List[structs.FrozenTrial]
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        Returns:
            A list of :class:`~optuna.structs.FrozenTrial` objects.
        """

        return self._storage.get_all_trials(self.study_id)

    @property
    def storage(self):
        # type: () -> storages.BaseStorage
        """Return the storage object used by the study.

        .. deprecated:: 0.15.0
            The direct use of storage is deprecated.
            Please access to storage via study's public methods
            (e.g., :meth:`~optuna.study.Study.set_user_attr`).

        Returns:
            A storage object.
        """

        warnings.warn("The direct use of storage is deprecated. "
                      "Please access to storage via study's public methods "
                      "(e.g., `Study.set_user_attr`)",
                      DeprecationWarning)

        logger = logging.get_logger(__name__)
        logger.warning("The direct use of storage is deprecated. "
                       "Please access to storage via study's public methods "
                       "(e.g., `Study.set_user_attr`)")

        return self._storage


class Study(BaseStudy):
    """A study corresponds to an optimization task, i.e., a set of trials.

    This object provides interfaces to run a new :class:`~optuna.trial.Trial`, access trials'
    history, set/get user-defined attributes of the study itself.

    Note that the direct use of this constructor is not recommended.
    To create and load a study, please refer to the documentation of
    :func:`~optuna.study.create_study` and :func:`~optuna.study.load_study` respectively.

    """

    def __init__(
            self,
            study_name,  # type: str
            storage,  # type: Union[str, storages.BaseStorage]
            sampler=None,  # type: samplers.BaseSampler
            pruner=None  # type: pruners.BasePruner
    ):
        # type: (...) -> None

        self.study_name = study_name
        storage = storages.get_storage(storage)
        study_id = storage.get_study_id_from_name(study_name)
        super(Study, self).__init__(study_id, storage)

        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self.logger = logging.get_logger(__name__)

        self._optimize_lock = threading.Lock()

    def __getstate__(self):
        # type: () -> Dict[Any, Any]

        state = self.__dict__.copy()
        del state['logger']
        del state['_optimize_lock']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None

        self.__dict__.update(state)
        self.logger = logging.get_logger(__name__)
        self._optimize_lock = threading.Lock()

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self._storage.get_study_user_attrs(self.study_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self._storage.get_study_system_attrs(self.study_id)

    def optimize(
            self,
            func,  # type: ObjectiveFuncType
            n_trials=None,  # type: Optional[int]
            timeout=None,  # type: Optional[float]
            n_jobs=1,  # type: int
            catch=(),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            callbacks=None,  # type: Optional[List[Callable[[Study, structs.FrozenTrial], None]]]
            gc_after_trial=True  # type: bool
    ):
        # type: (...) -> None
        """Optimize an objective function.

        Args:
            func:
                A callable that implements objective function.
            n_trials:
                The number of trials. If this argument is set to :obj:`None`, there is no
                limitation on the number of trials. If :obj:`timeout` is also set to :obj:`None`,
                the study continues to create trials until it receives a termination signal such
                as Ctrl+C or SIGTERM.
            timeout:
                Stop study after the given number of second(s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If :obj:`n_trials` is
                also set to :obj:`None`, the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            n_jobs:
                The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU counts.
            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~structs.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial.
            gc_after_trial:
                Flag to execute garbage collection at the end of each trial. By default, garbage
                collection is enabled, just in case. You can turn it off with this argument if
                memory is safely managed in your objective function.
        """

        if not self._optimize_lock.acquire(False):
            raise RuntimeError("Nested invocation of `Study.optimize` method isn't allowed.")
        if not isinstance(catch, tuple):
            raise TypeError("The catch argument is of type \'{}\' but must be a tuple.".format(
                type(catch).__name__))

        try:
            if n_jobs == 1:
                self._optimize_sequential(func, n_trials, timeout, catch, callbacks,
                                          gc_after_trial)
            else:
                self._optimize_parallel(func, n_trials, timeout, n_jobs, catch, callbacks,
                                        gc_after_trial)
        finally:
            self._optimize_lock.release()

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a user attribute to the study.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_user_attr(self.study_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a system attribute to the study.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.study.Study.set_user_attr` to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_system_attr(self.study_id, key, value)

    def trials_dataframe(self, include_internal_fields=False):
        # type: (bool) -> pd.DataFrame
        """Export trials as a pandas DataFrame_.

        The DataFrame_ provides various features to analyze studies. It is also useful to draw a
        histogram of objective values and to export trials as a CSV file. Note that DataFrames
        returned by :func:`~optuna.study.Study.trials_dataframe()` employ MultiIndex_, and columns
        have a hierarchical structure. Please refer to the example below to access DataFrame
        elements.

        Example:

            Get an objective value and a value of parameter ``x`` in the first row.

            >>> df = study.trials_dataframe()
            >>> df
            >>> df.value[0]
            0.0
            >>> df.params.x[0]
            1.0

        Args:
            include_internal_fields:
                By default, internal fields of :class:`~optuna.structs.FrozenTrial` are excluded
                from a DataFrame of trials. If this argument is :obj:`True`, they will be included
                in the DataFrame.

        Returns:
            A pandas DataFrame_ of trials in the :class:`~optuna.study.Study`.

        .. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
        .. _MultiIndex: https://pandas.pydata.org/pandas-docs/stable/advanced.html
        """
        _check_pandas_availability()

        # If no trials, return an empty dataframe.
        if not len(self.trials):
            return pd.DataFrame()

        # column_agg is an aggregator of column names.
        # Keys of column agg are attributes of FrozenTrial such as 'trial_id' and 'params'.
        # Values are dataframe columns such as ('trial_id', '') and ('params', 'n_layers').
        column_agg = collections.defaultdict(set)  # type: Dict[str, Set]
        non_nested_field = ''

        records = []  # type: List[Dict[Tuple[str, str], Any]]
        for trial in self.trials:
            trial_dict = trial._asdict()

            record = {}
            for field, value in trial_dict.items():
                if not include_internal_fields and field in structs.FrozenTrial.internal_fields:
                    continue
                if isinstance(value, dict):
                    for in_field, in_value in value.items():
                        record[(field, in_field)] = in_value
                        column_agg[field].add((field, in_field))
                else:
                    record[(field, non_nested_field)] = value
                    column_agg[field].add((field, non_nested_field))
            records.append(record)

        columns = sum((sorted(column_agg[k]) for k in structs.FrozenTrial._fields),
                      [])  # type: List[Tuple['str', 'str']]

        return pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))

    def _append_trial(
            self,
            value=None,  # type: Optional[float]
            params=None,  # type: Optional[Dict[str, Any]]
            distributions=None,  # type: Optional[Dict[str, BaseDistribution]]
            user_attrs=None,  # type: Optional[Dict[str, Any]]
            system_attrs=None,  # type: Optional[Dict[str, Any]]
            intermediate_values=None,  # type: Optional[Dict[int, float]]
            state=structs.TrialState.COMPLETE,  # type: structs.TrialState
            datetime_start=None,  # type: Optional[datetime.datetime]
            datetime_complete=None  # type: Optional[datetime.datetime]
    ):
        # type: (...) -> None

        params = params or {}
        distributions = distributions or {}
        user_attrs = user_attrs or {}
        system_attrs = system_attrs or {}
        intermediate_values = intermediate_values or {}
        datetime_start = datetime_start or datetime.datetime.now()

        if state.is_finished():
            datetime_complete = datetime_complete or datetime.datetime.now()

        trial = structs.FrozenTrial(
            number=-1,  # dummy value.
            trial_id=-1,  # dummy value.
            state=state,
            value=value,
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
            system_attrs=system_attrs,
            intermediate_values=intermediate_values)

        trial._validate()

        self.storage.create_new_trial(self.study_id, template_trial=trial)

    def _optimize_sequential(
            self,
            func,  # type: ObjectiveFuncType
            n_trials,  # type: Optional[int]
            timeout,  # type: Optional[float]
            catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            callbacks,  # type: Optional[List[Callable[[Study, structs.FrozenTrial], None]]]
            gc_after_trial  # type: bool
    ):
        # type: (...) -> None

        i_trial = 0
        time_start = datetime.datetime.now()
        while True:
            if n_trials is not None:
                if i_trial >= n_trials:
                    break
                i_trial += 1

            if timeout is not None:
                elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
                if elapsed_seconds >= timeout:
                    break

            self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)

    def _optimize_parallel(
            self,
            func,  # type: ObjectiveFuncType
            n_trials,  # type: Optional[int]
            timeout,  # type: Optional[float]
            n_jobs,  # type: int
            catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            callbacks,  # type: Optional[List[Callable[[Study, structs.FrozenTrial], None]]]
            gc_after_trial  # type: bool
    ):
        # type: (...) -> None

        self.start_datetime = datetime.datetime.now()

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if n_trials is not None:
            # The number of threads needs not to be larger than trials.
            n_jobs = min(n_jobs, n_trials)

            if n_trials == 0:
                return  # When n_jobs is zero, ThreadPool fails.

        pool = multiprocessing.pool.ThreadPool(n_jobs)  # type: ignore

        # A queue is passed to each thread. When True is received, then the thread continues
        # the evaluation. When False is received, then it quits.
        def func_child_thread(que):
            # type: (Queue) -> None

            while que.get():
                self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)
            self._storage.remove_session()

        que = multiprocessing.Queue(maxsize=n_jobs)  # type: ignore
        for _ in range(n_jobs):
            que.put(True)
        n_enqueued_trials = n_jobs
        imap_ite = pool.imap(func_child_thread, [que] * n_jobs, chunksize=1)

        while True:
            if timeout is not None:
                elapsed_timedelta = datetime.datetime.now() - self.start_datetime
                elapsed_seconds = elapsed_timedelta.total_seconds()
                if elapsed_seconds > timeout:
                    break

            if n_trials is not None:
                if n_enqueued_trials >= n_trials:
                    break

            try:
                que.put_nowait(True)
                n_enqueued_trials += 1
            except queue.Full:
                time.sleep(1)

        for _ in range(n_jobs):
            que.put(False)

        collections.deque(imap_ite, maxlen=0)  # Consume the iterator to wait for all threads.
        pool.terminate()
        que.close()
        que.join_thread()

    def _run_trial_and_callbacks(
            self,
            func,  # type: ObjectiveFuncType
            catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            callbacks,  # type: Optional[List[Callable[[Study, structs.FrozenTrial], None]]]
            gc_after_trial  # type: bool
    ):
        # type: (...) -> None

        trial = self._run_trial(func, catch, gc_after_trial)
        if callbacks is not None:
            frozen_trial = self._storage.get_trial(trial._trial_id)
            for callback in callbacks:
                callback(self, frozen_trial)

    def _run_trial(
            self,
            func,  # type: ObjectiveFuncType
            catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
            gc_after_trial  # type: bool
    ):
        # type: (...) -> trial_module.Trial

        trial_id = self._storage.create_new_trial(self.study_id)
        trial = trial_module.Trial(self, trial_id)
        trial_number = trial.number

        try:
            result = func(trial)
        except structs.TrialPruned as e:
            message = 'Setting status of trial#{} as {}. {}'.format(trial_number,
                                                                    structs.TrialState.PRUNED,
                                                                    str(e))
            self.logger.info(message)
            self._storage.set_trial_state(trial_id, structs.TrialState.PRUNED)
            return trial
        except Exception as e:
            message = 'Setting status of trial#{} as {} because of the following error: {}'\
                .format(trial_number, structs.TrialState.FAIL, repr(e))
            self.logger.warning(message, exc_info=True)
            self._storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            self._storage.set_trial_state(trial_id, structs.TrialState.FAIL)

            if isinstance(e, catch):
                return trial
            raise
        finally:
            # The following line mitigates memory problems that can be occurred in some
            # environments (e.g., services that use computing containers such as CircleCI).
            # Please refer to the following PR for further details:
            # https://github.com/pfnet/optuna/pull/325.
            if gc_after_trial:
                gc.collect()

        try:
            result = float(result)
        except (
                ValueError,
                TypeError,
        ):
            message = 'Setting status of trial#{} as {} because the returned value from the ' \
                      'objective function cannot be casted to float. Returned value is: ' \
                      '{}'.format(trial_number, structs.TrialState.FAIL, repr(result))
            self.logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            self._storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            return trial

        if math.isnan(result):
            message = 'Setting status of trial#{} as {} because the objective function ' \
                      'returned {}.'.format(trial_number, structs.TrialState.FAIL, result)
            self.logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            self._storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            return trial

        trial.report(result)
        self._storage.set_trial_state(trial_id, structs.TrialState.COMPLETE)
        self._log_completed_trial(trial_number, result)

        return trial

    def _log_completed_trial(self, trial_number, value):
        # type: (int, float) -> None

        self.logger.info('Finished trial#{} resulted in value: {}. '
                         'Current best value is {} with parameters: {}.'.format(
                             trial_number, value, self.best_value, self.best_params))


def create_study(
        storage=None,  # type: Union[None, str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
        study_name=None,  # type: Optional[str]
        direction='minimize',  # type: str
        load_if_exists=False,  # type: bool
):
    # type: (...) -> Study
    """Create a new :class:`~optuna.study.Study`.

    Args:
        storage:
            Database URL. If this argument is set to None, in-memory storage is used, and the
            :class:`~optuna.study.Study` will not be persistent.
        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used
            as the default. See also :class:`~optuna.samplers`.
        pruner:
            A pruner object that decides early stopping of unpromising trials. See also
            :class:`~optuna.pruners`.
        study_name:
            Study's name. If this argument is set to None, a unique name is generated
            automatically.
        direction:
            Direction of optimization. Set ``minimize`` for minimization and ``maximize`` for
            maximization.
        load_if_exists:
            Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.structs.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.

    Returns:
        A :class:`~optuna.study.Study` object.

    """

    storage = storages.get_storage(storage)
    try:
        study_id = storage.create_new_study(study_name)
    except structs.DuplicatedStudyError:
        if load_if_exists:
            assert study_name is not None

            logger = logging.get_logger(__name__)
            logger.info("Using an existing study with name '{}' instead of "
                        "creating a new one.".format(study_name))
            study_id = storage.get_study_id_from_name(study_name)
        else:
            raise

    study_name = storage.get_study_name_from_id(study_id)
    study = Study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner)

    if direction == 'minimize':
        _direction = structs.StudyDirection.MINIMIZE
    elif direction == 'maximize':
        _direction = structs.StudyDirection.MAXIMIZE
    else:
        raise ValueError('Please set either \'minimize\' or \'maximize\' to direction.')

    study._storage.set_study_direction(study_id, _direction)

    return study


def load_study(
        study_name,  # type: str
        storage,  # type: Union[str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
):
    # type: (...) -> Study
    """Load the existing :class:`~optuna.study.Study` that has the specified name.

    Args:
        study_name:
            Study's name. Each study has a unique name as an identifier.
        storage:
            Database URL such as ``sqlite:///example.db``. Optuna internally uses `SQLAlchemy
            <https://www.sqlalchemy.org/>`_ to handle databases. Please refer to `SQLAlchemy's
            document <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ for
            further details.
        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used
            as the default. See also :class:`~optuna.samplers`.
        pruner:
            A pruner object that decides early stopping of unpromising trials.
            If :obj:`None` is specified, :class:`~optuna.pruners.MedianPruner` is used
            as the default. See also :class:`~optuna.pruners`.

    """

    return Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)


def delete_study(
        study_name,  # type: str
        storage,  # type: Union[str, storages.BaseStorage]
):
    # type: (...) -> None
    """Delete a :class:`~optuna.study.Study` object.

    Args:
        study_name:
            Study's name.
        storage:
            Database URL such as ``sqlite:///example.db``. Optuna internally uses `SQLAlchemy
            <https://www.sqlalchemy.org/>`_ to handle databases. Please refer to `SQLAlchemy's
            document <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ for
            further details.

    """

    storage = storages.get_storage(storage)
    study_id = storage.get_study_id_from_name(study_name)
    storage.delete_study(study_id)


def get_all_study_summaries(storage):
    # type: (Union[str, storages.BaseStorage]) -> List[structs.StudySummary]
    """Get all history of studies stored in a specified storage.

    Args:
        storage:
            Database URL.

    Returns:
        List of study history summarized as :class:`~optuna.structs.StudySummary` objects.

    """

    storage = storages.get_storage(storage)
    return storage.get_all_study_summaries()


def _check_pandas_availability():
    # type: () -> None

    if not _pandas_available:
        raise ImportError(
            'pandas is not available. Please install pandas to use this feature. '
            'pandas can be installed by executing `$ pip install pandas`. '
            'For further information, please refer to the installation guide of pandas. '
            '(The actual import error is as follows: ' + str(_pandas_import_error) + ')')
