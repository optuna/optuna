import abc
import collections
import datetime
import gc
import math
import multiprocessing
import multiprocessing.pool
import pandas as pd
import six
from six.moves import queue
import time

from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import structs
from optuna import trial as trial_module
from optuna import types

if types.TYPE_CHECKING:
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


@six.add_metaclass(abc.ABCMeta)
class BaseStudy(object):
    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the :class:`~optuna.study.BaseStudy`.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.params

    @property
    def best_value(self):
        # type: () -> float
        """Return the best objective value in the :class:`~optuna.study.BaseStudy`.

        Returns:
            A float representing the best objective value.
        """

        best_value = self.best_trial.value
        if best_value is None:
            raise ValueError('No trials are completed yet.')

        return best_value

    @property
    @abc.abstractmethod
    def best_trial(self):
        # type: () -> structs.FrozenTrial
        """Return the best trial in the :class:`~optuna.study.BaseStudy`.

        Returns:
            A :class:`~optuna.structs.FrozenTrial` object of the best trial.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def direction(self):
        # type: () -> structs.StudyDirection
        """Return the direction of the :class:`~optuna.study.BaseStudy`.

        Returns:
            A :class:`~optuna.structs.StudyDirection` object.
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def trials(self):
        # type: () -> List[structs.FrozenTrial]
        """Return all trials in the :class:`~optuna.study.BaseStudy`.

        Returns:
            A list of :class:`~optuna.structs.FrozenTrial` objects.
        """

        raise NotImplementedError


class Study(BaseStudy):
    """A study corresponds to an optimization task, i.e., a set of trials.

    Note that the direct use of this constructor is not recommended.

    This object provides interfaces to run a new :class:`~optuna.trial.Trial`, access trials'
    history, set/get user-defined attributes of the study itself.

    For creating and loading studies, please refer to the documentation of
    :func:`~optuna.study.create_study` and :func:`~optuna.study.load_study` respectively.

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

    def __init__(
            self,
            study_name,  # type: str
            storage,  # type: Union[str, storages.BaseStorage]
            sampler=None,  # type: samplers.BaseSampler
            pruner=None,  # type: pruners.BasePruner
    ):
        # type: (...) -> None

        self.study_name = study_name
        self.storage = storages.get_storage(storage)
        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self.study_id = self.storage.get_study_id_from_name(study_name)
        self.logger = logging.get_logger(__name__)

    def __getstate__(self):
        # type: () -> Dict[Any, Any]

        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None

        self.__dict__.update(state)
        self.logger = logging.get_logger(__name__)

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the :class:`~optuna.study.Study`.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.params

    @property
    def best_value(self):
        # type: () -> float
        """Return the best objective value in the :class:`~optuna.study.Study`.

        Returns:
            A float representing the best objective value.
        """

        best_value = self.best_trial.value
        if best_value is None:
            raise ValueError('No trials are completed yet.')

        return best_value

    @property
    def best_trial(self):
        # type: () -> structs.FrozenTrial
        """Return the best trial in the :class:`~optuna.study.Study`.

        Returns:
            A :class:`~optuna.structs.FrozenTrial` object of the best trial.
        """

        return self.storage.get_best_trial(self.study_id)

    @property
    def direction(self):
        # type: () -> structs.StudyDirection
        """Return the direction of the :class:`~optuna.study.Study`.

        Returns:
            A :class:`~optuna.structs.StudyDirection` object.
        """

        return self.storage.get_study_direction(self.study_id)

    @property
    def trials(self):
        # type: () -> List[structs.FrozenTrial]
        """Return all trials in the :class:`~optuna.study.Study`.

        Returns:
            A list of :class:`~optuna.structs.FrozenTrial` objects.
        """

        return self.storage.get_all_trials(self.study_id)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self.storage.get_study_user_attrs(self.study_id)

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self.storage.get_study_system_attrs(self.study_id)

    def optimize(
            self,
            func,  # type: ObjectiveFuncType
            n_trials=None,  # type: Optional[int]
            timeout=None,  # type: Optional[float]
            n_jobs=1,  # type: int
            catch=(Exception, )  # type: Union[Tuple[()], Tuple[Type[Exception]]]
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
                A study continues to run even when a trial raises one of exceptions specified in
                this argument. Default is (`Exception <https://docs.python.org/3/library/
                exceptions.html#Exception>`_,), where all non-exit exceptions are handled
                by this logic.

        """

        if n_jobs == 1:
            self._optimize_sequential(func, n_trials, timeout, catch)
        else:
            self._optimize_parallel(func, n_trials, timeout, n_jobs, catch)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a user attribute to the :class:`~optuna.study.Study`.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self.storage.set_study_user_attr(self.study_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a system attribute to the :class:`~optuna.study.Study`.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.study.Study.set_user_attr` to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self.storage.set_study_system_attr(self.study_id, key, value)

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

    def _optimize_sequential(
            self,
            func,  # type: ObjectiveFuncType
            n_trials,  # type: Optional[int]
            timeout,  # type: Optional[float]
            catch  # type: Union[Tuple[()], Tuple[Type[Exception]]]
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

            self._run_trial(func, catch)

    def _optimize_parallel(
            self,
            func,  # type: ObjectiveFuncType
            n_trials,  # type: Optional[int]
            timeout,  # type: Optional[float]
            n_jobs,  # type: int
            catch  # type: Union[Tuple[()], Tuple[Type[Exception]]]
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
                self._run_trial(func, catch)
            self.storage.remove_session()

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

    def _run_trial(self, func, catch):
        # type: (ObjectiveFuncType, Union[Tuple[()], Tuple[Type[Exception]]]) -> trial_module.Trial

        trial_id = self.storage.create_new_trial_id(self.study_id)
        trial = trial_module.Trial(self, trial_id)
        trial_number = trial.number

        try:
            result = func(trial)
        except structs.TrialPruned as e:
            message = 'Setting status of trial#{} as {}. {}'.format(trial_number,
                                                                    structs.TrialState.PRUNED,
                                                                    str(e))
            self.logger.info(message)
            self.storage.set_trial_state(trial_id, structs.TrialState.PRUNED)
            return trial
        except catch as e:
            message = 'Setting status of trial#{} as {} because of the following error: {}'\
                .format(trial_number, structs.TrialState.FAIL, repr(e))
            self.logger.warning(message, exc_info=True)
            self.storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            self.storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            return trial
        finally:
            # The following line mitigates memory problems that can be occurred in some
            # environments (e.g., services that use computing containers such as CircleCI).
            # Please refer to the following PR for further details:
            # https://github.com/pfnet/optuna/pull/325.
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
            self.storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            self.storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            return trial

        if math.isnan(result):
            message = 'Setting status of trial#{} as {} because the objective function ' \
                      'returned {}.'.format(trial_number, structs.TrialState.FAIL, result)
            self.logger.warning(message)
            self.storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            self.storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            return trial

        trial.report(result)
        self.storage.set_trial_state(trial_id, structs.TrialState.COMPLETE)
        self._log_completed_trial(trial_number, result)

        return trial

    def _log_completed_trial(self, trial_number, value):
        # type: (int, float) -> None

        self.logger.info('Finished trial#{} resulted in value: {}. '
                         'Current best value is {} with parameters: {}.'.format(
                             trial_number, value, self.best_value, self.best_params))


class InTrialStudy(BaseStudy):
    """An object to access a study instance inside a trial instance safely.

    To prevent unexpected recursive calls of :func:`~optuna.study.Study.optimize()`, this class
    does not allow to call :func:`~optuna.study.InTrialStudy.optimize()` unlike
    :class:`~optuna.study.Study`.

    Note that this object is created within Optuna library, so
    it is not intended that library users directly use this constructor.

    Args:
        study:
            A :class:`~optuna.study.Study` object.

    """

    def __init__(self, study):
        # type: (Study) -> None

        self.study_name = study.study_name
        self.study_id = study.study_id
        self.storage = study.storage

    @property
    def best_trial(self):
        # type: () -> structs.FrozenTrial
        """Return the best trial in the :class:`~optuna.study.InTrialStudy`.

        Returns:
            A :class:`~optuna.structs.FrozenTrial` object of the best trial.
        """

        return self.storage.get_best_trial(self.study_id)

    @property
    def direction(self):
        # type: () -> structs.StudyDirection
        """Return the direction of the :class:`~optuna.study.InTrialStudy`.

        Returns:
            A :class:`~optuna.structs.StudyDirection` object.
        """

        return self.storage.get_study_direction(self.study_id)

    @property
    def trials(self):
        # type: () -> List[structs.FrozenTrial]
        """Return all trials in the :class:`~optuna.study.InTrialStudy`.

        Returns:
            A list of :class:`~optuna.structs.FrozenTrial` objects.
        """

        return self.storage.get_all_trials(self.study_id)


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
            A sampler object that implements background algorithm for value suggestion. See also
            :class:`~optuna.samplers`.
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
        study_id = storage.create_new_study_id(study_name)
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

    study.storage.set_study_direction(study_id, _direction)

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
