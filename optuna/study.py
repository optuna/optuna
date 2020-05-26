import collections
import copy
import datetime
import gc
import math
import threading
import warnings

import joblib
from joblib import delayed
from joblib import Parallel

from optuna._experimental import experimental
from optuna._study_direction import StudyDirection
from optuna._study_summary import StudySummary  # NOQA

try:
    import pandas as pd  # NOQA

    _pandas_available = True
except ImportError as e:
    _pandas_import_error = e
    # trials_dataframe is disabled because pandas is not available.
    _pandas_available = False

from optuna import exceptions
from optuna import logging
from optuna import progress_bar as pbar_module
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import trial as trial_module
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
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


_logger = logging.get_logger(__name__)


class BaseStudy(object):
    def __init__(self, study_id, storage):
        # type: (int, storages.BaseStorage) -> None

        self._study_id = study_id
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
        # type: () -> FrozenTrial
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.FrozenTrial` object of the best trial.
        """

        return copy.deepcopy(self._storage.get_best_trial(self._study_id))

    @property
    def direction(self):
        # type: () -> StudyDirection
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.study.StudyDirection` object.
        """

        return self._storage.get_study_direction(self._study_id)

    @property
    def trials(self):
        # type: () -> List[FrozenTrial]
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True)``.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self.get_trials()

    def get_trials(self, deepcopy=True):
        # type: (bool) -> List[FrozenTrial]
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        For library users, it's recommended to use more handy
        :attr:`~optuna.study.Study.trials` property to get the trials instead.

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
                Note that if you set the flag to :obj:`False`, you shouldn't mutate
                any fields of the returned trial. Otherwise the internal state of
                the study may corrupt and unexpected behavior may happen.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self._storage.get_all_trials(self._study_id, deepcopy=deepcopy)

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

        warnings.warn(
            "The direct use of storage is deprecated. "
            "Please access to storage via study's public methods "
            "(e.g., `Study.set_user_attr`)",
            DeprecationWarning,
        )

        _logger.warning(
            "The direct use of storage is deprecated. "
            "Please access to storage via study's public methods "
            "(e.g., `Study.set_user_attr`)"
        )

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
        pruner=None,  # type: pruners.BasePruner
    ):
        # type: (...) -> None

        self.study_name = study_name
        storage = storages.get_storage(storage)
        study_id = storage.get_study_id_from_name(study_name)
        super(Study, self).__init__(study_id, storage)

        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self._optimize_lock = threading.Lock()
        self._stop_flag = False

    def __getstate__(self):
        # type: () -> Dict[Any, Any]

        state = self.__dict__.copy()
        del state["_optimize_lock"]
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None

        self.__dict__.update(state)
        self._optimize_lock = threading.Lock()

    @property
    def study_id(self):
        # type: () -> int
        """Return the study ID.

        .. deprecated:: 0.20.0
            The direct use of this attribute is deprecated and it is recommended that you use
            :attr:`~optuna.study.Study.study_name` instead.

        Returns:
            The study ID.
        """

        message = (
            "The use of `Study.study_id` is deprecated. Please use `Study.study_name` instead."
        )
        warnings.warn(message, DeprecationWarning)
        _logger.warning(message)

        return self._study_id

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return copy.deepcopy(self._storage.get_study_user_attrs(self._study_id))

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return copy.deepcopy(self._storage.get_study_system_attrs(self._study_id))

    def optimize(
        self,
        func,  # type: ObjectiveFuncType
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        n_jobs=1,  # type: int
        catch=(),  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        callbacks=None,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial=True,  # type: bool
        show_progress_bar=False,  # type: bool
    ):
        # type: (...) -> None
        """Optimize an objective function.

        Optimization is done by choosing a suitable set of hyperparameter values from a given
        range. Uses a sampler which implements the task of value suggestion based on a specified
        distribution. The sampler is specified in :func:`~optuna.study.create_study` and the
        default choice for the sampler is TPE.
        See also :class:`~optuna.samplers.TPESampler` for more details on 'TPE'.

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
                set to CPU count.
            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
                :class:`~optuna.study.Study` and :class:`~optuna.FrozenTrial`.
            gc_after_trial:
                Flag to execute garbage collection at the end of each trial. By default, garbage
                collection is enabled, just in case. You can turn it off with this argument if
                memory is safely managed in your objective function.
            show_progress_bar:
                Flag to show progress bars or not. To disable progress bar, set this ``False``.
                Currently, progress bar is experimental feature and disabled
                when ``n_jobs`` :math:`\\ne 1`.
        """

        if not isinstance(catch, tuple):
            raise TypeError(
                "The catch argument is of type '{}' but must be a tuple.".format(
                    type(catch).__name__
                )
            )

        if not self._optimize_lock.acquire(False):
            raise RuntimeError("Nested invocation of `Study.optimize` method isn't allowed.")

        # TODO(crcrpar): Make progress bar work when n_jobs != 1.
        self._progress_bar = pbar_module._ProgressBar(
            show_progress_bar and n_jobs == 1, n_trials, timeout
        )

        self._stop_flag = False

        try:
            if n_jobs == 1:
                self._optimize_sequential(
                    func, n_trials, timeout, catch, callbacks, gc_after_trial, None
                )
            else:
                if show_progress_bar:
                    msg = "Progress bar only supports serial execution (`n_jobs=1`)."
                    warnings.warn(msg)
                    _logger.warning(msg)

                time_start = datetime.datetime.now()

                def _should_stop() -> bool:
                    if self._stop_flag:
                        return True

                    if timeout is not None:
                        # This is needed for mypy.
                        t = timeout  # type: float
                        return (datetime.datetime.now() - time_start).total_seconds() > t

                    return False

                if n_trials is not None:
                    _iter = iter(range(n_trials))
                else:
                    _iter = iter(_should_stop, True)

                with Parallel(n_jobs=n_jobs, prefer="threads") as parallel:
                    if not isinstance(
                        parallel._backend, joblib.parallel.ThreadingBackend
                    ) and isinstance(self._storage, storages.InMemoryStorage):
                        msg = (
                            "The default storage cannot be shared by multiple processes. "
                            "Please use an RDB (RDBStorage) when you use joblib for "
                            "multi-processing. The usage of RDBStorage can be found in "
                            "https://optuna.readthedocs.io/en/stable/tutorial/rdb.html."
                        )
                        warnings.warn(msg, UserWarning)
                        _logger.warning(msg)

                    parallel(
                        delayed(self._reseed_and_optimize_sequential)(
                            func, 1, timeout, catch, callbacks, gc_after_trial, time_start
                        )
                        for _ in _iter
                    )
        finally:
            self._optimize_lock.release()
            self._progress_bar.close()
            del self._progress_bar

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a user attribute to the study.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_user_attr(self._study_id, key, value)

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None
        """Set a system attribute to the study.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.study.Study.set_user_attr` to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_system_attr(self._study_id, key, value)

    def trials_dataframe(
        self,
        attrs=(
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),  # type: Tuple[str, ...]
        multi_index=False,  # type: bool
    ):
        # type: (...) -> pd.DataFrame
        """Export trials as a pandas DataFrame_.

        The DataFrame_ provides various features to analyze studies. It is also useful to draw a
        histogram of objective values and to export trials as a CSV file.
        If there are no trials, an empty DataFrame_ is returned.

        Example:

            .. testcode::

                import optuna
                import pandas

                def objective(trial):
                    x = trial.suggest_uniform('x', -1, 1)
                    return x ** 2

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                # Create a dataframe from the study.
                df = study.trials_dataframe()
                assert isinstance(df, pandas.DataFrame)
                assert df.shape[0] == 3  # n_trials.

        Args:
            attrs:
                Specifies field names of :class:`~optuna.FrozenTrial` to include them to a
                DataFrame of trials.
            multi_index:
                Specifies whether the returned DataFrame_ employs MultiIndex_ or not. Columns that
                are hierarchical by nature such as ``(params, x)`` will be flattened to
                ``params_x`` when set to :obj:`False`.

        Returns:
            A pandas DataFrame_ of trials in the :class:`~optuna.study.Study`.

        .. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
        .. _MultiIndex: https://pandas.pydata.org/pandas-docs/stable/advanced.html
        """

        _check_pandas_availability()

        trials = self.get_trials(deepcopy=False)

        # If no trials, return an empty dataframe.
        if not len(trials):
            return pd.DataFrame()

        assert all(isinstance(trial, FrozenTrial) for trial in trials)
        attrs_to_df_columns = collections.OrderedDict()  # type: Dict[str, str]
        for attr in attrs:
            if attr.startswith("_"):
                # Python conventional underscores are omitted in the dataframe.
                df_column = attr[1:]
            else:
                df_column = attr
            attrs_to_df_columns[attr] = df_column

        # column_agg is an aggregator of column names.
        # Keys of column agg are attributes of `FrozenTrial` such as 'trial_id' and 'params'.
        # Values are dataframe columns such as ('trial_id', '') and ('params', 'n_layers').
        column_agg = collections.defaultdict(set)  # type: Dict[str, Set]
        non_nested_attr = ""

        def _create_record_and_aggregate_column(trial):
            # type: (FrozenTrial) -> Dict[Tuple[str, str], Any]

            record = {}
            for attr, df_column in attrs_to_df_columns.items():
                value = getattr(trial, attr)
                if isinstance(value, TrialState):
                    # Convert TrialState to str and remove the common prefix.
                    value = str(value).split(".")[-1]
                if isinstance(value, dict):
                    for nested_attr, nested_value in value.items():
                        record[(df_column, nested_attr)] = nested_value
                        column_agg[attr].add((df_column, nested_attr))
                else:
                    record[(df_column, non_nested_attr)] = value
                    column_agg[attr].add((df_column, non_nested_attr))
            return record

        records = list([_create_record_and_aggregate_column(trial) for trial in trials])

        columns = sum(
            (sorted(column_agg[k]) for k in attrs if k in column_agg), []
        )  # type: List[Tuple[str, str]]

        df = pd.DataFrame(records, columns=pd.MultiIndex.from_tuples(columns))

        if not multi_index:
            # Flatten the `MultiIndex` columns where names are concatenated with underscores.
            # Filtering is required to omit non-nested columns avoiding unwanted trailing
            # underscores.
            df.columns = [
                "_".join(filter(lambda c: c, map(lambda c: str(c), col))) for col in columns
            ]

        return df

    @experimental("1.4.0")
    def stop(self) -> None:

        """Exit from the current optimization loop after the running trials finish.

        This method lets the running :meth:`~optuna.study.Study.optimize` method return
        immediately after all trials which the :meth:`~optuna.study.Study.optimize` method
        spawned finishes.
        This method does not affect any behaviors of parallel or successive study processes.

        Raises:
            RuntimeError:
                If this method is called outside an objective function or callback.
        """

        if self._optimize_lock.acquire(False):
            self._optimize_lock.release()
            raise RuntimeError(
                "`Study.stop` is supposed to be invoked inside an objective function or a "
                "callback."
            )

        self._stop_flag = True

    @experimental("1.2.0")
    def enqueue_trial(self, params):
        # type: (Dict[str, Any]) -> None
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters which will be evaluated in your
        objective function.

        Example:

            .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform('x', 0, 10)
                    return x ** 2

                study = optuna.create_study()
                study.enqueue_trial({'x': 5})
                study.enqueue_trial({'x': 0})
                study.optimize(objective, n_trials=2)

                assert study.trials[0].params == {'x': 5}
                assert study.trials[1].params == {'x': 0}

        Args:
            params:
                Parameter values to pass your objective function.
        """

        system_attrs = {"fixed_params": params}
        self._append_trial(state=TrialState.WAITING, system_attrs=system_attrs)

    def _append_trial(
        self,
        value=None,  # type: Optional[float]
        params=None,  # type: Optional[Dict[str, Any]]
        distributions=None,  # type: Optional[Dict[str, BaseDistribution]]
        user_attrs=None,  # type: Optional[Dict[str, Any]]
        system_attrs=None,  # type: Optional[Dict[str, Any]]
        intermediate_values=None,  # type: Optional[Dict[int, float]]
        state=TrialState.COMPLETE,  # type: TrialState
        datetime_start=None,  # type: Optional[datetime.datetime]
        datetime_complete=None,  # type: Optional[datetime.datetime]
    ):
        # type: (...) -> int

        params = params or {}
        distributions = distributions or {}
        user_attrs = user_attrs or {}
        system_attrs = system_attrs or {}
        intermediate_values = intermediate_values or {}
        datetime_start = datetime_start or datetime.datetime.now()

        if state.is_finished():
            datetime_complete = datetime_complete or datetime.datetime.now()

        trial = FrozenTrial(
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
            intermediate_values=intermediate_values,
        )

        trial._validate()

        trial_id = self._storage.create_new_trial(self._study_id, template_trial=trial)
        return trial_id

    def _reseed_and_optimize_sequential(
        self,
        func,  # type: ObjectiveFuncType
        n_trials,  # type: Optional[int]
        timeout,  # type: Optional[float]
        catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        callbacks,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial,  # type: bool
        time_start,  # type: Optional[datetime.datetime]
    ):
        # type: (...) -> None

        self.sampler.reseed_rng()
        self._optimize_sequential(
            func, n_trials, timeout, catch, callbacks, gc_after_trial, time_start
        )

    def _optimize_sequential(
        self,
        func,  # type: ObjectiveFuncType
        n_trials,  # type: Optional[int]
        timeout,  # type: Optional[float]
        catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        callbacks,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial,  # type: bool
        time_start,  # type: Optional[datetime.datetime]
    ):
        # type: (...) -> None

        i_trial = 0

        if time_start is None:
            time_start = datetime.datetime.now()

        while True:
            if self._stop_flag:
                break

            if n_trials is not None:
                if i_trial >= n_trials:
                    break
                i_trial += 1

            if timeout is not None:
                elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
                if elapsed_seconds >= timeout:
                    break

            self._run_trial_and_callbacks(func, catch, callbacks, gc_after_trial)

            self._progress_bar.update((datetime.datetime.now() - time_start).total_seconds())

        self._storage.remove_session()

    def _pop_waiting_trial_id(self):
        # type: () -> Optional[int]

        # TODO(c-bata): Reduce database query counts for extracting waiting trials.
        for trial in self.get_trials(deepcopy=False):
            if trial.state != TrialState.WAITING:
                continue

            if not self._storage.set_trial_state(trial._trial_id, TrialState.RUNNING):
                continue

            _logger.debug("Trial#{} is popped from the trial queue.".format(trial.number))
            return trial._trial_id

        return None

    def _run_trial_and_callbacks(
        self,
        func,  # type: ObjectiveFuncType
        catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        callbacks,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial,  # type: bool
    ):
        # type: (...) -> None

        trial = self._run_trial(func, catch, gc_after_trial)
        if callbacks is not None:
            frozen_trial = copy.deepcopy(self._storage.get_trial(trial._trial_id))
            for callback in callbacks:
                callback(self, frozen_trial)

    def _run_trial(
        self,
        func,  # type: ObjectiveFuncType
        catch,  # type: Union[Tuple[()], Tuple[Type[Exception]]]
        gc_after_trial,  # type: bool
    ):
        # type: (...) -> trial_module.Trial

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = trial_module.Trial(self, trial_id)
        trial_number = trial.number

        try:
            result = func(trial)
        except exceptions.TrialPruned as e:
            message = "Setting status of trial#{} as {}. {}".format(
                trial_number, TrialState.PRUNED, str(e)
            )
            _logger.info(message)

            # Register the last intermediate value if present as the value of the trial.
            # TODO(hvy): Whether a pruned trials should have an actual value can be discussed.
            frozen_trial = self._storage.get_trial(trial_id)
            last_step = frozen_trial.last_step
            if last_step is not None:
                self._storage.set_trial_value(
                    trial_id, frozen_trial.intermediate_values[last_step]
                )
            self._storage.set_trial_state(trial_id, TrialState.PRUNED)
            return trial
        except Exception as e:
            message = "Setting status of trial#{} as {} because of the following error: {}".format(
                trial_number, TrialState.FAIL, repr(e)
            )
            _logger.warning(message, exc_info=True)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)

            if isinstance(e, catch):
                return trial
            raise
        finally:
            # The following line mitigates memory problems that can be occurred in some
            # environments (e.g., services that use computing containers such as CircleCI).
            # Please refer to the following PR for further details:
            # https://github.com/optuna/optuna/pull/325.
            if gc_after_trial:
                gc.collect()

        try:
            result = float(result)
        except (
            ValueError,
            TypeError,
        ):
            message = (
                "Setting status of trial#{} as {} because the returned value from the "
                "objective function cannot be casted to float. Returned value is: "
                "{}".format(trial_number, TrialState.FAIL, repr(result))
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)
            return trial

        if math.isnan(result):
            message = (
                "Setting status of trial#{} as {} because the objective function "
                "returned {}.".format(trial_number, TrialState.FAIL, result)
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)
            return trial

        self._storage.set_trial_value(trial_id, result)
        self._storage.set_trial_state(trial_id, TrialState.COMPLETE)
        self._log_completed_trial(trial, result)

        return trial

    def _log_completed_trial(self, trial, result):
        # type: (trial_module.Trial, float) -> None

        _logger.info(
            "Finished trial#{} with value: {} with parameters: {}. "
            "Best is trial#{} with value: {}.".format(
                trial.number, result, trial.params, self.best_trial.number, self.best_value
            )
        )


def create_study(
    storage=None,  # type: Union[None, str, storages.BaseStorage]
    sampler=None,  # type: samplers.BaseSampler
    pruner=None,  # type: pruners.BasePruner
    study_name=None,  # type: Optional[str]
    direction="minimize",  # type: str
    load_if_exists=False,  # type: bool
):
    # type: (...) -> Study
    """Create a new :class:`~optuna.study.Study`.

    Args:
        storage:
            Database URL. If this argument is set to None, in-memory storage is used, and the
            :class:`~optuna.study.Study` will not be persistent.

            .. note::
                When a database URL is passed, Optuna internally uses `SQLAlchemy`_ to handle
                the database. Please refer to `SQLAlchemy's document`_ for further details.
                If you want to specify non-default options to `SQLAlchemy Engine`_, you can
                instantiate :class:`~optuna.storages.RDBStorage` with your desired options and
                pass it to the ``storage`` argument instead of a URL.

             .. _SQLAlchemy: https://www.sqlalchemy.org/
             .. _SQLAlchemy's document:
                 https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls
             .. _SQLAlchemy Engine: https://docs.sqlalchemy.org/en/latest/core/engines.html

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
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.

    Returns:
        A :class:`~optuna.study.Study` object.

    """

    storage = storages.get_storage(storage)
    try:
        study_id = storage.create_new_study(study_name)
    except exceptions.DuplicatedStudyError:
        if load_if_exists:
            assert study_name is not None

            _logger.info(
                "Using an existing study with name '{}' instead of "
                "creating a new one.".format(study_name)
            )
            study_id = storage.get_study_id_from_name(study_name)
        else:
            raise

    study_name = storage.get_study_name_from_id(study_id)
    study = Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)

    if direction == "minimize":
        _direction = StudyDirection.MINIMIZE
    elif direction == "maximize":
        _direction = StudyDirection.MAXIMIZE
    else:
        raise ValueError("Please set either 'minimize' or 'maximize' to direction.")

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
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.
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
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.

    """

    storage = storages.get_storage(storage)
    study_id = storage.get_study_id_from_name(study_name)
    storage.delete_study(study_id)


def get_all_study_summaries(storage):
    # type: (Union[str, storages.BaseStorage]) -> List[StudySummary]
    """Get all history of studies stored in a specified storage.

    Args:
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.

    Returns:
        List of study history summarized as :class:`~optuna.study.StudySummary` objects.

    """

    storage = storages.get_storage(storage)
    return storage.get_all_study_summaries()


def _check_pandas_availability():
    # type: () -> None

    if not _pandas_available:
        raise ImportError(
            "pandas is not available. Please install pandas to use this feature. "
            "pandas can be installed by executing `$ pip install pandas`. "
            "For further information, please refer to the installation guide of pandas. "
            "(The actual import error is as follows: " + str(_pandas_import_error) + ")"
        )
