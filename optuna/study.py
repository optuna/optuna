import collections
import copy
import datetime
import gc
import math
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union
import warnings

import joblib
from joblib import delayed
from joblib import Parallel

from optuna._experimental import experimental
from optuna._imports import try_import
from optuna._study_direction import StudyDirection
from optuna._study_summary import StudySummary  # NOQA
from optuna import exceptions
from optuna import logging
from optuna import progress_bar as pbar_module
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import trial as trial_module
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

ObjectiveFuncType = Callable[[trial_module.Trial], float]

with try_import() as _pandas_imports:
    # `trials_dataframe` is disabled if pandas is not available.
    import pandas as pd  # NOQA

_logger = logging.get_logger(__name__)


class BaseStudy(object):
    def __init__(self, study_id: int, storage: storages.BaseStorage) -> None:

        self._study_id = study_id
        self._storage = storage

    @property
    def best_params(self) -> Dict[str, Any]:
        """Return parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self.best_trial.params

    @property
    def best_value(self) -> float:
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.
        """

        best_value = self.best_trial.value
        assert best_value is not None

        return best_value

    @property
    def best_trial(self) -> FrozenTrial:
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.FrozenTrial` object of the best trial.
        """

        return copy.deepcopy(self._storage.get_best_trial(self._study_id))

    @property
    def direction(self) -> StudyDirection:
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.study.StudyDirection` object.
        """

        return self._storage.get_study_direction(self._study_id)

    @property
    def trials(self) -> List[FrozenTrial]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True)``.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self.get_trials()

    def get_trials(self, deepcopy: bool = True) -> List[FrozenTrial]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        For library users, it's recommended to use more handy
        :attr:`~optuna.study.Study.trials` property to get the trials instead.

        Example:
            .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform("x", -1, 1)
                    return x ** 2

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                trials = study.get_trials()
                assert len(trials) == 3
        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
                Note that if you set the flag to :obj:`False`, you shouldn't mutate
                any fields of the returned trial. Otherwise the internal state of
                the study may corrupt and unexpected behavior may happen.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        self._storage.read_trials_from_remote_storage(self._study_id)
        return self._storage.get_all_trials(self._study_id, deepcopy=deepcopy)


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
        study_name: str,
        storage: Union[str, storages.BaseStorage],
        sampler: Optional["samplers.BaseSampler"] = None,
        pruner: Optional[pruners.BasePruner] = None,
    ) -> None:

        self.study_name = study_name
        storage = storages.get_storage(storage)
        study_id = storage.get_study_id_from_name(study_name)
        super(Study, self).__init__(study_id, storage)

        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self._optimize_lock = threading.Lock()
        self._stop_flag = False

    def __getstate__(self) -> Dict[Any, Any]:

        state = self.__dict__.copy()
        del state["_optimize_lock"]
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:

        self.__dict__.update(state)
        self._optimize_lock = threading.Lock()

    @property
    def user_attrs(self) -> Dict[str, Any]:
        """Return user attributes.

        .. seealso::

            See :func:`~optuna.study.Study.set_user_attr` for related method.

        Example:

            .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_float("x", 0, 1)
                    y = trial.suggest_float("y", 0, 1)
                    return x ** 2 + y ** 2

                study = optuna.create_study()

                study.set_user_attr("objective function", "quadratic function")
                study.set_user_attr("dimensions", 2)
                study.set_user_attr("contributors", ["Akiba", "Sano"])

                assert study.user_attrs == {
                    "objective function": "quadratic function",
                    "dimensions": 2,
                    "contributors": ["Akiba", "Sano"]
                }

        Returns:
            A dictionary containing all user attributes.
        """

        return copy.deepcopy(self._storage.get_study_user_attrs(self._study_id))

    @property
    def system_attrs(self) -> Dict[str, Any]:
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return copy.deepcopy(self._storage.get_study_system_attrs(self._study_id))

    def optimize(
        self,
        func: ObjectiveFuncType,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[["Study", FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """Optimize an objective function.

        Optimization is done by choosing a suitable set of hyperparameter values from a given
        range. Uses a sampler which implements the task of value suggestion based on a specified
        distribution. The sampler is specified in :func:`~optuna.study.create_study` and the
        default choice for the sampler is TPE.
        See also :class:`~optuna.samplers.TPESampler` for more details on 'TPE'.

        Example:

            .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform("x", -1, 1)
                    return x ** 2

                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

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
                Flag to determine whether to automatically run garbage collection after each trial.
                Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling :func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to :obj:`True`.

                .. seealso::

                    :ref:`out-of-memory-gc-collect`

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

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute to the study.

        .. seealso::

            See :attr:`~optuna.study.Study.user_attrs` for related attribute.

        Example:

            .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_float("x", 0, 1)
                    y = trial.suggest_float("y", 0, 1)
                    return x ** 2 + y ** 2

                study = optuna.create_study()

                study.set_user_attr("objective function", "quadratic function")
                study.set_user_attr("dimensions", 2)
                study.set_user_attr("contributors", ["Akiba", "Sano"])

                assert study.user_attrs == {
                    "objective function": "quadratic function",
                    "dimensions": 2,
                    "contributors": ["Akiba", "Sano"]
                }

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._storage.set_study_user_attr(self._study_id, key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
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
        attrs: Tuple[str, ...] = (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),
        multi_index: bool = False,
    ) -> "pd.DataFrame":
        """Export trials as a pandas DataFrame_.

        The DataFrame_ provides various features to analyze studies. It is also useful to draw a
        histogram of objective values and to export trials as a CSV file.
        If there are no trials, an empty DataFrame_ is returned.

        Example:

            .. testcode::

                import optuna
                import pandas

                def objective(trial):
                    x = trial.suggest_uniform("x", -1, 1)
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

        _pandas_imports.check()

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

        def _create_record_and_aggregate_column(trial: FrozenTrial) -> Dict[Tuple[str, str], Any]:

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

    def stop(self) -> None:

        """Exit from the current optimization loop after the running trials finish.

        This method lets the running :meth:`~optuna.study.Study.optimize` method return
        immediately after all trials which the :meth:`~optuna.study.Study.optimize` method
        spawned finishes.
        This method does not affect any behaviors of parallel or successive study processes.

        Example:

            .. testcode::

                import optuna

                def objective(trial):
                    if trial.number == 4:
                        study.stop()
                    x = trial.suggest_uniform("x", 0, 10)
                    return x ** 2

                study = optuna.create_study()
                study.optimize(objective, n_trials=10)
                assert len(study.trials) == 5

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
    def enqueue_trial(self, params: Dict[str, Any]) -> None:
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters which will be evaluated in your
        objective function.

        Example:

            .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform("x", 0, 10)
                    return x ** 2

                study = optuna.create_study()
                study.enqueue_trial({"x": 5})
                study.enqueue_trial({"x": 0})
                study.optimize(objective, n_trials=2)

                assert study.trials[0].params == {"x": 5}
                assert study.trials[1].params == {"x": 0}

        Args:
            params:
                Parameter values to pass your objective function.
        """

        self.add_trial(
            create_trial(state=TrialState.WAITING, system_attrs={"fixed_params": params})
        )

    @experimental("2.0.0")
    def add_trial(self, trial: FrozenTrial) -> None:
        """Add trial to study.

        The trial is validated before being added.

        Example:

            .. testcode::

                import optuna
                from optuna.distributions import UniformDistribution

                def objective(trial):
                    x = trial.suggest_uniform("x", 0, 10)
                    return x ** 2

                study = optuna.create_study()
                assert len(study.trials) == 0

                trial = optuna.trial.create_trial(
                    params={"x": 2.0},
                    distributions={"x": UniformDistribution(0, 10)},
                    value=4.0,
                )

                study.add_trial(trial)
                assert len(study.trials) == 1

                study.optimize(objective, n_trials=3)
                assert len(study.trials) == 4

                other_study = optuna.create_study()

                for trial in study.trials:
                    other_study.add_trial(trial)
                assert len(other_study.trials) == len(study.trials)

                other_study.optimize(objective, n_trials=2)
                assert len(other_study.trials) == len(study.trials) + 2

        .. seealso::

            This method should in general be used to add already evaluated trials
            (``trial.state.is_finished() == True``). To queue trials for evaluation,
            please refer to :func:`~optuna.study.Study.enqueue_trial`.

        .. seealso::

            See :func:`~optuna.trial.create_trial` for how to create trials.

        Args:
            trial: Trial to add.

        Raises:
            :exc:`ValueError`:
                If trial is an invalid state.

        """

        trial._validate()

        self._storage.create_new_trial(self._study_id, template_trial=trial)

    def _reseed_and_optimize_sequential(
        self,
        func: ObjectiveFuncType,
        n_trials: Optional[int],
        timeout: Optional[float],
        catch: Tuple[Type[Exception], ...],
        callbacks: Optional[List[Callable[["Study", FrozenTrial], None]]],
        gc_after_trial: bool,
        time_start: Optional[datetime.datetime],
    ) -> None:

        self.sampler.reseed_rng()
        self._optimize_sequential(
            func, n_trials, timeout, catch, callbacks, gc_after_trial, time_start
        )

    def _optimize_sequential(
        self,
        func: ObjectiveFuncType,
        n_trials: Optional[int],
        timeout: Optional[float],
        catch: Tuple[Type[Exception], ...],
        callbacks: Optional[List[Callable[["Study", FrozenTrial], None]]],
        gc_after_trial: bool,
        time_start: Optional[datetime.datetime],
    ) -> None:

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

    def _pop_waiting_trial_id(self) -> Optional[int]:

        # TODO(c-bata): Reduce database query counts for extracting waiting trials.
        for trial in self._storage.get_all_trials(self._study_id, deepcopy=False):
            if trial.state != TrialState.WAITING:
                continue

            if not self._storage.set_trial_state(trial._trial_id, TrialState.RUNNING):
                continue

            _logger.debug("Trial {} popped from the trial queue.".format(trial.number))
            return trial._trial_id

        return None

    def _run_trial_and_callbacks(
        self,
        func: ObjectiveFuncType,
        catch: Tuple[Type[Exception], ...],
        callbacks: Optional[List[Callable[["Study", FrozenTrial], None]]],
        gc_after_trial: bool,
    ) -> None:

        trial = self._run_trial(func, catch, gc_after_trial)
        if callbacks is not None:
            frozen_trial = copy.deepcopy(self._storage.get_trial(trial._trial_id))
            for callback in callbacks:
                callback(self, frozen_trial)

    def _run_trial(
        self,
        func: ObjectiveFuncType,
        catch: Tuple[Type[Exception], ...],
        gc_after_trial: bool,
    ) -> trial_module.Trial:

        # Sync storage once at the beginning of the objective evaluation.
        self._storage.read_trials_from_remote_storage(self._study_id)

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = trial_module.Trial(self, trial_id)
        trial_number = trial.number

        try:
            result = func(trial)
        except exceptions.TrialPruned as e:
            message = "Trial {} pruned. {}".format(trial_number, str(e))
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
            message = "Trial {} failed because of the following error: {}".format(
                trial_number, repr(e)
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
                "Trial {} failed, because the returned value from the "
                "objective function cannot be cast to float. Returned value is: "
                "{}".format(trial_number, repr(result))
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)
            return trial

        if math.isnan(result):
            message = "Trial {} failed, because the objective function returned {}.".format(
                trial_number, result
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, TrialState.FAIL)
            return trial

        self._storage.set_trial_value(trial_id, result)
        self._storage.set_trial_state(trial_id, TrialState.COMPLETE)
        self._log_completed_trial(trial, result)

        return trial

    def _log_completed_trial(self, trial: trial_module.Trial, result: float) -> None:

        if not _logger.isEnabledFor(logging.INFO):
            return

        _logger.info(
            "Trial {} finished with value: {} and parameters: {}. "
            "Best is trial {} with value: {}.".format(
                trial.number, result, trial.params, self.best_trial.number, self.best_value
            )
        )


def create_study(
    storage: Optional[Union[str, storages.BaseStorage]] = None,
    sampler: Optional["samplers.BaseSampler"] = None,
    pruner: Optional[pruners.BasePruner] = None,
    study_name: Optional[str] = None,
    direction: str = "minimize",
    load_if_exists: bool = False,
) -> Study:
    """Create a new :class:`~optuna.study.Study`.

    Example:

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform("x", 0, 10)
                return x ** 2

            study = optuna.create_study()
            study.optimize(objective, n_trials=3)

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
            A pruner object that decides early stopping of unpromising trials. If :obj:`None`
            is specified, :class:`~optuna.pruners.MedianPruner` is used as the default. See
            also :class:`~optuna.pruners`.
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

    See also:
        :func:`optuna.create_study` is an alias of :func:`optuna.study.create_study`.

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
    study_name: str,
    storage: Union[str, storages.BaseStorage],
    sampler: Optional["samplers.BaseSampler"] = None,
    pruner: Optional[pruners.BasePruner] = None,
) -> Study:
    """Load the existing :class:`~optuna.study.Study` that has the specified name.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_float("x", 0, 10)
                return x ** 2

            study = optuna.create_study(storage="sqlite:///example.db", study_name="my_study")
            study.optimize(objective, n_trials=3)

            loaded_study = optuna.load_study(study_name="my_study", storage="sqlite:///example.db")
            assert len(loaded_study.trials) == len(study.trials)

        .. testcleanup::

            os.remove("example.db")

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

    See also:
        :func:`optuna.load_study` is an alias of :func:`optuna.study.load_study`.

    """

    return Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)


def delete_study(
    study_name: str,
    storage: Union[str, storages.BaseStorage],
) -> None:
    """Delete a :class:`~optuna.study.Study` object.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2

            study = optuna.create_study(study_name="example-study", storage="sqlite:///example.db")
            study.optimize(objective, n_trials=3)

            optuna.delete_study(study_name="example-study", storage="sqlite:///example.db")

        .. testcleanup::

            os.remove("example.db")

    Args:
        study_name:
            Study's name.
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.

    See also:
        :func:`optuna.delete_study` is an alias of :func:`optuna.study.delete_study`.

    """

    storage = storages.get_storage(storage)
    study_id = storage.get_study_id_from_name(study_name)
    storage.delete_study(study_id)


def get_all_study_summaries(storage: Union[str, storages.BaseStorage]) -> List[StudySummary]:
    """Get all history of studies stored in a specified storage.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2

            study = optuna.create_study(study_name="example-study", storage="sqlite:///example.db")
            study.optimize(objective, n_trials=3)

            study_summaries = optuna.study.get_all_study_summaries(storage="sqlite:///example.db")
            assert len(study_summaries) == 1

            study_summary = study_summaries[0]
            assert study_summary.study_name == "example-study"

        .. testcleanup::

            os.remove("example.db")

    Args:
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.study.create_study` for further details.

    Returns:
        List of study history summarized as :class:`~optuna.study.StudySummary` objects.

    See also:
        :func:`optuna.get_all_study_summaries` is an alias of
        :func:`optuna.study.get_all_study_summaries`.

    """

    storage = storages.get_storage(storage)
    return storage.get_all_study_summaries()
