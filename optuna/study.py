import copy
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from optuna import exceptions
from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import trial as trial_module
from optuna._dataframe import _trials_dataframe
from optuna._dataframe import pd
from optuna._experimental import experimental
from optuna._multi_objective import _get_pareto_front_trials
from optuna._optimize import _optimize
from optuna._study_direction import StudyDirection
from optuna._study_summary import StudySummary  # NOQA
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


ObjectiveFuncType = Callable[[trial_module.Trial], Union[float, Sequence[float]]]


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

        Raises:
            :exc:`RuntimeError`:
                If the study has more than one direction.
        """

        return self.best_trial.params

    @property
    def best_value(self) -> float:
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.

        Raises:
            :exc:`RuntimeError`:
                If the study has more than one direction.
        """

        best_value = self.best_trial.value
        assert best_value is not None

        return best_value

    @property
    def best_trial(self) -> FrozenTrial:
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.FrozenTrial` object of the best trial.

        Raises:
            :exc:`RuntimeError`:
                If the study has more than one direction.
        """

        if self._is_multi_objective():
            raise RuntimeError(
                "The best trial of a `study` is only supported for single-objective optimization."
            )

        return copy.deepcopy(self._storage.get_best_trial(self._study_id))

    @property
    def best_trials(self) -> List[FrozenTrial]:
        """Return trials located at the Pareto front in the study.

        A trial is located at the Pareto front if there are no trials that dominate the trial.
        It's called that a trial ``t0`` dominates another trial ``t1`` if
        ``all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values)`` and
        ``any(v0 < v1) for v0, v1 in zip(t0.values, t1.values)`` are held.

        Returns:
            A list of :class:`~optuna.trial.FrozenTrial` objects.
        """

        return _get_pareto_front_trials(self)

    @property
    def direction(self) -> StudyDirection:
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.study.StudyDirection` object.

        Raises:
            :exc:`RuntimeError`:
                If the study has more than one direction.
        """

        if self._is_multi_objective():
            raise RuntimeError(
                "The single direction of a `study` is only supported for single-objective "
                "optimization."
            )

        return self.directions[0]

    @property
    def directions(self) -> List[StudyDirection]:
        """Return the directions of the study.

        Returns:
            A list of :class:`~optuna.study.StudyDirection` objects.
        """

        return self._storage.get_study_directions(self._study_id)

    @property
    def trials(self) -> List[FrozenTrial]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True, states=None)``.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self.get_trials(deepcopy=True, states=None)

    def _is_multi_objective(self) -> bool:
        """Return :obj:`True` if the study has multiple objectives.

        Returns:
            A boolean value indicates if `self.directions` has more than 1 element or not.
        """

        return len(self.directions) > 1

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Optional[Tuple[TrialState, ...]] = None,
    ) -> List[FrozenTrial]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

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
            states:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        self._storage.read_trials_from_remote_storage(self._study_id)
        return self._storage.get_all_trials(self._study_id, deepcopy=deepcopy, states=states)


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
                    "contributors": ["Akiba", "Sano"],
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

        Raises:
            RuntimeError:
                If nested invocation of this method occurs.
        """
        _optimize(
            study=self,
            func=func,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

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
                    "contributors": ["Akiba", "Sano"],
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

        Note:
            If ``value`` is in ``attrs`` during multi-objective optimization, it is implicitly
            replaced with ``values``.
        """
        return _trials_dataframe(self, attrs, multi_index)

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
                        trial.study.stop()
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

    def _ask(self) -> trial_module.Trial:
        # Sync storage once at the beginning of the objective evaluation.
        self._storage.read_trials_from_remote_storage(self._study_id)

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        return trial_module.Trial(self, trial_id)

    def _tell(
        self, trial: trial_module.Trial, state: TrialState, values: Optional[List[float]]
    ) -> None:
        if state == TrialState.COMPLETE:
            assert values is not None
        if values is not None:
            self._storage.set_trial_values(trial._trial_id, values)
        self._storage.set_trial_state(trial._trial_id, state)

    def _log_completed_trial(self, trial: trial_module.Trial, values: Sequence[float]) -> None:

        if not _logger.isEnabledFor(logging.INFO):
            return

        if len(values) > 1:
            _logger.info(
                "Trial {} finished with values: {} and parameters: {}. ".format(
                    trial.number, values, trial.params
                )
            )
        elif len(values) == 1:
            _logger.info(
                "Trial {} finished with value: {} and parameters: {}. "
                "Best is trial {} with value: {}.".format(
                    trial.number,
                    values[0],
                    trial.params,
                    self.best_trial.number,
                    self.best_value,
                )
            )
        else:
            assert False, "Should not reach."


def create_study(
    storage: Optional[Union[str, storages.BaseStorage]] = None,
    sampler: Optional["samplers.BaseSampler"] = None,
    pruner: Optional[pruners.BasePruner] = None,
    study_name: Optional[str] = None,
    direction: Optional[str] = None,
    load_if_exists: bool = False,
    *,
    directions: Optional[Sequence[str]] = None,
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
            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used during
            single-objective optimization and :class:`~optuna.samplers.NSGAIISampler` during
            multi-objective optimization. See also :class:`~optuna.samplers`.
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

            .. note::
                If none of `direction` and `directions` are specified, the direction of the study
                is set to "minimize".
        directions:
            A sequence of directions during multi-objective optimization.
        load_if_exists:
            Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.

    Returns:
        A :class:`~optuna.study.Study` object.

    Raises:
        :exc:`ValueError`:
            If the length of ``directions`` is zero.
            Or, if ``direction`` is neither 'minimize' nor 'maximize' when it is a string.
            Or, if the element of ``directions`` is neither `minimize` nor `maximize`.
            Or, if both ``direction`` and ``directions`` are specified.

    See also:
        :func:`optuna.create_study` is an alias of :func:`optuna.study.create_study`.

    """

    if direction is None and directions is None:
        directions = ["minimize"]
    elif direction is not None and directions is not None:
        raise ValueError("Specify only one of `direction` and `directions`.")
    elif direction is not None:
        directions = [direction]
    elif directions is not None:
        directions = list(directions)
    else:
        assert False

    if len(directions) < 1:
        raise ValueError("The number of objectives must be greater than 0.")
    elif any(d != "minimize" and d != "maximize" for d in directions):
        raise ValueError("Please set either 'minimize' or 'maximize' to direction.")

    direction_objects = [
        StudyDirection.MINIMIZE if d == "minimize" else StudyDirection.MAXIMIZE for d in directions
    ]

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

    if sampler is None and len(direction_objects) > 1:
        sampler = samplers.NSGAIISampler()

    study_name = storage.get_study_name_from_id(study_id)
    study = Study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)

    study._storage.set_study_directions(study_id, direction_objects)

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
