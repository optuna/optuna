import copy
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings

from optuna import exceptions
from optuna import logging
from optuna import pruners
from optuna import samplers
from optuna import storages
from optuna import trial as trial_module
from optuna._deprecated import deprecated
from optuna._experimental import experimental
from optuna._imports import _LazyImport
from optuna.distributions import BaseDistribution
from optuna.study._multi_objective import _get_pareto_front_trials
from optuna.study._optimize import _check_and_convert_to_values
from optuna.study._optimize import _optimize
from optuna.study._study_direction import StudyDirection
from optuna.study._study_summary import StudySummary  # NOQA
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_dataframe = _LazyImport("optuna.study._dataframe")

if TYPE_CHECKING:
    from optuna.study._dataframe import pd


ObjectiveFuncType = Callable[[trial_module.Trial], Union[float, Sequence[float]]]


_logger = logging.get_logger(__name__)


class Study:
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
        self._study_id = study_id
        self._storage = storage

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
            A :class:`~optuna.trial.FrozenTrial` object of the best trial.

        Raises:
            :exc:`RuntimeError`:
                If the study has more than one direction.
        """

        if self._is_multi_objective():
            raise RuntimeError(
                "A single best trial cannot be retrieved from a multi-objective study. Consider "
                "using Study.best_trials to retrieve a list containing the best trials."
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
                "A single direction cannot be retrieved from a multi-objective study. Consider "
                "using Study.directions to retrieve a list containing all directions."
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
            A list of :class:`~optuna.trial.FrozenTrial` objects.
        """

        return self.get_trials(deepcopy=True, states=None)

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
                    x = trial.suggest_float("x", -1, 1)
                    return x**2


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
            A list of :class:`~optuna.trial.FrozenTrial` objects.
        """

        self._storage.read_trials_from_remote_storage(self._study_id)
        return self._storage.get_all_trials(self._study_id, deepcopy=deepcopy, states=states)

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
                    return x**2 + y**2


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
                    x = trial.suggest_float("x", -1, 1)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

        Args:
            func:
                A callable that implements objective function.
            n_trials:
                The number of trials for each process. If this argument is set to :obj:`None`,
                there is no limitation on the number of trials. If ``timeout`` is also set to
                :obj:`None`, the study continues to create trials until it receives a termination
                signal such as Ctrl+C or SIGTERM.

                .. seealso::
                    :class:`optuna.study.MaxTrialsCallback` can ensure how many times trials
                    will be performed across all processes.
            timeout:
                Stop study after the given number of second(s). If this argument is set to
                :obj:`None`, the study is executed without time limitation. If :obj:`n_trials` is
                also set to :obj:`None`, the study continues to create trials until it receives a
                termination signal such as Ctrl+C or SIGTERM.
            n_jobs:
                The number of parallel jobs. If this argument is set to :obj:`-1`, the number is
                set to CPU count.

                .. note::
                    ``n_jobs`` allows parallelization using :obj:`threading` and may suffer from
                    `Python's GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_.
                    It is recommended to use :ref:`process-based parallelization<distributed>`
                    if ``func`` is CPU bound.

            catch:
                A study continues to run even when a trial raises one of the exceptions specified
                in this argument. Default is an empty tuple, i.e. the study will stop for any
                exception except for :class:`~optuna.exceptions.TrialPruned`.
            callbacks:
                List of callback functions that are invoked at the end of each trial. Each function
                must accept two parameters with the following types in this order:
                :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.

                .. seealso::

                    See the tutorial of :ref:`optuna_callback` for how to use and implement
                    callback functions.

            gc_after_trial:
                Flag to determine whether to automatically run garbage collection after each trial.
                Set to :obj:`True` to run the garbage collection, :obj:`False` otherwise.
                When it runs, it runs a full collection by internally calling :func:`gc.collect`.
                If you see an increase in memory consumption over several trials, try setting this
                flag to :obj:`True`.

                .. seealso::

                    :ref:`out-of-memory-gc-collect`

            show_progress_bar:
                Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.
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

    def ask(
        self, fixed_distributions: Optional[Dict[str, BaseDistribution]] = None
    ) -> trial_module.Trial:
        """Create a new trial from which hyperparameters can be suggested.

        This method is part of an alternative to :func:`~optuna.study.Study.optimize` that allows
        controlling the lifetime of a trial outside the scope of ``func``. Each call to this
        method should be followed by a call to :func:`~optuna.study.Study.tell` to finish the
        created trial.

        .. seealso::

            The :ref:`ask_and_tell` tutorial provides use-cases with examples.

        Example:

            Getting the trial object with the :func:`~optuna.study.Study.ask` method.

            .. testcode::

                import optuna


                study = optuna.create_study()

                trial = study.ask()

                x = trial.suggest_float("x", -1, 1)

                study.tell(trial, x**2)

        Example:

            Passing previously defined distributions to the :func:`~optuna.study.Study.ask`
            method.

            .. testcode::

                import optuna


                study = optuna.create_study()

                distributions = {
                    "optimizer": optuna.distributions.CategoricalDistribution(["adam", "sgd"]),
                    "lr": optuna.distributions.LogUniformDistribution(0.0001, 0.1),
                }

                # You can pass the distributions previously defined.
                trial = study.ask(fixed_distributions=distributions)

                # `optimizer` and `lr` are already suggested and accessible with `trial.params`.
                assert "optimizer" in trial.params
                assert "lr" in trial.params

        Args:
            fixed_distributions:
                A dictionary containing the parameter names and parameter's distributions. Each
                parameter in this dictionary is automatically suggested for the returned trial,
                even when the suggest method is not explicitly invoked by the user. If this
                argument is set to :obj:`None`, no parameter is automatically suggested.

        Returns:
            A :class:`~optuna.trial.Trial`.
        """

        fixed_distributions = fixed_distributions or {}

        # Sync storage once every trial.
        self._storage.read_trials_from_remote_storage(self._study_id)

        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = trial_module.Trial(self, trial_id)

        for name, param in fixed_distributions.items():
            trial._suggest(name, param)

        return trial

    def tell(
        self,
        trial: Union[trial_module.Trial, int],
        values: Optional[Union[float, Sequence[float]]] = None,
        state: TrialState = TrialState.COMPLETE,
        skip_if_finished: bool = False,
    ) -> None:
        """Finish a trial created with :func:`~optuna.study.Study.ask`.

        .. seealso::

            The :ref:`ask_and_tell` tutorial provides use-cases with examples.

        Example:

            .. testcode::

                import optuna
                from optuna.trial import TrialState


                def f(x):
                    return (x - 2) ** 2


                def df(x):
                    return 2 * x - 4


                study = optuna.create_study()

                n_trials = 30

                for _ in range(n_trials):
                    trial = study.ask()

                    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

                    # Iterative gradient descent objective function.
                    x = 3  # Initial value.
                    for step in range(128):
                        y = f(x)

                        trial.report(y, step=step)

                        if trial.should_prune():
                            # Finish the trial with the pruned state.
                            study.tell(trial, state=TrialState.PRUNED)
                            break

                        gy = df(x)
                        x -= gy * lr
                    else:
                        # Finish the trial with the final value after all iterations.
                        study.tell(trial, y)

        Args:
            trial:
                A :class:`~optuna.trial.Trial` object or a trial number.
            values:
                Optional objective value or a sequence of such values in case the study is used
                for multi-objective optimization. Argument must be provided if ``state`` is
                :class:`~optuna.trial.TrialState.COMPLETE` and should be :obj:`None` if ``state``
                is :class:`~optuna.trial.TrialState.FAIL` or
                :class:`~optuna.trial.TrialState.PRUNED`.
            state:
                State to be reported. Must be :class:`~optuna.trial.TrialState.COMPLETE`,
                :class:`~optuna.trial.TrialState.FAIL` or
                :class:`~optuna.trial.TrialState.PRUNED`.
            skip_if_finished:
                Flag to control whether exception should be raised when values for already
                finished trial are told. If :obj:`True`, tell is skipped without any error
                when the trial is already finished.

        Raises:
            TypeError:
                If ``trial`` is not a :class:`~optuna.trial.Trial` or an :obj:`int`.
            ValueError:
                If any of the following.
                ``values`` is a sequence but its length does not match the number of objectives
                for its associated study.
                ``state`` is :class:`~optuna.trial.TrialState.COMPLETE` but
                ``values`` is :obj:`None`.
                ``state`` is :class:`~optuna.trial.TrialState.FAIL` or
                :class:`~optuna.trial.TrialState.PRUNED` but
                ``values`` is not :obj:`None`.
                ``state`` is not
                :class:`~optuna.trial.TrialState.COMPLETE`,
                :class:`~optuna.trial.TrialState.FAIL` or
                :class:`~optuna.trial.TrialState.PRUNED`.
                ``trial`` is a trial number but no
                trial exists with that number.
        """

        if not isinstance(trial, (trial_module.Trial, int)):
            raise TypeError("Trial must be a trial object or trial number.")

        if state == TrialState.COMPLETE:
            if values is None:
                raise ValueError(
                    "No values were told. Values are required when state is TrialState.COMPLETE."
                )
        elif state in (TrialState.PRUNED, TrialState.FAIL):
            if values is not None:
                raise ValueError(
                    "Values were told. Values cannot be specified when state is "
                    "TrialState.PRUNED or TrialState.FAIL."
                )
        else:
            raise ValueError(f"Cannot tell with state {state}.")

        if isinstance(trial, trial_module.Trial):
            trial_number = trial.number
            trial_id = trial._trial_id
        elif isinstance(trial, int):
            trial_number = trial
            try:
                trial_id = self._storage.get_trial_id_from_study_id_trial_number(
                    self._study_id, trial_number
                )
            except NotImplementedError as e:
                warnings.warn(
                    "Study.tell may be slow because the trial was represented by its number but "
                    f"the storage {self._storage.__class__.__name__} does not implement the "
                    "method required to map numbers back. Please provide the trial object "
                    "to avoid performance degradation."
                )

                trials = self.get_trials(deepcopy=False)

                if len(trials) <= trial_number:
                    raise ValueError(
                        f"Cannot tell for trial with number {trial_number} since it has not been "
                        "created."
                    ) from e

                trial_id = trials[trial_number]._trial_id
            except KeyError as e:
                raise ValueError(
                    f"Cannot tell for trial with number {trial_number} since it has not been "
                    "created."
                ) from e
        else:
            assert False, "Should not reach."

        frozen_trial = self._storage.get_trial(trial_id)

        if frozen_trial.state.is_finished() and skip_if_finished:
            _logger.info(
                f"Skipped telling trial {trial_number} with values "
                f"{values} and state {state} since trial was already finished. "
                f"Finished trial has values {frozen_trial.values} and state {frozen_trial.state}."
            )
            return

        if state == TrialState.PRUNED:
            # Register the last intermediate value if present as the value of the trial.
            # TODO(hvy): Whether a pruned trials should have an actual value can be discussed.
            assert values is None

            last_step = frozen_trial.last_step
            if last_step is not None:
                values = [frozen_trial.intermediate_values[last_step]]

        if values is not None:
            values, values_conversion_failure_message = _check_and_convert_to_values(
                len(self.directions), values, trial_number
            )
            # When called from `Study.optimize` and `state` is pruned, the given `values` contains
            # the intermediate value with the largest step so far. In this case, the value is
            # allowed to be NaN and errors should not be raised.
            if state != TrialState.PRUNED and values_conversion_failure_message is not None:
                raise ValueError(values_conversion_failure_message)

        try:
            # Sampler defined trial post-processing.
            study = pruners._filter_study(self, frozen_trial)
            self.sampler.after_trial(study, frozen_trial, state, values)
        except Exception:
            raise
        finally:
            if values is not None:
                self._storage.set_trial_values(trial_id, values)

            self._storage.set_trial_state(trial_id, state)

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute to the study.

        .. seealso::

            See :attr:`~optuna.study.Study.user_attrs` for related attribute.

        .. seealso::

            See the recipe on :ref:`attributes`.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 1)
                    y = trial.suggest_float("y", 0, 1)
                    return x**2 + y**2


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
                    x = trial.suggest_float("x", -1, 1)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                # Create a dataframe from the study.
                df = study.trials_dataframe()
                assert isinstance(df, pandas.DataFrame)
                assert df.shape[0] == 3  # n_trials.

        Args:
            attrs:
                Specifies field names of :class:`~optuna.trial.FrozenTrial` to include them to a
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
        return _dataframe._trials_dataframe(self, attrs, multi_index)

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
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


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
    def enqueue_trial(
        self, params: Dict[str, Any], user_attrs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters which will be evaluated in your
        objective function.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                study.enqueue_trial({"x": 5})
                study.enqueue_trial({"x": 0}, user_attrs={"memo": "optimal"})
                study.optimize(objective, n_trials=2)

                assert study.trials[0].params == {"x": 5}
                assert study.trials[1].params == {"x": 0}
                assert study.trials[1].user_attrs == {"memo": "optimal"}

        Args:
            params:
                Parameter values to pass your objective function.
            user_attrs:
                A dictionary of user-specific attributes other than ``params``.

        .. seealso::
            Please refer to :ref:`specify_params` for the tutorial of specifying hyperparameters
            manually.
        """

        self.add_trial(
            create_trial(
                state=TrialState.WAITING,
                system_attrs={"fixed_params": params},
                user_attrs=user_attrs,
            )
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
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


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

    @experimental("2.5.0")
    def add_trials(self, trials: Iterable[FrozenTrial]) -> None:
        """Add trials to study.

        The trials are validated before being added.

        Example:

            .. testcode::

                import optuna
                from optuna.distributions import UniformDistribution


                def objective(trial):
                    x = trial.suggest_float("x", 0, 10)
                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)
                assert len(study.trials) == 3

                other_study = optuna.create_study()
                other_study.add_trials(study.trials)
                assert len(other_study.trials) == len(study.trials)

                other_study.optimize(objective, n_trials=2)
                assert len(other_study.trials) == len(study.trials) + 2

        .. seealso::

            See :func:`~optuna.study.Study.add_trial` for addition of each trial.

        Args:
            trials: Trials to add.

        Raises:
            :exc:`ValueError`:
                If ``trials`` include invalid trial.
        """

        for trial in trials:
            self.add_trial(trial)

    def _is_multi_objective(self) -> bool:
        """Return :obj:`True` if the study has multiple objectives.

        Returns:
            A boolean value indicates if `self.directions` has more than 1 element or not.
        """

        return len(self.directions) > 1

    def _pop_waiting_trial_id(self) -> Optional[int]:

        for trial in self._storage.get_all_trials(
            self._study_id, deepcopy=False, states=(TrialState.WAITING,)
        ):
            if not self._storage.set_trial_state(trial._trial_id, TrialState.RUNNING):
                continue

            _logger.debug("Trial {} popped from the trial queue.".format(trial.number))
            return trial._trial_id

        return None

    @deprecated("2.5.0", "4.0.0")
    def _ask(self) -> trial_module.Trial:
        return self.ask()

    @deprecated("2.5.0", "4.0.0")
    def _tell(
        self, trial: trial_module.Trial, state: TrialState, values: Optional[List[float]]
    ) -> None:
        self.tell(trial, values, state)

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
    direction: Optional[Union[str, StudyDirection]] = None,
    load_if_exists: bool = False,
    *,
    directions: Optional[Sequence[Union[str, StudyDirection]]] = None,
) -> Study:
    """Create a new :class:`~optuna.study.Study`.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 10)
                return x**2


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
            maximization. You can also pass the corresponding :class:`~optuna.study.StudyDirection`
            object.

            .. note::
                If none of `direction` and `directions` are specified, the direction of the study
                is set to "minimize".
        load_if_exists:
            Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.
        directions:
            A sequence of directions during multi-objective optimization.

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
    elif any(
        d not in ["minimize", "maximize", StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
        for d in directions
    ):
        raise ValueError(
            "Please set either 'minimize' or 'maximize' to direction. You can also set the "
            "corresponding `StudyDirection` member."
        )

    direction_objects = [
        d if isinstance(d, StudyDirection) else StudyDirection[d.upper()] for d in directions
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
    study_name: Optional[str],
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
                return x**2


            study = optuna.create_study(storage="sqlite:///example.db", study_name="my_study")
            study.optimize(objective, n_trials=3)

            loaded_study = optuna.load_study(study_name="my_study", storage="sqlite:///example.db")
            assert len(loaded_study.trials) == len(study.trials)

        .. testcleanup::

            os.remove("example.db")

    Args:
        study_name:
            Study's name. Each study has a unique name as an identifier. If :obj:`None`, checks
            whether the storage contains a single study, and if so loads that study.
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

    Raises:
        :exc:`ValueError`:
            If ``study_name`` is :obj:`None` and the storage contains more than 1 study.

    See also:
        :func:`optuna.load_study` is an alias of :func:`optuna.study.load_study`.

    """
    if study_name is None:
        study_summaries = get_all_study_summaries(storage)
        if len(study_summaries) != 1:
            raise ValueError(
                f"Could not determine the study name since the storage {storage} does not "
                "contain exactly 1 study. Specify `study_name`."
            )
        study_name = study_summaries[0].study_name
        _logger.info(
            f"Study name was omitted but trying to load '{study_name}' because that was the only "
            "study found in the storage."
        )

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


@experimental("2.8.0")
def copy_study(
    from_study_name: str,
    from_storage: Union[str, storages.BaseStorage],
    to_storage: Union[str, storages.BaseStorage],
    to_study_name: Optional[str] = None,
) -> None:
    """Copy study from one storage to another.

    The direction(s) of the objective(s) in the study, trials, user attributes and system
    attributes are copied.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")
            if os.path.exists("example_copy.db"):
                raise RuntimeError("'example_copy.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study(
                study_name="example-study",
                storage="sqlite:///example.db",
            )
            study.optimize(objective, n_trials=3)

            optuna.copy_study(
                from_study_name="example-study",
                from_storage="sqlite:///example.db",
                to_storage="sqlite:///example_copy.db",
            )

            study = optuna.load_study(
                study_name=None,
                storage="sqlite:///example_copy.db",
            )

        .. testcleanup::

            os.remove("example.db")
            os.remove("example_copy.db")

    Args:
        from_study_name:
            Name of study.
        from_storage:
            Source database URL such as ``sqlite:///example.db``. Please see also the
            documentation of :func:`~optuna.study.create_study` for further details.
        to_storage:
            Destination database URL.
        to_study_name:
            Name of the created study. If omitted, ``from_study_name`` is used.

    Raises:
        :class:`~optuna.exceptions.DuplicatedStudyError`:
            If a study with a conflicting name already exists in the destination storage.

    """

    from_study = load_study(study_name=from_study_name, storage=from_storage)
    to_study = create_study(
        study_name=to_study_name or from_study_name,
        storage=to_storage,
        directions=from_study.directions,
        load_if_exists=False,
    )

    for key, value in from_study.system_attrs.items():
        to_study.set_system_attr(key, value)

    for key, value in from_study.user_attrs.items():
        to_study.set_user_attr(key, value)

    # Trials are deep copied on `add_trials`.
    to_study.add_trials(from_study.get_trials(deepcopy=False))


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
