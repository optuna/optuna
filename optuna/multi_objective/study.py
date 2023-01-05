import types
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

from optuna import logging
from optuna import multi_objective
from optuna._deprecated import deprecated_class
from optuna._deprecated import deprecated_func
from optuna.pruners import NopPruner
from optuna.storages import BaseStorage
from optuna.study import create_study as _create_study
from optuna.study import load_study as _load_study
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState


ObjectiveFuncType = Callable[["multi_objective.trial.MultiObjectiveTrial"], Sequence[float]]
CallbackFuncType = Callable[
    [
        "multi_objective.study.MultiObjectiveStudy",
        "multi_objective.trial.FrozenMultiObjectiveTrial",
    ],
    None,
]

_logger = logging.get_logger(__name__)


# TODO(ohta): Reconsider the API design.
# See https://github.com/optuna/optuna/pull/1054/files#r407255282 for the detail.
#
# TODO(ohta): Consider to add `objective_labels` argument.
# See: https://github.com/optuna/optuna/pull/1054#issuecomment-616382152
@deprecated_func("2.4.0", "4.0.0")
def create_study(
    directions: List[str],
    study_name: Optional[str] = None,
    storage: Optional[Union[str, BaseStorage]] = None,
    sampler: Optional["multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    load_if_exists: bool = False,
) -> "multi_objective.study.MultiObjectiveStudy":
    """Create a new :class:`~optuna.multi_objective.study.MultiObjectiveStudy`.

    Example:

        .. testcode::

            import optuna


            def objective(trial):
                # Binh and Korn function.
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x**2 + 4 * y**2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.multi_objective.create_study(["minimize", "minimize"])
            study.optimize(objective, n_trials=3)

    Args:
        directions:
            Optimization direction for each objective value.
            Set ``minimize`` for minimization and ``maximize`` for maximization.
        study_name:
            Study's name. If this argument is set to None, a unique name is generated
            automatically.
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
            If :obj:`None` is specified,
            :class:`~optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler` is used
            as the default. See also :class:`~optuna.multi_objective.samplers`.
        load_if_exists:
            Flag to control the behavior to handle a conflict of study names.
            In the case where a study named ``study_name`` already exists in the ``storage``,
            a :class:`~optuna.exceptions.DuplicatedStudyError` is raised if ``load_if_exists`` is
            set to :obj:`False`.
            Otherwise, the creation of the study is skipped, and the existing one is returned.

    Returns:
        A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object.
    """

    # TODO(ohta): Support pruner.
    mo_sampler = sampler or multi_objective.samplers.NSGAIIMultiObjectiveSampler()
    sampler_adapter = multi_objective.samplers._MultiObjectiveSamplerAdapter(mo_sampler)

    if not isinstance(directions, Iterable):
        raise TypeError("`directions` must be a list or other iterable types.")

    if not all(d in ["minimize", "maximize"] for d in directions):
        raise ValueError("`directions` includes unknown direction names.")

    study = _create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler_adapter,
        pruner=NopPruner(),
        load_if_exists=load_if_exists,
    )

    study._storage.set_study_system_attr(
        study._study_id, "multi_objective:study:directions", list(directions)
    )

    return MultiObjectiveStudy(study)


@deprecated_func("2.4.0", "4.0.0")
def load_study(
    study_name: str,
    storage: Union[str, BaseStorage],
    sampler: Optional["multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
) -> "multi_objective.study.MultiObjectiveStudy":
    """Load the existing :class:`MultiObjectiveStudy` that has the specified name.

    Example:

        .. testsetup::

            import os

            if os.path.exists("example.db"):
                raise RuntimeError("'example.db' already exists. Please remove it.")

        .. testcode::

            import optuna


            def objective(trial):
                # Binh and Korn function.
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x**2 + 4 * y**2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.multi_objective.create_study(
                directions=["minimize", "minimize"],
                study_name="my_study",
                storage="sqlite:///example.db",
            )
            study.optimize(objective, n_trials=3)

            loaded_study = optuna.multi_objective.study.load_study(
                study_name="my_study", storage="sqlite:///example.db"
            )
            assert len(loaded_study.trials) == len(study.trials)

        .. testcleanup::

            os.remove("example.db")

    Args:
        study_name:
            Study's name. Each study has a unique name as an identifier.
        storage:
            Database URL such as ``sqlite:///example.db``. Please see also the documentation of
            :func:`~optuna.multi_objective.study.create_study` for further details.
        sampler:
            A sampler object that implements background algorithm for value suggestion.
            If :obj:`None` is specified,
            :class:`~optuna.multi_objective.samplers.RandomMultiObjectiveSampler` is used
            as the default. See also :class:`~optuna.multi_objective.samplers`.

    Returns:
        A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object.
    """

    mo_sampler = sampler or multi_objective.samplers.RandomMultiObjectiveSampler()
    sampler_adapter = multi_objective.samplers._MultiObjectiveSamplerAdapter(mo_sampler)

    study = _load_study(study_name=study_name, storage=storage, sampler=sampler_adapter)

    return MultiObjectiveStudy(study)


@deprecated_class("2.4.0", "4.0.0")
class MultiObjectiveStudy:
    """A study corresponds to a multi-objective optimization task, i.e., a set of trials.

    This object provides interfaces to run a new
    :class:`~optuna.multi_objective.trial.Trial`, access trials'
    history, set/get user-defined attributes of the study itself.

    Note that the direct use of this constructor is not recommended.
    To create and load a study, please refer to the documentation of
    :func:`~optuna.multi_objective.study.create_study` and
    :func:`~optuna.multi_objective.study.load_study` respectively.
    """

    def __init__(self, study: Study):
        self._study = study

        self._directions = []
        for d in study._storage.get_study_system_attrs(study._study_id)[
            "multi_objective:study:directions"
        ]:
            if d == "minimize":
                self._directions.append(StudyDirection.MINIMIZE)
            elif d == "maximize":
                self._directions.append(StudyDirection.MAXIMIZE)
            else:
                raise ValueError("Unknown direction ({}) is specified.".format(d))

        n_objectives = len(self._directions)
        if n_objectives < 1:
            raise ValueError("The number of objectives must be greater than 0.")

        self._study._log_completed_trial = types.MethodType(  # type: ignore
            _log_completed_trial, self._study
        )

    @property
    def n_objectives(self) -> int:
        """Return the number of objectives.

        Returns:
            Number of objectives.
        """

        return len(self._directions)

    @property
    def directions(self) -> List[StudyDirection]:
        """Return the optimization direction list.

        Returns:
            A list that contains the optimization direction for each objective value.
        """

        return self._directions

    @property
    def sampler(self) -> "multi_objective.samplers.BaseMultiObjectiveSampler":
        """Return the sampler.

        Returns:
            A :class:`~multi_objective.samplers.BaseMultiObjectiveSampler` object.
        """

        adapter = self._study.sampler
        assert isinstance(adapter, multi_objective.samplers._MultiObjectiveSamplerAdapter)

        return adapter._mo_sampler

    def optimize(
        self,
        objective: ObjectiveFuncType,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[CallbackFuncType]] = None,
        gc_after_trial: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        """Optimize an objective function.

        This method is the same as :func:`optuna.study.Study.optimize` except for
        taking an objective function that returns multi-objective values as the argument.

        Please refer to the documentation of :func:`optuna.study.Study.optimize`
        for further details.

        Example:

            .. testcode::

                import optuna


                def objective(trial):
                    # Binh and Korn function.
                    x = trial.suggest_float("x", 0, 5)
                    y = trial.suggest_float("y", 0, 3)

                    v0 = 4 * x**2 + 4 * y**2
                    v1 = (x - 5) ** 2 + (y - 5) ** 2
                    return v0, v1


                study = optuna.multi_objective.create_study(["minimize", "minimize"])
                study.optimize(objective, n_trials=3)
        """

        def mo_objective(trial: Trial) -> float:
            mo_trial = multi_objective.trial.MultiObjectiveTrial(trial)
            values = objective(mo_trial)
            mo_trial._report_complete_values(values)
            return 0.0  # Dummy value.

        # Wraps a multi-objective callback so that we can pass it to the `Study.optimize` method.
        def wrap_mo_callback(callback: CallbackFuncType) -> Callable[[Study, FrozenTrial], None]:
            return lambda study, trial: callback(
                MultiObjectiveStudy(study),
                multi_objective.trial.FrozenMultiObjectiveTrial(self.n_objectives, trial),
            )

        if callbacks is None:
            wrapped_callbacks = None
        else:
            wrapped_callbacks = [wrap_mo_callback(callback) for callback in callbacks]

        self._study.optimize(
            mo_objective,
            timeout=timeout,
            n_trials=n_trials,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=wrapped_callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

    @property
    def user_attrs(self) -> Dict[str, Any]:
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self._study.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self._study._storage.get_study_system_attrs(self._study._study_id)

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute to the study.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.
        """

        self._study.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        """Set a system attribute to the study.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.multi_objective.study.MultiObjectiveStudy.set_user_attr`
        to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._study._storage.set_study_system_attr(self._study._study_id, key, value)

    def enqueue_trial(self, params: Dict[str, Any]) -> None:
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters which will be evaluated in your
        objective function.

        Please refer to the documentation of :func:`optuna.study.Study.enqueue_trial`
        for further details.

        Args:
            params:
                Parameter values to pass your objective function.
        """

        self._study.enqueue_trial(params, skip_if_exists=False)

    @property
    def trials(self) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True, states=None)``.

        Returns:
            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.
        """

        return self.get_trials(deepcopy=True, states=None)

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
                Note that if you set the flag to :obj:`False`, you shouldn't mutate
                any fields of the returned trial. Otherwise the internal state of
                the study may corrupt and unexpected behavior may happen.
            states:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.
        """

        return [
            multi_objective.trial.FrozenMultiObjectiveTrial(self.n_objectives, t)
            for t in self._study.get_trials(deepcopy=deepcopy, states=states)
        ]

    def get_pareto_front_trials(self) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
        """Return trials located at the pareto front in the study.

        A trial is located at the pareto front if there are no trials that dominate the trial.
        It's called that a trial ``t0`` dominates another trial ``t1`` if
        ``all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values)`` and
        ``any(v0 < v1) for v0, v1 in zip(t0.values, t1.values)`` are held.

        Returns:
            A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.
        """

        pareto_front = []
        trials = [t for t in self.trials if t.state == TrialState.COMPLETE]

        # TODO(ohta): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
        for trial in trials:
            dominated = False
            for other in trials:
                if other._dominates(trial, self.directions):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(trial)

        return pareto_front

    @property
    def _storage(self) -> BaseStorage:
        return self._study._storage

    @property
    def _study_id(self) -> int:
        return self._study._study_id


def _log_completed_trial(self: Study, trial: FrozenTrial) -> None:
    if not _logger.isEnabledFor(logging.INFO):
        return

    n_objectives = len(self.directions)
    frozen_multi_objective_trial = multi_objective.trial.FrozenMultiObjectiveTrial(
        n_objectives,
        trial,
    )
    actual_values = frozen_multi_objective_trial.values
    _logger.info(
        "Trial {} finished with values: {} with parameters: {}.".format(
            trial.number, actual_values, trial.params
        )
    )
