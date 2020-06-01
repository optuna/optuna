from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState

CategoricalChoiceType = Union[None, bool, int, float, str]


@experimental("1.4.0")
class MultiObjectiveTrial(object):
    """A trial is a process of evaluating an objective function.

    This object is passed to an objective function and provides interfaces to get parameter
    suggestion, manage the trial's state, and set/get user-defined attributes of the trial.

    Note that the direct use of this constructor is not recommended.
    This object is seamlessly instantiated and passed to the objective function behind
    the :func:`optuna.multi_objective.study.MultiObjectiveStudy.optimize()` method;
    hence library users do not care about instantiation of this object.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` object.
    """

    def __init__(self, trial: Trial):
        self._trial = trial

        # TODO(ohta): Optimize the code below to eliminate the `MultiObjectiveStudy` construction.
        # See also: https://github.com/optuna/optuna/pull/1054/files#r407982636
        self._n_objectives = multi_objective.study.MultiObjectiveStudy(trial.study).n_objectives

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> float:
        """Suggest a value for the floating point parameter.

        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_float`
        for further details.
        """

        return self._trial.suggest_float(name, low, high, step=step, log=log)

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_uniform`
        for further details.
        """

        return self._trial.suggest_uniform(name, low, high)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_loguniform`
        for further details.
        """

        return self._trial.suggest_loguniform(name, low, high)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        """Suggest a value for the discrete parameter.

        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_discrete_uniform`
        for further details.
        """

        return self._trial.suggest_discrete_uniform(name, low, high, q)

    def suggest_int(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False,
    ) -> int:
        """Suggest a value for the integer parameter.

        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_int`
        for further details.
        """

        return self._trial.suggest_int(name, low, high, step=step, log=log)

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        """Suggest a value for the categorical parameter.

        Please refer to the documentation of :func:`optuna.trial.Trial.suggest_categorical`
        for further details.
        """

        return self._trial.suggest_categorical(name, choices)

    def report(self, values: Tuple[float], step: int) -> None:
        """Report intermediate objective function values for a given step.

        The reported values are used by the pruners to determine whether this trial should be
        pruned.

        .. seealso::
            Please refer to :class:`~optuna.pruners.BasePruner`.

        .. note::
            The reported values are converted to ``float`` type by applying ``float()``
            function internally. Thus, it accepts all float-like types (e.g., ``numpy.float32``).
            If the conversion fails, a ``TypeError`` is raised.

        Args:
            values:
                Intermediate objective function values for a given step.
            step:
                Step of the trial (e.g., Epoch of neural network training).
        """

        # TODO(ohta): Allow users reporting a subset of target values.
        # See https://github.com/optuna/optuna/pull/1054/files#r401594785 for the detail.

        if len(values) != self._n_objectives:
            raise ValueError(
                "The number of the intermediate values {} at step {} is mismatched with"
                "the number of the objectives {}.",
                len(values),
                step,
                self._n_objectives,
            )

        for i, value in enumerate(values):
            self._trial.report(value, self._n_objectives * (step + 1) + i)

    def _report_complete_values(self, values: Tuple[float]) -> None:
        if len(values) != self._n_objectives:
            raise ValueError(
                "The number of the values {} is mismatched with the number of the objectives {}.",
                len(values),
                self._n_objectives,
            )

        for i, value in enumerate(values):
            self._trial.report(value, i)

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set user attributes to the trial.

        Please refer to the documentation of :func:`optuna.trial.Trial.set_user_attr`
        for further details.
        """

        self._trial.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        """Set system attributes to the trial.

        Please refer to the documentation of :func:`optuna.trial.Trial.set_system_attr`
        for further details.
        """

        self._trial.set_system_attr(key, value)

    @property
    def number(self) -> int:
        """Return trial's number which is consecutive and unique in a study.

        Returns:
            A trial number.
        """

        return self._trial.number

    @property
    def params(self) -> Dict[str, Any]:
        """Return parameters to be optimized.

        Returns:
            A dictionary containing all parameters.
        """

        return self._trial.params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        """Return distributions of parameters to be optimized.

        Returns:
            A dictionary containing all distributions.
        """

        return self._trial.distributions

    @property
    def user_attrs(self) -> Dict[str, Any]:
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self._trial.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self._trial.system_attrs

    @property
    def datetime_start(self) -> Optional[datetime]:
        """Return start datetime.

        Returns:
            Datetime where the :class:`~optuna.trial.Trial` started.
        """

        return self._trial.datetime_start

    # TODO(ohta): Add `to_single_objective` method.
    # This method would be helpful to use the existing pruning
    # integrations for multi-objective optimization.

    def _get_values(self) -> List[Optional[float]]:
        trial = self._trial.study._storage.get_trial(self._trial._trial_id)
        return [trial.intermediate_values.get(i) for i in range(self._n_objectives)]


@experimental("1.4.0")
class FrozenMultiObjectiveTrial(object):
    """Status and results of a :class:`~optuna.multi_objective.trial.MultiObjectiveTrial`.

    Attributes:
        number:
            Unique and consecutive number of
            :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` for each
            :class:`~optuna.multi_objective.study.MultiObjectiveStudy`.
            Note that this field uses zero-based numbering.
        state:
            :class:`~optuna.trial.TrialState` of the
            :class:`~optuna.multi_objective.trial.MultiObjectiveTrial`.
        values:
            Objective values of the :class:`~optuna.multi_objective.trial.MultiObjectiveTrial`.
        datetime_start:
            Datetime where the :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` started.
        datetime_complete:
            Datetime where the :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` finished.
        params:
            Dictionary that contains suggested parameters.
        distributions:
            Dictionary that contains the distributions of :attr:`params`.
        user_attrs:
            Dictionary that contains the attributes of the
            :class:`~optuna.multi_objective.trial.MultiObjectiveTrial` set with
            :func:`optuna.multi_objective.trial.MultiObjectiveTrial.set_user_attr`.
        intermediate_values:
            Intermediate objective values set with
            :func:`optuna.multi_objective.trial.MultiObjectiveTrial.report`.
    """

    def __init__(self, n_objectives: int, trial: FrozenTrial):
        self.n_objectives = n_objectives
        self._trial = trial

        self.values = tuple(trial.intermediate_values.get(i) for i in range(n_objectives))

        intermediate_values = {}  # type: Dict[int, List[Optional[float]]]
        for key, value in trial.intermediate_values.items():
            if key < n_objectives:
                continue

            step = key // n_objectives - 1
            if step not in intermediate_values:
                intermediate_values[step] = list(None for _ in range(n_objectives))

            intermediate_values[step][key % n_objectives] = value
        self.intermediate_values = {k: tuple(v) for k, v in intermediate_values.items()}

    @property
    def number(self) -> int:
        return self._trial.number

    @property
    def _trial_id(self) -> int:
        return self._trial._trial_id

    @property
    def state(self) -> TrialState:
        return self._trial.state

    @property
    def datetime_start(self) -> Optional[datetime]:
        return self._trial.datetime_start

    @property
    def datetime_complete(self) -> Optional[datetime]:
        return self._trial.datetime_complete

    @property
    def params(self) -> Dict[str, Any]:
        return self._trial.params

    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self._trial.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self._trial.system_attrs

    @property
    def last_step(self) -> Optional[int]:
        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        return self._trial.distributions

    def _dominates(
        self,
        other: "multi_objective.trial.FrozenMultiObjectiveTrial",
        directions: List[StudyDirection],
    ) -> bool:
        if len(self.values) != len(other.values):
            raise ValueError("Trials with different numbers of objectives cannot be compared.")

        if len(self.values) != len(directions):
            raise ValueError(
                "The number of the values and the number of the objectives are mismatched."
            )

        if self.state != TrialState.COMPLETE:
            return False

        if other.state != TrialState.COMPLETE:
            return True

        values0 = [_normalize_value(v, d) for v, d in zip(self.values, directions)]
        values1 = [_normalize_value(v, d) for v, d in zip(other.values, directions)]

        if values0 == values1:
            return False

        return all([v0 <= v1 for v0, v1 in zip(values0, values1)])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMultiObjectiveTrial):
            return NotImplemented
        return self._trial == other._trial

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMultiObjectiveTrial):
            return NotImplemented

        return self._trial < other._trial

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMultiObjectiveTrial):
            return NotImplemented

        return self._trial <= other._trial

    def __hash__(self) -> int:
        return hash(self._trial)

    # TODO(ohta): Implement `__repr__` method.


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
