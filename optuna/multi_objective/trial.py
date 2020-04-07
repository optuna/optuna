from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.structs import FrozenTrial
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.trial import Trial

CategoricalChoiceType = Union[None, bool, int, float, str]


@experimental("1.4.0")
class MultiObjectiveTrial(object):
    def __init__(self, trial: Trial):
        self._trial = trial
        self._n_objectives = multi_objective.study.MultiObjectiveStudy(trial.study).n_objectives

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        return self._trial.suggest_uniform(name, low, high)

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        return self._trial.suggest_loguniform(name, low, high)

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        return self._trial.suggest_discrete_uniform(name, low, high, q)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self._trial.suggest_int(name, low, high)

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        return self._trial.suggest_categorical(name, choices)

    def report(self, values: List[float], step: int) -> None:
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

    def _report_complete_values(self, values: List[float]) -> None:
        if len(values) != self._n_objectives:
            raise ValueError(
                "The number of the values {} is mismatched with the number of the objectives {}.",
                len(values),
                self._n_objectives,
            )

        for i, value in enumerate(values):
            self._trial.report(value, i)

    def set_user_attr(self, key: str, value: Any) -> None:
        self._trial.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        self._trial.set_system_attr(key, value)

    @property
    def params(self) -> Dict[str, Any]:
        return self._trial.params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        return self._trial.distributions

    @property
    def user_attrs(self) -> Dict[str, Any]:
        return self._trial.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        return self._trial.system_attrs

    @property
    def datetime_start(self) -> Optional[datetime]:
        return self._trial.datetime_start

    # TODO(ohta): Add `to_single_objective` method.
    # This method would be helpful to use the existing pruning
    # integrations for multi-objective optimization.

    def _get_values(self) -> List[Optional[float]]:
        trial = self._trial.study._storage.get_trial(self._trial._trial_id)
        return [trial.intermediate_values.get(i) for i in range(self._n_objectives)]


@experimental("1.4.0")
class FrozenMultiObjectiveTrial(object):
    def __init__(self, n_objectives: int, trial: FrozenTrial):
        self.n_objectives = n_objectives
        self._trial = trial

        self.values = [trial.intermediate_values.get(i) for i in range(n_objectives)]

        self.intermediate_values = {}  # type: Dict[int, List[Optional[float]]]
        for key, value in trial.intermediate_values.items():
            if key < n_objectives:
                continue

            step = key // n_objectives - 1
            if step not in trial.intermediate_values:
                self.intermediate_values[step] = list(None for _ in range(n_objectives))

            self.intermediate_values[step][key % n_objectives] = value

    @property
    def number(self) -> int:
        return self._trial.number

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

        values0 = [_normalize_value(v, d) for v, d in zip(self.values, directions)]
        values1 = [_normalize_value(v, d) for v, d in zip(other.values, directions)]

        if self.state != TrialState.COMPLETE:
            return False

        if other.state != TrialState.COMPLETE:
            return True

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
