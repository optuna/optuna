from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def _calculate_union(
    trials: list[FrozenTrial],
    include_pruned: bool = False,
    search_space: dict[str, BaseDistribution] | None = None,
    cached_trial_number: int = -1,
) -> tuple[dict[str, BaseDistribution] | None, int]:
    states_of_interest = [TrialState.COMPLETE, TrialState.WAITING, TrialState.RUNNING]
    if include_pruned:
        states_of_interest.append(TrialState.PRUNED)

    trials_of_interest = [t for t in trials if t.state in states_of_interest]
    if not trials_of_interest:
        return search_space, -1

    next_cached_trial_number = trials_of_interest[-1].number + 1
    for trial in reversed(trials_of_interest):
        if cached_trial_number > trial.number:
            break

        if not trial.state.is_finished():
            next_cached_trial_number = trial.number
            continue

        if search_space is None:
            search_space = copy.copy(trial.distributions)
            continue

        search_space = {
            name: dist
            for name, dist in search_space.items()
            if name not in trial.distributions or trial.distributions.get(name) == dist
        } | {
            name: dist
            for name, dist in trial.distributions.items()
            if name not in search_space or search_space.get(name) == dist
        }

    return search_space, next_cached_trial_number


class UnionSearchSpace:
    """Calculates the union search space of a Study."""

    def __init__(self, include_pruned: bool = True) -> None:
        self._cached_trial_number: int = -1
        self._search_space: dict[str, BaseDistribution] | None = None
        self._study_id: int | None = None
        self._include_pruned = include_pruned

    def calculate(self, study: Study) -> dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        elif self._study_id != study._study_id:
            raise ValueError("UnionSearchSpace cannot handle multiple studies.")

        self._search_space, self._cached_trial_number = _calculate_union(
            study.get_trials(deepcopy=False),
            self._include_pruned,
            self._search_space,
            self._cached_trial_number,
        )
        search_space = self._search_space or {}
        return copy.deepcopy(dict(sorted(search_space.items(), key=lambda x: x[0])))
