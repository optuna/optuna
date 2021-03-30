import copy
from typing import Dict
from typing import List
from typing import Optional

from optuna.distributions import BaseDistribution
from optuna.study import BaseStudy
from optuna.trial import TrialState


class _SearchSpaceGroup(object):
    def __init__(self) -> None:
        self._group: List[Dict[str, BaseDistribution]] = []

    @property
    def group(self) -> List[Dict[str, BaseDistribution]]:
        return self._group

    def add_distributions(self, distributions: Dict[str, BaseDistribution]) -> None:
        self._group = _add_distributions(self.group, distributions)


def _add_distributions(
    group: List[Dict[str, BaseDistribution]], distributions: Dict[str, BaseDistribution]
) -> List[Dict[str, BaseDistribution]]:
    if len(distributions) == 0:
        return group

    for search_space in group:
        _keys = set(search_space.keys())
        _dist_keys = set(distributions.keys())

        if _keys.isdisjoint(_dist_keys):
            continue

        if _keys < _dist_keys:
            return _add_distributions(
                group, {name: distributions[name] for name in _dist_keys - _keys}
            )

        if _keys > _dist_keys:
            group.append(distributions)
            group.append({name: search_space[name] for name in _keys - _dist_keys})
            group.remove(search_space)
            return group

        _intersection = _keys & _dist_keys
        group.append({name: search_space[name] for name in _intersection})
        if len(_keys - _intersection) > 0:
            group.append({name: search_space[name] for name in _keys - _intersection})
        group.remove(search_space)
        return _add_distributions(
            group, {name: distributions[name] for name in _dist_keys - _intersection}
        )

    group.append(distributions)

    return group


class _GroupDecomposedSearchSpace(object):
    def __init__(self, include_pruned: bool = False) -> None:
        self._cursor: int = -1
        self._search_space: Optional[_SearchSpaceGroup] = None
        self._study_id: Optional[int] = None
        self._include_pruned = include_pruned

    def calculate(self, study: BaseStudy) -> _SearchSpaceGroup:
        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`_GroupDecomposedSearchSpace` cannot handle multiple studies.")

        states_of_interest = [TrialState.COMPLETE]

        if self._include_pruned:
            states_of_interest.append(TrialState.PRUNED)

        next_cursor = self._cursor
        for trial in reversed(study.get_trials(deepcopy=False)):
            if self._cursor > trial.number:
                break

            if not trial.state.is_finished():
                next_cursor = trial.number

            if trial.state not in states_of_interest:
                continue

            if self._search_space is None:
                self._search_space = _SearchSpaceGroup()
                self._search_space.add_distributions(trial.distributions)
                continue

            self._search_space.add_distributions(trial.distributions)

        self._cursor = next_cursor
        search_space = self._search_space or _SearchSpaceGroup()

        return copy.deepcopy(search_space)
