import copy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from optuna.distributions import BaseDistribution
from optuna.study import BaseStudy
from optuna.trial import TrialState


class _SearchSpaceGroup(object):
    def __init__(self) -> None:
        self._search_spaces: List[Dict[str, BaseDistribution]] = []

    @property
    def search_spaces(self) -> List[Dict[str, BaseDistribution]]:
        return self._search_spaces

    def add_distributions(self, distributions: Dict[str, BaseDistribution]) -> None:
        self._search_spaces = _add_distributions(self.search_spaces, distributions)


def _add_distributions(
    search_spaces: List[Dict[str, BaseDistribution]], distributions: Dict[str, BaseDistribution]
) -> List[Dict[str, BaseDistribution]]:
    # Assumes that the elements of `search_spaces` are disjoint.

    if len(distributions) == 0:
        return search_spaces

    for search_space in search_spaces:
        keys = set(search_space.keys())
        dist_keys = set(distributions.keys())

        if keys.isdisjoint(dist_keys):
            continue

        if keys < dist_keys:
            return _add_distributions(
                search_spaces, {name: distributions[name] for name in dist_keys - keys}
            )

        if keys > dist_keys:
            search_spaces.append(distributions)
            search_spaces.append({name: search_space[name] for name in keys - dist_keys})
            search_spaces.remove(search_space)
            return search_spaces

        intersection = keys & dist_keys
        search_spaces.append({name: search_space[name] for name in intersection})
        if len(keys - intersection) > 0:
            search_spaces.append({name: search_space[name] for name in keys - intersection})
        search_spaces.remove(search_space)
        return _add_distributions(
            search_spaces, {name: distributions[name] for name in dist_keys - intersection}
        )

    search_spaces.append(distributions)

    return search_spaces


class _GroupDecomposedSearchSpace(object):
    def __init__(self, include_pruned: bool = False) -> None:
        self._search_space = _SearchSpaceGroup()
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

        states_of_interest: Tuple[TrialState, ...]
        if self._include_pruned:
            states_of_interest = (TrialState.COMPLETE, TrialState.PRUNED)
        else:
            states_of_interest = (TrialState.COMPLETE,)

        for trial in study.get_trials(deepcopy=False, states=states_of_interest):
            self._search_space.add_distributions(trial.distributions)

        return copy.deepcopy(self._search_space)
