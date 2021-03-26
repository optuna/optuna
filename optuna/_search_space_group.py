from typing import Dict
from typing import List

from optuna.distributions import BaseDistribution


class SearchSpaceGroup(object):
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
