import copy
from typing import Optional

import optuna
from optuna._search_space_group import SearchSpaceGroup
from optuna.study import BaseStudy


class _GroupDecomposedSearchSpace(object):
    def __init__(self, include_pruned: bool = False) -> None:
        self._cursor: int = -1
        self._search_space: Optional[SearchSpaceGroup] = None
        self._study_id: Optional[int] = None
        self._include_pruned = include_pruned

    def calculate(self, study: BaseStudy) -> SearchSpaceGroup:
        if self._study_id is None:
            self._study_id = study._study_id
        else:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            if self._study_id != study._study_id:
                raise ValueError("`_GroupDecomposedSearchSpace` cannot handle multiple studies.")

        states_of_interest = [optuna.trial.TrialState.COMPLETE]

        if self._include_pruned:
            states_of_interest.append(optuna.trial.TrialState.PRUNED)

        next_cursor = self._cursor
        for trial in reversed(study.get_trials(deepcopy=False)):
            if self._cursor > trial.number:
                break

            if not trial.state.is_finished():
                next_cursor = trial.number

            if trial.state not in states_of_interest:
                continue

            if self._search_space is None:
                self._search_space = SearchSpaceGroup()
                self._search_space.add_distributions(trial.distributions)
                continue

            self._search_space.add_distributions(trial.distributions)

        self._cursor = next_cursor
        search_space = self._search_space or SearchSpaceGroup()

        return copy.deepcopy(search_space)
