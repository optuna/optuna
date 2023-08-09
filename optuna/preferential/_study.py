from __future__ import annotations

import copy
from typing import Any
from typing import Container
from typing import Iterable

import optuna
from optuna import logging
from optuna.distributions import BaseDistribution
from optuna.preferential._system_attrs import get_preferences
from optuna.preferential._system_attrs import report_preferences
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)
_SYSTEM_ATTR_PREFERENTIAL_STUDY = "preference:is_preferential"
_SYSTEM_ATTR_COMPARISON_READY = "preference:comparison_ready"


class PreferentialStudy:
    def __init__(self, study: optuna.Study) -> None:
        self._study = study

    @property
    def trials(self) -> list[FrozenTrial]:
        return self._study.trials

    @property
    def best_trials(self) -> list[FrozenTrial]:
        ready_trials = [
            t
            for t in self._study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE, TrialState.RUNNING)
            )
            if t.system_attrs.get(_SYSTEM_ATTR_COMPARISON_READY) is True
        ]
        preferences = get_preferences(self._study, deepcopy=False)
        worse_numbers = {worse.number for _, worse in preferences}
        return [copy.deepcopy(t) for t in ready_trials if t.number not in worse_numbers]

    @property
    def study_name(self) -> str:
        return self._study.study_name

    @property
    def user_attrs(self) -> dict[str, Any]:
        return self._study.user_attrs

    @property
    def preferences(self) -> list[tuple[FrozenTrial, FrozenTrial]]:
        return self.get_preferences(deepcopy=True)

    def get_trials(
        self,
        deepcopy: bool = True,
        states: Container[optuna.trial.TrialState] | None = None,
    ) -> list[FrozenTrial]:
        return self._study.get_trials(deepcopy, states)

    def ask(self, fixed_distributions: dict[str, BaseDistribution] | None = None) -> optuna.Trial:
        return self._study.ask(fixed_distributions)

    def add_trial(self, trial: FrozenTrial) -> None:
        self._study.add_trial(trial)

    def add_trials(self, trials: Iterable[FrozenTrial]) -> None:
        self._study.add_trials(trials)

    def report_preference(
        self,
        better_trials: FrozenTrial | list[FrozenTrial],
        worse_trials: FrozenTrial | list[FrozenTrial],
    ) -> None:
        if not isinstance(better_trials, list):
            better_trials = [better_trials]
        if not isinstance(worse_trials, list):
            worse_trials = [worse_trials]

        report_preferences(self._study, [(b, w) for b in better_trials for w in worse_trials])

    def get_preferences(self, *, deepcopy: bool = True) -> list[tuple[FrozenTrial, FrozenTrial]]:
        return get_preferences(self._study, deepcopy=deepcopy)

    def set_user_attr(self, key: str, value: Any) -> None:
        self._study.set_user_attr(key, value)

    def mark_comparison_ready(self, trial_or_number: optuna.Trial | int) -> None:
        storage = self._study._storage
        if isinstance(trial_or_number, optuna.Trial):
            trial_id = trial_or_number._trial_id
        elif isinstance(trial_or_number, int):
            trial_id = storage.get_trial_id_from_study_id_trial_number(
                self._study._study_id, trial_or_number
            )
        else:
            raise RuntimeError("Unexpected trial type")
        storage.set_trial_system_attr(trial_id, _SYSTEM_ATTR_COMPARISON_READY, True)


def create_study(
    *,
    storage: str | optuna.storages.BaseStorage | None = None,
    sampler: BaseSampler | None = None,
    study_name: str | None = None,
    load_if_exists: bool = False,
) -> PreferentialStudy:
    try:
        study = optuna.create_study(
            storage=storage,
            sampler=sampler or RandomSampler(),
            study_name=study_name,
        )
        study._storage.set_study_system_attr(
            study._study_id, _SYSTEM_ATTR_PREFERENTIAL_STUDY, True
        )
        return PreferentialStudy(study)

    except optuna.exceptions.DuplicatedStudyError:
        if load_if_exists:
            assert study_name is not None
            assert storage is not None

            _logger.info(
                "Using an existing study with name '{}' instead of "
                "creating a new one.".format(study_name)
            )
            return load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
            )
        else:
            raise


def load_study(
    *,
    study_name: str | None,
    storage: str | optuna.storages.BaseStorage,
    sampler: BaseSampler | None = None,
) -> PreferentialStudy:
    study = optuna.load_study(
        study_name=study_name, storage=storage, sampler=sampler or RandomSampler()
    )
    system_attrs = study._storage.get_study_system_attrs(study._study_id)
    if not system_attrs.get(_SYSTEM_ATTR_PREFERENTIAL_STUDY):
        raise ValueError("The study is not a PreferentialStudy.")
    return PreferentialStudy(study)
