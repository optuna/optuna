from __future__ import annotations
from collections import defaultdict

import optuna

from optuna._experimental import experimental_class
from optuna.samplers._base import BaseSampler
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


@experimental_class("4.2.0")
class BaseGASampler(BaseSampler):
    _GENERATION_KEY = "BaseGASampler:generation"
    _POPULATION_CACHE_KEY_PREFIX = "BaseGASampler:population:"
    _PARENT_CACHE_KEY_PREFIX = "BaseGASampler:parent:"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._GENERATION_KEY = f"{cls.__name__}:generation"
        cls._POPULATION_CACHE_KEY_PREFIX = f"{cls.__name__}:population:"
        cls._PARENT_CACHE_KEY_PREFIX = f"{cls.__name__}:parent:"

    @classmethod
    def _get_generation_key(cls):
        return cls._GENERATION_KEY

    @classmethod
    def _get_population_cache_key_prefix(cls):
        return cls._POPULATION_CACHE_KEY_PREFIX

    def __init__(self, population_size: int):
        self._population_size = population_size

    def select_parent(self, study: optuna.Study, generation: int) -> list[FrozenTrial]:
        raise NotImplementedError

    def get_generation(self, study: optuna.Study, trial: FrozenTrial) -> int:
        """
        get the current generation number
        """
        trials = study._get_trials(deepcopy=False, use_cache=True)
        generation_to_population = defaultdict(list)
        for trial in filter(lambda trial: trial.state == TrialState.COMPLETE, trials):
            generation = trial.system_attrs.get(self._get_generation_key(), None)
            if generation is not None:
                generation_to_population[generation].append(trial)

        if len(generation_to_population) == 0:
            return 0
        elif (
            len(generation_to_population[max(generation_to_population.keys())])
            < self._population_size
        ):
            return max(generation_to_population.keys())
        else:
            return max(generation_to_population.keys()) + 1

    def set_generation_key(
        self, study: optuna.Study, trial: FrozenTrial, generation: int
    ):
        study._storage.set_trial_system_attr(
            trial._trial_id, self._get_generation_key(), generation
        )

    def get_population(self, study: optuna.Study, generation: int) -> list[FrozenTrial]:
        """ """
        return [
            trial
            for trial in study._get_trials(deepcopy=False, use_cache=True)
            if trial.system_attrs.get(self._get_generation_key(), None) == generation
        ]

    def get_parent_population(
        self, study: optuna.Study, generation: int
    ) -> list[FrozenTrial]:
        """ """
        if generation == 0:
            return []

        study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
        cached_parent_population = study_system_attrs.get(
            self._PARENT_CACHE_KEY_PREFIX + str(generation), None
        )

        if cached_parent_population is not None:
            return [
                study._storage.get_trial(trial_id)
                for trial_id in cached_parent_population
            ]
        else:
            parent_population = self.select_parent(study, generation)
            study._storage.set_study_system_attr(
                study._study_id,
                self._PARENT_CACHE_KEY_PREFIX + str(generation),
                [trial._trial_id for trial in parent_population],
            )
            return parent_population
