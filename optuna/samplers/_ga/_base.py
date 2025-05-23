from __future__ import annotations

import abc
from typing import Any

import optuna
from optuna.samplers._base import BaseSampler
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


# TODO(gen740): Add the experimental decorator?
class BaseGASampler(BaseSampler, abc.ABC):
    """Base class for Genetic Algorithm (GA) samplers.

    Genetic Algorithm samplers generate new trials by mimicking natural selection, using
    generations and populations to iteratively improve solutions. This base class defines the
    interface for GA samplers in Optuna and provides utility methods for managing generations and
    populations.

    The selection process is handled by :meth:`~BaseGASampler.select_parent`, which must be
    implemented by subclasses to define the parent selection strategy.

    Generation and population management is facilitated by methods like
    :meth:`~BaseGASampler.get_generation` and :meth:`~BaseGASampler.get_population`, ensuring
    consistent tracking and selection.

    Note:
        This class should be extended by subclasses that define specific GA sampling strategies,
        including parent selection and crossover operations.
    """

    _GENERATION_KEY = "BaseGASampler:generation"
    _PARENT_CACHE_KEY_PREFIX = "BaseGASampler:parent:"

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._GENERATION_KEY = f"{cls.__name__}:generation"
        cls._PARENT_CACHE_KEY_PREFIX = f"{cls.__name__}:parent:"

    @classmethod
    def _get_generation_key(cls) -> str:
        return cls._GENERATION_KEY

    @classmethod
    def _get_parent_cache_key_prefix(cls) -> str:
        return cls._PARENT_CACHE_KEY_PREFIX

    def __init__(self, population_size: int | None):
        self._population_size = population_size

    @property
    def population_size(self) -> int | None:
        return self._population_size

    @population_size.setter
    def population_size(self, value: int) -> None:
        self._population_size = value

    @abc.abstractmethod
    def select_parent(self, study: optuna.Study, generation: int) -> list[FrozenTrial]:
        """Select parent trials from the population for the given generation.

        This method is called once per generation to select parents from
        the population of the current generation.

        Output of this function is cached in the study system attributes.

        This method must be implemented in a subclass to define the specific selection strategy.

        Args:
            study:
                Target study object.
            generation:
                Target generation number.

        Returns:
            List of parent frozen trials.
        """
        raise NotImplementedError

    def get_trial_generation(self, study: optuna.Study, trial: FrozenTrial) -> int:
        """Get the generation number of the given trial.

        This method returns the generation number of the specified trial. If the generation number
        is not set in the trial's system attributes, it will calculate and set the generation
        number.

        The current generation number depends on the maximum generation number of all completed
        trials.

        Args:
            study:
                Study object which trial belongs to.
            trial:
                Trial object to get the generation number.

        Returns:
            Generation number of the given trial.
        """
        generation = trial.system_attrs.get(self._get_generation_key(), None)
        if generation is not None:
            return generation

        trials = study._get_trials(deepcopy=False, states=[TrialState.COMPLETE], use_cache=True)

        max_generation, max_generation_count = 0, 0

        for t in reversed(trials):
            generation = t.system_attrs.get(self._get_generation_key(), -1)

            if generation < max_generation:
                continue
            elif generation > max_generation:
                max_generation = generation
                max_generation_count = 1
            else:
                max_generation_count += 1

        assert self._population_size is not None, "Population size must be set."
        if max_generation_count < self._population_size:
            generation = max_generation
        else:
            generation = max_generation + 1
        study._storage.set_trial_system_attr(
            trial._trial_id, self._get_generation_key(), generation
        )
        return generation

    def get_population(self, study: optuna.Study, generation: int) -> list[FrozenTrial]:
        """Get the population of the given generation.

        Args:
            study:
                Target study object.
            generation:
                Target generation number.

        Returns:
            List of frozen trials in the given generation.
        """
        return [
            trial
            for trial in study._get_trials(
                deepcopy=False, states=[TrialState.COMPLETE], use_cache=True
            )
            if trial.system_attrs.get(self._get_generation_key(), None) == generation
        ]

    def get_parent_population(self, study: optuna.Study, generation: int) -> list[FrozenTrial]:
        """Get the parent population of the given generation.

        This method caches the parent population in the study's system attributes.

        Args:
            study:
                Target study object.
            generation:
                Target generation number.

        Returns:
            List of parent frozen trials. If `generation == 0`, returns an empty list.
        """
        if generation == 0:
            return []

        study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
        cached_parent_population_ids = study_system_attrs.get(
            self._get_parent_cache_key_prefix() + str(generation), None
        )

        if cached_parent_population_ids is not None:
            trials = study._get_trials(deepcopy=False)
            parent_population_ids = set(cached_parent_population_ids)
            return [trial for trial in trials if trial._trial_id in parent_population_ids]
        else:
            parent_population = self.select_parent(study, generation)
            study._storage.set_study_system_attr(
                study._study_id,
                self._get_parent_cache_key_prefix() + str(generation),
                [trial._trial_id for trial in parent_population],
            )
            return parent_population
