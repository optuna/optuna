from __future__ import annotations

from collections.abc import Sequence
import itertools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna import logging
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)


class BruteForceSampler(BaseSampler):
    """Sampler using brute-force search.

    The sampler iterates over all possible combinations of parameter values.

    Note:
        This sampler requires that the search space is determined in the first trial.
        It does not support dynamic search spaces where trial suggestions depend on
        the values of previously suggested parameters within the same trial.

    Args:
        seed:
            A seed to fix the order of trials as the search order is randomly shuffled.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = LazyRandomState(seed)
        self._grid: list[dict[str, Any]] | None = None
        self._shuffled_grid: list[dict[str, Any]] | None = None
        self._trial_params_cache: dict[int, dict[str, Any]] = {}

    def infer_relative_search_space(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> dict[str, BaseDistribution]:
        # In this design, the search space is inferred after the first trial,
        # so this method returns an empty dictionary.
        return {}

    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        # All sampling is handled by `sample_independent` in this design.
        return {}

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # If the grid is not built yet (i.e., during Trial 0), we must run in a
        # 'discovery' mode to define the search space. We sample deterministically.
        if self._shuffled_grid is None:
            if isinstance(param_distribution, CategoricalDistribution):
                return param_distribution.choices[0]
            # Extend with other distribution types if needed, e.g., return low value.
            raise NotImplementedError(
                "BruteForceSampler only supports CategoricalDistribution in the first trial."
            )

        trial_id = trial._trial_id
        if trial_id not in self._trial_params_cache:
            # This is the first suggest call for this trial. Time to select
            # a full parameter set for it from our pre-computed grid.
            if not self._shuffled_grid:
                _logger.warning(
                    "All combinations have been sampled. Reshuffling and starting over."
                )
                self._shuffled_grid = list(self._grid)
                self._rng.rng.shuffle(self._shuffled_grid)

            # Pop the next pre-computed parameter set.
            params_for_this_trial = self._shuffled_grid.pop()
            self._trial_params_cache[trial_id] = params_for_this_trial

        # Return the specific value for the parameter that was asked for.
        return self._trial_params_cache[trial_id][param_name]

    def after_trial(
        self,
        study: "optuna.study.Study",
        trial: "FrozenTrial",
        state: "TrialState",
        values: Sequence[float] | None,
    ) -> None:
        # After the first trial is complete, we can inspect its distributions
        # to build our full grid of parameters for all subsequent trials.
        if trial.number == 0 and state == TrialState.COMPLETE:
            search_space = trial.distributions
            self._grid = _get_all_search_params(search_space)
            self._shuffled_grid = list(self._grid)
            self._rng.rng.shuffle(self._shuffled_grid)
            _logger.info(f"Brute force grid built with {len(self._grid)} trial combinations.")

        # Clean up the cache for the trial that just finished.
        if trial._trial_id in self._trial_params_cache:
            del self._trial_params_cache[trial._trial_id]


def _get_all_search_params(search_space: dict[str, BaseDistribution]) -> list[dict[str, Any]]:
    param_names = list(search_space.keys())
    param_values = []
    for param_name in param_names:
        dist = search_space[param_name]
        if not isinstance(dist, CategoricalDistribution):
            raise ValueError(
                f"Parameter '{param_name}' has an unsupported distribution: {type(dist).__name__}."
            )
        param_values.append(dist.choices)

    all_combinations = itertools.product(*param_values)
    all_params = [dict(zip(param_names, combo)) for combo in all_combinations]
    return all_params
