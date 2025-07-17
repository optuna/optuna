from __future__ import annotations

import decimal
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Sequence

import optuna
from optuna import logging
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
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
        self._grid: List[Dict[str, Any]] | None = None
        self._shuffled_grid: List[Dict[str, Any]] | None = None
        self._trial_params_cache: Dict[int, Dict[str, Any]] = {}

    def infer_relative_search_space(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._shuffled_grid is None:
            # Discovery phase for Trial 0.
            if isinstance(param_distribution, CategoricalDistribution):
                return param_distribution.choices[0]
            elif isinstance(param_distribution, (IntDistribution, FloatDistribution)):
                return param_distribution.low
            raise NotImplementedError(f"Unsupported distribution {type(param_distribution)}.")

        trial_id = trial._trial_id
        if trial_id not in self._trial_params_cache:
            assert self._grid is not None, "Grid should be initialized after the first trial."
            if not self._shuffled_grid:
                _logger.warning(
                    "All combinations have been sampled. Reshuffling and starting over."
                )
                self._shuffled_grid = list(self._grid)
                self._rng.rng.shuffle(cast(Any, self._shuffled_grid))

            params_for_this_trial = self._shuffled_grid.pop(0)
            self._trial_params_cache[trial_id] = params_for_this_trial

        return self._trial_params_cache[trial_id][param_name]

    def after_trial(
        self,
        study: "optuna.study.Study",
        trial: "FrozenTrial",
        state: "TrialState",
        values: Sequence[float] | None,
    ) -> None:
        if trial.number == 0 and state == TrialState.COMPLETE:
            search_space = trial.distributions
            self._grid = _get_all_search_params(search_space)
            self._shuffled_grid = list(self._grid)
            self._rng.rng.shuffle(cast(Any, self._shuffled_grid))
            _logger.info(f"Brute force grid built with {len(self._grid)} trial combinations.")

        if trial._trial_id in self._trial_params_cache:
            del self._trial_params_cache[trial._trial_id]


def _get_all_search_params(search_space: Dict[str, BaseDistribution]) -> List[Dict[str, Any]]:
    param_names = list(search_space.keys())
    # This type hint is the fix for the mypy error.
    param_values: List[Sequence[Any]] = []
    for param_name in param_names:
        dist = search_space[param_name]
        if isinstance(dist, CategoricalDistribution):
            param_values.append(dist.choices)
        elif isinstance(dist, IntDistribution):
            param_values.append(list(range(dist.low, dist.high + 1, dist.step)))
        elif isinstance(dist, FloatDistribution):
            if dist.step is None:
                raise ValueError("FloatDistribution must have a step for brute-force search.")
            low = decimal.Decimal(str(dist.low))
            high = decimal.Decimal(str(dist.high))
            step = decimal.Decimal(str(dist.step))
            values: List[Any] = []
            val = low
            while val <= high:
                values.append(float(val))
                val += step
            param_values.append(values)
        else:
            raise ValueError(f"Unsupported distribution: {type(dist).__name__}.")

    all_combinations = itertools.product(*param_values)
    all_params = [dict(zip(param_names, combo)) for combo in all_combinations]
    return all_params
