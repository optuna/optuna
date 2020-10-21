from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np

from optuna import distributions
from optuna import pruners
from optuna import samplers
from optuna import stepwise
from optuna.trial import Trial


__all__ = [
    "step_feature_fraction",
    "step_num_leaves",
    "step_bagging",
    "step_feature_fraction_stage2",
    "step_regularization_factors",
    "step_min_data_in_leaf",
    "default_lgb_steps",
]


class _MakeStep:
    def __init__(
        self,
        sampler: Optional[samplers.BaseSampler] = None,
        pruner: Optional[pruners.BasePruner] = None,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.sampler = sampler
        self.pruner = pruner
        self.n_trials = n_trials
        self.timeout = timeout


class _InclusiveUniformStep(stepwise.Step):
    """Include `high` value from uniform distributions."""

    _eps = 1e-12

    def suggest(self, trial: Trial) -> Dict[str, Any]:
        values = {}
        for name, dist in self.distributions.items():
            if isinstance(dist, distributions.UniformDistribution):
                dist.high += self._eps
                value = trial._suggest(name, dist)
                value = min(value, dist.high)
            else:
                value = trial._suggest(name, dist)
            values[name] = value
        return values


def step_feature_fraction(
    pruner: Optional[pruners.BasePruner] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Tuple[str, stepwise.GridStep]:
    return "feature_fraction", stepwise.GridStep(
        search_space={"feature_fraction": np.round(np.linspace(0.4, 1, 7), decimals=1).tolist()},
        pruner=pruner,
        n_trials=n_trials,
        timeout=timeout,
    )


class _FeatureFractionStage2(_MakeStep):
    _interval_size = 0.16

    def __call__(self, params: Dict[str, Any]) -> stepwise.Step:
        feature_fraction = params.get("feature_fraction", 1.0)
        search_space = np.linspace(
            feature_fraction - self._interval_size / 2,
            feature_fraction + self._interval_size / 2,
            self.n_trials,
        )
        search_space = search_space[(search_space >= 0.4) & (search_space <= 1.0)]
        search_space = np.round(search_space, decimals=2).tolist()
        return stepwise.GridStep(
            search_space={"feature_fraction": search_space},
            pruner=self.pruner,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )


def step_feature_fraction_stage2(
    pruner: Optional[pruners.BasePruner] = None,
    n_trials: Optional[int] = 6,
    timeout: Optional[int] = None,
) -> Tuple[str, stepwise.StepType]:
    return (
        "feature_fraction_stage2",
        _FeatureFractionStage2(pruner=pruner, n_trials=n_trials, timeout=timeout),
    )


class _NumLeavesStep(_MakeStep):
    _default_tree_depth: int = 8

    def __call__(self, params: Dict[str, Any]) -> stepwise.Step:
        tree_depth = params.get("max_depth", self._default_tree_depth)
        max_num_leaves = 2 ** tree_depth

        return stepwise.Step(
            distributions={"max_depth": distributions.IntUniformDistribution(2, max_num_leaves)},
            pruner=self.pruner,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )


def step_num_leaves(
    sampler: Optional[samplers.BaseSampler] = None,
    pruner: Optional[pruners.BasePruner] = None,
    n_trials: Optional[int] = 20,
    timeout: Optional[int] = None,
) -> Tuple[str, stepwise.StepType]:
    return "num_leaves", _NumLeavesStep(sampler, pruner, n_trials, timeout)


def step_bagging(
    sampler: Optional[samplers.BaseSampler] = None,
    pruner: Optional[pruners.BasePruner] = None,
    n_trials: Optional[int] = 10,
    timeout: Optional[int] = None,
) -> Tuple[str, stepwise.Step]:
    dists = {
        "bagging_fraction": distributions.UniformDistribution(0.4, 1.0),
        "bagging_freq": distributions.IntUniformDistribution(1, 7),
    }
    return (
        "bagging",
        _InclusiveUniformStep(
            distributions=dists,
            sampler=sampler,
            pruner=pruner,
            n_trials=n_trials,
            timeout=timeout,
        ),
    )


def step_regularization_factors(
    sampler: Optional[samplers.BaseSampler] = None,
    pruner: Optional[pruners.BasePruner] = None,
    n_trials: Optional[int] = 20,
    timeout: Optional[int] = None,
) -> Tuple[str, stepwise.Step]:
    return (
        "regularization_factors",
        stepwise.Step(
            distributions={
                "lambda_l1": distributions.LogUniformDistribution(1e-8, 10.0),
                "lambda_l2": distributions.LogUniformDistribution(1e-8, 10.0),
            },
            sampler=sampler,
            pruner=pruner,
            n_trials=n_trials,
            timeout=timeout,
        ),
    )


def step_min_data_in_leaf(
    pruner: Optional[pruners.BasePruner] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Tuple[str, stepwise.GridStep]:

    return (
        "min_data_in_leaf",
        stepwise.GridStep(
            search_space={"min_child_samples": [5, 10, 25, 50, 100]},
            pruner=pruner,
            n_trials=n_trials,
            timeout=timeout,
        ),
    )


def default_lgb_steps() -> stepwise.StepListType:
    return [
        step_feature_fraction(),
        step_num_leaves(),
        step_bagging(),
        step_feature_fraction_stage2(),
        step_regularization_factors(),
        step_min_data_in_leaf(),
    ]
