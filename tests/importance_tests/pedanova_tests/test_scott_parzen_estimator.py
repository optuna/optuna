from __future__ import annotations

import pytest

import numpy as np

from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution
from optuna.importance._ped_anova.scott_parzen_estimator import _ScottParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _BatchedCategoricalDistributions
from optuna.samplers._tpe.parzen_estimator import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.parzen_estimator import _MixtureOfProductDistribution


DIST_TYPES = ["int", "cat"]


@pytest.mark.parametrize("dist_type", DIST_TYPES)
def test_init_scott_parzen_estimator(dist_type: str) -> None:
    counts = np.array([1, 1, 1, 1]).astype(float)
    is_cat = dist_type == "cat"
    pe = _ScottParzenEstimator(
        param_name="a",
        dist=(
            IntDistribution(low=0, high=counts.size - 1) if not is_cat
            else CategoricalDistribution(choices=["a"*i for i in range(counts.size)])
        ),
        counts=counts,
        consider_prior=False,
        prior_weight=0.0,
    )
    assert len(pe._mixture_distribution.distributions) == 1
    assert pe.n_steps == counts.size
    target_pe = pe._mixture_distribution.distributions[0]
    if is_cat:
        assert isinstance(target_pe, _BatchedCategoricalDistributions)
    else:
        assert isinstance(target_pe, _BatchedDiscreteTruncNormDistributions)


