import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from ConfigSpace import Float

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution

import optuna.samplers._tpe.parzen_estimator as new_parzen_estimator
import optuna.samplers._tpe.old_parzen_estimator as old_parzen_estimator
from optuna.samplers._tpe.sampler import default_weights
from scipy import stats

import pytest
parametrize_observations = pytest.mark.parametrize("observations", [
    {"x0": np.array([0., 0., 1., 1., 2., 2.]),
     "x1": np.array([2., 7., 3., 4., 9., 1.]),
     "x2": np.array([4., 3., 6.6, 7.2, 1., 10.]),
     "x3": np.array([5., 1., 10., 2., 1., 3.]),},
])

parametrize_search_space = pytest.mark.parametrize("search_space", [{
    "x0": CategoricalDistribution(choices=[-1, 0, 1]),
    "x1": FloatDistribution(1, 10),
    "x2": FloatDistribution(1, 10, step=0.2),
    "x3": IntDistribution(1, 10),
    },
    {
    "x0": CategoricalDistribution(choices=[-1, 0, 1]),
    "x1": FloatDistribution(1, 10, log=True),
    "x2": FloatDistribution(1, 10, step=0.2),
    "x3": IntDistribution(1, 10, log=True),}
    ])


parametrize_test_points = pytest.mark.parametrize("test_points", [
    {"x0": np.array([0, 0, 1, 1, 2, 2]),
     "x1": np.array([4, 1, 2, 5, 7, 9]),
     "x2": np.array([3, 8, 7, 8, 9, 2]),
     "x3": np.array([6, 2, 2, 1, 4, 5]),},
])
parametrize_predetermined_weights = pytest.mark.parametrize("predetermined_weights", [None, np.array([4., 2., 3., 4., 1., 3.])])

@parametrize_observations
@parametrize_search_space
@parametrize_test_points
@pytest.mark.parametrize("consider_prior", [True, False])
@pytest.mark.parametrize("prior_weight", [1.0, 2.0])
@pytest.mark.parametrize("consider_magic_clip", [True, False])
@pytest.mark.parametrize("consider_endpoints", [True, False])
@pytest.mark.parametrize("weights", [default_weights, lambda x: np.ones(x)])
@pytest.mark.parametrize("multivariate", [True, False])
@parametrize_predetermined_weights
def test_logpdf_unchanged(
    observations: Dict[str, np.ndarray],
    search_space: Dict[str, BaseDistribution],
    consider_prior: bool,
    prior_weight: Optional[float],
    consider_magic_clip: bool,
    consider_endpoints: bool,
    weights: Callable[[int], np.ndarray],
    multivariate: bool,
    predetermined_weights: Optional[np.ndarray],
    test_points: Dict[str, np.array]
) -> None:
    old_pe = old_parzen_estimator._ParzenEstimator(
        observations=observations,
        search_space=search_space,
        parameters=old_parzen_estimator._ParzenEstimatorParameters(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            weights=weights,
            multivariate=multivariate,
        ),
        predetermined_weights=predetermined_weights,
    )

    old_logpdf = old_pe.log_pdf(test_points)

    new_pe = new_parzen_estimator._ParzenEstimator(
        observations=observations,
        search_space=search_space,
        parameters=new_parzen_estimator._ParzenEstimatorParameters(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            weights=weights,
            multivariate=multivariate,
        ),
        predetermined_weights=predetermined_weights,
    )

    new_logpdf = new_pe.log_pdf(test_points)

    assert np.allclose(old_logpdf, new_logpdf)


@parametrize_observations
@parametrize_search_space
@pytest.mark.parametrize("consider_prior", [True, False])
@pytest.mark.parametrize("prior_weight", [1.0, 2.0])
@pytest.mark.parametrize("consider_magic_clip", [True, False])
@pytest.mark.parametrize("consider_endpoints", [True, False])
@pytest.mark.parametrize("weights", [default_weights, lambda x: np.ones(x)])
@pytest.mark.parametrize("multivariate", [True, False])
@parametrize_predetermined_weights
def test_sample_unchanged(
    observations: Dict[str, np.ndarray],
    search_space: Dict[str, BaseDistribution],
    consider_prior: bool,
    prior_weight: Optional[float],
    consider_magic_clip: bool,
    consider_endpoints: bool,
    weights: Callable[[int], np.ndarray],
    multivariate: bool,
    predetermined_weights: Optional[np.ndarray],
):
    seed = 0

    old2_pe = old_parzen_estimator._ParzenEstimator(
        observations=observations,
        search_space=search_space,
        parameters=old_parzen_estimator._ParzenEstimatorParameters(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            weights=weights,
            multivariate=multivariate,
        ),
        predetermined_weights=predetermined_weights,
    )

    old2_samples = old2_pe.sample(rng=np.random.RandomState(seed), size=100)
    
    new_pe = new_parzen_estimator._ParzenEstimator(
        observations=observations,
        search_space=search_space,
        parameters=new_parzen_estimator._ParzenEstimatorParameters(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            weights=weights,
            multivariate=multivariate,
        ),
        predetermined_weights=predetermined_weights,
    )

    new_samples = new_pe.sample(rng=np.random.RandomState(seed), size=100)
    for param_name in search_space.keys():
        assert np.allclose(old2_samples[param_name], new_samples[param_name])

