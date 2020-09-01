from typing import Any
from unittest.mock import patch

import numpy as np

from optuna import distributions
from optuna.samplers._tpe.multivariate_parzen_estimator import _MultivariateParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters

# We skip the test of `precomputat_sigma0`.
target = (
    "optuna.samplers._tpe.multivariate_parzen_estimator."
    "_MultivariateParzenEstimator._precompute_sigmas0"
)


@patch(target, return_value=1.0)
def test_init_MultivariateParzenEstimator(mock: Any) -> None:

    search_space = {
        "a": distributions.UniformDistribution(1.0, 100.0),
        "b": distributions.LogUniformDistribution(1.0, 100.0),
        "c": distributions.DiscreteUniformDistribution(1.0, 100.0, 3.0),
        "d": distributions.IntUniformDistribution(1, 100),
        "e": distributions.IntLogUniformDistribution(1, 100),
        "f": distributions.CategoricalDistribution(["x", "y", "z"]),
    }

    multivariate_samples = {
        "a": np.array([1.0, 2.0]),
        "b": np.array([1.0, 2.0]),
        "c": np.array([1.0, 2.0]),
        "d": np.array([1, 2]),
        "e": np.array([1, 2]),
        "f": np.array([1, 2]),
    }

    # Test a case when consider prior is True
    parameters = _ParzenEstimatorParameters(
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=lambda x: np.arange(x) + 1.0,
    )

    mpe = _MultivariateParzenEstimator(multivariate_samples, search_space, parameters)

    weights = [0.25, 0.5, 0.25]
    q = {"a": None, "b": None, "c": 3.0, "d": 1.0, "e": None, "f": None}
    low = {
        "a": 1.0,
        "b": np.log(1.0),
        "c": 1.0 - 1.5,
        "d": 1.0 - 0.5,
        "e": np.log(1.0 - 0.5),
        "f": None,
    }
    high = {
        "a": 100.0,
        "b": np.log(100.0),
        "c": 100.0 + 1.5,
        "d": 100.0 + 0.5,
        "e": np.log(100.0 + 0.5),
        "f": None,
    }

    assert np.all(mpe._weights == weights)
    assert mpe._q == q
    assert mpe._low == low
    assert mpe._high == high

    sigmas = {
        "a": [99.0, 99.0, 99.0],
        "b": [np.log(100.0), np.log(100), np.log(100)],
        "c": [102.0, 102.0, 102.0],
        "d": [100.0, 100.0, 100.0],
        "e": [
            np.log(100.5) - np.log(0.5),
            np.log(100.5) - np.log(0.5),
            np.log(100.5) - np.log(0.5),
        ],
        "f": None,
    }
    mus = {
        "a": [1.0, 2.0, 50.5],
        "b": [np.log(1.0), np.log(2.0), np.log(100) / 2.0],
        "c": [1.0, 2.0, 50.5],
        "d": [1.0, 2.0, 50.5],
        "e": [np.log(1.0), np.log(2.0), (np.log(100.5) + np.log(0.5)) / 2.0],
        "f": None,
    }
    categorical_weights = {
        "a": None,
        "b": None,
        "c": None,
        "d": None,
        "e": None,
        "f": np.array([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
    }

    for param_name, values in mpe._sigmas.items():
        assert np.all(
            np.equal(mpe._sigmas[param_name], sigmas[param_name])
        ), 'parameter "{}"'.format(param_name)
        assert np.all(np.equal(mpe._mus[param_name], mus[param_name])), "parameter: {}".format(
            param_name
        )
        assert np.all(
            np.equal(mpe._categorical_weights[param_name], categorical_weights[param_name])
        ), "parameter: {}".format(param_name)

    # Test a case when consider prior is False
    parameters = _ParzenEstimatorParameters(
        consider_prior=False,
        prior_weight=1.0,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=lambda x: np.arange(x) + 1.0,
    )

    mpe = _MultivariateParzenEstimator(multivariate_samples, search_space, parameters)

    weights = [1.0 / 3.0, 2.0 / 3.0]
    assert np.all(mpe._weights == weights)

    sigmas = {
        "a": [99.0, 99.0],
        "b": [np.log(100.0), np.log(100)],
        "c": [102.0, 102.0],
        "d": [100.0, 100.0],
        "e": [np.log(100.5) - np.log(0.5), np.log(100.5) - np.log(0.5)],
        "f": None,
    }
    mus = {
        "a": [1.0, 2.0],
        "b": [np.log(1.0), np.log(2.0)],
        "c": [1.0, 2.0],
        "d": [1.0, 2.0],
        "e": [np.log(1.0), np.log(2.0)],
        "f": None,
    }
    categorical_weights = {
        "a": None,
        "b": None,
        "c": None,
        "d": None,
        "e": None,
        "f": np.array([[0.2, 0.6, 0.2], [0.2, 0.2, 0.6]]),
    }

    for param_name, values in mpe._sigmas.items():
        assert np.all(
            np.equal(mpe._sigmas[param_name], sigmas[param_name])
        ), 'parameter "{}"'.format(param_name)
        assert np.all(np.equal(mpe._mus[param_name], mus[param_name])), "parameter: {}".format(
            param_name
        )
        assert np.all(
            np.equal(mpe._categorical_weights[param_name], categorical_weights[param_name])
        ), "parameter: {}".format(param_name)


# We skip the test of `precomputat_sigma0`.
target = (
    "optuna.samplers._tpe.multivariate_parzen_estimator."
    "_MultivariateParzenEstimator._precompute_sigmas0"
)


@patch(target, return_value=np.array([1e-8]))
def test_sample_MultivariateParzenEstimator(mock: Any) -> None:

    search_space = {
        "a": distributions.UniformDistribution(1.0, 100.0),
        "b": distributions.LogUniformDistribution(1.0, 100.0),
        "c": distributions.DiscreteUniformDistribution(1.0, 100.0, 3.0),
        "d": distributions.IntUniformDistribution(1, 100),
        "e": distributions.IntLogUniformDistribution(1, 100),
        "f": distributions.CategoricalDistribution(["x", "y", "z"]),
    }

    multivariate_samples = {
        "a": np.array([1.0]),
        "b": np.array([1.0]),
        "c": np.array([1.0]),
        "d": np.array([1]),
        "e": np.array([1]),
        "f": np.array([1]),
    }

    # Test a case when consider prior is True
    parameters = _ParzenEstimatorParameters(
        consider_prior=False,
        prior_weight=0.0,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=lambda x: np.arange(x) + 1.0,
    )

    mpe = _MultivariateParzenEstimator(multivariate_samples, search_space, parameters)

    # We test the shape of the output.
    output_multivariate_samples = mpe.sample(np.random.RandomState(0), 3)
    for param_name in output_multivariate_samples.keys():
        assert output_multivariate_samples[param_name].shape == (3,)

    # We test the values of the output.
    # As we set `consider_prior` = False, and sigmas to be 1e-8,
    # the samples almost equals to the input to the `__init__`.
    output_multivariate_samples = mpe.sample(np.random.RandomState(0), 1)
    for param_name, samples in output_multivariate_samples.items():
        if samples.dtype == str:
            assert samples[0] == "y", "parameter {}".format(param_name)
        else:
            assert np.allclose(samples, multivariate_samples[param_name]), "parameter {}".format(
                param_name
            )

    # We test the output when the seeds are fixed.
    assert output_multivariate_samples == mpe.sample(np.random.RandomState(0), 1)


# We skip the test of `precomputat_sigma0`.
target = (
    "optuna.samplers._tpe.multivariate_parzen_estimator."
    "_MultivariateParzenEstimator._precompute_sigmas0"
)


@patch(target, return_value=np.array([1]))
def test_log_pdf_MultivariateParzenEstimator(mock: Any) -> None:

    search_space = {
        "a": distributions.UniformDistribution(1.0, 100.0),
        "b": distributions.LogUniformDistribution(1.0, 100.0),
        "c": distributions.DiscreteUniformDistribution(1.0, 100.0, 3.0),
        "d": distributions.IntUniformDistribution(1, 100),
        "e": distributions.IntLogUniformDistribution(1, 100),
        "f": distributions.CategoricalDistribution(["x", "y", "z"]),
    }

    multivariate_samples = {
        "a": np.array([1.0]),
        "b": np.array([1.0]),
        "c": np.array([1.0]),
        "d": np.array([1]),
        "e": np.array([1]),
        "f": np.array([1]),
    }

    # Test a case when consider prior is True
    parameters = _ParzenEstimatorParameters(
        consider_prior=False,
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=True,
        weights=lambda x: np.arange(x) + 1.0,
    )

    mpe = _MultivariateParzenEstimator(multivariate_samples, search_space, parameters)

    # We test the values of the output.
    # As we set `consider_prior` = False, and sigmas to be 1e-8,
    # the samples almost equals to the input to the `__init__`.
    output_multivariate_samples = mpe.sample(np.random.RandomState(0), 100)
    log_pdf = mpe.log_pdf(multivariate_samples)
    output_log_pdf = mpe.log_pdf(output_multivariate_samples)
    for param_name, samples in output_multivariate_samples.items():
        assert np.all(log_pdf >= output_log_pdf), "parameter {}".format(param_name)
