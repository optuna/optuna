import numpy as np

from optuna.samplers.tpe.parzen_estimator import ParzenEstimator
from optuna.samplers.tpe.sampler import default_weights


class TestParzenEstimator(object):

    @staticmethod
    def test_calculate_with_prior():
        # type: () -> None

        consider_prior = True
        consider_magic_clip = True
        consider_endpoints = False
        # If no observation exists, the function returns the prior Gaussian distribution.
        weights, mus, sigma = ParzenEstimator._calculate([], -1.0, 1.0, prior_weight=1.0,
                                                         consider_prior=consider_prior,
                                                         consider_magic_clip=consider_magic_clip,
                                                         consider_endpoints=consider_endpoints,
                                                         weights_func=default_weights)
        np.testing.assert_almost_equal(weights, [1.0])
        np.testing.assert_almost_equal(mus, [0.0])
        np.testing.assert_almost_equal(sigma, [2.0])

        # If a single observation exists, the function returns two Gaussian distributions.
        weights, mus, sigma = ParzenEstimator._calculate([0.4], -1., 1., prior_weight=1.,
                                                         consider_prior=consider_prior,
                                                         consider_magic_clip=consider_magic_clip,
                                                         consider_endpoints=consider_endpoints,
                                                         weights_func=default_weights)
        # TODO(Yanase): Check values
        assert len(weights) == 2
        assert len(mus) == 2
        assert len(sigma) == 2

        # If two observation exist, the function returns three Gaussian distributions.
        weights, mus, sigma = ParzenEstimator._calculate([-0.4, 0.4], -1., 1., prior_weight=1.,
                                                         consider_prior=consider_prior,
                                                         consider_magic_clip=consider_magic_clip,
                                                         consider_endpoints=consider_endpoints,
                                                         weights_func=default_weights)
        # TODO(Yanase): Check values
        assert len(weights) == 3
        assert len(mus) == 3
        assert len(sigma) == 3

    @staticmethod
    def test_calculate_without_prior():
        # type: () -> None

        consider_prior = False
        consider_magic_clip = True
        consider_endpoints = False
        # If no observation exists, the function returns no distribution.
        weights, mus, sigma = ParzenEstimator._calculate([], -1.0, 1.0, prior_weight=1.0,
                                                         consider_prior=consider_prior,
                                                         consider_magic_clip=consider_magic_clip,
                                                         consider_endpoints=consider_endpoints,
                                                         weights_func=default_weights)
        assert len(weights) == 0
        assert len(mus) == 0
        assert len(sigma) == 0

        # If a single observation exists, the function returns one Gaussian distributions.
        weights, mus, sigma = ParzenEstimator._calculate([0.4], -1., 1., prior_weight=1.,
                                                         consider_prior=consider_prior,
                                                         consider_magic_clip=consider_magic_clip,
                                                         consider_endpoints=consider_endpoints,
                                                         weights_func=default_weights)
        # TODO(Yanase): Check values
        assert len(weights) == 1
        assert len(mus) == 1
        assert len(sigma) == 1

        # If two observation exist, the function returns two Gaussian distributions.
        weights, mus, sigma = ParzenEstimator._calculate([-0.4, 0.4], -1., 1., prior_weight=1.,
                                                         consider_prior=consider_prior,
                                                         consider_magic_clip=consider_magic_clip,
                                                         consider_endpoints=consider_endpoints,
                                                         weights_func=default_weights)
        # TODO(Yanase): Check values
        assert len(weights) == 2
        assert len(mus) == 2
        assert len(sigma) == 2
