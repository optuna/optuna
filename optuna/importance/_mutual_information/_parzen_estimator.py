from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.special

from optuna import distributions
from optuna.distributions import BaseDistribution

EPS = 1e-12
_DISTRIBUTION_CLASSES = (
    distributions.UniformDistribution,
    distributions.LogUniformDistribution,
    distributions.DiscreteUniformDistribution,
    distributions.IntUniformDistribution,
    distributions.IntLogUniformDistribution,
    distributions.CategoricalDistribution,
)
_NUMERICAL_DISTRIBUTION_CLASSES = (
    distributions.UniformDistribution,
    distributions.LogUniformDistribution,
    distributions.IntUniformDistribution,
    distributions.IntLogUniformDistribution,
    distributions.DiscreteUniformDistribution,
)


class _ParzenEstimatorParameters(
    NamedTuple(
        "_ParzenEstimatorParameters",
        [
            ("consider_prior", bool),
            ("prior_weight", Optional[float]),
            ("consider_magic_clip", bool),
            ("consider_endpoints", bool),
            ("weights", Callable[[int], np.ndarray]),
        ],
    )
):
    pass


class _MultivariateParzenEstimator:
    def __init__(
        self,
        multivariate_observations: Dict[str, np.ndarray],
        search_space: Dict[str, BaseDistribution],
    ) -> None:

        self._search_space = search_space
        self._n_observations = next(iter(multivariate_observations.values())).size
        self._parameters = _ParzenEstimatorParameters(
            False, 0.0, False, False, lambda n: np.ones(n)
        )
        self._weights = self._calculate_weights(multivariate_observations)

        self._low = {}  # type: Dict[str, Optional[float]]
        self._high = {}  # type: Dict[str, Optional[float]]
        self._q = {}  # type: Dict[str, Optional[float]]
        for param_name, dist in search_space.items():
            if isinstance(dist, distributions.CategoricalDistribution):
                low = high = q = None
            else:
                low, high, q = self._calculate_parzen_bounds(dist)
            self._low[param_name] = low
            self._high[param_name] = high
            self._q[param_name] = q

        # `_low`, `_high`, `_q` are needed for transformation.
        multivariate_observations = self._transform_to_uniform(multivariate_observations)

        # Transformed multivariate_observations are needed for following operations.
        self._sigmas0 = self._precompute_sigmas0(multivariate_observations)

        self._mus = {}  # type: Dict[str, Optional[np.ndarray]]
        self._sigmas = {}  # type: Dict[str, Optional[np.ndarray]]
        self._categorical_weights = {}  # type: Dict[str, Optional[np.ndarray]]
        for param_name, dist in search_space.items():
            observations = multivariate_observations[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                mus = sigmas = None
                categorical_weights = self._calculate_categorical_params(observations, param_name)
            else:
                mus, sigmas = self._calculate_parzen_est_params(observations, param_name)
                categorical_weights = None
            self._mus[param_name] = mus
            self._sigmas[param_name] = sigmas
            self._categorical_weights[param_name] = categorical_weights

    def log_pdf(self, multivariate_samples: Dict[str, np.ndarray]) -> np.ndarray:

        multivariate_samples = self._transform_to_uniform(multivariate_samples)
        n_weights = len(self._weights)
        n_samples = next(iter(multivariate_samples.values())).size
        if n_samples == 0:
            return np.asarray([], dtype=float)
        # We compute log pdf (compoment_log_pdf)
        # for each sample in multivariate_samples (of size n_samples)
        # for each component of `_MultivariateParzenEstimator` (of size n_weights).
        component_log_pdf = np.zeros((n_samples, n_weights))
        for param_name, dist in self._search_space.items():
            samples = multivariate_samples[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                log_pdf = np.log(categorical_weights.T[samples, :])
            else:
                # We restore parameters of parzen estimators.
                low = self._low[param_name]
                high = self._high[param_name]
                q = self._q[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                assert low is not None
                assert high is not None
                assert mus is not None
                assert sigmas is not None

                cdf_func = _MultivariateParzenEstimator._normal_cdf
                p_accept = cdf_func(high, mus, sigmas) - cdf_func(low, mus, sigmas)
                if q is None:
                    distance = samples[:, None] - mus
                    mahalanobis = distance / sigmas
                    z = np.sqrt(2 * np.pi) * sigmas
                    coefficient = 1 / z / p_accept
                    log_pdf = -0.5 * mahalanobis ** 2 + np.log(coefficient)
                else:
                    upper_bound = np.minimum(samples + q / 2.0, high)
                    lower_bound = np.maximum(samples - q / 2.0, low)
                    cdf = cdf_func(upper_bound[:, None], mus[None], sigmas[None]) - cdf_func(
                        lower_bound[:, None], mus[None], sigmas[None]
                    )
                    log_pdf = np.log(cdf) - np.log(p_accept)
            component_log_pdf += log_pdf
        ret = scipy.special.logsumexp(component_log_pdf + np.log(self._weights), axis=1)
        return ret

    def _calculate_weights(self, multivariate_observations: Dict[str, np.ndarray]) -> np.ndarray:

        # We decide the weights.
        consider_prior = self._parameters.consider_prior
        prior_weight = self._parameters.prior_weight
        weights_func = self._parameters.weights
        n_observations = self._n_observations
        if consider_prior:
            weights = np.empty(n_observations + 1)
            weights[:-1] = weights_func(n_observations)
            weights[-1] = prior_weight
        else:
            weights = weights_func(n_observations)
        weights /= weights.sum()
        return weights

    def _calculate_parzen_bounds(
        self, distribution: BaseDistribution
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:

        # We calculate low and high.
        if isinstance(distribution, distributions.UniformDistribution):
            low = distribution.low
            high = distribution.high
            q = None
        elif isinstance(distribution, distributions.LogUniformDistribution):
            low = np.log(distribution.low)
            high = np.log(distribution.high)
            q = None
        elif isinstance(distribution, distributions.DiscreteUniformDistribution):
            q = distribution.q
            low = distribution.low - 0.5 * q
            high = distribution.high + 0.5 * q
        elif isinstance(distribution, distributions.IntUniformDistribution):
            q = distribution.step
            low = distribution.low - 0.5 * q
            high = distribution.high + 0.5 * q
        elif isinstance(distribution, distributions.IntLogUniformDistribution):
            low = np.log(distribution.low - 0.5)
            high = np.log(distribution.high + 0.5)
            q = None
        else:
            distribution_list = [
                distributions.UniformDistribution.__name__,
                distributions.LogUniformDistribution.__name__,
                distributions.DiscreteUniformDistribution.__name__,
                distributions.IntUniformDistribution.__name__,
                distributions.IntLogUniformDistribution.__name__,
                distributions.CategoricalDistribution.__name__,
            ]
            raise NotImplementedError(
                "The distribution {} is not implemented. "
                "The parameter distribution should be one of the {}".format(
                    distribution, distribution_list
                )
            )

        assert low < high

        return low, high, q

    def _transform_to_uniform(
        self, multivariate_samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in multivariate_samples.items():
            distribution = self._search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(
                distribution,
                (distributions.LogUniformDistribution, distributions.IntLogUniformDistribution),
            ):
                samples = np.log(samples)

            transformed[param_name] = samples
        return transformed

    def _precompute_sigmas0(self, multivariate_samples: Dict[str, np.ndarray]) -> np.ndarray:
        # We use Scott's rule for bandwidth selection.
        # This rule was used in the BOHB paper.
        n_samples = next(iter(multivariate_samples.values())).size
        n_samples = max(n_samples, 1)
        n_params = len(multivariate_samples)
        # TODO(kstoneriv3): The constant factor 0.2 might not be optimal.
        return 0.2 * n_samples ** (-1.0 / (n_params + 4)) * np.ones(n_samples)

    def _calculate_categorical_params(
        self, observations: np.ndarray, param_name: str
    ) -> np.ndarray:

        observations = observations.astype(int)
        n_observations = self._n_observations
        distribution = self._search_space[param_name]
        assert isinstance(distribution, distributions.CategoricalDistribution)
        choices = distribution.choices
        consider_prior = self._parameters.consider_prior
        prior_weights = self._parameters.prior_weight
        if consider_prior:
            shape = (n_observations + 1, len(choices))
        else:
            shape = (n_observations, len(choices))
        weights = np.full(shape, fill_value=prior_weights / n_observations)
        weights[np.arange(n_observations), observations] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _calculate_parzen_est_params(
        self, observations: np.ndarray, param_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_observations = self._n_observations
        consider_prior = self._parameters.consider_prior
        consider_magic_clip = self._parameters.consider_magic_clip
        sigmas0 = self._sigmas0
        low = self._low[param_name]
        high = self._high[param_name]
        assert low is not None
        assert high is not None

        if n_observations == 0:
            consider_prior = True

        if consider_prior:

            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low)

            mus = np.empty(n_observations + 1)
            mus[:n_observations] = observations
            mus[n_observations] = prior_mu

            sigmas = np.empty(n_observations + 1)
            sigmas[:n_observations] = sigmas0 * (high - low)
            sigmas[n_observations] = prior_sigma

        else:
            mus = observations
            sigmas = sigmas0 * (high - low)

        # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(mus)))
        else:
            minsigma = EPS
        sigmas = np.clip(sigmas, minsigma, maxsigma)

        return mus, sigmas

    @staticmethod
    def _normal_cdf(x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + scipy.special.erf(z))
