from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy.special
from scipy.stats import truncnorm

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters

EPS = 1e-12

_NUMERICAL_DISTRIBUTION_CLASSES = (
    distributions.UniformDistribution,
    distributions.LogUniformDistribution,
    distributions.IntUniformDistribution,
    distributions.IntLogUniformDistribution,
    distributions.DiscreteUniformDistribution,
)


# TODO(kstoenriv3): We need to consider if we really need `_ParzenEstimatorParameters` class
# as there are alreaky so many attributes in `_MultivariateParzenEstimator` class.
# `_ParzenEstimatorParameters` class is not greatly recuding the complexity of the code.


class _MultivariateParzenEstimator:
    def __init__(
        self,
        multivariate_samples: Dict[str, np.ndarray],
        search_space: Dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
    ) -> None:

        self._search_space = search_space
        self._parameters = parameters
        self._weights = self._calculate_weights(multivariate_samples)

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
        multivariate_samples = self._transform_to_uniform(multivariate_samples)

        # Transformed multivariate_samples are needed for following operations.
        self._sigmas0 = self._precompute_sigmas0(multivariate_samples)

        self._mus = {}  # type: Dict[str, Optional[np.ndarray]]
        self._sigmas = {}  # type: Dict[str, Optional[np.ndarray]]
        self._categorical_weights = {}  # type: Dict[str, Optional[np.ndarray]]
        for param_name, dist in search_space.items():
            samples = multivariate_samples[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                mus = sigmas = None
                categorical_weights = self._calculate_categorical_params(samples, param_name)
            else:
                mus, sigmas = self._calculate_parzen_est_params(samples, param_name)
                categorical_weights = None
            self._mus[param_name] = mus
            self._sigmas[param_name] = sigmas
            self._categorical_weights[param_name] = categorical_weights

    def _calculate_weights(self, multivariate_samples: Dict[str, np.ndarray]) -> np.ndarray:

        # We decide the weights.
        consider_prior = self._parameters.consider_prior
        prior_weight = self._parameters.prior_weight
        weights_func = self._parameters.weights
        sample_size = next(iter(multivariate_samples.values())).size
        if consider_prior:
            weights = np.zeros(sample_size + 1)
            weights[:-1] = weights_func(sample_size)
            weights[-1] = prior_weight
        else:
            weights = weights_func(sample_size)
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
            if isinstance(distribution, distributions.UniformDistribution):
                transformed[param_name] = samples
            elif isinstance(distribution, distributions.LogUniformDistribution):
                transformed[param_name] = np.log(samples)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                transformed[param_name] = samples
            elif isinstance(distribution, distributions.IntUniformDistribution):
                transformed[param_name] = samples
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                transformed[param_name] = np.log(samples)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                transformed[param_name] = samples
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
        return transformed

    def _transform_from_uniform(
        self, multivariate_samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in multivariate_samples.items():
            distribution = self._search_space[param_name]
            if isinstance(distribution, distributions.UniformDistribution):
                transformed[param_name] = samples
            elif isinstance(distribution, distributions.LogUniformDistribution):
                transformed[param_name] = np.exp(samples)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                q = self._q[param_name]
                low = self._low[param_name]
                high = self._high[param_name]
                samples = (np.round((samples - low) / q - 0.5) + 0.5) * q + low
                transformed[param_name] = np.clip(samples, low, high)
            elif isinstance(distribution, distributions.IntUniformDistribution):
                q = self._q[param_name]
                samples = np.round(samples / q) * q
                samples = np.clip(samples, self._low[param_name], self._high[param_name])
                transformed[param_name] = samples.astype(int)
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                transformed[param_name] = np.round(np.exp(samples))
            elif isinstance(distribution, distributions.CategoricalDistribution):
                transformed[param_name] = samples
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
        return transformed

    def _precompute_sigmas0(
        self, multivariate_samples: Dict[str, np.ndarray]
    ) -> Union[np.ndarray, float]:

        # Categorical parameters are not considered.
        param_names = list(multivariate_samples.keys())
        rescaled_samples_list = []
        for param_name in param_names:
            if isinstance(self._search_space[param_name], _NUMERICAL_DISTRIBUTION_CLASSES):
                high = self._high[param_name]
                low = self._low[param_name]
                assert high is not None
                assert low is not None
                samples = multivariate_samples[param_name]
                samples = (samples - low) / (high - low)
                rescaled_samples_list.append(samples)
            else:  # Categorical parameters are ignored.
                continue

        # When the number of parameters is zero, we cannot determine sigma0.
        if len(rescaled_samples_list) == 0:
            raise ValueError(
                "multivariate_samples must contain at least one parameters."
                "But multivariate_samples = {}.".format(multivariate_samples)
            )
        # When the number of samples is zero, we return 1.0.
        elif len(rescaled_samples_list[0]) == 0:
            return 1.0

        rescaled_samples = np.array(rescaled_samples_list).T

        # compute distance matrix of samples
        distances = np.linalg.norm(
            rescaled_samples[:, None, :] - rescaled_samples[None, :, :], axis=2
        )
        distances += np.diag(np.ones(len(samples)) * np.infty)

        return np.min(distances, axis=1)

    def _calculate_categorical_params(self, samples: np.ndarray, param_name: str) -> np.ndarray:

        samples = samples.astype(int)
        distribution = self._search_space[param_name]
        assert isinstance(distribution, distributions.CategoricalDistribution)
        choices = distribution.choices
        consider_prior = self._parameters.consider_prior
        prior_weights = self._parameters.prior_weight
        if consider_prior:
            weights = prior_weights / samples.size * np.ones((samples.size + 1, len(choices)))
            weights[np.arange(samples.size), samples] += 1
        else:
            weights = prior_weights / samples.size * np.ones((samples.size, len(choices)))
            weights[np.arange(samples.size), samples] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _calculate_parzen_est_params(
        self, samples: np.ndarray, param_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        samples_size = samples.size
        consider_prior = self._parameters.consider_prior
        consider_magic_clip = self._parameters.consider_magic_clip
        sigmas0 = self._sigmas0
        low = self._low[param_name]
        high = self._high[param_name]
        assert low is not None
        assert high is not None

        if samples_size == 0:
            consider_prior = True

        if consider_prior:

            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low)

            mus = np.zeros(samples_size + 1)
            mus[:samples_size] = samples
            mus[samples_size] = prior_mu

            sigmas = np.zeros(samples_size + 1)
            sigmas[:samples_size] = sigmas0 * (high - low)
            sigmas[samples_size] = prior_sigma

        else:
            mus = samples
            sigmas = sigmas0 * (high - low)

        # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(mus)))
        else:
            minsigma = EPS
        sigmas = np.clip(sigmas, minsigma, maxsigma)

        return mus, sigmas

    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:

        multivariate_samples = {}
        active = rng.choice(len(self._weights), size, p=self._weights)

        for param_name, dist in self._search_space.items():

            if isinstance(dist, distributions.CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                weights = categorical_weights[active, :]
                samples = _MultivariateParzenEstimator._sample_from_categorical_dist(rng, weights)

            else:
                # We restore parameters of parzen estimators.
                low = self._low[param_name]
                high = self._high[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                assert low is not None
                assert high is not None
                assert mus is not None
                assert sigmas is not None

                # We sample from truncnorm.
                trunc_low = (low - mus[active]) / sigmas[active]
                trunc_high = (high - mus[active]) / sigmas[active]
                samples = np.full((), fill_value=high + 1.0, dtype=np.float64)
                while (samples >= high).any():
                    samples = np.where(
                        samples < high,
                        samples,
                        truncnorm.rvs(
                            trunc_low,
                            trunc_high,
                            size=size,
                            loc=mus[active],
                            scale=sigmas[active],
                            random_state=rng,
                        ),
                    )
            multivariate_samples[param_name] = samples
        multivariate_samples = self._transform_from_uniform(multivariate_samples)
        return multivariate_samples

    def log_pdf(self, multivariate_samples: Dict[str, np.ndarray]) -> np.ndarray:

        multivariate_samples = self._transform_to_uniform(multivariate_samples)
        weight_size = len(self._weights)
        sample_size = next(iter(multivariate_samples.values())).size
        if sample_size == 0:
            return np.asarray([], dtype=float)
        # We compute log pdf (compoment_log_pdf)
        # for each sample in multivariate_samples (of size sample_size)
        # for each component of `_MultivariateParzenEstimator` (of size weight_size).
        component_log_pdf = np.zeros((sample_size, weight_size))
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

    @staticmethod
    def _normal_cdf(x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + scipy.special.erf(z))

    @staticmethod
    def _sample_from_categorical_dist(
        rng: np.random.RandomState, probabilities: np.ndarray
    ) -> np.ndarray:

        sample_size = probabilities.shape[0]
        rnd_quantile = rng.rand(sample_size)
        cum_probs = np.cumsum(probabilities, axis=1)
        return np.sum(cum_probs < rnd_quantile[..., None], axis=1)
