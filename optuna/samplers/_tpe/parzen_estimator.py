from typing import Any, Callable, List
from typing import Dict
from typing import NamedTuple
from typing import Optional
import numpy as np

from optuna import distributions
from optuna.distributions import BaseDistribution

from optuna.samplers._tpe.probability_distributions import BaseProbabilityDistribution
from optuna.samplers._tpe.probability_distributions import CategoricalDistribution
from optuna.samplers._tpe.probability_distributions import UnivariateGaussianDistribution
from optuna.samplers._tpe.probability_distributions import DiscreteUnivariateGaussianDistribution
from optuna.samplers._tpe.probability_distributions import MixtureDistribution
from optuna.samplers._tpe.probability_distributions import ProductDistribution

EPS = 1e-12
SIGMA0_MAGNITUDE = 0.2

_DISTRIBUTION_CLASSES = (
    distributions.CategoricalDistribution,
    distributions.FloatDistribution,
    distributions.IntDistribution,
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
            ("multivariate", bool),
        ],
    )
):
    pass


class _ParzenEstimator:
    def __init__(
        self,
        observations: Dict[str, np.ndarray],
        search_space: Dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: Optional[np.ndarray] = None,
    ) -> None:
        self._search_space = search_space

        n_observations = next(iter(observations.values())).size
        if predetermined_weights is not None:
            assert next(iter(observations.values())).size == len(predetermined_weights)

        weights = _ParzenEstimator._calculate_weights(predetermined_weights, n_observations, parameters)

        transformed_observations = _ParzenEstimator._transform_to_uniform(observations, search_space)

        product_distributions: List[List[BaseProbabilityDistribution]]
        product_distributions = [self._calculate_distributions(transformed_observations[param], search_space[param], parameters) for param in search_space]
        print(weights)
        self._mixture_distribution = MixtureDistribution([
                                        (weights[i], ProductDistribution([dists[i] for dists in product_distributions])) 
                                        for i in range(weights.size)])

    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:
        sampled = [self._mixture_distribution.sample(rng) for _ in range(size)]
        return _ParzenEstimator._transform_from_uniform({param_name: np.array([sample[i] for sample in sampled]) for i, param_name in enumerate(self._search_space)}, self._search_space)

    def log_pdf(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        n_samples = len(next(iter(samples_dict.values())))
        transformed_samples_dict = _ParzenEstimator._transform_to_uniform(samples_dict, self._search_space)
        transformed_samples = [[transformed_samples_dict[param_name][i] for param_name in self._search_space] for i in range(n_samples)]
        return np.array([self._mixture_distribution.log_pdf(transformed_sample) for transformed_sample in transformed_samples])

    @classmethod
    def _calculate_weights(cls, predetermined_weights: Optional[np.ndarray], n_observations: int, parameters: _ParzenEstimatorParameters) -> np.ndarray:

        # We decide the weights.
        consider_prior = parameters.consider_prior
        prior_weight = parameters.prior_weight
        weights_func = parameters.weights

        if n_observations == 0:
            consider_prior = True

        if predetermined_weights is None:
            w = weights_func(n_observations)[:n_observations]
            if w is not None:
                if np.any(w < 0):
                    raise ValueError(
                        f"The `weights` function is not allowed to return negative values {w}. "
                        + f"The argument of the `weights` function is {n_observations}."
                    )
                if len(w) > 0 and np.sum(w) <= 0:
                    raise ValueError(
                        f"The `weight` function is not allowed to return all-zero values {w}."
                        + f" The argument of the `weights` function is {n_observations}."
                    )
                if not np.all(np.isfinite(w)):
                    raise ValueError(
                        "The `weights`function is not allowed to return infinite or NaN values "
                        + f"{w}. The argument of the `weights` function is {n_observations}."
                    )
        else:
            w = predetermined_weights[:n_observations]

        if consider_prior:
            # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
            # unexpected size.
            weights = np.zeros(n_observations + 1)
            weights[:-1] = w
            weights[-1] = prior_weight
        else:
            weights = w
        weights /= weights.sum()
        return weights

    @classmethod
    def _transform_to_uniform(cls, samples_dict: Dict[str, np.ndarray], search_space: Dict[str, BaseDistribution]) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in samples_dict.items():
            distribution = search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(
                distribution,
                (distributions.FloatDistribution, distributions.IntDistribution),
            ):
                if distribution.log:
                    samples = np.log(samples)

            transformed[param_name] = samples
        return transformed


    @classmethod
    def _transform_from_uniform(
        self, samples_dict: Dict[str, np.ndarray], search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in samples_dict.items():
            distribution = search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(distribution, distributions.FloatDistribution):
                if distribution.log:
                    transformed[param_name] = np.exp(samples)
                elif distribution.step is not None:
                    q = distribution.step
                    samples = np.round((samples - distribution.low) / q) * q + distribution.low
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
                else:
                    transformed[param_name] = samples
            elif isinstance(distribution, distributions.IntDistribution):
                if distribution.log:
                    samples = np.round(np.exp(samples))
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
                else:
                    q = distribution.step
                    samples = np.round((samples - distribution.low) / q) * q + distribution.low
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
            elif isinstance(distribution, distributions.CategoricalDistribution):
                transformed[param_name] = samples

        return transformed

    def _calculate_distributions(
        self,
        transformed_observations: np.ndarray,
        search_space: BaseDistribution,
        parameters: _ParzenEstimatorParameters,
    ):
        assert isinstance(search_space, _DISTRIBUTION_CLASSES)
        if isinstance(search_space, distributions.CategoricalDistribution):
            return self._calculate_categorical_distributions(transformed_observations, search_space.choices, parameters)
        else:

            if search_space.log:
                low = np.log(search_space.low)
                high = np.log(search_space.high)
            else:
                low = search_space.low
                high = search_space.high
            step = search_space.step
            
            # TODO(contramundum53): This is a hack and should be fixed.
            if step is not None and search_space.log:
                low = np.log(search_space.low - step / 2)
                high = np.log(search_space.high + step / 2)
                step = None

            return self._calculate_numerical_distributions(transformed_observations, low, high, step, parameters)

    def _calculate_categorical_distributions(
        self, observations: np.ndarray, choices: List[Any], parameters: _ParzenEstimatorParameters
    ) -> List[BaseProbabilityDistribution]:

        # TODO(kstoneriv3): This the bandwidth selection rule might not be optimal.
        observations = observations.astype(int)
        n_observations = len(observations)
        consider_prior = parameters.consider_prior
        prior_weight = parameters.prior_weight

        if n_observations == 0:
            consider_prior = True

        if consider_prior:
            shape = (n_observations + 1, len(choices))
            assert prior_weight is not None
            value = prior_weight / (n_observations + 1)
        else:
            shape = (n_observations, len(choices))
            assert prior_weight is not None
            value = prior_weight / n_observations
        weights = np.full(shape, fill_value=value)
        weights[np.arange(n_observations), observations] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return [CategoricalDistribution(w) for w in weights]

    def _calculate_numerical_distributions(
        self, observations: np.ndarray, low: float, high: float, step: Optional[float], parameters: _ParzenEstimatorParameters
    ) -> List[BaseProbabilityDistribution]:

        n_observations = len(observations)
        consider_prior = parameters.consider_prior
        consider_endpoints = parameters.consider_endpoints
        consider_magic_clip = parameters.consider_magic_clip
        multivariate = parameters.multivariate
        assert low is not None
        assert high is not None

        if n_observations == 0:
            consider_prior = True

        prior_mu = 0.5 * (low + high)
        prior_sigma = 1.0 * (high - low)

        if consider_prior:
            mus = np.empty(n_observations + 1)
            mus[:n_observations] = observations
            mus[n_observations] = prior_mu
            sigmas = np.empty(n_observations + 1)
        else:
            mus = observations
            sigmas = np.empty(n_observations)

        if multivariate:
            sigmas[:] = SIGMA0_MAGNITUDE * max(n_observations, 1) ** (-1.0 / (len(self._search_space) + 4)) * (high - low)
        else:
            sorted_indices = np.argsort(mus)
            sorted_mus = mus[sorted_indices]
            sorted_mus_with_endpoints = np.empty(len(mus) + 2, dtype=float)
            sorted_mus_with_endpoints[0] = low
            sorted_mus_with_endpoints[1:-1] = sorted_mus
            sorted_mus_with_endpoints[-1] = high

            sorted_sigmas = np.maximum(
                sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
            )

            if not consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                sorted_sigmas[-1] = sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]

            sigmas[:] = sorted_sigmas[np.argsort(sorted_indices)]

        # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(mus)))
        else:
            minsigma = EPS
        sigmas = np.asarray(np.clip(sigmas, minsigma, maxsigma))

        if consider_prior:
            sigmas[n_observations] = prior_sigma

        if step is None:
            return [UnivariateGaussianDistribution(mu, sigma, low, high) for mu, sigma in zip(mus, sigmas)]
        else:
            return [DiscreteUnivariateGaussianDistribution(mu, sigma, low, high, step) for mu, sigma in zip(mus, sigmas)]
