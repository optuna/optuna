from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np

from optuna import distributions
from optuna._imports import _LazyImport
from optuna.distributions import BaseDistribution


if TYPE_CHECKING:
    import scipy.special as special
    import scipy.stats as stats
else:
    special = _LazyImport("scipy.special")
    stats = _LazyImport("scipy.stats")


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
        self._parameters = parameters
        self._n_observations = next(iter(observations.values())).size
        if predetermined_weights is not None:
            assert self._n_observations == len(predetermined_weights)
        self._weights = self._calculate_weights(predetermined_weights)

        self._low: Dict[str, Optional[float]] = {}
        self._high: Dict[str, Optional[float]] = {}
        self._q: Dict[str, Optional[float]] = {}
        for param_name, dist in search_space.items():
            if isinstance(dist, distributions.CategoricalDistribution):
                low = high = q = None
            else:
                low, high, q = self._calculate_parzen_bounds(dist)
            self._low[param_name] = low
            self._high[param_name] = high
            self._q[param_name] = q

        # `_low`, `_high`, `_q` are needed for transformation.
        observations = self._transform_to_uniform(observations)

        # Transformed `observations` might be needed for following operations.
        self._sigmas0 = self._precompute_sigmas0(observations)

        self._mus: Dict[str, Optional[np.ndarray]] = {}
        self._sigmas: Dict[str, Optional[np.ndarray]] = {}
        self._categorical_weights: Dict[str, Optional[np.ndarray]] = {}
        categorical_weights: Optional[np.ndarray]
        for param_name, dist in search_space.items():
            param_observations = observations[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                mus = sigmas = None
                categorical_weights = self._calculate_categorical_params(
                    param_observations, param_name
                )
            else:
                mus, sigmas = self._calculate_numerical_params(param_observations, param_name)
                categorical_weights = None
            self._mus[param_name] = mus
            self._sigmas[param_name] = sigmas
            self._categorical_weights[param_name] = categorical_weights

    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:

        samples_dict = {}
        active = rng.choice(len(self._weights), size, p=self._weights)

        for param_name, dist in self._search_space.items():

            if isinstance(dist, distributions.CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                weights = categorical_weights[active, :]
                samples = _ParzenEstimator._sample_from_categorical_dist(rng, weights)

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
                        stats.truncnorm.rvs(
                            trunc_low,
                            trunc_high,
                            size=size,
                            loc=mus[active],
                            scale=sigmas[active],
                            random_state=rng,
                        ),
                    )
            samples_dict[param_name] = samples
        samples_dict = self._transform_from_uniform(samples_dict)
        return samples_dict

    def log_pdf(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:

        samples_dict = self._transform_to_uniform(samples_dict)
        n_observations = len(self._weights)
        n_samples = next(iter(samples_dict.values())).size
        if n_samples == 0:
            return np.asarray([], dtype=float)

        # When the search space is one CategoricalDistribution, we use the faster processing,
        # whose computation result is equivalent to the general one.
        if len(self._search_space.items()) == 1:
            param_name, dist = list(self._search_space.items())[0]
            if isinstance(dist, distributions.CategoricalDistribution):
                samples = samples_dict[param_name]
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                ret = np.log(np.inner(categorical_weights.T, self._weights))[samples]
                return ret

        # We compute log pdf (component_log_pdf)
        # for each sample in samples_dict (of size n_samples)
        # for each component of `_MultivariateParzenEstimator` (of size n_observations).
        component_log_pdf = np.zeros((n_samples, n_observations))
        for param_name, dist in self._search_space.items():
            samples = samples_dict[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                log_pdf = np.log(categorical_weights.T[samples, :])
            else:
                # We restore parameters of parzen estimators.
                low = np.asarray(self._low[param_name])
                high = np.asarray(self._high[param_name])
                q = self._q[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                assert low is not None
                assert high is not None
                assert mus is not None
                assert sigmas is not None

                cdf_func = _ParzenEstimator._normal_cdf
                p_accept = cdf_func(high, mus, sigmas) - cdf_func(low, mus, sigmas)
                if q is None:
                    distance = samples[:, None] - mus
                    mahalanobis = distance / np.maximum(sigmas, EPS)
                    z = np.sqrt(2 * np.pi) * sigmas
                    coefficient = 1 / z / p_accept
                    log_pdf = -0.5 * mahalanobis**2 + np.log(coefficient)
                else:
                    upper_bound = np.minimum(samples + q / 2.0, high)
                    lower_bound = np.maximum(samples - q / 2.0, low)
                    cdf = cdf_func(upper_bound[:, None], mus[None], sigmas[None]) - cdf_func(
                        lower_bound[:, None], mus[None], sigmas[None]
                    )
                    log_pdf = np.log(cdf + EPS) - np.log(p_accept + EPS)
            component_log_pdf += log_pdf
        ret = special.logsumexp(component_log_pdf + np.log(self._weights), axis=1)
        return ret

    def _calculate_weights(self, predetermined_weights: Optional[np.ndarray]) -> np.ndarray:

        # We decide the weights.
        consider_prior = self._parameters.consider_prior
        prior_weight = self._parameters.prior_weight
        weights_func = self._parameters.weights
        n_observations = self._n_observations

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

    def _calculate_parzen_bounds(
        self, distribution: BaseDistribution
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:

        # We calculate low and high.
        if isinstance(distribution, distributions.FloatDistribution):
            if distribution.log:
                low = np.log(distribution.low)
                high = np.log(distribution.high)
                q = None
            elif distribution.step is not None:
                q = distribution.step
                low = distribution.low - 0.5 * q
                high = distribution.high + 0.5 * q
            else:
                low = distribution.low
                high = distribution.high
                q = None
        elif isinstance(distribution, distributions.IntDistribution):
            if distribution.log:
                low = np.log(distribution.low - 0.5)
                high = np.log(distribution.high + 0.5)
                q = None
            else:
                q = distribution.step
                low = distribution.low - 0.5 * q
                high = distribution.high + 0.5 * q
        else:
            distribution_list = [
                distributions.CategoricalDistribution.__name__,
                distributions.FloatDistribution.__name__,
                distributions.IntDistribution.__name__,
            ]
            raise NotImplementedError(
                "The distribution {} is not implemented. "
                "The parameter distribution should be one of the {}".format(
                    distribution, distribution_list
                )
            )

        assert low < high

        return low, high, q

    def _transform_to_uniform(self, samples_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in samples_dict.items():
            distribution = self._search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(
                distribution,
                (distributions.FloatDistribution, distributions.IntDistribution),
            ):
                if distribution.log:
                    samples = np.log(samples)

            transformed[param_name] = samples
        return transformed

    def _transform_from_uniform(
        self, samples_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in samples_dict.items():
            distribution = self._search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(distribution, distributions.FloatDistribution):
                if distribution.log:
                    transformed[param_name] = np.exp(samples)
                elif distribution.step is not None:
                    q = self._q[param_name]
                    assert q is not None
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
                    q = self._q[param_name]
                    assert q is not None
                    samples = np.round((samples - distribution.low) / q) * q + distribution.low
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
            elif isinstance(distribution, distributions.CategoricalDistribution):
                transformed[param_name] = samples

        return transformed

    def _precompute_sigmas0(self, observations: Dict[str, np.ndarray]) -> Optional[float]:

        n_observations = next(iter(observations.values())).size
        n_observations = max(n_observations, 1)
        n_params = len(observations)

        # If it is univariate, there is no need to precompute sigmas0, so this method returns None.
        if not self._parameters.multivariate:
            return None

        # We use Scott's rule for bandwidth selection if the number of parameters > 1.
        # This rule was used in the BOHB paper.
        # TODO(kstoneriv3): The constant factor SIGMA0_MAGNITUDE=0.2 might not be optimal.
        return SIGMA0_MAGNITUDE * n_observations ** (-1.0 / (n_params + 4))

    def _calculate_categorical_params(
        self, observations: np.ndarray, param_name: str
    ) -> np.ndarray:

        # TODO(kstoneriv3): This the bandwidth selection rule might not be optimal.
        observations = observations.astype(int)
        n_observations = self._n_observations
        consider_prior = self._parameters.consider_prior
        prior_weight = self._parameters.prior_weight
        distribution = self._search_space[param_name]
        assert isinstance(distribution, distributions.CategoricalDistribution)
        choices = distribution.choices

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
        return weights

    def _calculate_numerical_params(
        self, observations: np.ndarray, param_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_observations = self._n_observations
        consider_prior = self._parameters.consider_prior
        consider_endpoints = self._parameters.consider_endpoints
        consider_magic_clip = self._parameters.consider_magic_clip
        multivariate = self._parameters.multivariate
        sigmas0 = self._sigmas0
        low = self._low[param_name]
        high = self._high[param_name]
        assert low is not None
        assert high is not None
        assert len(observations) == self._n_observations

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
            assert sigmas0 is not None
            sigmas[:] = sigmas0 * (high - low)
        else:
            assert sigmas0 is None
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

        return mus, sigmas

    @staticmethod
    def _normal_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + special.erf(z))

    @staticmethod
    def _sample_from_categorical_dist(
        rng: np.random.RandomState, probabilities: np.ndarray
    ) -> np.ndarray:

        n_samples = probabilities.shape[0]
        rnd_quantile = rng.rand(n_samples)
        cum_probs = np.cumsum(probabilities, axis=1)
        return np.sum(cum_probs < rnd_quantile[..., None], axis=1)
