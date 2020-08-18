from typing import Callable
from typing import NamedTuple
from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.stats import truncnorm
from scipy.special import erf

from optuna import distributions
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA
    from typing import Tuple  # NOQA

EPS = 1e-12


class _ParzenEstimatorParameters(
    NamedTuple(
        "_ParzenEstimatorParameters",
        [
            ("consider_prior", bool),
            ("prior_weight", Optional[float]),
            ("consider_magic_clip", bool),
            ("consider_endpoints", bool),
            ("weights", Callable[[int], ndarray]),
        ],
    )
):
    pass


class _ParzenEstimator(object):
    def __init__(
        self,
        mus,  # type: ndarray
        low,  # type: float
        high,  # type: float
        parameters,  # type: _ParzenEstimatorParameters
    ):
        # type: (...) -> None

        self.weights, self.mus, self.sigmas = _ParzenEstimator._calculate(
            mus,
            low,
            high,
            parameters.consider_prior,
            parameters.prior_weight,
            parameters.consider_magic_clip,
            parameters.consider_endpoints,
            parameters.weights,
        )

    @classmethod
    def _calculate(
        cls,
        mus,  # type: ndarray
        low,  # type: float
        high,  # type: float
        consider_prior,  # type: bool
        prior_weight,  # type: Optional[float]
        consider_magic_clip,  # type: bool
        consider_endpoints,  # type: bool
        weights_func,  # type: Callable[[int], ndarray]
    ):
        # type: (...) -> Tuple[ndarray, ndarray, ndarray]
        """Calculates the weights, mus and sigma for the Parzen estimator.

           Note: When the number of observations is zero, the Parzen estimator ignores the
           `consider_prior` flag and utilizes a prior. Validation of this approach is future work.
        """

        mus = np.asarray(mus)
        sigma = np.asarray([], dtype=float)
        prior_pos = 0

        # Parzen estimator construction requires at least one observation or a priror.
        if mus.size == 0:
            consider_prior = True

        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low)
            if mus.size == 0:
                low_sorted_mus_high = np.zeros(3)
                sorted_mus = low_sorted_mus_high[1:-1]
                sorted_mus[0] = prior_mu
                sigma = np.asarray([prior_sigma])
                prior_pos = 0
                order = []  # type: List[int]
            else:  # When mus.size is greater than 0.
                # We decide the place of the  prior.
                order = np.argsort(mus).astype(int)
                ordered_mus = mus[order]
                prior_pos = np.searchsorted(ordered_mus, prior_mu)
                # We decide the mus.
                low_sorted_mus_high = np.zeros(len(mus) + 3)
                sorted_mus = low_sorted_mus_high[1:-1]
                sorted_mus[:prior_pos] = ordered_mus[:prior_pos]
                sorted_mus[prior_pos] = prior_mu
                sorted_mus[prior_pos + 1 :] = ordered_mus[prior_pos:]
        else:
            order = np.argsort(mus)
            # We decide the mus.
            low_sorted_mus_high = np.zeros(len(mus) + 2)
            sorted_mus = low_sorted_mus_high[1:-1]
            sorted_mus[:] = mus[order]

        # We decide the sigma.
        if mus.size > 0:
            low_sorted_mus_high[-1] = high
            low_sorted_mus_high[0] = low
            sigma = np.maximum(
                low_sorted_mus_high[1:-1] - low_sorted_mus_high[0:-2],
                low_sorted_mus_high[2:] - low_sorted_mus_high[1:-1],
            )
            if not consider_endpoints and low_sorted_mus_high.size > 2:
                sigma[0] = low_sorted_mus_high[2] - low_sorted_mus_high[1]
                sigma[-1] = low_sorted_mus_high[-2] - low_sorted_mus_high[-3]

        # We decide the weights.
        unsorted_weights = weights_func(mus.size)
        if consider_prior:
            sorted_weights = np.zeros_like(sorted_mus)
            sorted_weights[:prior_pos] = unsorted_weights[order[:prior_pos]]
            sorted_weights[prior_pos] = prior_weight
            sorted_weights[prior_pos + 1 :] = unsorted_weights[order[prior_pos:]]
        else:
            sorted_weights = unsorted_weights[order]
        sorted_weights /= sorted_weights.sum()

        # We adjust the range of the 'sigma' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(sorted_mus)))
        else:
            minsigma = EPS
        sigma = np.clip(sigma, minsigma, maxsigma)
        if consider_prior:
            sigma[prior_pos] = prior_sigma

        return sorted_weights, sorted_mus, sigma


class _MultivariateParzenEstimator:
    
    def __init__(
        self, 
        search_space, 
        multivariate_samples,
        parameters,
    ):
        # weights
        # search_space: Dict[str, BaseDistribution]
        # mus: Dict[str, Optional[np.ndarray]]
        # sigma: Dict[str, Optional[np.ndarray]]
        # low: Dict[str, Optional[float]]
        # high: Dict[str, Optional[float]]
        # q: Dict[str, Optional[float]]
        # categorical_weights: Dict[str, Optional[float]]


        self._search_space = search_space
        self._parameters = parameters
        self._high = {}
        self._q = {}
        self._mus = {}
        self._sigmas = {}
        self._low = {}
        self._categorical_weights = {}
        
        self._weights = self._calculate_weights(multivariate_samples)
        
        multivariate_samples = self._transform_to_uniform(multivariate_samples)

        for param_name, dist in search_space.items():
            samples = multivariate_samples[param_name]
            # カテゴリ分布の場合は別処理
            if isinstance(dist, distributions.CategoricalDistribution):
                low = high = q = mus = sigmas = None
                raise NotImplementedError
            else:
                low, high, q, mus, sigmas = self._calculate_parzen_est_params(samples)
                categorical_weights = None
            self._low[param_name] = low
            self._high[param_name] = high
            self._q[param_name] = q
            self._mus[param_name] = mus
            self._sigmas[param_name] = sigmas
            self._categorical_weights[param_name] = categorical_weights

    def sample(self, rng, size):
        # TODO(kstoneriv3): maybe divide this functions into smaller ones 
        multivariate_samples = {}
        # active componentの決定
        active = rng.choice(len(self._weights), size, self._weights)
        for param_name, dist in self._search_space.items(): 
            # カテゴリ分布と場合分けしつつ、
            if isinstance(dist, distributions.CategoricalDistribution):
                pass
            else:
                # restore parameters of parzen estimators
                low = self._low[param_name]
                high = self._high[param_name]
                q = self._q[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                # sample from truncnorm
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
                if q:
                    samples = np.round(samples / q) * q
            multivariate_samples[param_name] = samples
        return self._transform_from_uniform(multivariate_samples)

    def log_pdf(self, multivariate_samples):
        _log_pdf = 0
        for param_name, dist in self._search_space.items(): 
            weights = self._weights
            samples = multivariate_samples[param_name]
            # カテゴリ分布と場合分けしつつ、
            if isinstance(dist, distributions.CategoricalDistribution):
                pass
            else:
                # restore parameters of parzen estimators
                low = self._low[param_name]
                high = self._high[param_name]
                q = self._q[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                if samples.size == 0:
                    return np.asarray([], dtype=float)
                if weights.ndim != 1:
                    raise ValueError(
                        "the 'weights' should be 1-dimension. "
                        "but weights.shape = {}".format(weights.shape)
                    )
                if mus.ndim != 1:
                    raise ValueError(
                        "the 'mus' should be 1-dimension. but mus.shape = {}".format(mus.shape)
                    )
                if sigmas.ndim != 1:
                    raise ValueError(
                        "the 'sigmas' should be 1-dimension. but sigmas.shape = {}".format(
                            sigmas.shape
                        )
                    )
                p_accept = np.sum(
                    weights
                    * (
                        _MultivariateParzenEstimator._normal_cdf(high, mus, sigmas)
                        - _MultivariateParzenEstimator._normal_cdf(low, mus, sigmas)
                    )
                )
                if q is None:
                    distance = samples[..., None] - mus
                    mahalanobis = (distance / np.maximum(sigmas, EPS)) ** 2
                    z = np.sqrt(2 * np.pi) * sigmas
                    coefficient = weights / z / p_accept
                    return _MultivariateParzenEstimator._logsum_rows(-0.5 * mahalanobis + np.log(coefficient))
                else:
                    cdf_func = _MultivariateParzenEstimator._normal_cdf
                    upper_bound = np.minimum(samples + q / 2.0, high)
                    lower_bound = np.maximum(samples - q / 2.0, low)
                    probabilities = np.sum(
                        weights[..., None]
                        * (
                            cdf_func(upper_bound[None], mus[..., None], sigmas[..., None])
                            - cdf_func(lower_bound[None], mus[..., None], sigmas[..., None])
                        ),
                        axis=0,
                    )
                    return np.log(probabilities + EPS) - np.log(p_accept + EPS)

    def _transform_to_uniform(self, multivariate_samples):
        transformed = {}
        # 分布ごとの場合分け
        for param_name, samples in multivariate_samples.items():
            if isinstance(self._search_space[param_name], distributions.UniformDistribution):
                transformed[param_name] = samples
            else:
                raise NotImplementedError
        return transformed
        
    def _transform_from_uniform(self, multivariate_samples):
        transformed = {}
        # 分布ごとの場合分け
        for param_name, samples in multivariate_samples.items():
            if isinstance(self._search_space[param_name], distributions.UniformDistribution):
                transformed[param_name] = samples
            else:
                raise NotImplementedError
        return transformed

    def _calculate_weights(self, multivariate_samples):
        pass

    def _calculate_parzen_est_params(self, samples):
        pass

    @classmethod
    def _logsum_rows(cls, x):
        # type: (np.ndarray) -> np.ndarray

        x = np.asarray(x)
        m = x.max(axis=1)
        return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m

    @classmethod
    def _normal_cdf(cls, x, mu, sigma):
        # type: (float, np.ndarray, np.ndarray) -> np.ndarray

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + erf(z))
