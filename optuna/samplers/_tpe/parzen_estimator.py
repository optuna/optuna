from typing import Any
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np

from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.samplers._tpe.probability_distributions import _categorical_distribution, _discrete_truncnorm_distribution, _truncnorm_distribution, _product_distribution, _mixture_distribution, _sample, _logpdf

EPS = 1e-12
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

        transformed_observations = self._transform_to_uniform(observations)

        assert predetermined_weights is None or len(transformed_observations) == len(predetermined_weights)
        weights = predetermined_weights if predetermined_weights is not None \
                    else self._call_weights_func(parameters.weights, len(transformed_observations))

        if parameters.consider_prior or len(transformed_observations) == 0:
            weights = np.append(weights, [parameters.prior_weight])

        self._mixture_distribution = _mixture_distribution(
                weights=weights,
                distributions=_product_distribution(
                                {param: self._calculate_distributions(
                                            transformed_observations[param], 
                                            search_space[param],
                                            parameters)
                                    for param in search_space}))

    
    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:
        sampled = _sample(self._mixture_distribution, rng, (size,))
        return self._transform_from_uniform(sampled)

    def log_pdf(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        transformed_samples_sa = self._transform_to_uniform(samples_dict)
        return _logpdf(np.broadcast_to(self._mixture_distribution, transformed_samples_sa.shape, subok=True),
                        transformed_samples_sa)


    @staticmethod
    def _call_weights_func(weights_func: Callable[[int], np.ndarray], n: int) -> np.ndarray:
        w = weights_func(n)[:n]
        if np.any(w < 0):
            raise ValueError(
                f"The `weights` function is not allowed to return negative values {w}. "
                + f"The argument of the `weights` function is {n}."
            )
        if len(w) > 0 and np.sum(w) <= 0:
            raise ValueError(
                f"The `weight` function is not allowed to return all-zero values {w}."
                + f" The argument of the `weights` function is {n}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "The `weights`function is not allowed to return infinite or NaN values "
                + f"{w}. The argument of the `weights` function is {n}."
            )

        # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
        # unexpected size.
        return w

    @staticmethod
    def _is_log(dist: BaseDistribution) -> bool:
        return isinstance(dist, (FloatDistribution, IntDistribution)) and dist.log

    def _transform_to_uniform(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.rec.fromarrays([np.log(samples_dict[param]) 
                                            if self._is_log(self._search_space[param]) 
                                            else samples_dict[param] 
                                        for param in self._search_space],
                                    names=list(self._search_space.keys()))

    def _transform_from_uniform(self, samples_sa: np.ndarray) -> Dict[str, np.ndarray]:
        res = {param: np.exp(samples_sa[param]) 
                        if self._is_log(self._search_space[param]) 
                        else samples_sa[param] 
                for param in self._search_space}

        # TODO(contramundum53): Remove this line after fixing log-Int hack.
        return {param: np.clip(
                    np.round(res[param] / self._search_space[param].step) * self._search_space[param].step,
                    self._search_space[param].low, self._search_space[param].high
                ) if isinstance(self._search_space[param], IntDistribution) else res[param]
                for param in self._search_space}

    def _calculate_distributions(
        self,
        transformed_observations: np.ndarray,
        search_space: BaseDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> np.ndarray:
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(
                transformed_observations, search_space.choices, parameters
            )
        else:   
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
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

            return self._calculate_numerical_distributions(
                transformed_observations, low, high, step, parameters
            )

    def _calculate_categorical_distributions(
        self,
        observations: np.ndarray,
        choices: Tuple[Any, ...],
        parameters: _ParzenEstimatorParameters,
    ) -> np.ndarray:

        consider_prior = parameters.consider_prior or len(observations) == 0

        assert parameters.prior_weight is not None
        weights = np.full(shape=(len(observations) + consider_prior, len(choices)), 
                            fill_value=parameters.prior_weight / (len(observations) + consider_prior))

        weights[np.arange(len(observations)), observations.astype(int)] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return _categorical_distribution(weights)

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,
        high: float,
        step: Optional[float],
        parameters: _ParzenEstimatorParameters,
    ) -> np.ndarray:
        step = step or 0

        mus = observations
        consider_prior = parameters.consider_prior or len(observations) == 0

        def compute_sigmas() -> np.ndarray:
            if parameters.multivariate:
                SIGMA0_MAGNITUDE = 0.2
                sigma = SIGMA0_MAGNITUDE \
                        * max(len(observations), 1) ** (-1.0 / (len(self._search_space) + 4)) \
                        * (high - low + step)
                sigmas = np.full(shape=(len(observations),), fill_value=sigma)
            else:
                # Why include prior_mu???
                prior_mu = 0.5 * (low + high)
                mus_with_prior = np.append(mus, prior_mu) if consider_prior else mus

                sorted_indices = np.argsort(mus_with_prior)
                sorted_mus = mus_with_prior[sorted_indices]
                sorted_mus_with_endpoints = np.empty(len(mus_with_prior) + 2, dtype=float)
                sorted_mus_with_endpoints[0] = low - step / 2
                sorted_mus_with_endpoints[1:-1] = sorted_mus 
                sorted_mus_with_endpoints[-1] = high + step / 2

                sorted_sigmas = np.maximum(
                    sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                    sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
                )

                if not parameters.consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                    sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                    sorted_sigmas[-1] = sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]

                sigmas = sorted_sigmas[np.argsort(sorted_indices)][:len(observations)]

            # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
            maxsigma = 1.0 * (high - low + step)
            if parameters.consider_magic_clip:
                # Why change minsigma depending on consider_prior???
                minsigma = 1.0 * (high - low + step) / min(100.0, (1.0 + len(observations) + consider_prior))
            else:
                minsigma = EPS
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))

        sigmas = compute_sigmas()


        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low + step)
            mus = np.append(mus, [prior_mu])
            sigmas = np.append(sigmas, [prior_sigma])

        if step == 0:
            return _truncnorm_distribution(
                mus, 
                sigmas, 
                np.full(shape=len(mus), fill_value=low),
                np.full(shape=len(mus), fill_value=high),
            )
        else:
            return _discrete_truncnorm_distribution(
                mus, 
                sigmas, 
                np.full(shape=len(mus), fill_value=low),
                np.full(shape=len(mus), fill_value=high),
                np.full(shape=len(mus), fill_value=step),
            )