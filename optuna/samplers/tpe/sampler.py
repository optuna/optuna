import numpy as np
import scipy.special

from optuna import distributions  # NOQA
from optuna.distributions import BaseDistribution  # NOQA
from optuna.samplers import base  # NOQA
from optuna.samplers import random  # NOQA
from optuna.samplers.tpe.parzen_estimator import ParzenEstimator  # NOQA
from optuna.samplers.tpe.parzen_estimator import ParzenEstimatorParameters  # NOQA
from optuna.storages.base import BaseStorage  # NOQA
from optuna.structs import StudyDirection
from optuna import types

if types.TYPE_CHECKING:
    from typing import Callable  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA
    from typing import Union  # NOQA

EPS = 1e-12


def default_gamma(x):
    # type: (int) -> int

    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def default_weights(x):
    # type: (int) -> np.ndarray

    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


class TPESampler(base.BaseSampler):
    def __init__(
            self,
            consider_prior=True,  # type: bool
            prior_weight=1.0,  # type: float
            consider_magic_clip=True,  # type: bool
            consider_endpoints=False,  # type: bool
            n_startup_trials=10,  # type: int
            n_ei_candidates=24,  # type: int
            gamma=default_gamma,  # type: Callable[[int], int]
            weights=default_weights,  # type: Callable[[int], np.ndarray]
            seed=None  # type: Optional[int]
    ):
        # type: (...) -> None

        self.parzen_estimator_parameters = ParzenEstimatorParameters(
            consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights)
        self.prior_weight = prior_weight
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.weights = weights
        self.seed = seed

        self.rng = np.random.RandomState(seed)
        self.random_sampler = random.RandomSampler(seed=seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> float

        observation_pairs = storage.get_trial_param_result_pairs(study_id, param_name)
        if storage.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            observation_pairs = [(p, -v) for p, v in observation_pairs]

        n = len(observation_pairs)

        if n < self.n_startup_trials:
            return self.random_sampler.sample(storage, study_id, param_name, param_distribution)

        below_param_values, above_param_values = self._split_observation_pairs(
            list(range(n)), [p[0] for p in observation_pairs], list(range(n)),
            [p[1] for p in observation_pairs])

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._sample_uniform(param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(param_distribution, below_param_values,
                                           above_param_values)
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            return self._sample_discrete_uniform(param_distribution, below_param_values,
                                                 above_param_values)
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            return self._sample_int(param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            return self._sample_categorical(param_distribution, below_param_values,
                                            above_param_values)
        else:
            distribution_list = [
                distributions.UniformDistribution.__name__,
                distributions.LogUniformDistribution.__name__,
                distributions.DiscreteUniformDistribution.__name__,
                distributions.IntUniformDistribution.__name__,
                distributions.CategoricalDistribution.__name__
            ]
            raise NotImplementedError("The distribution {} is not implemented. "
                                      "The parameter distribution should be one of the {}".format(
                                          param_distribution, distribution_list))

    def _split_observation_pairs(
            self,
            config_idxs,  # type: List[int]
            config_vals,  # type: List[float]
            loss_idxs,  # type: List[int]
            loss_vals  # type: List[float]
    ):
        # type: (...) -> Tuple[np.ndarray, np.ndarray]

        config_idxs, config_vals, loss_idxs, loss_vals = map(
            np.asarray, [config_idxs, config_vals, loss_idxs, loss_vals])
        n_below = self.gamma(len(config_vals))
        loss_ascending = np.argsort(loss_vals)

        keep_idxs = set(loss_idxs[loss_ascending[:n_below]])
        below = [v for i, v in zip(config_idxs, config_vals) if i in keep_idxs]

        keep_idxs = set(loss_idxs[loss_ascending[n_below:]])
        above = [v for i, v in zip(config_idxs, config_vals) if i in keep_idxs]

        below = np.asarray(below, dtype=float)
        above = np.asarray(above, dtype=float)
        return below, above

    def _sample_uniform(self, distribution, below, above):
        # type: (distributions.UniformDistribution, np.ndarray, np.ndarray) -> float

        low = distribution.low
        high = distribution.high
        return self._sample_numerical(low, high, below, above)

    def _sample_loguniform(self, distribution, below, above):
        # type: (distributions.LogUniformDistribution, np.ndarray, np.ndarray) -> float

        low = distribution.low
        high = distribution.high
        return self._sample_numerical(low, high, below, above, is_log=True)

    def _sample_discrete_uniform(self, distribution, below, above):
        # type:(distributions.DiscreteUniformDistribution, np.ndarray, np.ndarray) -> float

        q = distribution.q
        r = distribution.high - distribution.low
        # [low, high] is shifted to [0, r] to align sampled values at regular intervals.
        low = 0 - 0.5 * q
        high = r + 0.5 * q
        best_sample = self._sample_numerical(low, high, below, above, q=q) + distribution.low
        return min(max(best_sample, distribution.low), distribution.high)

    def _sample_int(self, distribution, below, above):
        # type: (distributions.IntUniformDistribution, np.ndarray, np.ndarray) -> float

        q = 1.0
        low = distribution.low - 0.5 * q
        high = distribution.high + 0.5 * q
        return int(self._sample_numerical(low, high, below, above, q=q))

    def _sample_numerical(
            self,
            low,  # type: float
            high,  # type: float
            below,  # type: np.ndarray
            above,  # type: np.ndarray
            q=None,  # type: Optional[float]
            is_log=False  # type: bool
    ):
        # type: (...) -> float

        if is_log:
            low = np.log(low)
            high = np.log(high)
            below = np.log(below)
            above = np.log(above)

        size = (self.n_ei_candidates, )

        parzen_estimator_below = ParzenEstimator(
            mus=below, low=low, high=high, parameters=self.parzen_estimator_parameters)
        samples_below = self._sample_from_gmm(
            parzen_estimator=parzen_estimator_below,
            low=low,
            high=high,
            q=q,
            is_log=is_log,
            size=size)
        log_likelihoods_below = self._gmm_log_pdf(
            samples=samples_below,
            parzen_estimator=parzen_estimator_below,
            low=low,
            high=high,
            q=q,
            is_log=is_log)

        parzen_estimator_above = ParzenEstimator(
            mus=above, low=low, high=high, parameters=self.parzen_estimator_parameters)

        log_likelihoods_above = self._gmm_log_pdf(
            samples=samples_below,
            parzen_estimator=parzen_estimator_above,
            low=low,
            high=high,
            q=q,
            is_log=is_log)

        return float(
            TPESampler._compare(
                samples=samples_below, log_l=log_likelihoods_below,
                log_g=log_likelihoods_above)[0])

    def _sample_categorical(self, distribution, below, above):
        # type: (distributions.CategoricalDistribution, np.ndarray, np.ndarray) -> float

        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)
        size = (self.n_ei_candidates, )

        weights_below = self.weights(len(below))
        counts_below = np.bincount(below, minlength=upper, weights=weights_below)
        weighted_below = counts_below + self.prior_weight
        weighted_below /= weighted_below.sum()
        samples_below = self._sample_from_categorical_dist(weighted_below, size=size)
        log_likelihoods_below = TPESampler._categorical_log_pdf(samples_below, weighted_below)

        weights_above = self.weights(len(above))
        counts_above = np.bincount(above, minlength=upper, weights=weights_above)
        weighted_above = counts_above + self.prior_weight
        weighted_above /= weighted_above.sum()
        log_likelihoods_above = TPESampler._categorical_log_pdf(samples_below, weighted_above)

        return int(
            TPESampler._compare(
                samples=samples_below, log_l=log_likelihoods_below,
                log_g=log_likelihoods_above)[0])

    def _sample_from_gmm(
            self,
            parzen_estimator,  # type: ParzenEstimator
            low,  # type: float
            high,  # type: float
            q=None,  # type: Optional[float]
            size=(),  # type: Tuple
            is_log=False,  # type: bool
    ):
        # type: (...) -> np.ndarray

        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))
        n_samples = np.prod(size)

        if low >= high:
            raise ValueError("The 'low' should be lower than the 'high'. "
                             "But (low, high) = ({}, {}).".format(low, high))
        samples = np.asarray([], dtype=float)
        while samples.size < n_samples:
            active = np.argmax(self.rng.multinomial(1, weights))
            draw = self.rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw < high:
                samples = np.append(samples, draw)

        samples = np.reshape(samples, size)

        if is_log:
            samples = np.exp(samples)

        if q is None:
            return samples
        else:
            return np.round(samples / q) * q

    def _gmm_log_pdf(
            self,
            samples,  # type: np.ndarray
            parzen_estimator,  # type: ParzenEstimator
            low,  # type: float
            high,  # type: float
            q=None,  # type: Optional[float]
            is_log=False  # type: bool
    ):
        # type: (...) -> np.ndarray

        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))
        if samples.size == 0:
            return np.asarray([], dtype=float)
        if weights.ndim != 1:
            raise ValueError("The 'weights' should be 2-dimension. "
                             "But weights.shape = {}".format(weights.shape))
        if mus.ndim != 1:
            raise ValueError("The 'mus' should be 2-dimension. "
                             "But mus.shape = {}".format(mus.shape))
        if sigmas.ndim != 1:
            raise ValueError("The 'sigmas' should be 2-dimension. "
                             "But sigmas.shape = {}".format(sigmas.shape))
        _samples = samples
        samples = _samples.flatten()

        p_accept = np.sum(
            weights *
            (TPESampler._normal_cdf(high, mus, sigmas) - TPESampler._normal_cdf(low, mus, sigmas)))

        if q is None:
            jacobian = samples[:, None] if is_log else np.ones(samples.shape)[:, None]
            if is_log:
                distance = np.log(samples[:, None]) - mus
            else:
                distance = samples[:, None] - mus
            mahalanobis = (distance / np.maximum(sigmas, EPS))**2
            Z = np.sqrt(2 * np.pi) * sigmas * jacobian
            coefficient = weights / Z / p_accept
            return_val = TPESampler._logsum_rows(-0.5 * mahalanobis + np.log(coefficient))
        else:
            probabilities = np.zeros(samples.shape, dtype=float)
            cdf_func = TPESampler._log_normal_cdf if is_log else TPESampler._normal_cdf
            for w, mu, sigma in zip(weights, mus, sigmas):
                if is_log:
                    upper_bound = np.minimum(samples + q / 2.0, np.exp(high))
                    lower_bound = np.maximum(samples - q / 2.0, np.exp(low))
                    lower_bound = np.maximum(0, lower_bound)
                else:
                    upper_bound = np.minimum(samples + q / 2.0, high)
                    lower_bound = np.maximum(samples - q / 2.0, low)
                inc_amt = w * cdf_func(upper_bound, mu, sigma)
                inc_amt -= w * cdf_func(lower_bound, mu, sigma)
                probabilities += inc_amt
            return_val = np.log(probabilities + EPS) - np.log(p_accept + EPS)

        return_val.shape = _samples.shape
        return return_val

    def _sample_from_categorical_dist(self, probabilities, size=()):
        # type: (Union[np.ndarray, np.ndarray], Tuple) -> Union[np.ndarray, np.ndarray]

        if probabilities.size == 1 and isinstance(probabilities[0], np.ndarray):
            probabilities = probabilities[0]
        probabilities = np.asarray(probabilities)

        if size == ():
            size = (1, )
        elif isinstance(size, (int, np.number)):
            size = (size, )
        else:
            size = tuple(size)

        if size == (0, ):
            return np.asarray([], dtype=float)
        assert len(size)
        assert probabilities.ndim == 1

        n_draws = int(np.prod(size))
        sample = self.rng.multinomial(n=1, pvals=probabilities, size=int(n_draws))
        assert sample.shape == size + (probabilities.size, )
        return_val = np.dot(sample, np.arange(probabilities.size))
        return_val.shape = size
        return return_val

    @classmethod
    def _categorical_log_pdf(
            cls,
            sample,  # type: np.ndarray
            p  # type: np.ndarray
    ):
        # type: (...) -> np.ndarray

        if sample.size:
            return np.log(np.asarray(p)[sample])
        else:
            return np.asarray([])

    @classmethod
    def _compare(cls, samples, log_l, log_g):
        # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray

        samples, log_l, log_g = map(np.asarray, (samples, log_l, log_g))
        if samples.size:
            score = log_l - log_g
            if samples.size != score.size:
                raise ValueError("The size of the 'samples' and that of the 'score' "
                                 "should be same. "
                                 "But (samples.size, score.size) = ({}, {})".format(
                                     samples.size, score.size))

            best = np.argmax(score)
            return np.asarray([samples[best]] * samples.size)
        else:
            return np.asarray([])

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
        return 0.5 * (1 + scipy.special.erf(z))

    @classmethod
    def _log_normal_cdf(cls, x, mu, sigma):
        # type: (float, np.ndarray, np.ndarray) -> np.ndarray

        mu, sigma = map(np.asarray, (mu, sigma))
        if x < 0:
            raise ValueError("Negative argument is given to _lognormal_cdf. x: {}".format(x))
        denominator = np.log(np.maximum(x, EPS)) - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return .5 + .5 * scipy.special.erf(z)
