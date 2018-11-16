import numpy
import scipy.special
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Callable  # NOQA
from typing import Tuple  # NOQA
from typing import Union  # NOQA

from optuna import distributions  # NOQA
from optuna.distributions import BaseDistribution  # NOQA
from optuna.samplers import base  # NOQA
from optuna.samplers import random  # NOQA
from optuna.samplers.parzen_estimator import ParzenEstimator  # NOQA
from optuna.samplers.parzen_estimator import ParzenEstimatorParameters  # NOQA
from optuna.storages.base import BaseStorage  # NOQA

default_consider_prior = True
default_prior_weight = 1.0
default_consider_magic_clip = True
default_consider_endpoints = False
default_n_startup_trials = 4
default_n_ei_candidates = 24
EPS = 1e-12


def default_gamma(x):
    return min(int(numpy.ceil(0.25 * numpy.sqrt(x))), 25)


def default_weights(x):
    if x == 0:
        return numpy.asarray([])
    elif x < 25:
        return numpy.ones(x)
    else:
        ramp = numpy.linspace(1.0 / x, 1.0, num=x - 25)
        flat = numpy.ones(25)
        return numpy.concatenate([ramp, flat], axis=0)


GeneralizedList = Union[List, numpy.ndarray]


class MyTPESampler(base.BaseSampler):

    def __init__(self,
                 consider_prior=default_consider_prior,  # type: bool
                 prior_weight=default_prior_weight,  # type: Optional[float]
                 consider_magic_clip=default_consider_magic_clip,  # type: bool
                 consider_endpoints=default_consider_endpoints,  # type: bool
                 n_startup_trials=default_n_startup_trials,  # type: int
                 n_ei_candidates=default_n_ei_candidates,  # type: int
                 gamma=default_gamma,  # type: Callable[int, int]
                 weights=default_weights,  # type: Callable[[int], GeneralizedList]
                 seed=None  # type: Optional[int]
                 ):
        # type: (...) -> None
        self.parzen_estimator_parameters = ParzenEstimatorParameters(consider_prior,
                                                                     prior_weight,
                                                                     consider_magic_clip,
                                                                     consider_endpoints,
                                                                     weights)
        self.prior_weight = prior_weight
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.weights = weights
        self.seed = seed

        self.rng = numpy.random.RandomState(seed)
        self.random_sampler = random.RandomSampler(seed=seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> float
        observation_pairs = storage.get_trial_param_result_pairs(study_id, param_name)
        n = len(observation_pairs)

        if n < self.n_startup_trials:
            return self.random_sampler.sample(storage, study_id, param_name, param_distribution)

        below_param_values, above_param_values = self._split_observation_pairs(
            list(range(n)), [p[0] for p in observation_pairs],
            list(range(n)), [p[1] for p in observation_pairs])

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._sample_uniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            return self._sample_discrete_uniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            return self._sample_int(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            return self._sample_categorical(
                param_distribution, below_param_values, above_param_values)
        else:
            raise NotImplementedError

    def _split_observation_pairs(self, config_idxs, config_vals, loss_idxs, loss_vals):
        # type: (List[int], List[float], List[int], List[float]) -> (List[float], List[float])
        config_idxs, config_vals, loss_idxs, loss_vals = map(numpy.asarray,
                                                             [config_idxs,
                                                              config_vals,
                                                              loss_idxs,
                                                              loss_vals])
        n_below = self.gamma(len(config_vals))
        loss_ascending = numpy.argsort(loss_vals)

        keep_idxs = set(loss_idxs[loss_ascending[:n_below]])
        below = [v for i, v in zip(config_idxs, config_vals) if i in keep_idxs]

        keep_idxs = set(loss_idxs[loss_ascending[n_below:]])
        above = [v for i, v in zip(config_idxs, config_vals) if i in keep_idxs]

        return below, above

    def _sample_uniform(self, distribution, below, above):
        # type: (distributions.UniformDistribution, List[float], List[float]) -> float
        size = (self.n_ei_candidates,)
        # From below, make samples and log-likelihoods
        parzen_estimator_below = ParzenEstimator(mus=below,
                                                 low=distribution.low,
                                                 high=distribution.high,
                                                 parameters=self.parzen_estimator_parameters)
        samples_b = self._GMM(weights=parzen_estimator_below.weights,
                              mus=parzen_estimator_below.mus,
                              sigmas=parzen_estimator_below.sigmas,
                              low=distribution.low,
                              high=distribution.high,
                              q=None,
                              size=size)
        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_below.weights,
                                              mus=parzen_estimator_below.mus,
                                              sigmas=parzen_estimator_below.sigmas,
                                              low=distribution.low,
                                              high=distribution.high,
                                              q=None)

        # From above, make log-likelihoods
        parzen_estimator_above = ParzenEstimator(mus=above,
                                                 low=distribution.low,
                                                 high=distribution.high,
                                                 parameters=self.parzen_estimator_parameters)
        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_above.weights,
                                              mus=parzen_estimator_above.mus,
                                              sigmas=parzen_estimator_above.sigmas,
                                              low=distribution.low,
                                              high=distribution.high,
                                              q=None)
        return MyTPESampler._compare(
            samples=samples_b, log_l=log_likelihoods_b, log_g=log_likelihoods_a)[0]

    def _sample_loguniform(self, distribution, below, above):
        # type: (distributions.LogUniformDistribution, List[float], List[float]) -> float
        low = numpy.log(distribution.low)
        high = numpy.log(distribution.high)
        size = (self.n_ei_candidates,)
        # From below, make samples and log-likelihoods
        parzen_estimator_below = ParzenEstimator(mus=numpy.log(below),
                                                 low=low,
                                                 high=high,
                                                 parameters=self.parzen_estimator_parameters)
        samples_b = self._GMM(weights=parzen_estimator_below.weights,
                              mus=parzen_estimator_below.mus,
                              sigmas=parzen_estimator_below.sigmas,
                              low=low,
                              high=high,
                              q=None,
                              is_log=True,
                              size=size)
        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_below.weights,
                                              mus=parzen_estimator_below.mus,
                                              sigmas=parzen_estimator_below.sigmas,
                                              low=low,
                                              high=high,
                                              q=None,
                                              is_log=True)

        # From above, make log-likelihoods
        parzen_estimator_above = ParzenEstimator(mus=numpy.log(above),
                                                 low=low,
                                                 high=high,
                                                 parameters=self.parzen_estimator_parameters)

        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_above.weights,
                                              mus=parzen_estimator_above.mus,
                                              sigmas=parzen_estimator_above.sigmas,
                                              low=low,
                                              high=high,
                                              q=None,
                                              is_log=True)

        return MyTPESampler._compare(
            samples=samples_b, log_l=log_likelihoods_b, log_g=log_likelihoods_a)[0]

    def _sample_discrete_uniform(self, distribution, below, above):
        # type: (distributions.DiscreteUniformDistribution, List[float], List[float]) -> float
        # From below, make samples and log-likelihoods
        low = distribution.low - 0.5 * distribution.q
        high = distribution.high + 0.5 * distribution.q
        size = (self.n_ei_candidates,)

        parzen_estimator_below = ParzenEstimator(mus=below,
                                                 low=low,
                                                 high=high,
                                                 parameters=self.parzen_estimator_parameters)
        samples_b = self._GMM(weights=parzen_estimator_below.weights,
                              mus=parzen_estimator_below.mus,
                              sigmas=parzen_estimator_below.sigmas,
                              low=low,
                              high=high,
                              q=distribution.q,
                              size=size)
        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_below.weights,
                                              mus=parzen_estimator_below.mus,
                                              sigmas=parzen_estimator_below.sigmas,
                                              low=low,
                                              high=high,
                                              q=distribution.q)

        # From above, make log-likelihoods
        parzen_estimator_above = ParzenEstimator(mus=above,
                                                 low=low,
                                                 high=high,
                                                 parameters=self.parzen_estimator_parameters)

        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_above.weights,
                                              mus=parzen_estimator_above.mus,
                                              sigmas=parzen_estimator_above.sigmas,
                                              low=low,
                                              high=high,
                                              q=distribution.q)

        return min(max(MyTPESampler._compare(samples=samples_b,
                                             log_l=log_likelihoods_b,
                                             log_g=log_likelihoods_a)[0],
                       low),
                   high)

    def _sample_int(self, distribution, below, above):
        # type: (distributions.IntUniformDistribution, List[float], List[float]) -> float
        # From below, make samples and log-likelihoods
        q = 1.0
        low = distribution.low - 0.5 * q
        high = distribution.high + 0.5 * q
        size = (self.n_ei_candidates,)

        parzen_estimator_below = ParzenEstimator(mus=below,
                                                 low=low,
                                                 high=high,
                                                 parameters=self.parzen_estimator_parameters)
        samples_b = self._GMM(weights=parzen_estimator_below.weights,
                              mus=parzen_estimator_below.mus,
                              sigmas=parzen_estimator_below.sigmas,
                              low=low,
                              high=high,
                              q=q,
                              size=size)
        log_likelihoods_b = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_below.weights,
                                              mus=parzen_estimator_below.mus,
                                              sigmas=parzen_estimator_below.sigmas,
                                              low=low,
                                              high=high,
                                              q=q)

        # From above, make log-likelihoods
        parzen_estimator_above = ParzenEstimator(mus=above,
                                                 low=low,
                                                 high=high,
                                                 parameters=self.parzen_estimator_parameters)

        log_likelihoods_a = self._GMM_log_pdf(samples=samples_b,
                                              weights=parzen_estimator_above.weights,
                                              mus=parzen_estimator_above.mus,
                                              sigmas=parzen_estimator_above.sigmas,
                                              low=low,
                                              high=high,
                                              q=q)

        return int(MyTPESampler._compare(samples=samples_b,
                                         log_l=log_likelihoods_b,
                                         log_g=log_likelihoods_a)[0])

    def _sample_categorical(self, distribution, below, above):
        # type: (distributions.CategoricalDistribution, List[float], List[float]) -> float
        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)
        size = (self.n_ei_candidates,)

        weights_b = self.weights(len(below))
        counts_b = numpy.bincount(below, minlength=upper, weights=weights_b)
        pseudocounts_b = counts_b + self.prior_weight
        pseudocounts_b /= pseudocounts_b.sum()
        samples_b = self._categorical(pseudocounts_b, size=size)
        log_likelihoods_b = MyTPESampler._categorical_log_pdf(samples_b, list(pseudocounts_b))

        weights_a = self.weights(len(above))
        counts_a = numpy.bincount(above, minlength=upper, weights=weights_a)
        pseudocounts_a = counts_a + self.prior_weight
        pseudocounts_a /= pseudocounts_a.sum()
        log_likelihoods_a = MyTPESampler._categorical_log_pdf(samples_b, list(pseudocounts_a))

        return int(MyTPESampler._compare(samples=samples_b,
                                         log_l=log_likelihoods_b,
                                         log_g=log_likelihoods_a)[0])

    def _GMM(self,
             weights,  # type: List[float]
             mus,  # type: List[float]
             sigmas,  # type: List[float]
             low,  # type: float
             high,  # type: float
             q=None,  # type: Optional[float]
             size=(),  # type: Tuple
             is_log=False,  # type: bool
             ):
        # type: (...) -> List[float]
        weights, mus, sigmas = map(numpy.asarray, (weights, mus, sigmas))
        n_samples = numpy.prod(size)

        if low >= high:
            raise ValueError("low >= high", (low, high))
        samples = []
        while len(samples) < n_samples:
            active = numpy.argmax(self.rng.multinomial(1, weights))
            draw = self.rng.normal(loc=mus[active], scale=sigmas[active])
            if low <= draw <= high:
                samples.append(draw)
        samples = numpy.reshape(numpy.asarray(samples), size)

        if is_log:
            samples = numpy.exp(samples)

        if q is None:
            return list(samples)
        else:
            return list(numpy.round(samples / q) * q)

    def _GMM_log_pdf(self,
                     samples,  # type: List[float]
                     weights,  # type: List[float]
                     mus,  # type: List[float]
                     sigmas,  # type: List[float]
                     low,  # type: float
                     high,  # type: float
                     q=None,  # type: Optional[float]
                     is_log=False  # type: bool
                     ):
        # type: (...) -> List[float]
        samples, weights, mus, sigmas = map(numpy.asarray, (samples, weights, mus, sigmas))
        if samples.size == 0:
            return []
        if weights.ndim != 1:
            raise TypeError("need vector of weights", weights.shape)
        if mus.ndim != 1:
            raise TypeError("need vector of mus", mus.shape)
        if sigmas.ndim != 1:
            raise TypeError("need vector of sigmas", sigmas.shape)
        _samples = samples
        samples = _samples.flatten()

        p_accept = numpy.sum(weights * (MyTPESampler._normal_cdf(high, mus, sigmas)
                                        - MyTPESampler._normal_cdf(low, mus, sigmas)))

        if q is None:
            if is_log:
                distance = numpy.log(samples[:, None]) - mus
                mahalanobis = (distance / numpy.maximum(sigmas, EPS)) ** 2
                Z = numpy.sqrt(2 * numpy.pi) * sigmas * samples[:, None]
                coefficient = weights / Z / p_accept
                return_val = MyTPESampler._logsum_rows(- 0.5 *
                                                       mahalanobis + numpy.log(coefficient))
            else:
                distance = samples[:, None] - mus
                mahalanobis = (distance / numpy.maximum(sigmas, EPS)) ** 2
                Z = numpy.sqrt(2 * numpy.pi) * sigmas
                coefficient = weights / Z / p_accept
                return_val = MyTPESampler._logsum_rows(- 0.5 *
                                                       mahalanobis + numpy.log(coefficient))
        else:
            probabilities = numpy.zeros(samples.shape, dtype='float64')
            for w, mu, sigma in zip(weights, mus, sigmas):
                if is_log:
                    upper_bound = numpy.minimum(samples + q / 2.0, numpy.exp(high))
                    lower_bound = numpy.maximum(samples - q / 2.0, numpy.exp(low))
                    lower_bound = numpy.maximum(0, lower_bound)
                    inc_amt = w * MyTPESampler._log_normal_cdf(upper_bound, mu, sigma)
                    inc_amt -= w * MyTPESampler._log_normal_cdf(lower_bound, mu, sigma)
                else:
                    upper_bound = numpy.minimum(samples + q / 2.0, high)
                    lower_bound = numpy.maximum(samples - q / 2.0, low)
                    inc_amt = w * MyTPESampler._normal_cdf(upper_bound, mu, sigma)
                    inc_amt -= w * MyTPESampler._normal_cdf(lower_bound, mu, sigma)
                probabilities += inc_amt
            return_val = numpy.log(probabilities) - numpy.log(p_accept)

        return_val.shape = _samples.shape
        return return_val

    def _categorical(self, p, size=()):
        # type: (Union[List[float], numpy.ndarray], Tuple) -> Union[List[float], numpy.ndarray]
        if len(p) == 1 and isinstance(p[0], numpy.ndarray):
            p = p[0]
        p = numpy.asarray(p)

        if size == ():
            size = (1,)
        elif isinstance(size, (int, numpy.number)):
            size = (size,)
        else:
            size = tuple(size)

        if size == (0,):
            return numpy.asarray([])
        assert len(size)

        if p.ndim == 0:
            raise NotImplementedError
        elif p.ndim == 1:
            n_draws = int(numpy.prod(size))
            sample = self.rng.multinomial(n=1, pvals=p, size=int(n_draws))
            assert sample.shape == size + (len(p),)
            return_val = numpy.dot(sample, numpy.arange(len(p)))
            return_val.shape = size
            return return_val
        elif p.ndim == 2:
            n_draws_, n_choices = p.shape
            n_draws, = size
            assert n_draws_ == n_draws
            return_val = [numpy.where(self.rng.multinomial(pvals=[ii], n=1))[0][0]
                          for ii in range(n_draws_)]
            return_val = numpy.asarray(return_val)
            return_val.shape = size
            return return_val
        else:
            raise NotImplementedError

    @classmethod
    def _categorical_log_pdf(cls,
                             sample,  # type: Union[List[float], numpy.ndarray]
                             p  # type: List[float]
                             ):
        # type: (...) -> Union[List[float], numpy.ndarray]
        if sample.size:
            return numpy.log(numpy.asarray(p)[sample])
        else:
            return numpy.asarray([])

    @classmethod
    def _compare(cls, samples, log_l, log_g):
        # type: (List[float], List[float], List[float]) -> List[float]
        samples, log_l, log_g = map(numpy.asarray, (samples, log_l, log_g))
        if len(samples):
            score = log_l - log_g
            if len(samples) != len(score):
                raise ValueError()
            best = numpy.argmax(score)
            return [samples[best]] * len(samples)
        else:
            return []

    @classmethod
    def _logsum_rows(cls, x):
        # type: (List[float]) -> numpy.ndarray
        x = numpy.asarray(x)
        m = x.max(axis=1)
        return numpy.log(numpy.exp(x - m[:, None]).sum(axis=1)) + m

    @classmethod
    def _normal_cdf(cls, x, mu, sigma):
        # type: (float, List[float], List[float]) -> numpy.ndarray
        mu, sigma = map(numpy.asarray, (mu, sigma))
        top = x - mu
        bottom = numpy.maximum(numpy.sqrt(2) * sigma, EPS)
        z = top / bottom
        return 0.5 * (1 + scipy.special.erf(z))

    @classmethod
    def _lognormal_cdf(cls, x, mu, sigma):
        if x.min() < 0:
            raise ValueError("negative argument is given to _lognormal_cdf", x)
        olderr = numpy.seterr(divide='ignore')
        try:
            top = numpy.log(numpy.maximum(x, EPS)) - mu
            bottom = numpy.maximum(numpy.sqrt(2) * sigma, EPS)
            z = top / bottom
            return .5 + .5 * scipy.special.erf(z)
        finally:
            numpy.seterr(**olderr)
