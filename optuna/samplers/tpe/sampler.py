import math

import numpy as np
import scipy.special
from scipy.stats import truncnorm

from optuna import distributions
from optuna.samplers import base
from optuna.samplers import random
from optuna.samplers.tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers.tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA

EPS = 1e-12


def default_gamma(x):
    # type: (int) -> int

    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_default_gamma(x):
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
    """Sampler using TPE (Tree-structured Parzen Estimator) algorithm.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) ``l(x)`` to
    the set of parameter values associated with the best objective values, and another GMM
    ``g(x)`` to the remaining parameter values. It chooses the parameter value ``x`` that
    maximizes the ratio ``l(x)/g(x)``.

    For further information about TPE algorithm, please refer to the following papers:

    - `Algorithms for Hyper-Parameter Optimization
      <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
    - `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
      Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import TPESampler

            def objective(trial):
                x = trial.suggest_uniform('x', -10, 10)
                return x**2

            study = optuna.create_study(sampler=TPESampler())
            study.optimize(objective, n_trials=10)

    Args:
        consider_prior:
            Enhance the stability of Parzen estimator by imposing a Gaussian prior when
            :obj:`True`. The prior is only effective if the sampling distribution is
            either :class:`~optuna.distributions.UniformDistribution`,
            :class:`~optuna.distributions.DiscreteUniformDistribution`,
            :class:`~optuna.distributions.LogUniformDistribution`,
            :class:`~optuna.distributions.IntUniformDistribution`,
            or :class:`~optuna.distributions.IntLogUniformDistribution`.
        prior_weight:
            The weight of the prior. This argument is used in
            :class:`~optuna.distributions.UniformDistribution`,
            :class:`~optuna.distributions.DiscreteUniformDistribution`,
            :class:`~optuna.distributions.LogUniformDistribution`,
            :class:`~optuna.distributions.IntUniformDistribution`,
            :class:`~optuna.distributions.IntLogUniformDistribution`, and
            :class:`~optuna.distributions.CategoricalDistribution`.
        consider_magic_clip:
            Enable a heuristic to limit the smallest variances of Gaussians used in
            the Parzen estimator.
        consider_endpoints:
            Take endpoints of domains into account when calculating variances of Gaussians
            in Parzen estimator. See the original paper for details on the heuristics
            to calculate the variances.
        n_startup_trials:
            The random sampling is used instead of the TPE algorithm until the given number
            of trials finish in the same study.
        n_ei_candidates:
            Number of candidate samples used to calculate the expected improvement.
        gamma:
            A function that takes the number of finished trials and returns the number
            of trials to form a density function for samples with low grains.
            See the original paper for more details.
        weights:
            A function that takes the number of finished trials and returns a weight for them.
            See `Making a Science of Model Search: Hyperparameter Optimization in Hundreds of
            Dimensions for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_
            for more details.
        seed:
            Seed for random number generator.
    """

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
        seed=None,  # type: Optional[int]
    ):
        # type: (...) -> None

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights
        )
        self._prior_weight = prior_weight
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._gamma = gamma
        self._weights = weights

        self._rng = np.random.RandomState(seed)
        self._random_sampler = random.RandomSampler(seed=seed)

    def reseed_rng(self) -> None:

        self._rng = np.random.RandomState()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        return {}

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> Any

        values, scores = _get_observation_pairs(study, param_name, trial)

        n = len(values)

        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        below_param_values, above_param_values = self._split_observation_pairs(values, scores)

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._sample_uniform(param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(
                param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            return self._sample_discrete_uniform(
                param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            return self._sample_int(param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.IntLogUniformDistribution):
            return self._sample_int_loguniform(
                param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            index = self._sample_categorical_index(
                param_distribution, below_param_values, above_param_values
            )
            return param_distribution.choices[index]
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
                    param_distribution, distribution_list
                )
            )

    def _split_observation_pairs(
        self,
        config_vals,  # type: List[Optional[float]]
        loss_vals,  # type: List[Tuple[float, float]]
    ):
        # type: (...) -> Tuple[np.ndarray, np.ndarray]

        config_vals = np.asarray(config_vals)
        loss_vals = np.asarray(loss_vals, dtype=[("step", float), ("score", float)])

        n_below = self._gamma(len(config_vals))
        loss_ascending = np.argsort(loss_vals)
        below = config_vals[np.sort(loss_ascending[:n_below])]
        below = np.asarray([v for v in below if v is not None], dtype=float)
        above = config_vals[np.sort(loss_ascending[n_below:])]
        above = np.asarray([v for v in above if v is not None], dtype=float)
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

        # Shift below and above to [0, r]
        above -= distribution.low
        below -= distribution.low

        best_sample = self._sample_numerical(low, high, below, above, q=q) + distribution.low
        return min(max(best_sample, distribution.low), distribution.high)

    def _sample_int(self, distribution, below, above):
        # type: (distributions.IntUniformDistribution, np.ndarray, np.ndarray) -> int

        d = distributions.DiscreteUniformDistribution(
            low=distribution.low, high=distribution.high, q=distribution.step
        )
        return int(self._sample_discrete_uniform(d, below, above))

    def _sample_int_loguniform(self, distribution, below, above):
        # type: (distributions.IntLogUniformDistribution, np.ndarray, np.ndarray) -> int

        low = distribution.low - 0.5
        high = distribution.high + 0.5

        sample = self._sample_numerical(low, high, below, above, is_log=True)
        best_sample = (
            np.round((sample - distribution.low) / distribution.step) * distribution.step
            + distribution.low
        )
        return int(min(max(best_sample, distribution.low), distribution.high))

    def _sample_numerical(
        self,
        low,  # type: float
        high,  # type: float
        below,  # type: np.ndarray
        above,  # type: np.ndarray
        q=None,  # type: Optional[float]
        is_log=False,  # type: bool
    ):
        # type: (...) -> float

        if is_log:
            low = np.log(low)
            high = np.log(high)
            below = np.log(below)
            above = np.log(above)

        size = (self._n_ei_candidates,)

        parzen_estimator_below = _ParzenEstimator(
            mus=below, low=low, high=high, parameters=self._parzen_estimator_parameters
        )
        samples_below = self._sample_from_gmm(
            parzen_estimator=parzen_estimator_below, low=low, high=high, q=q, size=size,
        )
        log_likelihoods_below = self._gmm_log_pdf(
            samples=samples_below,
            parzen_estimator=parzen_estimator_below,
            low=low,
            high=high,
            q=q,
        )

        parzen_estimator_above = _ParzenEstimator(
            mus=above, low=low, high=high, parameters=self._parzen_estimator_parameters
        )

        log_likelihoods_above = self._gmm_log_pdf(
            samples=samples_below,
            parzen_estimator=parzen_estimator_above,
            low=low,
            high=high,
            q=q,
        )

        ret = float(
            TPESampler._compare(
                samples=samples_below, log_l=log_likelihoods_below, log_g=log_likelihoods_above
            )[0]
        )
        return math.exp(ret) if is_log else ret

    def _sample_categorical_index(self, distribution, below, above):
        # type: (distributions.CategoricalDistribution, np.ndarray, np.ndarray) -> int

        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)
        size = (self._n_ei_candidates,)

        weights_below = self._weights(len(below))
        counts_below = np.bincount(below, minlength=upper, weights=weights_below)
        weighted_below = counts_below + self._prior_weight
        weighted_below /= weighted_below.sum()
        samples_below = self._sample_from_categorical_dist(weighted_below, size)
        log_likelihoods_below = TPESampler._categorical_log_pdf(samples_below, weighted_below)

        weights_above = self._weights(len(above))
        counts_above = np.bincount(above, minlength=upper, weights=weights_above)
        weighted_above = counts_above + self._prior_weight
        weighted_above /= weighted_above.sum()
        log_likelihoods_above = TPESampler._categorical_log_pdf(samples_below, weighted_above)

        return int(
            TPESampler._compare(
                samples=samples_below, log_l=log_likelihoods_below, log_g=log_likelihoods_above
            )[0]
        )

    def _sample_from_gmm(
        self,
        parzen_estimator,  # type: _ParzenEstimator
        low,  # type: float
        high,  # type: float
        q=None,  # type: Optional[float]
        size=(),  # type: Tuple
    ):
        # type: (...) -> np.ndarray

        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        weights, mus, sigmas = map(np.asarray, (weights, mus, sigmas))

        if low >= high:
            raise ValueError(
                "The 'low' should be lower than the 'high'. "
                "But (low, high) = ({}, {}).".format(low, high)
            )

        active = np.argmax(self._rng.multinomial(1, weights, size=size), axis=-1)
        trunc_low = (low - mus[active]) / sigmas[active]
        trunc_high = (high - mus[active]) / sigmas[active]
        while True:
            samples = truncnorm.rvs(
                trunc_low,
                trunc_high,
                size=size,
                loc=mus[active],
                scale=sigmas[active],
                random_state=self._rng,
            )
            if (samples < high).all():
                break

        if q is None:
            return samples
        else:
            return np.round(samples / q) * q

    def _gmm_log_pdf(
        self,
        samples,  # type: np.ndarray
        parzen_estimator,  # type: _ParzenEstimator
        low,  # type: float
        high,  # type: float
        q=None,  # type: Optional[float]
    ):
        # type: (...) -> np.ndarray

        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))
        if samples.size == 0:
            return np.asarray([], dtype=float)
        if weights.ndim != 1:
            raise ValueError(
                "The 'weights' should be 2-dimension. "
                "But weights.shape = {}".format(weights.shape)
            )
        if mus.ndim != 1:
            raise ValueError(
                "The 'mus' should be 2-dimension. " "But mus.shape = {}".format(mus.shape)
            )
        if sigmas.ndim != 1:
            raise ValueError(
                "The 'sigmas' should be 2-dimension. " "But sigmas.shape = {}".format(sigmas.shape)
            )

        p_accept = np.sum(
            weights
            * (
                TPESampler._normal_cdf(high, mus, sigmas)
                - TPESampler._normal_cdf(low, mus, sigmas)
            )
        )

        if q is None:
            distance = samples[..., None] - mus
            mahalanobis = (distance / np.maximum(sigmas, EPS)) ** 2
            Z = np.sqrt(2 * np.pi) * sigmas
            coefficient = weights / Z / p_accept
            return TPESampler._logsum_rows(-0.5 * mahalanobis + np.log(coefficient))
        else:
            cdf_func = TPESampler._normal_cdf
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

    def _sample_from_categorical_dist(self, probabilities, size):
        # type: (np.ndarray, Tuple[int]) -> np.ndarray

        if probabilities.size == 1 and isinstance(probabilities[0], np.ndarray):
            probabilities = probabilities[0]
        probabilities = np.asarray(probabilities)

        if size == (0,):
            return np.asarray([], dtype=float)
        assert len(size)
        assert probabilities.ndim == 1

        n_draws = int(np.prod(size))
        sample = self._rng.multinomial(n=1, pvals=probabilities, size=int(n_draws))
        assert sample.shape == size + (probabilities.size,)
        return_val = np.dot(sample, np.arange(probabilities.size))
        return_val.shape = size
        return return_val

    @classmethod
    def _categorical_log_pdf(
        cls,
        sample,  # type: np.ndarray
        p,  # type: np.ndarray
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
                raise ValueError(
                    "The size of the 'samples' and that of the 'score' "
                    "should be same. "
                    "But (samples.size, score.size) = ({}, {})".format(samples.size, score.size)
                )

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
        return 0.5 + 0.5 * scipy.special.erf(z)

    @staticmethod
    def hyperopt_parameters():
        # type: () -> Dict[str, Any]
        """Return the the default parameters of hyperopt (v0.1.2).

        :class:`~optuna.samplers.TPESampler` can be instantiated with the parameters returned
        by this method.

        Example:

            Create a :class:`~optuna.samplers.TPESampler` instance with the default
            parameters of `hyperopt <https://github.com/hyperopt/hyperopt/tree/0.1.2>`_.

            .. testcode::

                    import optuna
                    from optuna.samplers import TPESampler

                    def objective(trial):
                        x = trial.suggest_uniform('x', -10, 10)
                        return x**2

                    sampler = TPESampler(**TPESampler.hyperopt_parameters())
                    study = optuna.create_study(sampler=sampler)
                    study.optimize(objective, n_trials=10)

        Returns:
            A dictionary containing the default parameters of hyperopt.

        """

        return {
            "consider_prior": True,
            "prior_weight": 1.0,
            "consider_magic_clip": True,
            "consider_endpoints": False,
            "n_startup_trials": 20,
            "n_ei_candidates": 24,
            "gamma": hyperopt_default_gamma,
            "weights": default_weights,
        }


def _get_observation_pairs(study, param_name, trial):
    # type: (Study, str, FrozenTrial) -> Tuple[List[Optional[float]], List[Tuple[float, float]]]
    """Get observation pairs from the study.

       This function collects observation pairs from the complete or pruned trials of the study.
       The values for trials that don't contain the parameter named ``param_name`` are set to None.

       An observation pair fundamentally consists of a parameter value and an objective value.
       However, due to the pruning mechanism of Optuna, final objective values are not always
       available. Therefore, this function uses intermediate values in addition to the final
       ones, and reports the value with its step count as ``(-step, value)``.
       Consequently, the structure of the observation pair is as follows:
       ``(param_value, (-step, value))``.

       The second element of an observation pair is used to rank observations in
       ``_split_observation_pairs`` method (i.e., observations are sorted lexicographically by
       ``(-step, value)``).
    """

    sign = 1
    if study.direction == StudyDirection.MAXIMIZE:
        sign = -1

    values = []
    scores = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state is TrialState.COMPLETE and trial.value is not None:
            score = (-float("inf"), sign * trial.value)
        elif trial.state is TrialState.PRUNED:
            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, float("inf"))
                else:
                    score = (-step, sign * intermediate_value)
            else:
                score = (float("inf"), 0.0)
        else:
            continue

        param_value = None  # type: Optional[float]
        if param_name in trial.params:
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])

        values.append(param_value)
        scores.append(score)

    return values, scores
