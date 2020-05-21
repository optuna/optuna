import math
from typing import Callable
from typing import Optional
from typing import Tuple

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
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Type  # NOQA

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
            or :class:`~optuna.distributions.IntUniformDistribution`.
        prior_weight:
            The weight of the prior. This argument is used in
            :class:`~optuna.distributions.UniformDistribution`,
            :class:`~optuna.distributions.DiscreteUniformDistribution`,
            :class:`~optuna.distributions.LogUniformDistribution`,
            :class:`~optuna.distributions.IntUniformDistribution` and
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
        n_ei_candidate:
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
        distribution:
            Type of underlying distribution. Either "gaussian" or "logistic".
            "logistic" runs faster.
    """

    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = False,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Callable[[int], np.ndarray] = default_weights,
        seed: Optional[int] = None,
        distribution: str = "gaussian",
    ) -> None:

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

        if distribution == "gaussian":
            self._dist = _GMM  # type: Type[_TruncatedMixturedDistribution]
        elif distribution == "logistic":
            self._dist = _LMM
        else:
            raise ValueError(
                'Distribution "{}" is not supported in TPE sampler.'.format(distribution)
            )

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
        dist_below = self._dist(
            low,
            high,
            parzen_estimator_below.mus,
            parzen_estimator_below.sigmas,
            parzen_estimator_below.weights,
        )
        samples_below = self._sample(distribution=dist_below, q=q, size=size)
        log_likelihoods_below = self._log_pdf(samples=samples_below, distribution=dist_below, q=q)

        parzen_estimator_above = _ParzenEstimator(
            mus=above, low=low, high=high, parameters=self._parzen_estimator_parameters
        )
        dist_above = self._dist(
            low,
            high,
            parzen_estimator_above.mus,
            parzen_estimator_above.sigmas,
            parzen_estimator_above.weights,
        )
        log_likelihoods_above = self._log_pdf(samples=samples_below, distribution=dist_above, q=q)

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

    def _sample(
        self,
        distribution: "_TruncatedMixturedDistribution",
        q: Optional[float] = None,
        size: Tuple = (),
    ) -> np.ndarray:

        samples = distribution.sample(size, self._rng)

        if q is None:
            return samples
        else:
            return np.round(samples / q) * q

    @staticmethod
    def _log_pdf(
        samples: np.ndarray,
        distribution: "_TruncatedMixturedDistribution",
        q: Optional[float] = None,
    ) -> np.ndarray:

        if samples.size == 0:
            return np.asarray([], dtype=float)

        if q is None:
            return distribution.log_pdf(samples)
        else:
            return distribution.quantized_log_pdf(samples, q)

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


class _TruncatedMixturedDistribution:
    def __init__(
        self, low: float, high: float, loc: np.ndarray, scale: np.ndarray, weights: np.ndarray
    ) -> None:
        if weights.ndim != 1:
            raise ValueError(
                "The 'weights' should be 1-dimension. "
                "But weights.shape = {}".format(weights.shape)
            )
        if loc.ndim != 1:
            raise ValueError(
                "The 'loc' should be 1-dimension. " "But loc.shape = {}".format(loc.shape)
            )
        if scale.ndim != 1:
            raise ValueError(
                "The 'scale' should be 1-dimension. " "But scale.shape = {}".format(scale.shape)
            )

        self.low = low
        self.high = high
        self.loc = loc
        self.scale = scale
        self.weights = weights
        self.normalization_constant = np.sum(
            self.unnormalized_cdf(np.asarray([high])) - self.unnormalized_cdf(np.asarray([low]))
        )

    def sample(self, size: Tuple, random_state: np.random.RandomState,) -> np.ndarray:
        raise NotImplementedError()

    def unnormalized_cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def quantized_log_pdf(self, x: np.ndarray, q: float) -> np.ndarray:
        upper_bound = np.minimum(x + q / 2.0, self.high)
        lower_bound = np.maximum(x - q / 2.0, self.low)
        probabilities = self.unnormalized_cdf(upper_bound) - self.unnormalized_cdf(lower_bound)
        return np.log(probabilities + EPS) - np.log(self.normalization_constant + EPS)


class _GMM(_TruncatedMixturedDistribution):
    def sample(self, size: Tuple, random_state: np.random.RandomState,) -> np.ndarray:
        active = np.argmax(random_state.multinomial(1, self.weights, size=size), axis=-1)
        loc = self.loc[active]
        scale = self.scale[active]
        trunc_low = (self.low - loc) / scale
        trunc_high = (self.high - loc) / scale

        while True:
            samples = truncnorm.rvs(
                trunc_low, trunc_high, size=size, loc=loc, scale=scale, random_state=random_state,
            )
            if (samples < self.high).all():
                break
        return samples

    def unnormalized_cdf(self, x: np.ndarray) -> np.ndarray:
        x = x[None]
        loc = self.loc[..., None]
        scale = self.scale[..., None]
        weights = self.weights[..., None]
        denominator = x - loc
        numerator = np.maximum(np.sqrt(2) * scale, EPS)
        z = denominator / numerator
        return np.sum(0.5 * (1 + scipy.special.erf(z)) * weights, axis=0)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        distance = x[..., None] - self.loc
        mahalanobis = (distance / np.maximum(self.scale, EPS)) ** 2
        Z = np.sqrt(2 * np.pi) * self.scale
        coefficient = self.weights / Z / self.normalization_constant
        return self._logsum_rows(-0.5 * mahalanobis + np.log(coefficient))

    @staticmethod
    def _logsum_rows(x: np.ndarray) -> np.ndarray:
        m = np.max(x, axis=1)
        return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m


class _LMM(_TruncatedMixturedDistribution):
    def sample(self, size: Tuple, random_state: np.random.RandomState,) -> np.ndarray:
        active = np.argmax(random_state.multinomial(1, self.weights, size=size), axis=-1)
        loc = self.loc[active]
        scale = self.scale[active]
        trunc_low = 1 / (1 + np.exp(-(self.low - loc) / scale))
        trunc_high = 1 / (1 + np.exp(-(self.high - loc) / scale))

        p = random_state.uniform(trunc_low, trunc_high, size=size)
        return loc - scale * np.log(1 / p - 1)

    def unnormalized_cdf(self, x: np.ndarray) -> np.ndarray:
        x = x[None]
        loc = self.loc[..., None]
        scale = self.scale[..., None]
        weights = self.weights[..., None]
        return np.sum(weights / (1 + np.exp((x - loc) / np.maximum(scale, EPS))), axis=0)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        p = self.unnormalized_cdf(x) * (1 - self.unnormalized_cdf(x))
        return np.log(p / np.maximum(self.normalization_constant, EPS))
