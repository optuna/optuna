import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
import scipy.special
from scipy.stats import truncnorm

from optuna import distributions
from optuna._study_direction import StudyDirection
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers._tpe.multivariate_parzen_estimator import _MultivariateParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


EPS = 1e-12
_DISTRIBUTION_CLASSES = (
    distributions.UniformDistribution,
    distributions.LogUniformDistribution,
    distributions.DiscreteUniformDistribution,
    distributions.IntUniformDistribution,
    distributions.IntLogUniformDistribution,
    distributions.CategoricalDistribution,
)
_logger = get_logger(__name__)


def default_gamma(x: int) -> int:

    return min(int(np.ceil(0.1 * x)), 25)


def hyperopt_default_gamma(x: int) -> int:

    return min(int(np.ceil(0.25 * np.sqrt(x))), 25)


def default_weights(x: int) -> np.ndarray:

    if x == 0:
        return np.asarray([])
    elif x < 25:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
        flat = np.ones(25)
        return np.concatenate([ramp, flat], axis=0)


class TPESampler(BaseSampler):
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
                x = trial.suggest_uniform("x", -10, 10)
                return x ** 2


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
        multivariate:
            If this is :obj:`True`, the multivariate TPE is used when suggesting parameters.
            The multivariate TPE is reported to outperform the independent TPE. See `BOHB: Robust
            and Efficient Hyperparameter Optimization at Scale
            <http://proceedings.mlr.press/v80/falkner18a.html>`_ for more details.

            .. note::
                Added in v2.2.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.2.0.
        warn_independent_sampling:
            If this is :obj:`True` and ``multivariate=True``, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.
            If ``multivariate=False``, this flag has no effect.
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
        *,
        multivariate: bool = False,
        warn_independent_sampling: bool = True,
    ) -> None:

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights
        )
        self._prior_weight = prior_weight
        self._n_startup_trials = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._gamma = gamma
        self._weights = weights

        self._warn_independent_sampling = warn_independent_sampling
        self._rng = np.random.RandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

        self._multivariate = multivariate
        self._search_space = IntersectionSearchSpace()

        if multivariate:
            warnings.warn(
                "``multivariate`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

    def reseed_rng(self) -> None:

        self._rng = np.random.RandomState()
        self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        if not self._multivariate:
            return {}

        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if not isinstance(distribution, _DISTRIBUTION_CLASSES):
                if self._warn_independent_sampling:
                    complete_trials = study.get_trials(deepcopy=False)
                    if len(complete_trials) >= self._n_startup_trials:
                        self._log_independent_sampling(trial, name)
                continue
            search_space[name] = distribution

        return search_space

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "instead of being sampled by multivariate TPE sampler. "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `TPESampler`, "
            "if this independent sampling is intended behavior.".format(param_name, trial.number)
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        self._raise_error_if_multi_objective(study)

        if search_space == {}:
            return {}

        param_names = list(search_space.keys())
        values, scores = _get_multivariate_observation_pairs(study, param_names)

        # If the number of samples is insufficient, we run random trial.
        n = len(scores)
        if n < self._n_startup_trials:
            return {}

        # We divide data into below and above.
        below, above = self._split_multivariate_observation_pairs(values, scores)
        # We then sample by maximizing log likelihood ratio.
        mpe_below = _MultivariateParzenEstimator(
            below, search_space, self._parzen_estimator_parameters
        )
        mpe_above = _MultivariateParzenEstimator(
            above, search_space, self._parzen_estimator_parameters
        )
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = TPESampler._compare_multivariate(
            samples_below, log_likelihoods_below, log_likelihoods_above
        )

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        self._raise_error_if_multi_objective(study)

        values, scores = _get_observation_pairs(study, param_name)

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
        self, config_vals: List[Optional[float]], loss_vals: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:

        config_vals = np.asarray(config_vals)
        loss_vals = np.asarray(loss_vals, dtype=[("step", float), ("score", float)])

        n_below = self._gamma(len(config_vals))
        loss_ascending = np.argsort(loss_vals)
        below = config_vals[np.sort(loss_ascending[:n_below])]
        below = np.asarray([v for v in below if v is not None], dtype=float)
        above = config_vals[np.sort(loss_ascending[n_below:])]
        above = np.asarray([v for v in above if v is not None], dtype=float)
        return below, above

    def _split_multivariate_observation_pairs(
        self,
        config_vals: Dict[str, List[Optional[float]]],
        loss_vals: List[Tuple[float, float]],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        config_vals = {k: np.asarray(v, dtype=float) for k, v in config_vals.items()}
        loss_vals = np.asarray(loss_vals, dtype=[("step", float), ("score", float)])

        n_below = self._gamma(len(loss_vals))
        index_loss_ascending = np.argsort(loss_vals)
        # `np.sort` is used to keep chronological order.
        index_below = np.sort(index_loss_ascending[:n_below])
        index_above = np.sort(index_loss_ascending[n_below:])
        below = {}
        above = {}
        for param_name, param_val in config_vals.items():
            below[param_name] = param_val[index_below]
            above[param_name] = param_val[index_above]

        return below, above

    def _sample_uniform(
        self, distribution: distributions.UniformDistribution, below: np.ndarray, above: np.ndarray
    ) -> float:

        low = distribution.low
        high = distribution.high
        return self._sample_numerical(low, high, below, above)

    def _sample_loguniform(
        self,
        distribution: distributions.LogUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> float:

        low = distribution.low
        high = distribution.high
        return self._sample_numerical(low, high, below, above, is_log=True)

    def _sample_discrete_uniform(
        self,
        distribution: distributions.DiscreteUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> float:

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

    def _sample_int(
        self,
        distribution: distributions.IntUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> int:

        d = distributions.DiscreteUniformDistribution(
            low=distribution.low, high=distribution.high, q=distribution.step
        )
        return int(self._sample_discrete_uniform(d, below, above))

    def _sample_int_loguniform(
        self,
        distribution: distributions.IntLogUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> int:

        low = distribution.low - 0.5
        high = distribution.high + 0.5

        sample = self._sample_numerical(low, high, below, above, is_log=True)
        best_sample = np.round(sample)

        return int(min(max(best_sample, distribution.low), distribution.high))

    def _sample_numerical(
        self,
        low: float,
        high: float,
        below: np.ndarray,
        above: np.ndarray,
        q: Optional[float] = None,
        is_log: bool = False,
    ) -> float:

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
            parzen_estimator=parzen_estimator_below, low=low, high=high, q=q, size=size
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

    def _sample_categorical_index(
        self,
        distribution: distributions.CategoricalDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> int:

        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)

        # We can use `np.arange(len(distribution.choices))` instead of sampling from `l(x)`
        # when the cardinality of categorical parameters is lower than `n_ei_candidates`.
        # Though it seems to be theoretically correct, it leads to performance degradation
        # on the NAS benchmark experiment in https://arxiv.org/abs/1902.09635.
        # See https://github.com/optuna/optuna/pull/1603 for more details.
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
        parzen_estimator: _ParzenEstimator,
        low: float,
        high: float,
        q: Optional[float] = None,
        size: Tuple = (),
    ) -> np.ndarray:

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
                    random_state=self._rng,
                ),
            )

        if q is None:
            return samples
        else:
            return np.round(samples / q) * q

    def _gmm_log_pdf(
        self,
        samples: np.ndarray,
        parzen_estimator: _ParzenEstimator,
        low: float,
        high: float,
        q: Optional[float] = None,
    ) -> np.ndarray:

        weights = parzen_estimator.weights
        mus = parzen_estimator.mus
        sigmas = parzen_estimator.sigmas
        samples, weights, mus, sigmas = map(np.asarray, (samples, weights, mus, sigmas))
        if samples.size == 0:
            return np.asarray([], dtype=float)
        if weights.ndim != 1:
            raise ValueError(
                "The 'weights' should be 1-dimension. "
                "But weights.shape = {}".format(weights.shape)
            )
        if mus.ndim != 1:
            raise ValueError(
                "The 'mus' should be 1-dimension. But mus.shape = {}".format(mus.shape)
            )
        if sigmas.ndim != 1:
            raise ValueError(
                "The 'sigmas' should be 1-dimension. But sigmas.shape = {}".format(sigmas.shape)
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

    def _sample_from_categorical_dist(
        self, probabilities: np.ndarray, size: Tuple[int]
    ) -> np.ndarray:

        if size == (0,):
            return np.asarray([], dtype=float)
        assert len(size)

        if probabilities.size == 1 and isinstance(probabilities[0], np.ndarray):
            probabilities = probabilities[0]
        assert probabilities.ndim == 1

        n_draws = np.prod(size).item()
        sample = self._rng.multinomial(n=1, pvals=probabilities, size=n_draws)
        assert sample.shape == size + probabilities.shape
        return_val = np.dot(sample, np.arange(probabilities.size)).reshape(size)
        return return_val

    @classmethod
    def _categorical_log_pdf(cls, sample: np.ndarray, p: np.ndarray) -> np.ndarray:

        if sample.size:
            return np.log(np.asarray(p)[sample])
        else:
            return np.asarray([])

    @classmethod
    def _compare(cls, samples: np.ndarray, log_l: np.ndarray, log_g: np.ndarray) -> np.ndarray:

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
    def _compare_multivariate(
        cls,
        multivariate_samples: Dict[str, np.ndarray],
        log_l: np.ndarray,
        log_g: np.ndarray,
    ) -> Dict[str, Union[float, int]]:

        sample_size = next(iter(multivariate_samples.values())).size
        if sample_size:
            score = log_l - log_g
            if sample_size != score.size:
                raise ValueError(
                    "The size of the 'samples' and that of the 'score' "
                    "should be same. "
                    "But (samples.size, score.size) = ({}, {})".format(sample_size, score.size)
                )
            best = np.argmax(score)
            return {k: v[best].item() for k, v in multivariate_samples.items()}
        else:
            raise ValueError(
                "The size of 'samples' should be more than 0."
                "But samples.size = {}".format(sample_size)
            )

    @classmethod
    def _logsum_rows(cls, x: np.ndarray) -> np.ndarray:

        x = np.asarray(x)
        m = x.max(axis=1)
        return np.log(np.exp(x - m[:, None]).sum(axis=1)) + m

    @classmethod
    def _normal_cdf(cls, x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + scipy.special.erf(z))

    @staticmethod
    def hyperopt_parameters() -> Dict[str, Any]:
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
                    x = trial.suggest_uniform("x", -10, 10)
                    return x ** 2


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


def _get_observation_pairs(
    study: Study, param_name: str
) -> Tuple[List[Optional[float]], List[Tuple[float, float]]]:
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
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)):
        if trial.state is TrialState.COMPLETE:
            if trial.value is None:
                continue
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
            assert False

        param_value: Optional[float] = None
        if param_name in trial.params:
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])

        values.append(param_value)
        scores.append(score)

    return values, scores


def _get_multivariate_observation_pairs(
    study: Study, param_names: List[str]
) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, float]]]:

    sign = 1
    if study.direction == StudyDirection.MAXIMIZE:
        sign = -1

    scores = []
    values: Dict[str, List[Optional[float]]] = {param_name: [] for param_name in param_names}
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)):

        # We extract score from the trial.
        if trial.state is TrialState.COMPLETE:
            if trial.value is None:
                continue
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
            assert False
        scores.append(score)

        # We extract param_value from the trial.
        for param_name in param_names:
            assert param_name in trial.params
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])
            values[param_name].append(param_value)

    return values, scores
