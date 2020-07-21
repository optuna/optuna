from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import math

import numpy as np
import scipy.special
from scipy.stats import truncnorm

from pygmo import hypervolume

import optuna
from optuna._experimental import experimental
from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers import RandomSampler
from optuna.study import StudyDirection

EPS = 1e-12
_SPLITCACHE_KEY = "multi_objective:motpe:splitcache"


def default_gamma(x: int, _: int) -> int:
    """Default gamma function for MOTPE.
    Args:
        The first argument is the number of trials.
        The second argument is the number of nondominated trials.

    The second argument can be used to change gamma dynamically based on the number of nondominated
    trials.

    Example:
        def dynamic_gamma(_, x):
            return max(0, int(np.ceil(0.10 * x) - 1))
    """

    return max(0, int(np.ceil(0.1 * x)) - 1)


def default_weights(x: int) -> np.ndarray:
    if x == 0:
        return np.asarray([])
    else:
        return np.ones(x)


def default_hssp_reference_point(observations: np.ndarray, _: np.ndarray) -> np.ndarray:
    """Default reference point for HSSP.
    Args:
        The first argument is the all of the observed function values so far.
        The second argument is a subset of the observed function values that are an input for a
        hypervolume subset selection problem (HSSP).

    The second argument can be used to set a reference_point based on the input set for HSSP.

    Example:
        def hssp_reference_point(_, observations):
            worst_point = np.max(observations, axis=0)
            reference_point = 1.1 * worst_point
            return reference_point
    """

    worst_point = np.max(observations, axis=0)
    reference_point = np.maximum(
        np.maximum(
            1.1 * worst_point,  # case: value > 0
            0.9 * worst_point  # case: value < 0
        ),
        np.full(len(worst_point), EPS)  # case: value = 0
    )
    return reference_point


@experimental("2.0.0")
class MOTPEMultiObjectiveSampler(BaseMultiObjectiveSampler):
    """Multi-objective sampler using the MOTPE algorithm.

    This sampler is a multiobjective version of :class:`~optuna.samplers.TPESampler`.

    For further information about MOTPE algorithm, please refer to the following paper:

    - `Multiobjective tree-structured parzen estimator for computationally expensive optimization
      problems <https://dl.acm.org/doi/abs/10.1145/3377930.3389817>`_

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
            The random sampling is used instead of the MOTPE algorithm until the given number
            of trials finish in the same study.
        n_ehvi_candidates:
            Number of candidate samples used to calculate the expected hypervolume improvement.
        gamma:
            A function that takes the number of finished trials and the number of nondominated
            trials and returns the number of trials to form a density function for samples with
            low grains.
            See the original paper for more details.
        weights:
            A function that takes the number of finished trials and returns a weight for them.
        hssp_reference_point:
            A function that takes the all of the observed function values and a subset of the
            observed function values that are an input for a hypervolume subset selection problem
            (HSSP) and returns a reference point for HSSP.
            See the original paper for more details.
        seed:
            Seed for random number generator.
    """

    def __init__(
            self,
            consider_prior: bool = True,
            prior_weight: float = 1.0,
            consider_magic_clip: bool = True,
            consider_endpoints: bool = False,
            n_startup_trials: int = 10,
            n_ehvi_candidates: int = 24,
            gamma: Callable[[int], int] = default_gamma,
            weights: Callable[[int], np.ndarray] = default_weights,
            hssp_reference_point: Callable[[np.ndarray, np.ndarray], np.ndarray] = \
                default_hssp_reference_point,
            seed: Optional[int] = None
        ) -> None:
        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior, prior_weight, consider_magic_clip, consider_endpoints, weights
        )
        self._prior_weight = prior_weight
        self._n_startup_trials = n_startup_trials
        self._n_ehvi_candidates = n_ehvi_candidates
        self._gamma = gamma
        self._weights = weights
        self._hssp_reference_point = hssp_reference_point

        self._rng = np.random.RandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

    def infer_relative_search_space(
            self,
            study: "multi_objective.study.MultiObjectiveStudy",
            trial: "multi_objective.trial.FrozenMultiObjectiveTrial"
        ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
            self,
            study: "multi_objective.study.MultiObjectiveStudy",
            trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
            search_space: Dict[str, BaseDistribution],
        ) -> Dict[str, Any]:
        return {}

    def sample_independent(
            self,
            study: "multi_objective.study.MultiObjectiveStudy",
            trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
            param_name: str,
            param_distribution: BaseDistribution
        ) -> Any:
        values, scores = _get_observation_pairs(study, param_name)
        n = len(values)
        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
        below_param_values, above_param_values = self._split_observation_pairs(
            study, trial, values, scores
        )

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
            study: "multi_objective.study.MultiObjectiveStudy",
            trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
            config_vals: List[Optional[float]],
            loss_vals: List[List[float]]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Split observations into observations for l(x) and observations for g(x) with the ratio
           of gamma:1-gamma.

            This splitting strategy consists of the following two steps:
                1. Nondonation rank-based selection
                2. Hypervolume subset selection problem (HSSP)-based selection

            Please refer to the original paper for more details.
        """
        config_vals = np.asarray(config_vals)
        loss_vals = np.asarray(loss_vals)

        # Solving HSSP the number of variables times is a waste of time.
        # Therefore, we cache the result of splitting.
        if _SPLITCACHE_KEY in trial.system_attrs:
            split_cache = trial.system_attrs[_SPLITCACHE_KEY]
            below_indices = split_cache['below_indices']
            above_indices = split_cache['above_indices']
        else:
            nondomination_ranks = _calculate_nondomination_rank(loss_vals)
            n_below = self._gamma(len(config_vals), len(nondomination_ranks))
            indices = np.array(range(len(loss_vals)))
            below_indices = np.array([], dtype=int)

            # Nondomination rank-based selection
            i = 0
            while len(below_indices) + sum(nondomination_ranks == i) < n_below:
                below_indices = np.append(below_indices, indices[nondomination_ranks == i])
                i += 1

            # Hypervolume subset selection problem (HSSP)-based selection
            subset_size = n_below - len(below_indices)
            if subset_size > 0:
                rank_i_loss_vals = loss_vals[nondomination_ranks == i]
                rank_i_indices = indices[nondomination_ranks == i]
                reference_point = self._hssp_reference_point(loss_vals, rank_i_loss_vals)
                below_indices = np.append(below_indices, _solve_hssp(
                    rank_i_loss_vals, rank_i_indices, subset_size, reference_point
                ))
            assert len(below_indices) == n_below
            above_indices = np.setdiff1d(indices, below_indices)

            study._storage.set_trial_system_attr(
                trial._trial_id, _SPLITCACHE_KEY,
                {'below_indices': below_indices, 'above_indices': above_indices}
            )

        below = config_vals[below_indices]
        below = np.asarray([v for v in below if v is not None], dtype=float)
        above = config_vals[above_indices]
        above = np.asarray([v for v in above if v is not None], dtype=float)

        return below, above

    def _sample_uniform(
            self,
            distribution: distributions.UniformDistribution,
            below: np.ndarray,
            above: np.ndarray
        ) -> float:
        low = distribution.low
        high = distribution.high
        return self._sample_numerical(low, high, below, above)

    def _sample_loguniform(
            self,
            distribution: distributions.LogUniformDistribution,
            below: np.ndarray,
            above: np.ndarray
        ) -> float:
        low = distribution.low
        high = distribution.high
        return self._sample_numerical(low, high, below, above, is_log=True)

    def _sample_discrete_uniform(
            self,
            distribution: distributions.DiscreteUniformDistribution,
            below: np.ndarray,
            above: np.ndarray
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
            above: np.ndarray
        ) -> int:
        d = distributions.DiscreteUniformDistribution(
            low=distribution.low, high=distribution.high, q=distribution.step
        )
        return int(self._sample_discrete_uniform(d, below, above))

    def _sample_int_loguniform(
            self,
            distribution: distributions.IntLogUniformDistribution,
            below: np.ndarray,
            above: np.ndarray
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
            is_log: bool = False
        ) -> float:
        if is_log:
            low = np.log(low)
            high = np.log(high)
            below = np.log(below)
            above = np.log(above)

        size = (self._n_ehvi_candidates,)

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
            MOTPEMultiObjectiveSampler._compare(
                samples=samples_below, log_l=log_likelihoods_below, log_g=log_likelihoods_above
            )
        )
        return math.exp(ret) if is_log else ret

    def _sample_categorical_index(
            self,
            distribution: distributions.CategoricalDistribution,
            below: np.ndarray,
            above: np.ndarray
        ) -> int:
        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)
        size = (self._n_ehvi_candidates,)

        weights_below = self._weights(len(below))
        counts_below = np.bincount(below, minlength=upper, weights=weights_below)
        weighted_below = counts_below + self._prior_weight
        weighted_below /= weighted_below.sum()
        samples_below = self._sample_from_categorical_dist(weighted_below, size)
        log_likelihoods_below = MOTPEMultiObjectiveSampler._categorical_log_pdf(
            samples_below, weighted_below)

        weights_above = self._weights(len(above))
        counts_above = np.bincount(above, minlength=upper, weights=weights_above)
        weighted_above = counts_above + self._prior_weight
        weighted_above /= weighted_above.sum()
        log_likelihoods_above = MOTPEMultiObjectiveSampler._categorical_log_pdf(
            samples_below, weighted_above)

        return int(
            MOTPEMultiObjectiveSampler._compare(
                samples=samples_below, log_l=log_likelihoods_below, log_g=log_likelihoods_above
            )
        )

    def _sample_from_gmm(
            self,
            parzen_estimator: _ParzenEstimator,
            low: float,
            high: float,
            q: Optional[float] = None,
            size: Tuple = ()
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
        while True:
            samples = truncnorm.rvs(
                trunc_low,
                trunc_high,
                size=size,
                loc=mus[active],
                scale=sigmas[active],
                random_state=self._rng,
            )
            if (samples >= low).all() and (samples < high).all():
                break

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
            q: Optional[float] = None
        ) -> np.ndarray:
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
                "The 'mus' should be 2-dimension. But mus.shape = {}".format(mus.shape)
            )
        if sigmas.ndim != 1:
            raise ValueError(
                "The 'sigmas' should be 2-dimension. But sigmas.shape = {}".format(sigmas.shape)
            )

        p_accept = np.sum(
            weights
            * (
                MOTPEMultiObjectiveSampler._normal_cdf(high, mus, sigmas)
                - MOTPEMultiObjectiveSampler._normal_cdf(low, mus, sigmas)
            )
        )
        if q is None:
            distance = samples[..., None] - mus
            mahalanobis = (distance / np.maximum(sigmas, EPS)) ** 2
            Z = np.sqrt(2 * np.pi) * sigmas
            coefficient = weights / Z / p_accept
            return MOTPEMultiObjectiveSampler._logsum_rows(-0.5 * mahalanobis + np.log(coefficient))
        else:
            cdf_func = MOTPEMultiObjectiveSampler._normal_cdf
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
            self,
            probabilities: np.ndarray,
            size: Tuple[int]
        ) -> np.ndarray:

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
            sample: np.ndarray,
            p: np.ndarray
        ) -> np.ndarray:
        if sample.size:
            return np.log(np.asarray(p)[sample])
        else:
            return np.asarray([])

    @classmethod
    def _compare(
            cls,
            samples: np.ndarray,
            log_l: np.ndarray,
            log_g: np.ndarray
        ) -> np.ndarray:
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
            return np.asarray([samples[best]])
        else:
            return np.asarray([])

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

    @classmethod
    def _log_normal_cdf(cls, x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        mu, sigma = map(np.asarray, (mu, sigma))
        if x < 0:
            raise ValueError("Negative argument is given to _lognormal_cdf. x: {}".format(x))
        denominator = np.log(np.maximum(x, EPS)) - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 + 0.5 * scipy.special.erf(z)


def _calculate_nondomination_rank(
        loss_vals: List[List[float]],
    ) -> np.ndarray:
    vecs = loss_vals.copy()
    ranks = np.zeros(len(vecs))

    num_unranked = len(vecs)
    rank = 0
    MARK_AS_RANKED = np.finfo(float).max
    while num_unranked > 0:
        extended = np.tile(vecs, (vecs.shape[0], 1, 1))
        counts = np.sum(np.logical_and(
            np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
            np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)
        vecs[counts == 0] = MARK_AS_RANKED
        ranks[counts == 0] = rank
        rank += 1
        num_unranked -= np.sum(counts == 0)
    return ranks


def _solve_hssp(
        rank_i_loss_vals: np.ndarray,
        rank_i_indices: np.ndarray,
        subset_size: int,
        reference_point: np.ndarray,
    ) -> np.ndarray:
    """Solve a hypervolume subset selection problem (HSSP) via a greedy algorithm.

    This method is a 1-1/e approximation algorithm to solve HSSP.

    For further information about algorithms to solve HSSP, please refer to the following paper:

    - `Greedy Hypervolume Subset Selection in Low Dimensions
       <https://ieeexplore.ieee.org/document/7570501>`_
    """

    selected_vecs = []
    selected_indices = []
    contributions = [hypervolume([v]).compute(reference_point) for v in rank_i_loss_vals]
    MARK_AS_SELECTED = np.finfo(float).min
    hv_selected = 0
    while len(selected_indices) < subset_size:
        max_index = np.argmax(contributions)
        contributions[max_index] = MARK_AS_SELECTED
        selected_index = rank_i_indices[max_index]
        selected_vec = rank_i_loss_vals[max_index]
        for j, v in enumerate(rank_i_loss_vals):
            if contributions[j] == MARK_AS_SELECTED:
                continue
            p = np.max([selected_vec, v], axis=0)
            contributions[j] -= (
                hypervolume(selected_vecs + [p]).compute(reference_point) - hv_selected
            )
        selected_vecs += [selected_vec]
        selected_indices += [selected_index]
        hv_selected = hypervolume(selected_vecs).compute(reference_point)

    return np.asarray(selected_indices, dtype=int)


def _get_observation_pairs(
        study: "multi_objective.study.MultiObjectiveStudy",
        param_name: str,
    ) -> Tuple[List[Optional[float]], List[List[float]]]:
    """Get observation pairs from the study.

       This function collects observation pairs from the complete trials of the study.
       Currently, pruning is not supported.
       The values for trials that don't contain the parameter named ``param_name`` are set to None.

       Objective values are negated if their directions are maximization and all objectives are
       treated as minimization in the MOTPE algorithm.
    """

    trials = [
        multi_objective.trial.FrozenMultiObjectiveTrial(study.n_objectives, trial)
        for trial in study._storage.get_all_trials(study._study_id, deepcopy=False)
    ]

    values = []
    scores = []
    for trial in trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        param_value = None  # type: Optional[float]
        if param_name in trial.params:
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])

        # Convert all objectives to minimization
        score = [v if d == StudyDirection.MINIMIZE else -v \
                 for d, v in zip(study.directions, trial.values)]

        values.append(param_value)
        scores.append(score)

    return values, scores
