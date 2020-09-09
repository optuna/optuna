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

import optuna
from optuna._experimental import experimental
from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.multi_objective import _hypervolume
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.multi_objective.samplers import RandomMultiObjectiveSampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.study import StudyDirection

try:
    import pygmo

    _IS_PYGMO_AVAILABLE = True
except ImportError:
    _IS_PYGMO_AVAILABLE = False

EPS = 1e-12
_SPLITCACHE_KEY = "multi_objective:motpe:splitcache"
_WEIGHTS_BELOW_KEY = "multi_objective:motpe:weights_below"


def default_gamma(x: int) -> int:
    return int(np.floor(0.1 * x))


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
            of trials finish in the same study. 11 * number of variables - 1 is recommended in the
            original paper.
        n_ehvi_candidates:
            Number of candidate samples used to calculate the expected hypervolume improvement.
        gamma:
            A function that takes the number of finished trials and returns the number of trials to
            form a density function for samples with low grains. See the original paper for more
            details.
        weights:
            A function that takes the number of finished trials and returns a weight for them. When
            None is set, weights are automatically calculated by a default algorithm.
        seed:
            Seed for random number generator.
        use_pygmo:
            Use pygmo to accelerate hypervolume computation. To use this option, pygmo must be
            installed.

    The original paper utilizes Latin hypercube sampling (LHS) for initialization. The following
    code is an example of initializing MOTPE using LHS. Here
    `pyDOE2 <https://pypi.org/project/pyDOE2/>`_ is used for LHS.

    Example:

        .. testcode::

                import numpy as np
                import pyDOE2

                seed = 128
                num_variables = 9
                n_startup_trials = 11 * num_variables - 1

                def objective(trial):
                    x = []
                    for i in range(1, num_variables + 1):
                        x.append(trial.suggest_uniform(f"x{i}", 0.0, 2.0 * i))
                    return x

                sampler = optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler(
                    n_startup_trials=n_startup_trials,
                    n_ehvi_candidates=24,
                    seed=seed
                )
                # NOTE: Sampled vectors are normalized. Denormalization is needed when a search
                # space is not normalized.
                initial_samples = pyDOE2.lhs(
                    num_variables, samples=n_startup_trials, criterion='maximin',
                    random_state=np.random.RandomState(seed)
                )
                for s in initial_samples:
                    study.enqueue_trial({f'x{i}': v for i, v in enumerate(s, start=1)})

                study.optimize(objective, n_trials=250)
    """

    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        consider_endpoints: bool = True,
        n_startup_trials: int = 10,
        n_ehvi_candidates: int = 24,
        gamma: Callable[[int], int] = default_gamma,
        weights: Optional[Callable[[int], np.ndarray]] = None,
        seed: Optional[int] = None,
        use_pygmo: bool = False,
    ) -> None:
        self._prior_weight = prior_weight
        self._n_startup_trials = n_startup_trials
        self._n_ehvi_candidates = n_ehvi_candidates
        self._gamma = gamma
        self._consider_prior = consider_prior
        self._prior_weight = prior_weight
        self._consider_magic_clip = consider_magic_clip
        self._consider_endpoints = consider_endpoints
        self._weights = weights
        self._rng = np.random.RandomState(seed)
        self._random_sampler = RandomMultiObjectiveSampler(seed=seed)
        self._use_pygmo = use_pygmo

    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
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
        param_distribution: BaseDistribution,
    ) -> Any:
        assert len(study.directions) >= 2, (
            "Number of objectives must be >= 2. "
            "Please use optuna.samplers.TPESampler for single-objective optimization."
        )

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
            return self._sample_uniform(
                study, trial, param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(
                study, trial, param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            return self._sample_discrete_uniform(
                study, trial, param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            return self._sample_int(
                study, trial, param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.IntLogUniformDistribution):
            return self._sample_int_loguniform(
                study, trial, param_distribution, below_param_values, above_param_values
            )
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            index = self._sample_categorical_index(
                study, trial, param_distribution, below_param_values, above_param_values
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
        loss_vals: List[List[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split observations into observations for l(x) and g(x) with the ratio of gamma:1-gamma.

        Weights for l(x) are also calculated in this method.

        This splitting strategy consists of the following two steps:
            1. Nondonation rank-based selection
            2. Hypervolume subset selection problem (HSSP)-based selection

        Please refer to the original paper for more details.
        """
        cvals = np.asarray(config_vals)
        lvals = np.asarray(loss_vals)

        # Solving HSSP the number of variables times is a waste of time.
        # Therefore, we cache the result of splitting.
        if _SPLITCACHE_KEY in trial.system_attrs:
            split_cache = trial.system_attrs[_SPLITCACHE_KEY]
            indices_below = split_cache["indices_below"]
            weights_below = split_cache["weights_below"]
            indices_above = split_cache["indices_above"]
        else:
            nondomination_ranks = _calculate_nondomination_rank(lvals)
            n_below = self._gamma(len(lvals))
            assert 0 <= n_below <= len(lvals)

            indices = np.array(range(len(lvals)))
            indices_below = np.array([], dtype=int)

            # Nondomination rank-based selection
            i = 0
            while len(indices_below) + sum(nondomination_ranks == i) <= n_below:
                indices_below = np.append(indices_below, indices[nondomination_ranks == i])
                i += 1

            # Hypervolume subset selection problem (HSSP)-based selection
            subset_size = n_below - len(indices_below)
            if subset_size > 0:
                rank_i_lvals = lvals[nondomination_ranks == i]
                rank_i_indices = indices[nondomination_ranks == i]
                worst_point = np.max(rank_i_lvals, axis=0)
                reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
                reference_point[reference_point == 0] = EPS
                selected_indices = self._solve_hssp(
                    rank_i_lvals, rank_i_indices, subset_size, reference_point
                )
                indices_below = np.append(indices_below, selected_indices)
            assert len(indices_below) == n_below

            indices_above = np.setdiff1d(indices, indices_below)

            attrs = {"indices_below": indices_below, "indices_above": indices_above}
            if self._weights is None:
                weights_below = self._calculate_default_weights_below(lvals, indices_below)
                attrs["weights_below"] = weights_below
            study._storage.set_trial_system_attr(trial._trial_id, _SPLITCACHE_KEY, attrs)

        below = cvals[indices_below]
        if self._weights is None:
            study._storage.set_trial_system_attr(
                trial._trial_id,
                _WEIGHTS_BELOW_KEY,
                lambda _: np.asarray(
                    [w for w, v in zip(weights_below, below) if v is not None], dtype=float
                ),
            )
        below = np.asarray([v for v in below if v is not None], dtype=float)
        above = cvals[indices_above]
        above = np.asarray([v for v in above if v is not None], dtype=float)
        return below, above

    def _sample_uniform(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        distribution: distributions.UniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> float:
        low = distribution.low
        high = distribution.high
        return self._sample_numerical(study, trial, low, high, below, above)

    def _sample_loguniform(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        distribution: distributions.LogUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> float:
        low = distribution.low
        high = distribution.high
        return self._sample_numerical(study, trial, low, high, below, above, is_log=True)

    def _sample_discrete_uniform(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
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

        best_sample = (
            self._sample_numerical(study, trial, low, high, below, above, q=q) + distribution.low
        )
        return min(max(best_sample, distribution.low), distribution.high)

    def _sample_int(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        distribution: distributions.IntUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> int:
        d = distributions.DiscreteUniformDistribution(
            low=distribution.low, high=distribution.high, q=distribution.step
        )
        return int(self._sample_discrete_uniform(study, trial, d, below, above))

    def _sample_int_loguniform(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        distribution: distributions.IntLogUniformDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> int:
        low = distribution.low - 0.5
        high = distribution.high + 0.5

        sample = self._sample_numerical(study, trial, low, high, below, above, is_log=True)
        best_sample = np.round(sample)

        return int(min(max(best_sample, distribution.low), distribution.high))

    def _sample_numerical(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
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

        size = (self._n_ehvi_candidates,)
        if self._weights is None:
            weights_below = study._storage.get_trial(trial._trial_id).system_attrs[
                _WEIGHTS_BELOW_KEY
            ]
        else:
            weights_below = self._weights
        parzen_estimator_parameters_below = _ParzenEstimatorParameters(
            self._consider_prior,
            self._prior_weight,
            self._consider_magic_clip,
            self._consider_endpoints,
            weights_below,
        )
        parzen_estimator_below = _ParzenEstimator(
            mus=below, low=low, high=high, parameters=parzen_estimator_parameters_below
        )
        samples_below = self._sample_from_gmm(
            parzen_estimator=parzen_estimator_below,
            low=low,
            high=high,
            q=q,
            size=size,
        )
        log_likelihoods_below = self._gmm_log_pdf(
            samples=samples_below,
            parzen_estimator=parzen_estimator_below,
            low=low,
            high=high,
            q=q,
        )

        if self._weights is None:
            weights_above = _default_weights_above
        else:
            weights_above = self._weights  # type: ignore
        parzen_estimator_parameters_above = _ParzenEstimatorParameters(
            self._consider_prior,
            self._prior_weight,
            self._consider_magic_clip,
            self._consider_endpoints,
            weights_above,
        )
        parzen_estimator_above = _ParzenEstimator(
            mus=above, low=low, high=high, parameters=parzen_estimator_parameters_above
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
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        distribution: distributions.CategoricalDistribution,
        below: np.ndarray,
        above: np.ndarray,
    ) -> int:
        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        upper = len(choices)
        size = (self._n_ehvi_candidates,)

        if self._weights is None:
            weights_below = study._storage.get_trial(trial._trial_id).system_attrs[
                _WEIGHTS_BELOW_KEY
            ](len(below))
        else:
            weights_below = self._weights(len(below))
        counts_below = np.bincount(below, minlength=upper, weights=weights_below)
        weighted_below = counts_below + self._prior_weight
        weighted_below /= weighted_below.sum()
        samples_below = self._sample_from_categorical_dist(weighted_below, size)
        log_likelihoods_below = MOTPEMultiObjectiveSampler._categorical_log_pdf(
            samples_below, weighted_below
        )

        if self._weights is None:
            weights_above = _default_weights_above(len(above))
        else:
            weights_above = self._weights(len(above))
        counts_above = np.bincount(above, minlength=upper, weights=weights_above)
        weighted_above = counts_above + self._prior_weight
        weighted_above /= weighted_above.sum()
        log_likelihoods_above = MOTPEMultiObjectiveSampler._categorical_log_pdf(
            samples_below, weighted_above
        )

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
            return MOTPEMultiObjectiveSampler._logsum_rows(
                -0.5 * mahalanobis + np.log(coefficient)
            )
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
        self, probabilities: np.ndarray, size: Tuple[int]
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

    def _compute_hypervolume(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        if self._use_pygmo:
            assert (
                _IS_PYGMO_AVAILABLE
            ), "Pygmo is not available. Please install pygmo to use this option."
            return pygmo.hypervolume(solution_set).compute(reference_point)
        else:
            return _hypervolume.WFG().compute(solution_set, reference_point)

    def _solve_hssp(
        self,
        rank_i_loss_vals: np.ndarray,
        rank_i_indices: np.ndarray,
        subset_size: int,
        reference_point: np.ndarray,
    ) -> np.ndarray:
        """Solve a hypervolume subset selection problem (HSSP) via a greedy algorithm.

        This method is a 1-1/e approximation algorithm to solve HSSP.

        For further information about algorithms to solve HSSP, please refer to the following
        paper:

        - `Greedy Hypervolume Subset Selection in Low Dimensions
           <https://ieeexplore.ieee.org/document/7570501>`_
        """
        selected_vecs = []  # type: List[np.ndarray]
        selected_indices = []  # type: List[int]
        contributions = [
            self._compute_hypervolume(np.asarray([v]), reference_point) for v in rank_i_loss_vals
        ]
        MARK_AS_SELECTED = -1
        hv_selected = 0.0
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
                    self._compute_hypervolume(np.asarray(selected_vecs + [p]), reference_point)
                    - hv_selected
                )
            selected_vecs += [selected_vec]
            selected_indices += [selected_index]
            hv_selected = self._compute_hypervolume(np.asarray(selected_vecs), reference_point)

        return np.asarray(selected_indices, dtype=int)

    def _calculate_default_weights_below(
        self,
        lvals: np.ndarray,
        indices_below: np.ndarray,
    ) -> np.ndarray:
        # Calculate weights based on hypervolume contributions.
        n_below = len(indices_below)
        if n_below == 0:
            return np.asarray([])
        elif n_below == 1:
            return np.asarray([1.0])
        else:
            lvals_below = lvals[indices_below].tolist()
            worst_point = np.max(lvals_below, axis=0)
            reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
            reference_point[reference_point == 0] = EPS
            hv = self._compute_hypervolume(np.asarray(lvals_below), reference_point)
            contributions = np.asarray(
                [
                    hv
                    - self._compute_hypervolume(
                        np.asarray(lvals_below[:i] + lvals_below[i + 1 :]), reference_point
                    )
                    for i in range(len(lvals))
                ]
            )
            weights_below = np.clip(contributions / np.max(contributions), 0, 1)
            return weights_below

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


def _calculate_nondomination_rank(loss_vals: np.ndarray) -> np.ndarray:
    vecs = loss_vals.copy()

    # Normalize values
    lb = vecs.min(axis=0, keepdims=True)
    ub = vecs.max(axis=0, keepdims=True)
    vecs = (vecs - lb) / (ub - lb)

    ranks = np.zeros(len(vecs))
    num_unranked = len(vecs)
    rank = 0
    MARK_AS_RANKED = 1.1
    while num_unranked > 0:
        extended = np.tile(vecs, (vecs.shape[0], 1, 1))
        counts = np.sum(
            np.logical_and(
                np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
                np.any(extended < np.swapaxes(extended, 0, 1), axis=2),
            ),
            axis=1,
        )
        vecs[counts == 0] = MARK_AS_RANKED
        ranks[counts == 0] = rank
        rank += 1
        num_unranked -= np.sum(counts == 0)
    return ranks


def _default_weights_above(x: int) -> np.ndarray:
    return np.ones(x)


def _get_observation_pairs(
    study: "multi_objective.study.MultiObjectiveStudy",
    param_name: str,
) -> Tuple[List[Optional[float]], List[List[float]]]:
    """Get observation pairs from the study.

    This function collects observation pairs from the complete trials of the study.
    Pruning is currently not supported.
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
        if trial.state != optuna.trial.TrialState.COMPLETE or None in trial.values:
            continue

        param_value = None  # type: Optional[float]
        if param_name in trial.params:
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])

        # Convert all objectives to minimization
        score = [
            v if d == StudyDirection.MINIMIZE else -v  # type: ignore
            for d, v in zip(study.directions, trial.values)
        ]

        values.append(param_value)
        scores.append(score)

    return values, scores
