import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

import numpy as np

from optuna._hypervolume import WFG
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers._search_space.group_decomposed import _GroupDecomposedSearchSpace
from optuna.samplers._search_space.group_decomposed import _SearchSpaceGroup
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


EPS = 1e-12
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
    - `Multiobjective tree-structured parzen estimator for computationally expensive optimization
      problems <https://dl.acm.org/doi/10.1145/3377930.3389817>`_

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import TPESampler


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
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

            .. note::
                In the multi-objective case, this argument is only used to compute the weights of
                bad trials, i.e., trials to construct `g(x)` in the `paper
                <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
                ). The weights of good trials, i.e., trials to construct `l(x)`, are computed by a
                rule based on the hypervolume contribution proposed in the `paper of MOTPE
                <https://dl.acm.org/doi/10.1145/3377930.3389817>`_.
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
        group:
            If this and ``multivariate`` are :obj:`True`, the multivariate TPE with the group
            decomposed search space is used when suggesting parameters.
            The sampling algorithm decomposes the search space based on past trials and samples
            from the joint distribution in each decomposed subspace.
            The decomposed subspaces are a partition of the whole search space. Each subspace
            is a maximal subset of the whole search space, which satisfies the following:
            for a trial in completed trials, the intersection of the subspace and the search space
            of the trial becomes subspace itself or an empty set.
            Sampling from the joint distribution on the subspace is realized by multivariate TPE.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.

            Example:

            .. testcode::

                import optuna


                def objective(trial):
                    x = trial.suggest_categorical("x", ["A", "B"])
                    if x == "A":
                        return trial.suggest_float("y", -10, 10)
                    else:
                        return trial.suggest_int("z", -10, 10)


                sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)
        warn_independent_sampling:
            If this is :obj:`True` and ``multivariate=True``, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.
            If ``multivariate=False``, this flag has no effect.
        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
            nearby.

            .. note::
                Abnormally terminated trials often leave behind a record with a state of
                `RUNNING` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                `FAIL`.

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.

    Raises:
        ValueError:
            If ``multivariate`` is :obj:`False` and ``group`` is :obj:`True`.
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
        group: bool = False,
        warn_independent_sampling: bool = True,
        constant_liar: bool = False,
    ) -> None:

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            consider_prior,
            prior_weight,
            consider_magic_clip,
            consider_endpoints,
            weights,
            multivariate,
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
        self._group = group
        self._group_decomposed_search_space: Optional[_GroupDecomposedSearchSpace] = None
        self._search_space_group: Optional[_SearchSpaceGroup] = None
        self._search_space = IntersectionSearchSpace(include_pruned=True)
        self._constant_liar = constant_liar

        if multivariate:
            warnings.warn(
                "``multivariate`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

        if group:
            if not multivariate:
                raise ValueError(
                    "``group`` option can only be enabled when ``multivariate`` is enabled."
                )
            warnings.warn(
                "``group`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )
            self._group_decomposed_search_space = _GroupDecomposedSearchSpace(True)

        if constant_liar:
            warnings.warn(
                "``constant_liar`` option is an experimental feature."
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

        if self._group:
            assert self._group_decomposed_search_space is not None
            self._search_space_group = self._group_decomposed_search_space.calculate(study)
            for sub_space in self._search_space_group.search_spaces:
                # Sort keys because Python's string hashing is nondeterministic.
                for name, distribution in sorted(sub_space.items()):
                    if distribution.single():
                        continue
                    search_space[name] = distribution
            return search_space

        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def _log_independent_sampling(
        self, n_complete_trials: int, trial: FrozenTrial, param_name: str
    ) -> None:
        if self._warn_independent_sampling and self._multivariate:
            # The first trial samples independently.
            if n_complete_trials >= max(self._n_startup_trials, 1):
                _logger.warning(
                    f"The parameter '{param_name}' in trial#{trial.number} is sampled "
                    "independently instead of being sampled by multivariate TPE sampler. "
                    "(optimization performance may be degraded). "
                    "You can suppress this warning by setting `warn_independent_sampling` "
                    "to `False` in the constructor of `TPESampler`, "
                    "if this independent sampling is intended behavior."
                )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        if self._group:
            assert self._search_space_group is not None
            params = {}
            for sub_space in self._search_space_group.search_spaces:
                search_space = {}
                # Sort keys because Python's string hashing is nondeterministic.
                for name, distribution in sorted(sub_space.items()):
                    if not distribution.single():
                        search_space[name] = distribution
                params.update(self._sample_relative(study, trial, search_space))
            return params
        else:
            return self._sample_relative(study, trial, search_space)

    def _sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        if search_space == {}:
            return {}

        param_names = list(search_space.keys())
        values, scores = _get_observation_pairs(
            study, param_names, self._multivariate, self._constant_liar
        )

        # If the number of samples is insufficient, we run random trial.
        n = len(scores)
        if n < self._n_startup_trials:
            return {}

        # We divide data into below and above.
        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n))
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        below = _build_observation_dict(config_values, indices_below)
        above = _build_observation_dict(config_values, indices_above)

        # We then sample by maximizing log likelihood ratio.
        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                config_values, scores, indices_below
            )
            mpe_below = _ParzenEstimator(
                below, search_space, self._parzen_estimator_parameters, weights_below
            )
        else:
            mpe_below = _ParzenEstimator(below, search_space, self._parzen_estimator_parameters)
        mpe_above = _ParzenEstimator(above, search_space, self._parzen_estimator_parameters)
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = TPESampler._compare(samples_below, log_likelihoods_below, log_likelihoods_above)

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

        values, scores = _get_observation_pairs(
            study, [param_name], self._multivariate, self._constant_liar
        )

        n = len(scores)

        self._log_independent_sampling(n, trial, param_name)

        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n))
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        below = _build_observation_dict(config_values, indices_below)
        above = _build_observation_dict(config_values, indices_above)

        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                config_values, scores, indices_below
            )
            mpe_below = _ParzenEstimator(
                below,
                {param_name: param_distribution},
                self._parzen_estimator_parameters,
                weights_below,
            )
        else:
            mpe_below = _ParzenEstimator(
                below, {param_name: param_distribution}, self._parzen_estimator_parameters
            )
        mpe_above = _ParzenEstimator(
            above, {param_name: param_distribution}, self._parzen_estimator_parameters
        )
        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)
        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        ret = TPESampler._compare(samples_below, log_likelihoods_below, log_likelihoods_above)

        return param_distribution.to_external_repr(ret[param_name])

    @classmethod
    def _compare(
        cls,
        samples: Dict[str, np.ndarray],
        log_l: np.ndarray,
        log_g: np.ndarray,
    ) -> Dict[str, Union[float, int]]:

        sample_size = next(iter(samples.values())).size
        if sample_size:
            score = log_l - log_g
            if sample_size != score.size:
                raise ValueError(
                    "The size of the 'samples' and that of the 'score' "
                    "should be same. "
                    "But (samples.size, score.size) = ({}, {})".format(sample_size, score.size)
                )
            best = np.argmax(score)
            return {k: v[best].item() for k, v in samples.items()}
        else:
            raise ValueError(
                "The size of 'samples' should be more than 0."
                "But samples.size = {}".format(sample_size)
            )

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
                    x = trial.suggest_float("x", -10, 10)
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

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:

        self._random_sampler.after_trial(study, trial, state, values)


def _calculate_nondomination_rank(loss_vals: np.ndarray) -> np.ndarray:
    vecs = loss_vals.copy()

    # Normalize values
    lb = vecs.min(axis=0, keepdims=True)
    ub = vecs.max(axis=0, keepdims=True)
    vecs = (vecs - lb) / (ub - lb)

    ranks = np.zeros(len(vecs))
    num_unranked = len(vecs)
    rank = 0
    while num_unranked > 0:
        extended = np.tile(vecs, (vecs.shape[0], 1, 1))
        counts = np.sum(
            np.logical_and(
                np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
                np.any(extended < np.swapaxes(extended, 0, 1), axis=2),
            ),
            axis=1,
        )
        vecs[counts == 0] = 1.1  # mark as ranked
        ranks[counts == 0] = rank
        rank += 1
        num_unranked -= np.sum(counts == 0)
    return ranks


def _get_observation_pairs(
    study: Study,
    param_names: List[str],
    multivariate: bool,
    constant_liar: bool = False,  # TODO(hvy): Remove default value and fix unit tests.
) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, List[float]]]]:
    """Get observation pairs from the study.

    This function collects observation pairs from the complete or pruned trials of the study.
    In addition, if ``constant_liar`` is :obj:`True`, the running trials are considered.
    The values for trials that don't contain the parameter in the ``param_names`` are skipped.

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

    if len(param_names) > 1:
        assert multivariate

    signs = []
    for d in study.directions:
        if d == StudyDirection.MINIMIZE:
            signs.append(1)
        else:
            signs.append(-1)

    states: Tuple[TrialState, ...]
    if constant_liar:
        states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
    else:
        states = (TrialState.COMPLETE, TrialState.PRUNED)

    scores = []
    values: Dict[str, List[Optional[float]]] = {param_name: [] for param_name in param_names}
    for trial in study.get_trials(deepcopy=False, states=states):
        # If ``multivariate`` = True and ``group`` = True, we ignore the trials that are not
        # included in each subspace.
        # If ``multivariate`` = False, we skip the check.
        if multivariate and any([param_name not in trial.params for param_name in param_names]):
            continue

        # We extract score from the trial.
        if trial.state is TrialState.COMPLETE:
            if trial.values is None:
                continue
            score = (-float("inf"), [sign * v for sign, v in zip(signs, trial.values)])
        elif trial.state is TrialState.PRUNED:
            if study._is_multi_objective():
                continue

            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, [float("inf")])
                else:
                    score = (-step, [signs[0] * intermediate_value])
            else:
                score = (float("inf"), [0.0])
        elif trial.state is TrialState.RUNNING:
            if study._is_multi_objective():
                continue

            assert constant_liar
            score = (-float("inf"), [signs[0] * float("inf")])
        else:
            assert False
        scores.append(score)

        # We extract param_value from the trial.
        for param_name in param_names:
            raw_param_value = trial.params.get(param_name, None)
            param_value: Optional[float]
            if raw_param_value is not None:
                distribution = trial.distributions[param_name]
                param_value = distribution.to_internal_repr(trial.params[param_name])
            else:
                param_value = None
            values[param_name].append(param_value)

    return values, scores


def _split_observation_pairs(
    loss_vals: List[Tuple[float, List[float]]],
    n_below: int,
) -> Tuple[np.ndarray, np.ndarray]:

    n_objectives = 1
    if len(loss_vals) > 0:
        n_objectives = len(loss_vals[0][1])

    if n_objectives <= 1:
        loss_values = np.asarray(
            [(s, v[0]) for s, v in loss_vals], dtype=[("step", float), ("score", float)]
        )

        index_loss_ascending = np.argsort(loss_values)
        # `np.sort` is used to keep chronological order.
        indices_below = np.sort(index_loss_ascending[:n_below])
        indices_above = np.sort(index_loss_ascending[n_below:])
    else:
        # Multi-objective TPE does not support pruning, so it ignores the ``step``.
        lvals = np.asarray([v for _, v in loss_vals])

        # Solving HSSP for variables number of times is a waste of time.
        nondomination_ranks = _calculate_nondomination_rank(lvals)
        assert 0 <= n_below <= len(lvals)

        indices = np.array(range(len(lvals)))
        indices_below = np.empty(n_below, dtype=int)

        # Nondomination rank-based selection
        i = 0
        last_idx = 0
        while last_idx + sum(nondomination_ranks == i) <= n_below:
            length = indices[nondomination_ranks == i].shape[0]
            indices_below[last_idx : last_idx + length] = indices[nondomination_ranks == i]
            last_idx += length
            i += 1

        # Hypervolume subset selection problem (HSSP)-based selection
        subset_size = n_below - last_idx
        if subset_size > 0:
            rank_i_lvals = lvals[nondomination_ranks == i]
            rank_i_indices = indices[nondomination_ranks == i]
            worst_point = np.max(rank_i_lvals, axis=0)
            reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
            reference_point[reference_point == 0] = EPS
            selected_indices = _solve_hssp(
                rank_i_lvals, rank_i_indices, subset_size, reference_point
            )
            indices_below[last_idx:] = selected_indices

        indices_above = np.setdiff1d(indices, indices_below)

    return indices_below, indices_above


def _build_observation_dict(
    config_values: Dict[str, np.ndarray], indices: np.ndarray
) -> Dict[str, np.ndarray]:

    observation_dict = {}
    for param_name, param_val in config_values.items():
        param_values = param_val[indices]
        observation_dict[param_name] = param_values[~np.isnan(param_values)]

    return observation_dict


def _compute_hypervolume(solution_set: np.ndarray, reference_point: np.ndarray) -> float:
    return WFG().compute(solution_set, reference_point)


def _solve_hssp(
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
        _compute_hypervolume(np.asarray([v]), reference_point) for v in rank_i_loss_vals
    ]
    hv_selected = 0.0
    while len(selected_indices) < subset_size:
        max_index = int(np.argmax(contributions))
        contributions[max_index] = -1  # mark as selected
        selected_index = rank_i_indices[max_index]
        selected_vec = rank_i_loss_vals[max_index]
        for j, v in enumerate(rank_i_loss_vals):
            if contributions[j] == -1:
                continue
            p = np.max([selected_vec, v], axis=0)
            contributions[j] -= (
                _compute_hypervolume(np.asarray(selected_vecs + [p]), reference_point)
                - hv_selected
            )
        selected_vecs += [selected_vec]
        selected_indices += [selected_index]
        hv_selected = _compute_hypervolume(np.asarray(selected_vecs), reference_point)

    return np.asarray(selected_indices, dtype=int)


def _calculate_weights_below_for_multi_objective(
    config_values: Dict[str, np.ndarray],
    loss_vals: List[Tuple[float, List[float]]],
    indices: np.ndarray,
) -> np.ndarray:
    # Multi-objective TPE only sees the first parameter to determine the weights.
    # In the call lf `sample_relative`, this logic makes sense because we only have the
    # intersection search space or group decomposed search space. This means one parameter
    # misses the one trial, then the other parameter must miss the trial, in this call of
    # `sample_relative`.
    # In the call of `sample_independent`, we only have one parameter so the logic makes sense.
    cvals = list(config_values.values())[0][indices]

    # Multi-objective TPE does not support pruning, so it ignores the ``step``.
    lvals = np.asarray([v for _, v in loss_vals])[indices]

    # Calculate weights based on hypervolume contributions.
    n_below = len(lvals)
    if n_below == 0:
        weights_below = np.asarray([])
    elif n_below == 1:
        weights_below = np.asarray([1.0])
    else:
        worst_point = np.max(lvals, axis=0)
        reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
        reference_point[reference_point == 0] = EPS
        hv = _compute_hypervolume(lvals, reference_point)
        indices = ~np.eye(n_below).astype(bool)
        contributions = np.asarray(
            [hv - _compute_hypervolume(lvals[indices[i]], reference_point) for i in range(n_below)]
        )
        contributions += EPS
        weights_below = np.clip(contributions / np.max(contributions), 0, 1)

    weights_below = weights_below[~np.isnan(cvals)]
    return weights_below
