import math
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

import numpy as np

from optuna._hypervolume import WFG
from optuna._hypervolume.hssp import _solve_hssp
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
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
    - `Multiobjective Tree-Structured Parzen Estimator <https://doi.org/10.1613/jair.1.13188>`_

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
            either :class:`~optuna.distributions.FloatDistribution`,
            or :class:`~optuna.distributions.IntDistribution`.
        prior_weight:
            The weight of the prior. This argument is used in
            :class:`~optuna.distributions.FloatDistribution`,
            :class:`~optuna.distributions.IntDistribution`, and
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
            If ``group`` is :obj:`True`, ``multivariate`` must be :obj:`True` as well.

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
                ``RUNNING`` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                ``FAIL``.

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.

            .. note::
                This feature can be used for only single-objective optimization; this argument is
                ignored for multi-objective optimization.

            .. note::
                Added in v2.8.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.8.0.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.

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
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
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
        self._constraints_func = constraints_func

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

        if constraints_func is not None:
            warnings.warn(
                "The ``constraints_func`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

    def reseed_rng(self) -> None:

        self._rng.seed()
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
        values, scores, violations = _get_observation_pairs(
            study,
            param_names,
            self._constant_liar,
            self._constraints_func is not None,
        )

        # If the number of samples is insufficient, we run random trial.
        n = sum(s < float("inf") for s, v in scores)  # Ignore running trials.
        if n < self._n_startup_trials:
            return {}

        # We divide data into below and above.
        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n), violations)
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        # `None` items appear only when `group=True`. We just use the first parameter because the
        # masks are the same for all parameters in one group.
        config_values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        param_mask = ~np.isnan(list(config_values.values())[0])
        param_mask_below, param_mask_above = param_mask[indices_below], param_mask[indices_above]
        below = {k: v[indices_below[param_mask_below]] for k, v in config_values.items()}
        above = {k: v[indices_above[param_mask_above]] for k, v in config_values.items()}

        # We then sample by maximizing log likelihood ratio.
        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                scores, indices_below, violations
            )[param_mask_below]
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

        values, scores, violations = _get_observation_pairs(
            study,
            [param_name],
            self._constant_liar,
            self._constraints_func is not None,
        )

        n = sum(s < float("inf") for s, v in scores)  # Ignore running trials.

        # Avoid independent warning at the first sampling of `param_name` when `group=True`.
        if any(param is not None for param in values[param_name]):
            self._log_independent_sampling(n, trial, param_name)

        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        indices_below, indices_above = _split_observation_pairs(scores, self._gamma(n), violations)
        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_value = np.asarray(values[param_name], dtype=float)
        param_mask = ~np.isnan(config_value)
        param_mask_below, param_mask_above = param_mask[indices_below], param_mask[indices_above]
        below = {param_name: config_value[indices_below[param_mask_below]]}
        above = {param_name: config_value[indices_above[param_mask_above]]}

        if study._is_multi_objective():
            weights_below = _calculate_weights_below_for_multi_objective(
                scores, indices_below, violations
            )[param_mask_below]
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
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._random_sampler.after_trial(study, trial, state, values)


def _calculate_nondomination_rank(loss_vals: np.ndarray) -> np.ndarray:
    ranks = np.full(len(loss_vals), -1)
    num_unranked = len(loss_vals)
    rank = 0
    domination_mat = np.all(loss_vals[:, None, :] >= loss_vals[None, :, :], axis=2) & np.any(
        loss_vals[:, None, :] > loss_vals[None, :, :], axis=2
    )
    while num_unranked > 0:
        counts = np.sum((ranks == -1)[None, :] & domination_mat, axis=1)
        num_unranked -= np.sum((counts == 0) & (ranks == -1))
        ranks[(counts == 0) & (ranks == -1)] = rank
        rank += 1
    return ranks


def _get_observation_pairs(
    study: Study,
    param_names: List[str],
    constant_liar: bool = False,  # TODO(hvy): Remove default value and fix unit tests.
    constraints_enabled: bool = False,
) -> Tuple[
    Dict[str, List[Optional[float]]],
    List[Tuple[float, List[float]]],
    Optional[List[float]],
]:
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

    When ``constraints_enabled`` is :obj:`True`, 1-dimensional violation values are returned
    as the third element (:obj:`None` otherwise). Each value is a float of 0 or greater and a
    trial is feasible if and only if its violation score is 0.
    """

    signs = []
    for d in study.directions:
        if d == StudyDirection.MINIMIZE:
            signs.append(1)
        else:
            signs.append(-1)

    states: Container[TrialState]
    if constant_liar:
        states = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.RUNNING)
    else:
        states = (TrialState.COMPLETE, TrialState.PRUNED)

    scores = []
    values: Dict[str, List[Optional[float]]] = {param_name: [] for param_name in param_names}
    violations: Optional[List[float]] = [] if constraints_enabled else None
    for trial in study.get_trials(deepcopy=False, states=states):
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
                score = (1, [0.0])
        elif trial.state is TrialState.RUNNING:
            if study._is_multi_objective():
                continue

            assert constant_liar
            score = (float("inf"), [signs[0] * float("inf")])
        else:
            assert False
        scores.append(score)

        # We extract param_value from the trial.
        for param_name in param_names:
            param_value: Optional[float]
            if param_name in trial.params:
                distribution = trial.distributions[param_name]
                param_value = distribution.to_internal_repr(trial.params[param_name])
            else:
                param_value = None
            values[param_name].append(param_value)

        if constraints_enabled:
            assert violations is not None
            constraint = trial.system_attrs.get(_CONSTRAINTS_KEY)
            if constraint is None:
                warnings.warn(
                    f"Trial {trial.number} does not have constraint values."
                    " It will be treated as a lower priority than other trials."
                )
                violation = float("inf")
            else:
                # Violation values of infeasible dimensions are summed up.
                violation = sum(v for v in constraint if v > 0)
            violations.append(violation)

    return values, scores, violations


def _split_observation_pairs(
    loss_vals: List[Tuple[float, List[float]]],
    n_below: int,
    violations: Optional[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    # When constrains is not None, trials are split into below and above
    # according to the following rules.
    # 1. Feasible trials are better than infeasible trials.
    # 2. Infeasible trials are sorted by sum of how much they violate each constraint.
    # 3. Feasible trials are sorted by loss_vals.
    if violations is not None:
        violation_1d = np.array(violations, dtype=float)
        idx = violation_1d.argsort(kind="stable")
        if n_below >= len(idx) or violation_1d[idx[n_below]] > 0:
            # Below is filled by all feasible trials and trials with smaller violation values.
            indices_below = idx[:n_below]
            indices_above = idx[n_below:]
        else:
            # All trials in below are feasible.
            # Feasible trials with smaller loss_vals are selected.
            (feasible_idx,) = (violation_1d == 0).nonzero()
            (infeasible_idx,) = (violation_1d > 0).nonzero()
            assert len(feasible_idx) >= n_below
            feasible_below, feasible_above = _split_observation_pairs(
                [loss_vals[i] for i in feasible_idx], n_below, None
            )
            indices_below = feasible_idx[feasible_below]
            indices_above = np.concatenate([feasible_idx[feasible_above], infeasible_idx])
        # `np.sort` is used to keep chronological order.
        return np.sort(indices_below), np.sort(indices_above)

    n_objectives = 1
    if len(loss_vals) > 0:
        n_objectives = len(loss_vals[0][1])

    if n_objectives <= 1:
        loss_values = np.asarray(
            [(s, v[0]) for s, v in loss_vals], dtype=[("step", float), ("score", float)]
        )

        index_loss_ascending = np.argsort(loss_values, kind="stable")
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
        while last_idx < n_below and last_idx + sum(nondomination_ranks == i) <= n_below:
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


def _calculate_weights_below_for_multi_objective(
    loss_vals: List[Tuple[float, List[float]]],
    indices: np.ndarray,
    violations: Optional[List[float]],
) -> np.ndarray:
    if violations is None:
        feasible_mask = np.ones(len(indices), dtype=bool)
    else:
        # Hypervolume contributions are calculated only using feasible trials.
        feasible_mask = np.array(violations, dtype=float)[indices] == 0

    # Multi-objective TPE does not support pruning, so it ignores the ``step``.
    lvals = np.asarray([v for _, v in loss_vals])[indices[feasible_mask]]

    # Calculate weights based on hypervolume contributions.
    n_below = len(lvals)
    weights_below: np.ndarray
    if n_below == 0:
        weights_below = np.asarray([])
    elif n_below == 1:
        weights_below = np.asarray([1.0])
    else:
        worst_point = np.max(lvals, axis=0)
        reference_point = np.maximum(1.1 * worst_point, 0.9 * worst_point)
        reference_point[reference_point == 0] = EPS
        hv = WFG().compute(lvals, reference_point)
        indices_mat = ~np.eye(n_below).astype(bool)
        contributions = np.asarray(
            [hv - WFG().compute(lvals[indices_mat[i]], reference_point) for i in range(n_below)]
        )
        contributions += EPS
        weights_below = np.clip(contributions / np.max(contributions), 0, 1)

    # For now, EPS weight is assigned to infeasible trials.
    weights_below_all = np.full(len(indices), EPS)
    weights_below_all[feasible_mask] = weights_below
    return weights_below_all
