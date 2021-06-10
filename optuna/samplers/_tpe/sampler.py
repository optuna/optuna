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

from optuna import distributions
from optuna._study_direction import StudyDirection
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
                x = trial.suggest_float("x", -10, 10)
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

        n_complete_trials = len(study.get_trials(deepcopy=False))
        search_space: Dict[str, BaseDistribution] = {}

        if self._group:
            assert self._group_decomposed_search_space is not None
            self._search_space_group = self._group_decomposed_search_space.calculate(study)
            for sub_space in self._search_space_group.search_spaces:
                for name, distribution in sub_space.items():
                    if not isinstance(distribution, _DISTRIBUTION_CLASSES):
                        self._log_independent_sampling(n_complete_trials, trial, name)
                        continue
                    search_space[name] = distribution
            return search_space

        for name, distribution in self._search_space.calculate(study).items():
            if not isinstance(distribution, _DISTRIBUTION_CLASSES):
                self._log_independent_sampling(n_complete_trials, trial, name)
                continue
            search_space[name] = distribution

        return search_space

    def _log_independent_sampling(
        self, n_complete_trials: int, trial: FrozenTrial, param_name: str
    ) -> None:
        if self._warn_independent_sampling:
            if n_complete_trials >= self._n_startup_trials:
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

        self._raise_error_if_multi_objective(study)

        if self._group:
            assert self._search_space_group is not None
            params = {}
            for sub_space in self._search_space_group.search_spaces:
                search_space = {}
                for name, distribution in sub_space.items():
                    if isinstance(distribution, _DISTRIBUTION_CLASSES):
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
        below, above = self._split_observation_pairs(values, scores)
        # We then sample by maximizing log likelihood ratio.
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

        self._raise_error_if_multi_objective(study)

        values, scores = _get_observation_pairs(
            study, [param_name], self._multivariate, self._constant_liar
        )

        n = len(scores)

        if n < self._n_startup_trials:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        below, above = self._split_observation_pairs(values, scores)
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

    def _split_observation_pairs(
        self, config_vals: Dict[str, List[Optional[float]]], loss_vals: List[Tuple[float, float]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        # `None` items are intentionally converted to `nan` and then filtered out.
        # For `nan` conversion, the dtype must be float.
        config_values = {k: np.asarray(v, dtype=float) for k, v in config_vals.items()}
        loss_values = np.asarray(loss_vals, dtype=[("step", float), ("score", float)])

        n_below = self._gamma(len(loss_values))
        index_loss_ascending = np.argsort(loss_values)
        # `np.sort` is used to keep chronological order.
        index_below = np.sort(index_loss_ascending[:n_below])
        index_above = np.sort(index_loss_ascending[n_below:])
        below = {}
        above = {}
        for param_name, param_val in config_values.items():
            param_val_below = param_val[index_below]
            param_val_above = param_val[index_above]
            below[param_name] = param_val_below[~np.isnan(param_val_below)]
            above[param_name] = param_val_above[~np.isnan(param_val_above)]

        return below, above

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

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:

        self._random_sampler.after_trial(study, trial, state, values)


def _get_observation_pairs(
    study: Study,
    param_names: List[str],
    multivariate: bool,
    constant_liar: bool = False,  # TODO(hvy): Remove default value and fix unit tests.
) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, float]]]:
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

    sign = 1
    if study.direction == StudyDirection.MAXIMIZE:
        sign = -1

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
        elif trial.state is TrialState.RUNNING:
            assert constant_liar
            score = (-float("inf"), sign * float("inf"))
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
