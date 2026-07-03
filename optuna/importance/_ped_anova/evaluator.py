from __future__ import annotations

from collections import defaultdict
import math
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._experimental import experimental_class
from optuna._warnings import optuna_warn
from optuna.importance._base import _check_evaluate_args
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova.scott_parzen_estimator import build_parzen_estimator_on_grid
from optuna.samplers._tpe.sampler import _split_complete_trials_multi_objective
from optuna.study import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class _QuantileFilter:
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        target: Callable[[FrozenTrial], float] | None,
    ) -> None:
        assert 0 < quantile <= 1, "quantile must be in (0, 1]."
        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._target = target

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        sign = 1.0 if self._is_lower_better else -1.0
        loss_values = sign * np.asarray([t.value if self._target is None else self._target(t) for t in trials])
        # TODO(nabenabe0928): After dropping Python3.10, replace below with
        # np.quantile(loss_values, self._quantile, method="inverted_cdf").
        cutoff_index = int(math.ceil(self._quantile * loss_values.size)) - 1
        cutoff_val = float(np.partition(loss_values, cutoff_index)[cutoff_index])
        should_keep_trials = loss_values <= cutoff_val
        return [t for t, should_keep in zip(trials, should_keep_trials) if should_keep]


@experimental_class("3.6.0")
class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm.

    PED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials better
    than a user-specified ``target_quantile``.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than ``target_quantile``.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`__ (IJCAI 2023)

    For further information on how conditional parameters are handled, please refer to the
    following paper:

    - `Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces
      <https://arxiv.org/abs/2601.20800>`__ (KDD 2026)

    ``target_quantile`` and ``region_quantile`` correspond to the parameters
    :math:`\\gamma'` and :math:`\\gamma` in the original paper, respectively.

    .. note::

        For multi-objective studies, if ``target`` is :obj:`None`, top-quantile trials are
        selected in the same manner as MOTPE, using non-domination ranks and the hypervolume
        subset selection problem (HSSP) for tie-breaking.

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above
        ``target_quantile``. To stabilize the analysis, it is preferable to include at least
        5 trials above ``target_quantile``.

    .. note::

        Please also refer to the original implementations:

        - `PED-ANOVA <https://github.com/nabenabe0928/local-anova>`__
        - `condPED-ANOVA <https://github.com/kAIto47802/condPED-ANOVA>`__

    Args:
        target_quantile:
            Compute the importance of achieving top-``target_quantile`` quantile objective value.
            For example, ``target_quantile=0.1`` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

        region_quantile:
            Define the region where we compute the importance. For example,
            ``region_quantile=0.5`` means that we compute the importance in the region where
            trials achieve top-50% performance. If ``region_quantile=1.0``, the importance is
            computed in the whole search space.

        baseline_quantile:
            Compute the importance of achieving top-``baseline_quantile`` quantile objective value.
            For example, ``baseline_quantile=0.1`` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            optimization.

            .. warning::
                Deprecated in v4.7.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v5.0.0, but this schedule is subject to
                change. ``baseline_quantile`` is currently ignored. Use ``target_quantile``
                instead. See https://github.com/optuna/optuna/releases/tag/v4.7.0.

        evaluate_on_local:
            Whether we measure the importance in the local or global space.
            If :obj:`True`, the importances imply how importance each parameter is during
            optimization. Meanwhile, ``evaluate_on_local=False`` gives the importances in the
            specified search_space. ``evaluate_on_local=True`` is especially useful when users
            modify search space during optimization.

    Example:
        An example of using PED-ANOVA is as follows:

        .. testcode::

            import optuna
            from optuna.importance import PedAnovaImportanceEvaluator


            def objective(trial):
                x1 = trial.suggest_float("x1", -10, 10)
                x2 = trial.suggest_float("x2", -10, 10)
                return x1 + x2 / 1000


            study = optuna.create_study()
            study.optimize(objective, n_trials=100)
            evaluator = PedAnovaImportanceEvaluator()
            importance = optuna.importance.get_param_importances(study, evaluator=evaluator)

    """

    def __init__(
        self,
        *,
        target_quantile: float = 0.1,  # gamma' in the original paper
        region_quantile: float = 1.0,  # gamma in the original paper
        baseline_quantile: float | None = None,
        evaluate_on_local: bool = True,
    ) -> None:
        assert 0.0 < target_quantile < region_quantile <= 1.0, (
            "condition 0.0 < `target_quantile` < `region_quantile` <= 1.0 must be satisfied"
        )
        if baseline_quantile is not None:
            msg = _DEPRECATION_WARNING_TEMPLATE.format(
                name="`baseline_quantile`", d_ver="4.7.0", r_ver="5.0.0"
            )
            optuna_warn(
                f"{msg} `baseline_quantile` is currently ignored. Use `target_quantile` instead.",
            )
        if region_quantile != 1.0 and not evaluate_on_local:
            optuna_warn("If `evaluate_on_local` is False, `region_quantile` has no effect.")

        self._target_quantile = target_quantile
        self._region_quantile = region_quantile
        self._evaluate_on_local = evaluate_on_local

        # Advanced Setups.
        # Discretize a domain [low, high] as `np.linspace(low, high, n_steps)`.
        self._n_steps: int = 50
        # Control the regularization effect by prior.
        self._prior_weight = 1.0
        # How many `trials` must be included in `top_trials`.
        self._min_n_top_trials = 2
        # How many `trials` must be included in each regime.
        self._min_n_trials_in_regime = 2

    def _get_top_quantile_trials(
        self,
        study: Study,
        trials: list[FrozenTrial],
        quantile: float,
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        if quantile == 1.0:
            return trials
        if study._is_multi_objective() and target is None:
            n_below = math.ceil(quantile * len(trials))
            # NOTE(kAIto47802): Since HSSP is implemented greedily, target trials could be
            # obtained by taking the top trials from region trials without solving HSSP again,
            # which would improve performance by a constant factor. However,
            # _split_complete_trials_multi_objective does not return trials in the selected
            # order, so this optimization would require a larger refactoring.
            top_trials, _ = _split_complete_trials_multi_objective(trials, study, n_below)
            return top_trials
        is_lower_better = study.directions[0] == StudyDirection.MINIMIZE
        if target is not None:
            optuna_warn(
                f"{self.__class__.__name__} computes the importances of params to achieve "
                "low `target` values. If this is not what you want, "
                "please modify target, e.g., by multiplying the output by -1."
            )
            is_lower_better = True

        top_trials = _QuantileFilter(
            quantile, is_lower_better, target
        ).filter(trials)

        return top_trials

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        target_trials: list[FrozenTrial],
        region_trials: list[FrozenTrial],
    ) -> float:
        # When pdf_all == pdf_top, i.e. all_trials == top_trials, this method will give 0.0.
        prior_weight = self._prior_weight
        pe_top, grid_size = build_parzen_estimator_on_grid(
            param_name, dist, target_trials, self._n_steps, prior_weight
        )
        grids = np.arange(grid_size)
        pdf_top = pe_top.pdf({param_name: grids}) + 1e-12

        if self._evaluate_on_local:  # The importance of param during the study.
            pe_local, _ = build_parzen_estimator_on_grid(
                param_name, dist, region_trials, self._n_steps, prior_weight
            )
            pdf_local = pe_local.pdf({param_name: grids}) + 1e-12
        else:  # The importance of param in the search space.
            pdf_local = np.full(grid_size, 1.0 / grid_size)

        return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        dists = _get_distributions_list(study, params=params)
        if params is None:
            params = list(dict.fromkeys(k for d in dists for k in d))

        assert params is not None

        trials = _get_filtered_trials(study, target=target)
        # The following should be tested at _get_filtered_trials.
        assert target is not None or max([len(t.values) for t in trials], default=1) == 1
        if len(trials) <= 1:
            return {k: 0.0 for k in params}

        target_trials = self._get_top_quantile_trials(study, trials, self._target_quantile, target)
        region_trials = self._get_top_quantile_trials(study, trials, self._region_quantile, target)
        if len(target_trials) == len(region_trials):
            optuna_warn(
                "Target and region quantiles select the same set of trials. "
                "Parameter importances will be equal."
            )
        if len(target_trials) == 0:
            return {k: 0.0 for k in params}
        target_trial_ids = set(t._trial_id for t in target_trials)
        region_trial_ids = set(t._trial_id for t in region_trials)
        # Since HSSP is approximately implemented using a greedy algorithm, target trials
        # are guaranteed to be included in region trials, even when target is None for
        # multi-objective studies.
        assert target_trial_ids.issubset(region_trial_ids)

        # Theorem 4.2 and Algorithm 1 in the original paper:
        # https://arxiv.org/abs/2601.20800
        quantile = len(target_trials) / len(region_trials)  # gamma' / gamma
        param_importances = {k: 0.0 for k in params}
        for param_name in params:
            regime_trials = _partition_by_regime(
                param_name, region_trials, self._min_n_trials_in_regime
            )
            for dist, region_trials_regime in regime_trials.items():
                target_trials_regime = [
                    t for t in region_trials_regime if t._trial_id in target_trial_ids
                ]
                target_prob_regime = len(target_trials_regime) / len(target_trials)  # alpha_i
                region_prob_regime = len(region_trials_regime) / len(region_trials)  # beta_i
                if dist is not None and not dist.single() and len(target_trials_regime):
                    param_importances[param_name] += (
                        target_prob_regime**2
                        / region_prob_regime
                        * self._compute_pearson_divergence(
                            param_name,
                            dist,
                            target_trials=target_trials_regime,
                            region_trials=region_trials_regime,
                        )
                    )
        param_importances = {k: v * quantile**2 for k, v in param_importances.items()}
        return _sort_dict_by_importance(param_importances)


def _partition_by_regime(
    param_name: str, trials: list[FrozenTrial], min_n_trials_in_regime: int
) -> dict[BaseDistribution | None, list[FrozenTrial]]:
    # None for the inactive regime
    regime_trials: dict[BaseDistribution | None, list[FrozenTrial]] = defaultdict(list)
    for trial in trials:
        regime_trials[trial.distributions.get(param_name)].append(trial)

    # NOTE(kAIto47802): We support the domain that takes one of several discrete values depending
    # on the condition. However, when the domain changes smoothly, some ranges need to be merged
    # into the same regime to stabilize the KDE within each regime.
    # TODO(kAIto47802): Implement this.
    if any(len(v) < min_n_trials_in_regime for v in regime_trials.values()):
        optuna_warn(
            f"Some regimes for parameter `{param_name}` have less than "
            f"{min_n_trials_in_regime} trials. "
            "The importance of the parameter may be inaccurate."
        )
    regime_trials = {k: v for k, v in regime_trials.items() if len(v) >= min_n_trials_in_regime}

    return regime_trials


def _get_filtered_trials(
    study: Study, target: Callable[[FrozenTrial], float] | None
) -> list[FrozenTrial]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return [
        trial
        for trial in trials
        if np.isfinite(target(trial) if target is not None else cast(float, trial.value))  # TC006
    ]


def _get_distributions_list(
    study: Study, params: list[str] | None
) -> list[dict[str, BaseDistribution]]:
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    _check_evaluate_args(trials, params)
    params_set = set(params) if params is not None else None
    return [
        {k: v for k, v in t.distributions.items() if params_set is None or k in params_set}
        for t in trials
    ]
