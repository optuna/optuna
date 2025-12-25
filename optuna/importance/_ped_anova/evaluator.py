from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._experimental import experimental_class
from optuna._warnings import optuna_warn
from optuna.distributions import BaseDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova.scott_parzen_estimator import _build_parzen_estimator
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


class _QuantileFilter:
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        min_n_top_trials: int,
        target: Callable[[FrozenTrial], float] | None,
    ) -> None:
        assert 0 < quantile <= 1, "quantile must be in (0, 1]."
        assert min_n_top_trials > 0, "min_n_top_trials must be positive."

        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials
        self._target = target

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        target, min_n_top_trials = self._target, self._min_n_top_trials
        sign = 1.0 if self._is_lower_better else -1.0
        loss_values = sign * np.asarray([t.value if target is None else target(t) for t in trials])
        err_msg = "len(trials) must be larger than or equal to min_n_top_trials"
        assert min_n_top_trials <= loss_values.size, err_msg

        def _quantile(v: np.ndarray, q: float) -> float:
            cutoff_index = int(np.ceil(q * loss_values.size)) - 1
            return float(np.partition(loss_values, cutoff_index)[cutoff_index])

        cutoff_val = max(
            np.partition(loss_values, min_n_top_trials - 1)[min_n_top_trials - 1],
            # TODO(nabenabe0928): After dropping Python3.10, replace below with
            # np.quantile(loss_values, self._quantile, method="inverted_cdf").
            _quantile(loss_values, self._quantile),
        )
        should_keep_trials = loss_values <= cutoff_val
        return [t for t, should_keep in zip(trials, should_keep_trials) if should_keep]


@experimental_class("3.6.0")
class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm.

    PED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials better
    than a user-specified `target_quantile`.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than `target_quantile`.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`__

    `target_quantile` and `region_quantile` correspond to the parameters ``gamma'`` and ``gamma``
    in the original paper, respectively.

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above
        `target_quantile`. To stabilize the analysis, it is preferable to include at least
        5 trials above `target_quantile`.

    .. note::

        Please refer to `the original work <https://github.com/nabenabe0928/local-anova>`__.

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
                change. `baseline_quantile` is currently ignored. Use `target_quantile` instead.
                See https://github.com/optuna/optuna/releases/tag/v4.7.0.

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

    def _get_top_quantile_trials(
        self,
        study: Study,
        trials: list[FrozenTrial],
        quantile: float,
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        is_lower_better = study.directions[0] == StudyDirection.MINIMIZE
        if target is not None:
            optuna_warn(
                f"{self.__class__.__name__} computes the importances of params to achieve "
                "low `target` values. If this is not what you want, "
                "please modify target, e.g., by multiplying the output by -1."
            )
            is_lower_better = True

        top_trials = _QuantileFilter(
            quantile, is_lower_better, self._min_n_top_trials, target
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
        pe_top = _build_parzen_estimator(
            param_name, dist, target_trials, self._n_steps, prior_weight
        )
        # NOTE: pe_top.n_steps could be different from self._n_steps.
        grids = np.arange(pe_top.n_steps)
        pdf_top = pe_top.pdf(grids) + 1e-12

        if self._evaluate_on_local:  # The importance of param during the study.
            pe_local = _build_parzen_estimator(
                param_name, dist, region_trials, self._n_steps, prior_weight
            )
            pdf_local = pe_local.pdf(grids) + 1e-12
        else:  # The importance of param in the search space.
            pdf_local = np.full(pe_top.n_steps, 1.0 / pe_top.n_steps)

        return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        dists = _get_distributions(study, params=params)
        if params is None:
            params = list(dists.keys())

        assert params is not None
        # PED-ANOVA does not support parameter distributions with a single value,
        # because the importance of such params become zero.
        non_single_dists = {name: dist for name, dist in dists.items() if not dist.single()}
        single_dists = {name: dist for name, dist in dists.items() if dist.single()}
        if len(non_single_dists) == 0:
            return {}

        trials = _get_filtered_trials(study, params=params, target=target)
        # The following should be tested at _get_filtered_trials.
        assert target is not None or max([len(t.values) for t in trials], default=1) == 1
        if len(trials) <= self._min_n_top_trials:
            return {k: 0.0 for k in dists}

        target_trials = self._get_top_quantile_trials(study, trials, self._target_quantile, target)
        region_trials = (
            trials
            if self._region_quantile == 1.0
            else self._get_top_quantile_trials(study, trials, self._region_quantile, target)
        )
        if len(target_trials) == len(region_trials):
            optuna_warn(
                "Target and region quantiles select the same set of trials. "
                "Parameter importances will be equal."
            )
        quantile = len(target_trials) / len(region_trials)
        param_importances = {}
        for param_name, dist in non_single_dists.items():
            param_importances[param_name] = quantile**2 * self._compute_pearson_divergence(
                param_name, dist, target_trials=target_trials, region_trials=region_trials
            )

        param_importances.update({k: 0.0 for k in single_dists})
        return _sort_dict_by_importance(param_importances)
