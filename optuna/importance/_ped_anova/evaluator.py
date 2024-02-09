from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova.scott_parzen_estimator import _build_parzen_estimator
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


class QuantileFilter:
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        min_n_top_trials: int,
        target: Callable[[FrozenTrial], float] | None,
    ):
        if not (0 <= quantile <= 1):  # nan must also be rejected.
            raise ValueError(f"quantile must be in [0, 1], but got {quantile}.")
        if min_n_top_trials <= 0:
            raise ValueError(f"min_n_top_trials must be positive, but got {min_n_top_trials}.")

        self._quantile = quantile
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials
        self._target = target

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        target, min_n_top_trials = self._target, self._min_n_top_trials
        if target is None and max([len(t.values) for t in trials], default=1) > 1:
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

        sign = 1.0 if self._is_lower_better else -1.0
        loss_values = sign * np.asarray([t.value if target is None else target(t) for t in trials])
        if min_n_top_trials > loss_values.size:
            raise ValueError(
                f"len(trials) must be larger than or equal to min_n_top_trials={min_n_top_trials}"
                f", but got len(trials)={len(trials)}."
            )

        def _quantile(v: np.ndarray, q: float) -> float:
            cutoff_index = int(np.ceil(q * loss_values.size)) - 1
            return float(np.partition(loss_values, cutoff_index)[cutoff_index])

        cutoff_val = max(
            np.partition(loss_values, min_n_top_trials - 1)[min_n_top_trials - 1],
            # TODO(nabenabe0928): After dropping Python3.7, replace below with
            # np.quantile(loss_values, self._quantile, method="inverted_cdf")
            _quantile(loss_values, self._quantile),
        )
        should_keep_trials = loss_values <= cutoff_val
        return [t for t, should_keep in zip(trials, should_keep_trials) if should_keep]


class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm in
    `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`_.

    PED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials better
    than a user-specified baseline. Users can specify the baseline either by a quantile.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than baseline.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`_

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above baseline.
        To stabilize the analysis, it is preferable to include at least 5 trials above baseline.

    .. note::

        Please refer to the original work available at https://github.com/nabenabe0928/local-anova.

    Args:
        baseline_quantile:
            Compute the importance of achieving top-`baseline_quantile` quantile `target_value`.
            For example, `baseline_quantile=0.1` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            the specified `study`.
        evaluate_on_local:
            Whether we measure the importance in the local or global space.
            If `True`, the importances imply how importance each parameter is during `study`.
            Meanwhile, `evaluate_on_local=False` gives the importances in the specified
            `search_space`. `evaluate_on_local=True` is especially useful when users modify search
            space during the specified `study`.
        custom_filter:
            If given, the importance of each parameter to achieve filtered trials will be computed.
            If not specified, we compute the importance of achieving `baseline_quantile` in trials.

            Example:
                The following custom_filter allows calculating the importance of achieving
                the value of lower than 1.0.

                .. testcode::

                    import optuna
                    from optuna.importance import PedAnovaImportanceEvaluator


                    def objective(trial):
                        x1 = trial.suggest_float("x1", -10, 10)
                        x2 = trial.suggest_float("x2", -10, 10)
                        return x1 + x2 / 1000


                    study = optuna.create_study()
                    study.optimize(objective, n_trials=100)
                    custom_filter = lambda trials: [t for t in trials if t.value < 1.0]
                    evaluator = PedAnovaImportanceEvaluator(custom_filter=custom_filter)
                    evaluator.evaluate(study)

    """

    def __init__(
        self,
        *,
        baseline_quantile: float = 0.1,
        evaluate_on_local: bool = True,
        custom_filter: Callable[[list[FrozenTrial]], list[FrozenTrial]] | None = None,
    ):
        self._baseline_quantile = baseline_quantile
        self._evaluate_on_local = evaluate_on_local
        self._custom_filter = custom_filter

        # Advanced Setups.
        # Discretize a domain [low, high] as `np.linspace(low, high, n_steps)`.
        self._n_steps: int = 50
        # Prior is used for regularization.
        self._consider_prior = True
        # Control the regularization effect.
        self._prior_weight = 1.0
        # How many `trials` must be included in `top_trials`.
        self._min_n_top_trials = 2

    def _get_top_trials(
        self,
        study: Study,
        trials: list[FrozenTrial],
        params: list[str],
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        is_lower_better = study.directions[0] == StudyDirection.MINIMIZE
        if target is not None:
            warnings.warn(
                f"{self.__class__.__name__} computes the importances of params to achieve "
                "low `target` values. If this is not what you want, "
                "please modify target, e.g., by multiplying the output by -1."
            )
            is_lower_better = True

        if self._custom_filter is not None:
            top_trials = self._custom_filter(trials)
        else:
            top_trials = QuantileFilter(
                self._baseline_quantile, is_lower_better, self._min_n_top_trials, target
            ).filter(trials)

        if len(trials) == len(top_trials):
            warnings.warn("All trials are in top trials, which gives equal importances.")
        elif len(top_trials) < self._min_n_top_trials:
            raise ValueError(
                f"custom_filter must give at least {self._min_n_top_trials} trials "
                f"after the filtering, but got {len(top_trials)} trials."
            )

        return top_trials

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        top_trials: list[FrozenTrial],
        all_trials: list[FrozenTrial],
    ) -> float:
        consider_prior, prior_weight = self._consider_prior, self._prior_weight
        pe_top = _build_parzen_estimator(
            param_name, dist, top_trials, self._n_steps, consider_prior, prior_weight
        )
        grids = np.arange(pe_top.n_steps)
        pdf_top = pe_top.pdf(grids) + 1e-12

        if self._evaluate_on_local:  # The importance of param during the study.
            pe_local = _build_parzen_estimator(
                param_name, dist, all_trials, self._n_steps, consider_prior, prior_weight
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
        distributions = _get_distributions(study, params=params)
        if params is None:
            params = list(distributions.keys())

        assert params is not None
        # PED-ANOVA does not support parameter distributions with a single value,
        # because the importance of such params become zero.
        non_single_distributions = {
            name: dist for name, dist in distributions.items() if not dist.single()
        }
        single_distributions = {
            name: dist for name, dist in distributions.items() if dist.single()
        }
        if len(non_single_distributions) == 0:
            return {}

        trials = _get_filtered_trials(study, params=params, target=target)
        top_trials = self._get_top_trials(study, trials, params, target)
        importance_sum = 0.0
        param_importances = {}
        for param_name, dist in non_single_distributions.items():
            param_importances[param_name] = self._compute_pearson_divergence(
                param_name, dist, top_trials=top_trials, all_trials=trials
            )
            importance_sum += param_importances[param_name]

        if importance_sum > 0.0:
            param_importances = {k: v / importance_sum for k, v in param_importances.items()}
        else:
            assert len(trials) == len(top_trials), "Unexpected Error."
            n_params = len(param_importances)
            param_importances = {k: 1.0 / n_params for k in param_importances}

        param_importances.update({k: 0.0 for k in single_distributions})
        return _sort_dict_by_importance(param_importances)
