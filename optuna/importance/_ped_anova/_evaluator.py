from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._ped_anova._scott_parzen_estimator import _build_parzen_estimator
from optuna.importance.filters import get_trial_filter
from optuna.study import Study
from optuna.trial import FrozenTrial


class PedAnovaImportanceEvaluator(BaseImportanceEvaluator):
    """PED-ANOVA importance evaluator.

    Implements the PED-ANOVA hyperparameter importance evaluation algorithm in
    `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`_.

    PED-ANOVA fits Parzen estimators of :class:`~optuna.trial.TrialState.COMPLETE` trials better
    than a user-specified baseline. Users can specify the baseline either by a quantile or a value.
    The importance can be interpreted as how important each hyperparameter is to get
    the performance better than baseline.
    Users can also remove trials worse than `cutoff` so that the interpretation removes the bias
    caused by the initial trials.

    For further information about PED-ANOVA algorithm, please refer to the following paper:

    - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
      <https://arxiv.org/abs/2304.10255>`_

    .. note::

        The performance of PED-ANOVA depends on how many trials to consider above baseline.
        To stabilize the analysis, it is preferable to include at least 5 trials above baseline.

    .. note::

        Please refer to the original work available at https://github.com/nabenabe0928/local-anova.

    Args:
        is_lower_better:
            Whether `target_value` is better when it is lower.
        n_steps:
            The number of grids in continuous domains.
            For example, if one of the parameters has the domain of [`low`, `high`],
            we discretize it as `np.linspace(low, high, n_steps)`.
        baseline_quantile:
            Compute the importance of achieving top-`baseline_quantile` quantile `target_value`.
            For example, `baseline_quantile=0.1` means that the importances give the information
            of which parameters were important to achieve the top-10% performance during
            the specified `study`.
        min_n_top_trials:
            How many `trials` must be included in `top_trials`.
        consider_prior:
            Whether we use non-informative prior to regularize the Parzen estimators.
            This might be helpful to avoid overfitting.
        prior_weight:
            How much we regularize the Parzen estimator fitting.
            The larger `prior_weight` becomes, the more we regularize the fitting.
            All the observations receive `weight=1.0`, so the default value is `prior_weight=1.0`.
        categorical_distance_func:
            A dictionary of distance functions for categorical parameters. The key is the name of
            the categorical parameter and the value is a distance function that takes two
            :class:`~optuna.distributions.CategoricalChoiceType` s and returns a :obj:`float`
            value. The distance function must return a non-negative value.

            While categorical choices are handled equally by default, this option allows users to
            specify prior knowledge on the structure of categorical parameters.
        evaluate_on_local:
            Whether we measure the importance in the local or global space.
            If `True`, the importances imply how importance each parameter is during `study`.
            Meanwhile, `evaluate_on_local=False` gives the importances in the specified
            `search_space`. `evaluate_on_local=True` is especially useful when users modify search
            space during the specified `study`.
    """

    def __init__(
        self,
        is_lower_better: bool,
        *,
        n_steps: int = 50,
        baseline_quantile: float = 0.1,
        consider_prior: bool = False,
        prior_weight: float = 1.0,
        categorical_distance_func: dict[
            str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        ]
        | None = None,
        evaluate_on_local: bool = True,
        min_n_top_trials: int = 2,
    ):
        if n_steps <= 1:
            raise ValueError(f"`n_steps` must be larger than 1, but got {n_steps}.")

        if min_n_top_trials < 2:
            raise ValueError(
                f"min_n_top_trials must be larger than 1, but got {min_n_top_trials}."
            )

        self._n_steps = n_steps
        self._categorical_distance_func = (
            categorical_distance_func if categorical_distance_func is not None else {}
        )
        self._consider_prior = consider_prior
        self._prior_weight = prior_weight
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials
        self._baseline_quantile = baseline_quantile
        self._evaluate_on_local = evaluate_on_local

    def _get_top_trials(
        self,
        trials: list[FrozenTrial],
        params: list[str],
        target: Callable[[FrozenTrial], float] | None,
    ) -> list[FrozenTrial]:
        trial_filter = get_trial_filter(
            quantile=self._baseline_quantile,
            is_lower_better=self._is_lower_better,
            min_n_top_trials=self._min_n_top_trials,
            target=target,
        )
        top_trials = trial_filter(trials)

        if len(trials) == len(top_trials):
            warnings.warn(
                "All the trials were considered to be in top and it gives equal importances."
            )

        return top_trials

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        top_trials: list[FrozenTrial],
        all_trials: list[FrozenTrial],
    ) -> float:
        cat_dist_func = self._categorical_distance_func.get(param_name, None)
        pe_top = _build_parzen_estimator(
            param_name=param_name,
            dist=dist,
            trials=top_trials,
            n_steps=self._n_steps,
            consider_prior=self._consider_prior,
            prior_weight=self._prior_weight,
            categorical_distance_func=cat_dist_func,
        )
        n_grids = pe_top.n_grids
        grids = np.arange(n_grids)
        pdf_top = pe_top.pdf(grids) + 1e-12

        if self._evaluate_on_local:
            # Compute the integral on the local space.
            # It gives us the importances of hyperparameters during the search.
            pe_local = _build_parzen_estimator(
                param_name=param_name,
                dist=dist,
                trials=all_trials,
                n_steps=self._n_steps,
                consider_prior=self._consider_prior,
                prior_weight=self._prior_weight,
                categorical_distance_func=cat_dist_func,
            )
            pdf_local = pe_local.pdf(grids) + 1e-12
        else:
            # Compute the integral on the global space.
            # It gives us the importances of hyperparameters in the search space.
            pdf_local = np.full(n_grids, 1.0 / n_grids)

        return float(pdf_local @ ((pdf_top / pdf_local - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        if target is None and study._is_multi_objective():
            raise ValueError(
                "If the `study` is being used for multi-objective optimization, "
                "please specify the `target`. For example, use "
                "`target=lambda t: t.values[0]` for the first objective value."
            )

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
        top_trials = self._get_top_trials(trials, params, target)
        importance_sum = 0.0
        param_importances = {}
        for param_name, dist in non_single_distributions.items():
            param_importances[param_name] = self._compute_pearson_divergence(
                param_name,
                dist,
                top_trials=top_trials,
                all_trials=trials,
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
