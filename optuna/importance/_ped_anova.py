from __future__ import annotations

from collections import Callable
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_filtered_trials
from optuna.importance._base import _get_target_values
from optuna.importance._base import _sort_dict_by_importance
from optuna.importance._base import BaseImportanceEvaluator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial


class _EfficientParzenEstimator(_ParzenEstimator):
    """Fast implementation for 1D ParzenEstimator."""

    def __init__(
        self,
        param_name: str,
        dist: IntDistribution | CategoricalDistribution,
        counts: np.ndarray,
        categorical_distance_func: Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        | None,
    ):
        self._param_name = param_name
        self._search_space = {param_name: dist}
        self._counts = counts.copy()
        self._n_trials = np.sum(self._counts)
        self._n_grids = len(counts)
        self._categorical_distance_func = categorical_distance_func

        if isinstance(dist, CategoricalDistribution):
            distribution = self._calculate_categorical_distributions_efficient()
        elif isinstance(dist, IntDistribution):
            distribution = self._calculate_numerical_distributions_efficient()
        else:
            raise ValueError(
                f"Only IntDistribution and CategoricalDistribution are supported, but got {dist}."
            )

        self._mixture_distribution = _MixtureOfProductDistribution(
            weights=counts / self._n_trials,
            distributions=[distribution],
        )

    @property
    def n_grids(self) -> int:
        return self._n_grids

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf({self._param_name: samples}))

    def _calculate_categorical_distributions_efficient(self) -> _BatchedDistributions:
        distribution = self._search_space[self._param_name]
        assert isinstance(distribution, CategoricalDistribution), "Mypy redefinition."
        choices = distribution.choices
        n_choices = len(choices)
        if n_choices != self._counts.size:
            raise ValueError(
                f"The shape of counts must be n_choices={n_choices}, "
                f"but got {self._counts.size}."
            )

        dist_func = self._categorical_distance_func
        if dist_func is None:
            weights = np.identity(n_choices)
        else:
            used_indices = set([i for i, c in enumerate(self._counts) if c != 0])
            dists = np.array(
                [
                    # If indices are not used, their weights will not be used.
                    [dist_func(choices[i], c) if i in used_indices else 1.0 for c in choices]
                    for i in range(n_choices)
                ]
            )
            max_dists = np.max(dists, axis=1)
            coef = np.log(self._n_trials) * np.log(n_choices) / np.log(6)
            weights = np.exp(-((dists / max_dists[:, np.newaxis]) ** 2) * coef)
            weights /= np.sum(weights, axis=1, keepdims=True)

        return _BatchedCategoricalDistributions(weights=weights)

    def _calculate_numerical_distributions_efficient(self) -> _BatchedDistributions:
        n_trials = self._n_trials
        weights = self._counts / n_trials
        values = np.arange(self.n_grids)
        mean_est = values @ weights
        sigma_est = np.sqrt((values - mean_est) ** 2 @ self._counts / max(1, n_trials - 1))

        count_cum = np.cumsum(self._counts)
        idx_q25, idx_q75 = np.searchsorted(count_cum, [n_trials // 4, n_trials * 3 // 4])
        IQR = values[idx_q75] - values[idx_q25]

        # Scott's rule by Scott, D.W. (1992),
        # Multivariate Density Estimation: Theory, Practice, and Visualization.
        sigma_est = 1.059 * min(IQR / 1.34, sigma_est) * n_trials ** (-0.2)
        # To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        sigma_est = max(sigma_est, 0.5 / 1.64)
        return _BatchedDiscreteTruncNormDistributions(
            mu=values,
            sigma=np.full(self.n_grids, sigma_est),
            low=0,
            high=self.n_grids - 1,
            step=1,
        )


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
        direction:
            TODO.
        step:
            TODO.
        baseline_quantile:
            TODO.
        cutoff_quantile:
            TODO.
        baseline_value:
            TODO.
        cutoff_value:
            TODO.
        categorical_distance_func:
            TODO.

    """

    def __init__(
        self,
        direction: str,
        *,
        step: int = 50,
        baseline_quantile: float | None = None,
        cutoff_quantile: float | None = None,
        baseline_value: float | None = None,
        cutoff_value: float | None = None,
        categorical_distance_func: dict[
            str, Callable[[CategoricalChoiceType, CategoricalChoiceType], float]
        ]
        | None = None,
    ):
        if step <= 1:
            raise ValueError(f"`step` must be larger than 1, but got {step}.")
        direction_choices = ["minimize", "maximize"]
        if direction not in direction_choices:
            raise ValueError(f"`direction` must be in {direction_choices}, but got {direction}.")

        self._step = step
        self._categorical_distance_func = (
            categorical_distance_func if categorical_distance_func is not None else {}
        )
        self._minimize = direction == direction_choices[0]
        self._validate_input(baseline_quantile, cutoff_quantile, baseline_value, cutoff_value)
        if baseline_value is None and baseline_quantile is None:
            baseline_quantile = 0.1
            warnings.warn(
                "both baseline_value and baseline_quantile were not specified, "
                f"so baseline_quantile={baseline_quantile} will be used."
            )
        if cutoff_value is None and cutoff_quantile is None:
            cutoff_value = np.inf if self._minimize else -np.inf

        self._baseline_quantile = baseline_quantile
        self._cutoff_quantile = cutoff_quantile
        self._baseline_value = baseline_value
        self._cutoff_value = cutoff_value

    def _validate_input(
        self,
        baseline_quantile: float | None,
        cutoff_quantile: float | None,
        baseline_value: float | None,
        cutoff_value: float | None,
    ) -> None:
        suffix = "cannot be specified simultaneously."
        if baseline_quantile is not None and baseline_value is not None:
            raise ValueError(f"baseline_quantile and baseline_value {suffix}")
        if cutoff_quantile is not None and cutoff_value is not None:
            raise ValueError(f"cutoff_quantile and cutoff_value {suffix}")

        suffix += " Only (baseline_quantile, cutoff_quantile) or "
        suffix += "(baseline_value, cutoff_value) can be specified simultaneously."
        if baseline_quantile is not None and cutoff_value is not None:
            raise ValueError(f"baseline_quantile and cutoff_value {suffix}")
        if cutoff_quantile is not None and baseline_value is not None:
            raise ValueError(f"baseline_quantile and cutoff_value {suffix}")

        if (
            baseline_quantile is not None
            and cutoff_quantile is not None
            and baseline_quantile > cutoff_quantile
        ):
            raise ValueError(
                "baseline_quantile must be smaller than cutoff_quantile, but got "
                f"baseline_quantile={baseline_quantile} and cutoff_quantile={cutoff_quantile}."
            )
        if baseline_quantile is not None and not (0.0 <= baseline_quantile <= 1.0):
            raise ValueError(f"baseline_quantile must be in [0, 1], but got {baseline_quantile}")
        if cutoff_quantile is not None and not (0.0 <= cutoff_quantile <= 1.0):
            raise ValueError(f"cutoff_quantile must be in [0, 1], but got {cutoff_quantile}")

        if (
            baseline_value is not None
            and cutoff_value is not None
            and (
                (self._minimize and baseline_value > cutoff_value)
                or (not self._minimize and baseline_value < cutoff_value)
            )
        ):
            raise ValueError(
                "baseline_value must be better than cutoff_value, but got "
                f"baseline_value={baseline_value} and cutoff_value={cutoff_value}."
            )

    @staticmethod
    def _validate_and_get_params_and_distributions(
        study: Study,
        params: list[str] | None,
        target: Callable[[FrozenTrial], float] | None,
    ) -> tuple[list[str], dict[str, BaseDistribution]]:
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
        return params, distributions

    def _get_grids_and_grid_indices_of_trials(
        self,
        param_name: str,
        dist: IntDistribution | FloatDistribution,
        trials: list[FrozenTrial],
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(dist, FloatDistribution):
            if dist.log:
                grids = np.linspace(np.log(dist.low), np.log(dist.high), self._step)
                params = np.log([t.params[param_name] for t in trials])
            else:
                grids = np.linspace(dist.low, dist.high, self._step)
                params = np.asarray([t.params[param_name] for t in trials])
        elif isinstance(dist, IntDistribution):
            if dist.log:
                exponent_of_2 = int(np.ceil(np.log(dist.high - dist.low + 1) / np.log(2)))
                step = min(exponent_of_2, self._step)
                grids = np.linspace(np.log(dist.low), np.log(dist.high), step)
                params = np.log([t.params[param_name] for t in trials])
            else:
                grids = np.arange(dist.low, dist.high + 1)[:: dist.step]
                params = np.asarray([t.params[param_name] for t in trials])
        else:
            assert False, "Should not be reached."

        step_size = grids[1] - grids[0]
        # grids[indices[n] - 1] < param - step_size / 2 <= grids[indices[n]]
        indices = np.searchsorted(grids, params - step_size / 2)
        return grids, indices

    def _count_numerical_param_in_grid(
        self,
        param_name: str,
        dist: IntDistribution | FloatDistribution,
        trials: list[FrozenTrial],
    ) -> np.ndarray:
        grids, grid_indices_of_trials = self._get_grids_and_grid_indices_of_trials(
            param_name,
            dist,
            trials,
        )
        unique_vals, counts_in_unique = np.unique(grid_indices_of_trials, return_counts=True)
        counts = np.zeros(grids.size, dtype=np.int32)
        counts[unique_vals] += counts_in_unique
        return counts

    def _count_categorical_param_in_grid(
        self,
        param_name: str,
        dist: CategoricalDistribution,
        trials: list[FrozenTrial],
    ) -> np.ndarray:
        choice_to_index = {c: i for i, c in enumerate(dist.choices)}
        unique_vals, counts_in_unique = np.unique(
            [choice_to_index[t.params[param_name]] for t in trials],
            return_counts=True,
        )
        counts = np.zeros(len(dist.choices), dtype=np.int32)
        counts[unique_vals] += counts_in_unique
        return counts

    def _build_parzen_estimator(
        self,
        param_name: str,
        dist: BaseDistribution,
        trials: list[FrozenTrial],
    ) -> _EfficientParzenEstimator:
        rounded_dist: IntDistribution | CategoricalDistribution
        if isinstance(dist, (IntDistribution, FloatDistribution)):
            counts = self._count_numerical_param_in_grid(param_name, dist, trials)
            rounded_dist = IntDistribution(low=0, high=counts.size - 1)
        elif isinstance(dist, CategoricalDistribution):
            counts = self._count_categorical_param_in_grid(param_name, dist, trials)
            rounded_dist = dist
        else:
            raise ValueError(f"Got an unknown dist with the type {type(dist)}.")

        categorical_distance_func = self._categorical_distance_func.get(param_name, None)
        return _EfficientParzenEstimator(
            param_name,
            rounded_dist,
            counts,
            categorical_distance_func,
        )

    def _get_trials_better_than_cutoff_or_baseline(
        self,
        study: Study,
        params: list[str],
        target: Callable[[FrozenTrial], float] | None,
    ) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
        trials = _get_filtered_trials(study, params=params, target=target)
        target_values = _get_target_values(trials, target)

        if self._cutoff_value is not None:
            cutoff_value = self._cutoff_value
        else:
            assert self._cutoff_quantile is not None, "Mypy redefinition."
            cutoff_quantile = (
                self._cutoff_quantile if self._minimize else 1 - self._cutoff_quantile
            )
            cutoff_value = np.quantile(
                target_values,
                cutoff_quantile,
                method="higher" if self._minimize else "lower",
            )

        if self._baseline_value is not None:
            baseline_value = self._baseline_value
        else:
            assert self._baseline_quantile is not None, "Mypy redefinition."
            baseline_quantile = (
                self._baseline_quantile if self._minimize else 1 - self._baseline_quantile
            )
            baseline_value = np.quantile(
                target_values,
                baseline_quantile,
                method="higher" if self._minimize else "lower",
            )

        n_trials = len(trials)
        if self._minimize:
            indices_for_cutoff = np.arange(n_trials)[target_values <= cutoff_value]
            indices_for_baseline = np.arange(n_trials)[target_values <= baseline_value]
        else:
            indices_for_cutoff = np.arange(n_trials)[target_values >= cutoff_value]
            indices_for_baseline = np.arange(n_trials)[target_values >= baseline_value]

        if indices_for_baseline.size == 0:
            raise ValueError(
                f"baseline_quantile={self._baseline_quantile} and "
                f"baseline_value={self._baseline_value} are too tight."
            )
        if indices_for_baseline.size < 5:
            warnings.warn(
                f"The number of trials better than baseline_quantile={self._baseline_quantile} "
                f"and baseline_value={self._baseline_value} is less than 5 and the evaluation"
                " might be inaccurate. Please relax these values."
            )

        return [trials[i] for i in indices_for_cutoff], [trials[i] for i in indices_for_baseline]

    def _compute_pearson_divergence(
        self,
        param_name: str,
        dist: BaseDistribution,
        trials_better_than_cutoff: list[FrozenTrial],
        trials_better_than_baseline: list[FrozenTrial],
    ) -> float:
        pe_cutoff = self._build_parzen_estimator(param_name, dist, trials_better_than_cutoff)
        pe_baseline = self._build_parzen_estimator(param_name, dist, trials_better_than_baseline)
        grids = np.arange(pe_cutoff.n_grids)
        pdf_baseline = pe_baseline.pdf(grids) + 1e-12
        pdf_cutoff = pe_cutoff.pdf(grids) + 1e-12
        return float(pdf_cutoff @ ((pdf_baseline / pdf_cutoff - 1) ** 2))

    def evaluate(
        self,
        study: Study,
        params: list[str] | None = None,
        *,
        target: Callable[[FrozenTrial], float] | None = None,
    ) -> dict[str, float]:
        params, distributions = self._validate_and_get_params_and_distributions(
            study, params, target
        )

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

        (
            trials_better_than_cutoff,
            trials_better_than_baseline,
        ) = self._get_trials_better_than_cutoff_or_baseline(study, params, target)
        importance_sum = 0.0
        param_importances = {}
        for param_name, dist in non_single_distributions.items():
            param_importances[param_name] = self._compute_pearson_divergence(
                param_name,
                dist,
                trials_better_than_cutoff,
                trials_better_than_baseline,
            )
            importance_sum += param_importances[param_name]

        param_importances = {k: v / importance_sum for k, v in param_importances.items()}
        param_importances.update({k: 0.0 for k in single_distributions})
        return _sort_dict_by_importance(param_importances)
