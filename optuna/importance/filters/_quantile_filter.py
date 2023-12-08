from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optuna.importance.filters._base import BaseFilter
from optuna.importance.filters._base import FilterRunner
from optuna.trial import FrozenTrial


class QuantileFilter(BaseFilter):
    def __init__(
        self,
        quantile: float,
        is_lower_better: bool,
        min_n_top_trials: int | None = None,
        target: Callable[[FrozenTrial], float] | None = None,
    ):
        if quantile < 0 or quantile > 1:
            raise ValueError(f"quantile must be in [0, 1], but got {quantile}.")
        if min_n_top_trials is not None and min_n_top_trials <= 0:
            raise ValueError(f"min_n_top_trials must be positive, but got {min_n_top_trials}.")

        self._filter_runner = FilterRunner(
            is_lower_better=is_lower_better,
            cond_name="quantile",
            cond_value=quantile,
            min_n_top_trials=min_n_top_trials,
            target=target,
            filter_name=self.__class__.__name__,
            cutoff_value_calculate_method=self._calculate_cutoff_value,
        )
        self._quantile = quantile

    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        return np.quantile(target_loss_values, self._quantile, method="higher")
