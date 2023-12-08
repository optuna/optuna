from __future__ import annotations

from collections.abc import Callable

from optuna.importance.filters import QuantileFilter
from optuna.trial import FrozenTrial


def get_trial_filter(
    quantile: float,
    is_lower_better: bool = True,
    min_n_top_trials: int | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
) -> Callable[[list[FrozenTrial]], list[FrozenTrial]]:
    """Get trial filter.

    Args:
        quantile:
            Filter top `quantile * 100`% trials.
            For example, `quantile=0.1` means trials better than top-10% will be filtered.
        is_lower_better:
            Whether `target_value` is better when it is lower.
        min_n_top_trials:
            The minimum number of trials to be included in the filtered trials.
        target:
            A function to specify the value to evaluate importances.
            If it is :obj:`None` and ``study`` is being used for single-objective optimization,
            the objective values are used. Can also be used for other trial attributes, such as
            the duration, like ``target=lambda t: t.duration.total_seconds()``.

    Returns:
        A list of filtered trials.
    """
    return QuantileFilter(quantile, is_lower_better, min_n_top_trials, target).filter
