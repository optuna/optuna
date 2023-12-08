from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Callable
import warnings

import numpy as np

from optuna.trial import FrozenTrial


def _get_topk_value(target_loss_values: np.ndarray, topk: int) -> float:
    if topk > target_loss_values.size or topk < 1:
        raise ValueError(f"topk must be in [1, {target_loss_values.size}], but got {topk}.")

    return np.partition(target_loss_values, topk - 1)[topk - 1]


class FilterRunner:
    def __init__(
        self,
        is_lower_better: bool,
        cond_name: str,
        cond_value: float,
        min_n_top_trials: int | None,
        filter_name: str,
        target: Callable[[FrozenTrial], float] | None,
        cutoff_value_calculate_method: Callable[[np.ndarray], float],
    ):
        self._is_lower_better = is_lower_better
        self._min_n_top_trials = min_n_top_trials
        self._target = target
        self._cutoff_value_calculate_method = cutoff_value_calculate_method
        self._warning_msg_generator: Callable[[float], str] = lambda value_instead: (
            f"The given {cond_name}={cond_value} was too tight to have {min_n_top_trials} trials "
            f"after applying {filter_name}, so {value_instead} was used as cutoff_value."
        )

    def _get_cutoff_value_with_warning(self, target_loss_values: np.ndarray) -> float:
        cutoff_value = self._cutoff_value_calculate_method(target_loss_values)
        if self._min_n_top_trials is None:
            return cutoff_value

        top_value = _get_topk_value(target_loss_values, self._min_n_top_trials)
        if cutoff_value < top_value:
            value_instead = top_value if self._is_lower_better else -top_value
            warnings.warn(self._warning_msg_generator(value_instead))

        return max(cutoff_value, top_value)

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        target = self._target
        target_values = np.array(
            [target(trial) if target is not None else trial.value for trial in trials]
        )
        if len(target_values.shape) != 1:
            raise ValueError(f"target_values must be 1d array, but got {target_values.shape}.")

        target_loss_values = target_values if self._is_lower_better else -target_values
        mask = target_loss_values <= self._get_cutoff_value_with_warning(target_loss_values)
        return [t for should_be_in, t in zip(mask, trials) if should_be_in]


class BaseFilter(metaclass=ABCMeta):
    _filter_runner: FilterRunner | None

    @abstractmethod
    def _calculate_cutoff_value(self, target_loss_values: np.ndarray) -> float:
        raise NotImplementedError

    def filter(self, trials: list[FrozenTrial]) -> list[FrozenTrial]:
        """Filter trials based on target_values.

        Args:
            trials:
                A list of trials to which the filter is applied.

        Returns:
            A list of filtered trials.
        """
        if self._filter_runner is None:
            raise ValueError("FilterRunner must be defined.")

        return self._filter_runner.filter(trials)
