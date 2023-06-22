from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

from optuna.samplers._base import _process_constraints_after_trial
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class NSGAIIAfterTrialStrategy:
    def __init__(
        self, *, constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None
    ) -> None:
        self._constraints_func = constraints_func

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None = None,
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
