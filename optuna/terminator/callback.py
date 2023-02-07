import math
from typing import Optional

from optuna.logging import get_logger
from optuna.study.study import Study
from optuna.terminator.regret.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.terminator import Terminator
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = get_logger(__name__)


class TerminatorCallback:
    def __init__(
        self,
        terminator: Optional[Terminator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self._n_startup_trials = min_n_trials
        self._terminator = terminator or Terminator()

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        trials = study.get_trials(states=[TrialState.COMPLETE])

        if len(trials) <= self._n_startup_trials:
            return

        should_terminate, regret_bound, variance = self._terminator.should_terminate_with_metrics(
            study=study,
        )

        _logger.debug(
            f"Optuna terminator has reported regret bound {regret_bound} and "
            f"sqrt variance {math.sqrt(variance)}."
        )

        if should_terminate:
            _logger.info("The terminator stopped the study because noise exceeds regret bound.")
            study.stop()
