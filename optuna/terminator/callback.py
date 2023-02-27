from typing import Optional

from optuna.logging import get_logger
from optuna.study.study import Study
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.terminator import BaseTerminator
from optuna.terminator.terminator import Terminator
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = get_logger(__name__)


class TerminatorCallback:
    def __init__(
        self,
        terminator: Optional[BaseTerminator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self._min_n_trials = min_n_trials
        self._terminator = terminator or Terminator()

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        trials = study.get_trials(states=[TrialState.COMPLETE])

        if len(trials) < self._min_n_trials:
            return

        should_terminate = self._terminator.should_terminate(study=study)

        if should_terminate:
            _logger.info("The study has been stopped by the terminator.")
            study.stop()
