from typing import Optional

from optuna._experimental import experimental_class
from optuna.logging import get_logger
from optuna.study.study import Study
from optuna.terminator.terminator import BaseTerminator
from optuna.terminator.terminator import Terminator
from optuna.trial import FrozenTrial


_logger = get_logger(__name__)


@experimental_class("3.2.0")
class TerminatorCallback:
    def __init__(
        self,
        terminator: Optional[BaseTerminator] = None,
    ) -> None:
        self._terminator = terminator or Terminator()

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        should_terminate = self._terminator.should_terminate(study=study)

        if should_terminate:
            _logger.info("The study has been stopped by the terminator.")
            study.stop()
