from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna._warnings import optuna_warn
from optuna.logging import get_logger
from optuna.terminator.terminator import Terminator


if TYPE_CHECKING:
    from optuna.study.study import Study
    from optuna.terminator.terminator import BaseTerminator
    from optuna.trial import FrozenTrial


_logger = get_logger(__name__)

_DEPRECATION_WARNING_MESSAGE = _DEPRECATION_WARNING_TEMPLATE.format(
    name="`optuna.terminator` module",
    d_ver="4.9.0",
    r_ver="6.0.0",
)


class TerminatorCallback:
    """A callback that terminates the optimization using Terminator.

    This class implements a callback which wraps :class:`~optuna.terminator.Terminator`
    so that it can be used with the :func:`~optuna.study.Study.optimize` method.

    Args:
        terminator:
            A terminator object which determines whether to terminate the optimization by
            assessing the room for optimization and statistical error. Defaults to a
            :class:`~optuna.terminator.Terminator` object with default
            ``improvement_evaluator`` and ``error_evaluator``.

    Example:

        .. testcode::

            from sklearn.datasets import load_wine
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold

            import optuna
            from optuna.terminator import TerminatorCallback
            from optuna.terminator import report_cross_validation_scores


            def objective(trial):
                X, y = load_wine(return_X_y=True)

                clf = RandomForestClassifier(
                    max_depth=trial.suggest_int("max_depth", 2, 32),
                    min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
                    criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
                )

                scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
                report_cross_validation_scores(trial, scores)
                return scores.mean()


            study = optuna.create_study(direction="maximize")
            terminator = TerminatorCallback()
            study.optimize(objective, n_trials=50, callbacks=[terminator])

    .. seealso::
        Please refer to :class:`~optuna.terminator.Terminator` for the details of
        the terminator mechanism.
    """

    def __init__(self, terminator: BaseTerminator | None = None) -> None:
        optuna_warn(_DEPRECATION_WARNING_MESSAGE, FutureWarning)
        self._terminator = terminator or Terminator()

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        should_terminate = self._terminator.should_terminate(study=study)

        if should_terminate:
            _logger.info("The study has been stopped by the terminator.")
            study.stop()
