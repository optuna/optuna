import abc
from typing import Optional

from optuna._experimental import experimental_class
from optuna.study.study import Study
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.trial import TrialState


class BaseTerminator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def should_terminate(self, study: Study) -> bool:
        pass


@experimental_class("3.2.0")
class Terminator(BaseTerminator):
    """Automatic stopping mechanism for Optuna studies

    This class implements an automatic stopping mechanism for Optuna studies, aiming to prevent
    unnecessary computation. The study is terminated when the statistical error, e.g.
    cross-validation error, exceeds the room left for optimization.

    For further information about the algorithm, please refer to the following paper:

    - `A. Makarova et al. Automatic termination for hyperparameter optimization.
      <https://arxiv.org/abs/2104.08166>`_

    Args:
        improvement_evaluator:
            An evaluator object for assessing the room left for optimization. Defaults to a
            :class:`~optuna.terminator.improvement.evaluator.RegretBoundEvaluator` object.
        error_evaluator:
            An evaluator for calculating the statistical error, e.g. cross-validation error.
            Defaults to a :class:`~optuna.terminator.erroreval.CrossValidationErrorEvaluator`
            object.
        min_n_trials:
            The minimum number of trials before termination is considered. Defaults to ``20``.

    Raises:
        ValueError: If ``min_n_trials`` is not a positive integer.

    Example:
        from sklearn.datasets import load_wine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold

        import optuna
        from optuna.terminator.terminator import Terminator
        from optuna.terminator.serror import report_cross_validation_scores


        study = optuna.create_study(direction="maximize")
        terminator = Terminator()
        min_n_trials = 20

        while True:
            trial = study.ask()

            X, y = load_wine(return_X_y=True)

            clf = RandomForestClassifier(
                max_depth=trial.suggest_int("max_depth", 2, 32),
                min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
                criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
            )

            scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
            report_cross_validation_scores(trial, scores)

            value = scores.mean()
            print(f"Trial #{trial.number} finished with value {value}.")
            study.tell(trial, value)

            if trial.number > min_n_trials and terminator.should_terminate(study):
                print("Terminated by Optuna Terminator!")
                break

    .. seealso::
        Please refer to :class:`~optuna.terminator.callbacks.TerminationCallback` for to use the
        terminator mechanism with the :func:`~optuna.study.Study.optimize` method.
    """

    def __init__(
        self,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
        error_evaluator: Optional[BaseErrorEvaluator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        if min_n_trials <= 0:
            raise ValueError("`min_n_trials` is expected to be a positive integer.")

        self._improvement_evaluator = improvement_evaluator or RegretBoundEvaluator()
        self._error_evaluator = error_evaluator or CrossValidationErrorEvaluator()
        self._min_n_trials = min_n_trials

    def should_terminate(self, study: Study) -> bool:
        trials = study.get_trials(states=[TrialState.COMPLETE])

        if len(trials) < self._min_n_trials:
            return False

        regret_bound = self._improvement_evaluator.evaluate(
            trials=study.trials,
            study_direction=study.direction,
        )
        error = self._error_evaluator.evaluate(
            trials=study.trials, study_direction=study.direction
        )
        should_terminate = regret_bound < error

        return should_terminate
