from datetime import datetime
import enum
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional


class TrialState(enum.Enum):

    """State of a trial.

    Attributes:
        RUNNING:
            Trial is running.
        COMPLETE:
            Trial has been finished without any error.
        PRUNED:
            Trial has been pruned with :class:`TrialPruned`.
        FAIL:
            Trial has failed due to an uncaught error.
    """

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3

    def is_finished(self):
        # type: () -> bool

        return self == TrialState.COMPLETE or self == TrialState.PRUNED


class StudyDirection(enum.Enum):

    """StudyDirection represents whether the study minimizes or maximizes the objective function.

    Attributes:
        NOT_SET:
            Direction has not been set.
        MNIMIZE:
            Study minimizes the objective function.
        MAXIMIZE:
            Study maximizes the objective function.
    """

    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2


class FrozenTrial(
    NamedTuple(
        '_BaseFrozenTrial',
        [('trial_id', int),
         ('state', TrialState),
         ('value', Optional[float]),
         ('datetime_start', Optional[datetime]),
         ('datetime_complete', Optional[datetime]),
         ('params', Dict[str, Any]),
         ('user_attrs', Dict[str, Any]),
         ('system_attrs', Dict[str, Any]),
         ('intermediate_values', Dict[int, float]),
         ('params_in_internal_repr', Dict[str, float]),
         ])):

    """A FrozenTrial object holds the status and results of a trial.

    Attributes:
        trial_id:
            Identifier of the trial.
        state:
            :class:`TrialState` of the trial.
        value:
            Objective value of the trial.
        datetime_start:
            Datetime where the trial started.
        datetime_complete:
            Datetime where the trial finished.
        params:
            Dictionary that contains suggested parameters.
        user_attrs:
            Dictionary that contains the trial's attributes set with
            :func:`optuna.trial.Trial.set_user_attr`.
        system_attrs:
            Dictionary that contains the trial's attributes internally set by Optuna.
        intermediate_values:
            Intermediate objective values set with :func:`optuna.trial.Trial.report`.
        prams_in_internal_repr:
            Optuna's internal representation of :attr:`params`.
    """

    internal_fields = ['params_in_internal_repr']


class StudySummary(
    NamedTuple(
        'StudySummary',
        [('study_id', int),
         ('study_name', str),
         ('direction', StudyDirection),
         ('best_trial', Optional[FrozenTrial]),
         ('user_attrs', Dict[str, Any]),
         ('system_attrs', Dict[str, Any]),
         ('n_trials', int),
         ('datetime_start', Optional[datetime])])):

    """A StudySummary object hold basic attributes and aggregated results of a study.

    .. seealso::
        :func:`optuna.study.get_all_study_summaries`

    Attributes:
        study_id:
            Identifier of the study.
        study_name:
            Name of the study.
        direction:
            :class:`StudyDirection` of the study.
        best_trial:
            :class:`FrozenTrial` with best objective value in the study.
        user_attrs:
            Dictionary that contains the trial's attributes set with
            :func:`optuna.study.Study.set_user_attr`.
        system_attrs:
            Dictionary that contains the study's attributes internally set by Optuna.
        n_trials:
            The number of trials ran in the study.
        datetime_start:
            Datetime where the study started.
    """


class OptunaError(Exception):

    """Base class for Optuna specific errors."""

    pass


class TrialPruned(OptunaError):

    """Exception for pruned trials.

    This error tells a trainer that the current trial was pruned. It is supposed to be raised
    after :func:`optuna.trial.Trial.should_prune` as shown in the following example.

    Example:

        .. code::

            >>> def objective(trial):
            >>>     ...
            >>>     for step in range(n_train_iter):
            >>>         ...
            >>>         if trial.should_prune(step):
            >>>             raise TrailPruned()
    """

    pass


class CLIUsageError(OptunaError):

    """Exception for CLI.

    CLI raises this exception when it receives invalid configuration.
    """

    pass


class StorageInternalError(OptunaError):

    """Exception for storage operation.

    This error is raised when an operation failed in backend DB of storage.
    """

    pass
