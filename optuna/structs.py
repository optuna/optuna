from datetime import datetime
import enum
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional

from optuna.distributions import BaseDistribution  # NOQA


class TrialState(enum.Enum):
    """State of a :class:`~optuna.trial.Trial`.

    Attributes:
        RUNNING:
            The :class:`~optuna.trial.Trial` is running.
        COMPLETE:
            The :class:`~optuna.trial.Trial` has been finished without any error.
        PRUNED:
            The :class:`~optuna.trial.Trial` has been pruned with :class:`TrialPruned`.
        FAIL:
            The :class:`~optuna.trial.Trial` has failed due to an uncaught error.
    """

    RUNNING = 0
    COMPLETE = 1
    PRUNED = 2
    FAIL = 3

    def is_finished(self):
        # type: () -> bool

        return self != TrialState.RUNNING


class StudyDirection(enum.Enum):
    """Direction of a :class:`~optuna.study.Study`.

    Attributes:
        NOT_SET:
            Direction has not been set.
        MNIMIZE:
            :class:`~optuna.study.Study` minimizes the objective function.
        MAXIMIZE:
            :class:`~optuna.study.Study` maximizes the objective function.
    """

    NOT_SET = 0
    MINIMIZE = 1
    MAXIMIZE = 2


class FrozenTrial(
        NamedTuple('_BaseFrozenTrial', [
            ('number', int),
            ('state', TrialState),
            ('value', Optional[float]),
            ('datetime_start', Optional[datetime]),
            ('datetime_complete', Optional[datetime]),
            ('params', Dict[str, Any]),
            ('distributions', Dict[str, BaseDistribution]),
            ('user_attrs', Dict[str, Any]),
            ('system_attrs', Dict[str, Any]),
            ('intermediate_values', Dict[int, float]),
            ('params_in_internal_repr', Dict[str, float]),
            ('trial_id', int),
        ])):
    """Status and results of a :class:`~optuna.trial.Trial`.

    Attributes:
        number:
            Unique and consecutive number of :class:`~optuna.trial.Trial` for each
            :class:`~optuna.study.Study`. Note that this field uses zero-based numbering.
        state:
            :class:`TrialState` of the :class:`~optuna.trial.Trial`.
        value:
            Objective value of the :class:`~optuna.trial.Trial`.
        datetime_start:
            Datetime where the :class:`~optuna.trial.Trial` started.
        datetime_complete:
            Datetime where the :class:`~optuna.trial.Trial` finished.
        params:
            Dictionary that contains suggested parameters.
        distributions:
            Dictionary that contains the distributions of :attr:`params`.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` set with
            :func:`optuna.trial.Trial.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` internally
            set by Optuna.
        intermediate_values:
            Intermediate objective values set with :func:`optuna.trial.Trial.report`.
        params_in_internal_repr:
            Optuna's internal representation of :attr:`params`. Note that this field is not
            supposed to be used by library users.
        trial_id:
            Optuna's internal identifier of the :class:`~optuna.trial.Trial`. Note that this field
            is not supposed to be used by library users. Instead, please use :attr:`number` and
            :class:`~optuna.study.Study.study_id` to identify a :class:`~optuna.trial.Trial`.
    """

    internal_fields = ['distributions', 'params_in_internal_repr', 'trial_id']


class StudySummary(
        NamedTuple('StudySummary', [('study_id', int), ('study_name', str),
                                    ('direction', StudyDirection),
                                    ('best_trial', Optional[FrozenTrial]),
                                    ('user_attrs', Dict[str, Any]),
                                    ('system_attrs', Dict[str, Any]), ('n_trials', int),
                                    ('datetime_start', Optional[datetime])])):
    """Basic attributes and aggregated results of a :class:`~optuna.study.Study`.

    See also :func:`optuna.study.get_all_study_summaries`.

    Attributes:
        study_id:
            Identifier of the :class:`~optuna.study.Study`.
        study_name:
            Name of the :class:`~optuna.study.Study`.
        direction:
            :class:`StudyDirection` of the :class:`~optuna.study.Study`.
        best_trial:
            :class:`FrozenTrial` with best objective value in the :class:`~optuna.study.Study`.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` set with
            :func:`optuna.study.Study.set_user_attr`.
        system_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` internally
            set by Optuna.
        n_trials:
            The number of trials ran in the :class:`~optuna.study.Study`.
        datetime_start:
            Datetime where the :class:`~optuna.study.Study` started.
    """


class OptunaError(Exception):
    """Base class for Optuna specific errors."""

    pass


class TrialPruned(OptunaError):
    """Exception for pruned trials.

    This error tells a trainer that the current :class:`~optuna.trial.Trial` was pruned. It is
    supposed to be raised after :func:`optuna.trial.Trial.should_prune` as shown in the following
    example.

    Example:

        .. code::

            >>> def objective(trial):
            >>>     ...
            >>>     for step in range(n_train_iter):
            >>>         ...
            >>>         if trial.should_prune():
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


class DuplicatedStudyError(OptunaError):
    """Exception for a duplicated study name.

    This error is raised when a specified study name already exists in the storage.
    """

    pass
