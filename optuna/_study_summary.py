import datetime
import warnings

from optuna._study_direction import StudyDirection

from optuna import logging
from optuna import structs
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA


class StudySummary(object):
    """Basic attributes and aggregated results of a :class:`~optuna.study.Study`.

    See also :func:`optuna.study.get_all_study_summaries`.

    Attributes:
        study_name:
            Name of the :class:`~optuna.study.Study`.
        direction:
            :class:`~optuna.study.StudyDirection` of the :class:`~optuna.study.Study`.
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

    def __init__(
        self,
        study_name,  # type: str
        direction,  # type: StudyDirection
        best_trial,  # type: Optional[structs.FrozenTrial]
        user_attrs,  # type: Dict[str, Any]
        system_attrs,  # type: Dict[str, Any]
        n_trials,  # type: int
        datetime_start,  # type: Optional[datetime.datetime]
        study_id,  # type: int
    ):
        # type: (...) -> None

        self.study_name = study_name
        self.direction = direction
        self.best_trial = best_trial
        self.user_attrs = user_attrs
        self.system_attrs = system_attrs
        self.n_trials = n_trials
        self.datetime_start = datetime_start
        self._study_id = study_id

    def __eq__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, StudySummary):
            return NotImplemented

        return other.__dict__ == self.__dict__

    def __lt__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, StudySummary):
            return NotImplemented

        return self._study_id < other._study_id

    def __le__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, StudySummary):
            return NotImplemented

        return self._study_id <= other._study_id

    @property
    def study_id(self):
        # type: () -> int
        """Return the study ID.

        .. deprecated:: 0.20.0
            The direct use of this attribute is deprecated and it is recommended that you use
            :attr:`~optuna.study.StudySummary.study_name` instead.

        Returns:
            The study ID.
        """

        message = (
            "The use of `StudySummary.study_id` is deprecated. "
            "Please use `StudySummary.study_name` instead."
        )
        warnings.warn(message, DeprecationWarning)

        logger = logging.get_logger(__name__)
        logger.warning(message)

        return self._study_id
