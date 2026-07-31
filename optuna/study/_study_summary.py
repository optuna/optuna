from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna import logging


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from optuna.study._study_direction import StudyDirection
    from optuna.trial import FrozenTrial

_logger = logging.get_logger(__name__)


class StudySummary:
    """Basic attributes and aggregated results of a :class:`~optuna.study.Study`.

    See also :func:`optuna.study.get_all_study_summaries`.

    Attributes:
        study_name:
            Name of the :class:`~optuna.study.Study`.
        direction:
            :class:`~optuna.study.StudyDirection` of the :class:`~optuna.study.Study`.

            .. note::
                This attribute is only available during single-objective optimization.
        directions:
            A sequence of :class:`~optuna.study.StudyDirection` objects.
        best_trial:
            :class:`optuna.trial.FrozenTrial` with best objective value in the
            :class:`~optuna.study.Study`.
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.study.Study` set with
            :func:`optuna.study.Study.set_user_attr`.
        n_trials:
            The number of trials ran in the :class:`~optuna.study.Study`.
        datetime_start:
            Datetime where the :class:`~optuna.study.Study` started.

    """

    def __init__(
        self,
        study_name: str,
        direction: StudyDirection | None,
        best_trial: FrozenTrial | None,
        user_attrs: dict[str, Any],
        n_trials: int,
        datetime_start: datetime | None,
        study_id: int,
        *,
        directions: Sequence[StudyDirection] | None = None,
    ):
        self.study_name = study_name
        if direction is None and directions is None:
            raise ValueError("Specify one of `direction` and `directions`.")
        elif directions is not None:
            self._directions = list(directions)
        elif direction is not None:
            self._directions = [direction]
        else:
            raise ValueError("Specify only one of `direction` and `directions`.")
        self.best_trial = best_trial
        self.user_attrs = user_attrs
        self.n_trials = n_trials
        self.datetime_start = datetime_start
        self._study_id = study_id

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StudySummary):
            return NotImplemented

        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, StudySummary):
            return NotImplemented

        return self._study_id < other._study_id

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, StudySummary):
            return NotImplemented

        return self._study_id <= other._study_id

    @property
    def direction(self) -> StudyDirection:
        if len(self._directions) > 1:
            raise RuntimeError(
                "This attribute is not available during multi-objective optimization."
            )

        return self._directions[0]

    @property
    def directions(self) -> Sequence[StudyDirection]:
        return self._directions
