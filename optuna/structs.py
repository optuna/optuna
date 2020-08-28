import warnings

from optuna._deprecated import deprecated
from optuna import _study_direction
from optuna import exceptions
from optuna import trial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from datetime import datetime  # NOQA
    from datetime import timedelta  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA


_message = (
    "`structs` is deprecated. Classes have moved to the following modules. "
    "`structs.StudyDirection`->`study.StudyDirection`, "
    "`structs.StudySummary`->`study.StudySummary`, "
    "`structs.FrozenTrial`->`trial.FrozenTrial`, "
    "`structs.TrialState`->`trial.TrialState`, "
    "`structs.TrialPruned`->`exceptions.TrialPruned`."
)
warnings.warn(_message, FutureWarning)

# The use of the structs.StudyDirection is deprecated and it is recommended that you use
# study.StudyDirection instead. See the API reference for more details.
StudyDirection = _study_direction.StudyDirection

# The use of the structs.TrialState is deprecated and it is recommended that you use
# trial.TrialState instead. See the API reference for more details.
TrialState = trial.TrialState


@deprecated(
    "1.4.0",
    text=(
        "This class was moved to :mod:`~optuna.trial`. Please use "
        ":class:`~optuna.trial.FrozenTrial` instead."
    ),
)
class FrozenTrial(trial.FrozenTrial):
    pass


@deprecated(
    "1.4.0",
    text=(
        "This class was moved to :mod:`~optuna.study`. Please use "
        ":class:`~optuna.study.StudySummary` instead."
    ),
)
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
        direction,  # type: _study_direction.StudyDirection
        best_trial,  # type: Optional[trial.FrozenTrial]
        user_attrs,  # type: Dict[str, Any]
        system_attrs,  # type: Dict[str, Any]
        n_trials,  # type: int
        datetime_start,  # type: Optional[datetime]
        study_id,  # type: int
    ) -> None:
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


@deprecated(
    "0.19.0",
    text=(
        "This class was moved to :mod:`~optuna.exceptions`. Please use "
        ":class:`~optuna.exceptions.TrialPruned` instead."
    ),
)
class TrialPruned(exceptions.TrialPruned):
    """Exception for pruned trials."""

    pass
