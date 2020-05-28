import warnings

from optuna import _study_direction
from optuna import exceptions
from optuna import logging
from optuna import trial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from datetime import datetime  # NOQA
    from datetime import timedelta  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA


_logger = logging.get_logger(__name__)

_message = (
    "`structs` is deprecated. Classes have moved to the following modules. "
    "`structs.StudyDirection`->`study.StudyDirection`, "
    "`structs.StudySummary`->`study.StudySummary`, "
    "`structs.FrozenTrial`->`trial.FrozenTrial`, "
    "`structs.TrialState`->`trial.TrialState`, "
    "`structs.TrialPruned`->`exceptions.TrialPruned`."
)
warnings.warn(_message, DeprecationWarning)
_logger.warning(_message)

# The use of the structs.StudyDirection is deprecated and it is recommended that you use
# study.StudyDirection instead. See the API reference for more details.
StudyDirection = _study_direction.StudyDirection

# The use of the structs.TrialState is deprecated and it is recommended that you use
# trial.TrialState instead. See the API reference for more details.
TrialState = trial.TrialState


class FrozenTrial(object):
    """Status and results of a :class:`~optuna.trial.Trial`.

    .. deprecated:: 1.4.0

        This class was moved to :mod:`~optuna.trial`. Please use
        :class:`~optuna.trial.FrozenTrial` instead.

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
        user_attrs:
            Dictionary that contains the attributes of the :class:`~optuna.trial.Trial` set with
            :func:`optuna.trial.Trial.set_user_attr`.
        intermediate_values:
            Intermediate objective values set with :func:`optuna.trial.Trial.report`.
    """

    def __init__(
        self,
        number,  # type: int
        state,  # type: TrialState
        value,  # type: Optional[float]
        datetime_start,  # type: Optional[datetime]
        datetime_complete,  # type: Optional[datetime]
        params,  # type: Dict[str, Any]
        distributions,  # type: Dict[str, BaseDistribution]
        user_attrs,  # type: Dict[str, Any]
        system_attrs,  # type: Dict[str, Any]
        intermediate_values,  # type: Dict[int, float]
        trial_id,  # type: int
    ):
        # type: (...) -> None

        message = (
            "The use of `structs.FrozenTrial` is deprecated. "
            "Please use `trial.FrozenTrial` instead."
        )
        warnings.warn(message, DeprecationWarning)
        _logger.warning(message)

        self.number = number
        self.state = state
        self.value = value
        self.datetime_start = datetime_start
        self.datetime_complete = datetime_complete
        self.params = params
        self.user_attrs = user_attrs
        self.system_attrs = system_attrs
        self.intermediate_values = intermediate_values
        self._distributions = distributions
        self._trial_id = trial_id

    # Ordered list of fields required for `__repr__`, `__hash__` and dataframe creation.
    # TODO(hvy): Remove this list in Python 3.6 as the order of `self.__dict__` is preserved.
    _ordered_fields = [
        "number",
        "value",
        "datetime_start",
        "datetime_complete",
        "params",
        "_distributions",
        "user_attrs",
        "system_attrs",
        "intermediate_values",
        "_trial_id",
        "state",
    ]

    def __eq__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, FrozenTrial):
            return NotImplemented
        return other.__dict__ == self.__dict__

    def __lt__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, FrozenTrial):
            return NotImplemented

        return self.number < other.number

    def __le__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, FrozenTrial):
            return NotImplemented

        return self.number <= other.number

    def __hash__(self):
        # type: () -> int

        return hash(tuple(getattr(self, field) for field in self._ordered_fields))

    def __repr__(self):
        # type: () -> str

        return "{cls}({kwargs})".format(
            cls=self.__class__.__name__,
            kwargs=", ".join(
                "{field}={value}".format(
                    field=field if not field.startswith("_") else field[1:],
                    value=repr(getattr(self, field)),
                )
                for field in self._ordered_fields
            ),
        )

    def _validate(self):
        # type: () -> None

        if self.datetime_start is None:
            raise ValueError("`datetime_start` is supposed to be set.")

        if self.state.is_finished():
            if self.datetime_complete is None:
                raise ValueError("`datetime_complete` is supposed to be set for a finished trial.")
        else:
            if self.datetime_complete is not None:
                raise ValueError(
                    "`datetime_complete` is supposed to be None for an unfinished trial."
                )

        if self.state == TrialState.COMPLETE and self.value is None:
            raise ValueError("`value` is supposed to be set for a complete trial.")

        if set(self.params.keys()) != set(self.distributions.keys()):
            raise ValueError(
                "Inconsistent parameters {} and distributions {}.".format(
                    set(self.params.keys()), set(self.distributions.keys())
                )
            )

        for param_name, param_value in self.params.items():
            distribution = self.distributions[param_name]

            param_value_in_internal_repr = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_value_in_internal_repr):
                raise ValueError(
                    "The value {} of parameter '{}' isn't contained in the distribution "
                    "{}.".format(param_value, param_name, distribution)
                )

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]
        """Dictionary that contains the distributions of :attr:`params`."""

        return self._distributions

    @distributions.setter
    def distributions(self, value):
        # type: (Dict[str, BaseDistribution]) -> None
        self._distributions = value

    @property
    def trial_id(self):
        # type: () -> int
        """Return the trial ID.

        .. deprecated:: 0.19.0
            The direct use of this attribute is deprecated and it is recommended that you use
            :attr:`~optuna.trial.FrozenTrial.number` instead.

        Returns:
            The trial ID.
        """

        warnings.warn(
            "The use of `FrozenTrial.trial_id` is deprecated. "
            "Please use `FrozenTrial.number` instead.",
            DeprecationWarning,
        )

        _logger.warning(
            "The use of `FrozenTrial.trial_id` is deprecated. "
            "Please use `FrozenTrial.number` instead."
        )

        return self._trial_id

    @property
    def last_step(self):
        # type: () -> Optional[int]

        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

    @property
    def duration(self):
        # type: () -> Optional[timedelta]
        """Return the elapsed time taken to complete the trial.

        Returns:
            The duration.
        """

        if self.datetime_start and self.datetime_complete:
            return self.datetime_complete - self.datetime_start
        else:
            return None


class StudySummary(object):
    """Basic attributes and aggregated results of a :class:`~optuna.study.Study`.

    .. deprecated:: 1.4.0

        This class was moved to :mod:`~optuna.study`. Please use
        :class:`~optuna.study.StudySummary` instead.

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
        best_trial,  # type: Optional[FrozenTrial]
        user_attrs,  # type: Dict[str, Any]
        system_attrs,  # type: Dict[str, Any]
        n_trials,  # type: int
        datetime_start,  # type: Optional[datetime]
        study_id,  # type: int
    ):
        # type: (...) -> None

        message = (
            "The use of `structs.StudySummary` is deprecated. "
            "Please use `study.StudySummary` instead."
        )
        warnings.warn(message, DeprecationWarning)
        _logger.warning(message)

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

        _logger.warning(message)

        return self._study_id


class TrialPruned(exceptions.TrialPruned):
    """Exception for pruned trials.

    .. deprecated:: 0.19.0

        This class was moved to :mod:`~optuna.exceptions`. Please use
        :class:`~optuna.exceptions.TrialPruned` instead.
    """

    def __init__(self, *args, **kwargs):
        # type: (Any, Any) -> None

        message = (
            "The use of `optuna.structs.TrialPruned` is deprecated. "
            "Please use `optuna.TrialPruned` instead."
        )
        warnings.warn(message, DeprecationWarning)
        _logger.warning(message)
