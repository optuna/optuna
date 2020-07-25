import datetime
from typing import Any
from typing import Dict
from typing import Optional

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import logging
from optuna.trial._state import TrialState

_logger = logging.get_logger(__name__)


class FrozenTrial(object):
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
        datetime_start,  # type: Optional[datetime.datetime]
        datetime_complete,  # type: Optional[datetime.datetime]
        params,  # type: Dict[str, Any]
        distributions,  # type: Dict[str, BaseDistribution]
        user_attrs,  # type: Dict[str, Any]
        system_attrs,  # type: Dict[str, Any]
        intermediate_values,  # type: Dict[int, float]
        trial_id,  # type: int
    ):
        # type: (...) -> None

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
    def last_step(self):
        # type: () -> Optional[int]

        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

    @property
    def duration(self):
        # type: () -> Optional[datetime.timedelta]
        """Return the elapsed time taken to complete the trial.

        Returns:
            The duration.
        """

        if self.datetime_start and self.datetime_complete:
            return self.datetime_complete - self.datetime_start
        else:
            return None


@experimental("2.0.0")
def create_trial(
    *,
    state: Optional[TrialState] = None,
    value: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
    distributions: Optional[Dict[str, BaseDistribution]] = None,
    user_attrs: Optional[Dict[str, Any]] = None,
    system_attrs: Optional[Dict[str, Any]] = None,
    intermediate_values: Optional[Dict[int, float]] = None
) -> FrozenTrial:
    """Create a new :class:`~optuna.trial.FrozenTrial`.

    Example:

        .. testcode::

            import optuna
            from optuna.distributions import CategoricalDistribution
            from optuna.distributions import UniformDistribution

            trial = optuna.trial.create_trial(
                params={"x": 1.0, "y": 0},
                distributions={
                    "x": UniformDistribution(0, 10),
                    "y": CategoricalDistribution([-1, 0, 1]),
                },
                value=5.0,
            )

            assert isinstance(trial, optuna.trial.FrozenTrial)
            assert trial.value == 5.0
            assert trial.params == {"x": 1.0, "y": 0}

    .. seealso::

        See :func:`~optuna.study.Study.add_trial` for how this function can be used to create a
        study from existing trials.

    .. note::

        Please note that this is a low-level API. In general, trials that are passed to objective
        functions are created inside :func:`~optuna.study.Study.optimize`.

    Args:
        state:
            Trial state.
        value:
            Trial objective value. Must be specified if ``state`` is :class:`TrialState.COMPLETE`.
        params:
            Dictionary with suggested parameters of the trial.
        distributions:
            Dictionary with parameter distributions of the trial.
        user_attrs:
            Dictionary with user attributes.
        system_attrs:
            Dictionary with system attributes. Should not have to be used for most users.
        intermediate_values:
            Dictionary with intermediate objective values of the trial.

    Returns:
        Created trial.

    """

    params = params or {}
    distributions = distributions or {}
    user_attrs = user_attrs or {}
    system_attrs = system_attrs or {}
    intermediate_values = intermediate_values or {}
    state = state or TrialState.COMPLETE

    datetime_start = datetime.datetime.now()
    if state.is_finished():
        datetime_complete = datetime_start  # type: Optional[datetime.datetime]
    else:
        datetime_complete = None

    trial = FrozenTrial(
        number=-1,
        trial_id=-1,
        state=state,
        value=value,
        datetime_start=datetime_start,
        datetime_complete=datetime_complete,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        intermediate_values=intermediate_values,
    )

    trial._validate()

    return trial
