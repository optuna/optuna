import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

from optuna import distributions
from optuna import logging
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.trial._base import BaseTrial
from optuna.trial._state import TrialState


_logger = logging.get_logger(__name__)

CategoricalChoiceType = Union[None, bool, int, float, str]


class BaseFrozenTrial(BaseTrial):
    def __init__(
        self,
        number: int,
        state: TrialState,
        value: Optional[Union[float, Sequence[float]]],
        datetime_start: Optional[datetime.datetime],
        datetime_complete: Optional[datetime.datetime],
        params: Dict[str, Any],
        distributions: Dict[str, BaseDistribution],
        user_attrs: Dict[str, Any],
        system_attrs: Dict[str, Any],
        intermediate_values: Dict[int, Union[float, Sequence[float]]],
        trial_id: int,
    ) -> None:

        self._number = number
        self.state = state
        self._value = value
        self._datetime_start = datetime_start
        self.datetime_complete = datetime_complete
        self._params = params
        self._user_attrs = user_attrs
        self._system_attrs = system_attrs
        self._intermediate_values = intermediate_values
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

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, BaseFrozenTrial):
            return NotImplemented
        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:

        if not isinstance(other, BaseFrozenTrial):
            return NotImplemented

        return self.number < other.number

    def __le__(self, other: Any) -> bool:

        if not isinstance(other, BaseFrozenTrial):
            return NotImplemented

        return self.number <= other.number

    def __hash__(self) -> int:

        return hash(tuple(getattr(self, field) for field in self._ordered_fields))

    def __repr__(self) -> str:

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

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> float:

        if step is not None:
            if log:
                raise ValueError("The parameter `step` is not supported when `log` is True.")
            else:
                return self._suggest(name, DiscreteUniformDistribution(low=low, high=high, q=step))
        else:
            if log:
                return self._suggest(name, LogUniformDistribution(low=low, high=high))
            else:
                return self._suggest(name, UniformDistribution(low=low, high=high))

    def suggest_uniform(self, name: str, low: float, high: float) -> float:

        return self._suggest(name, UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:

        return self._suggest(name, LogUniformDistribution(low=low, high=high))

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:

        discrete = DiscreteUniformDistribution(low=low, high=high, q=q)
        return self._suggest(name, discrete)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:

        if step != 1:
            if log:
                raise ValueError(
                    "The parameter `step != 1` is not supported when `log` is True."
                    "The specified `step` is {}.".format(step)
                )
            else:
                distribution: Union[
                    IntUniformDistribution, IntLogUniformDistribution
                ] = IntUniformDistribution(low=low, high=high, step=step)
        else:
            if log:
                distribution = IntLogUniformDistribution(low=low, high=high)
            else:
                distribution = IntUniformDistribution(low=low, high=high, step=step)
        return int(self._suggest(name, distribution))

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:

        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        """Interface of report function.

        Since :class:`~optuna.trial.FrozenTrial` is not pruned,
        this report function does nothing.

        .. seealso::
            Please refer to :func:`~optuna.trial.FrozenTrial.should_prune`.

        Args:
            value:
                A value returned from the objective function.
            step:
                Step of the trial (e.g., Epoch of neural network training). Note that pruners
                assume that ``step`` starts at zero. For example,
                :class:`~optuna.pruners.MedianPruner` simply checks if ``step`` is less than
                ``n_warmup_steps`` as the warmup mechanism.
        """

        pass

    def should_prune(self) -> bool:
        """Suggest whether the trial should be pruned or not.

        The suggestion is always :obj:`False` regardless of a pruning algorithm.

        .. note::
            :class:`~optuna.trial.FrozenTrial` only samples one combination of parameters.

        Returns:
            :obj:`False`.
        """

        return False

    def set_user_attr(self, key: str, value: Any) -> None:

        self._user_attrs[key] = value

    def set_system_attr(self, key: str, value: Any) -> None:

        self._system_attrs[key] = value

    def _validate(self) -> None:

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

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:

        if name not in self._params:
            raise ValueError(
                "The value of the parameter '{}' is not found. Please set it at "
                "the construction of the FrozenTrial object.".format(name)
            )

        value = self._params[name]
        param_value_in_internal_repr = distribution.to_internal_repr(value)
        if not distribution._contains(param_value_in_internal_repr):
            raise ValueError(
                "The value {} of the parameter '{}' is out of "
                "the range of the distribution {}.".format(value, name, distribution)
            )

        if name in self._distributions:
            distributions.check_distribution_compatibility(self._distributions[name], distribution)

        self._distributions[name] = distribution

        return value

    @property
    def number(self) -> int:

        return self._number

    @number.setter
    def number(self, value: int) -> None:

        self._number = value

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:

        return self._datetime_start

    @datetime_start.setter
    def datetime_start(self, value: Optional[datetime.datetime]) -> None:

        self._datetime_start = value

    @property
    def params(self) -> Dict[str, Any]:

        return self._params

    @params.setter
    def params(self, params: Dict[str, Any]) -> None:

        self._params = params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        """Dictionary that contains the distributions of :attr:`params`."""

        return self._distributions

    @distributions.setter
    def distributions(self, value: Dict[str, BaseDistribution]) -> None:

        self._distributions = value

    @property
    def user_attrs(self) -> Dict[str, Any]:

        return self._user_attrs

    @user_attrs.setter
    def user_attrs(self, value: Dict[str, Any]) -> None:

        self._user_attrs = value

    @property
    def system_attrs(self) -> Dict[str, Any]:

        return self._system_attrs

    @system_attrs.setter
    def system_attrs(self, value: Dict[str, Any]) -> None:

        self._system_attrs = value

    @property
    def last_step(self) -> Optional[int]:
        """Return the maximum step of `intermediate_values` in the trial.

        Returns:
            The maximum step of intermediates.
        """

        if len(self._intermediate_values) == 0:
            return None
        else:
            return max(self._intermediate_values.keys())

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        """Return the elapsed time taken to complete the trial.

        Returns:
            The duration.
        """

        if self.datetime_start and self.datetime_complete:
            return self.datetime_complete - self.datetime_start
        else:
            return None
