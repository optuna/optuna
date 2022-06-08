import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

from optuna import distributions
from optuna import logging
from optuna._deprecated import deprecated_func
from optuna.distributions import _convert_old_distribution_to_new_distribution
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial._base import BaseTrial
from optuna.trial._state import TrialState


_logger = logging.get_logger(__name__)
_suggest_deprecated_msg = "Use :func:`~optuna.trial.FrozenTrial.suggest_float` instead."

CategoricalChoiceType = Union[None, bool, int, float, str]


class FrozenTrial(BaseTrial):
    """Status and results of a :class:`~optuna.trial.Trial`.

    This object has the same methods as :class:`~optuna.trial.Trial`, and it suggests the same
    parameter values as in :attr:`params`; it does not sample any value from a distribution.
    In contrast to :class:`~optuna.trial.Trial`,
    :class:`~optuna.trial.FrozenTrial` does not depend on :class:`~optuna.study.Study`, and it is
    useful for deploying optimization results.

    Example:

        Re-evaluate an objective function with parameter values optimized study.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -1, 1)
                return x**2


            study = optuna.create_study()
            study.optimize(objective, n_trials=3)

            assert objective(study.best_trial) == study.best_value

    .. note::
        Instances are mutable, despite the name.
        For instance, :func:`~optuna.trial.FrozenTrial.set_user_attr` will update user attributes
        of objects in-place.


        Example:

            Overwritten attributes.

            .. testcode::

                import copy
                import datetime

                import optuna


                def objective(trial):
                    x = trial.suggest_float("x", -1, 1)

                    # this user attribute always differs
                    trial.set_user_attr("evaluation time", datetime.datetime.now())

                    return x**2


                study = optuna.create_study()
                study.optimize(objective, n_trials=3)

                best_trial = study.best_trial
                best_trial_copy = copy.deepcopy(best_trial)

                # re-evaluate
                objective(best_trial)

                # the user attribute is overwritten by re-evaluation
                assert best_trial.user_attrs != best_trial_copy.user_attrs

    .. note::
        Please refer to :class:`~optuna.trial.Trial` for details of methods and properties.


    Attributes:
        number:
            Unique and consecutive number of :class:`~optuna.trial.Trial` for each
            :class:`~optuna.study.Study`. Note that this field uses zero-based numbering.
        state:
            :class:`TrialState` of the :class:`~optuna.trial.Trial`.
        value:
            Objective value of the :class:`~optuna.trial.Trial`.
            ``value`` and ``values`` must not be specified at the same time.
        values:
            Sequence of objective values of the :class:`~optuna.trial.Trial`.
            The length is greater than 1 if the problem is multi-objective optimization.
            ``value`` and ``values`` must not be specified at the same time.
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
        number: int,
        state: TrialState,
        value: Optional[float],
        datetime_start: Optional[datetime.datetime],
        datetime_complete: Optional[datetime.datetime],
        params: Dict[str, Any],
        distributions: Dict[str, BaseDistribution],
        user_attrs: Dict[str, Any],
        system_attrs: Dict[str, Any],
        intermediate_values: Dict[int, float],
        trial_id: int,
        *,
        values: Optional[Sequence[float]] = None,
    ) -> None:

        self._number = number
        self.state = state
        self._values: Optional[List[float]] = None
        if value is not None and values is not None:
            raise ValueError("Specify only one of `value` and `values`.")
        elif value is not None:
            self._values = [value]
        elif values is not None:
            self._values = list(values)
        self._datetime_start = datetime_start
        self.datetime_complete = datetime_complete
        self._params = params
        self._user_attrs = user_attrs
        self._system_attrs = system_attrs
        self.intermediate_values = intermediate_values
        self._distributions = distributions
        self._trial_id = trial_id

    # Ordered list of fields required for `__repr__`, `__hash__` and dataframe creation.
    # TODO(hvy): Remove this list in Python 3.7 as the order of `self.__dict__` is preserved.
    _ordered_fields = [
        "number",
        "_values",
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

        if not isinstance(other, FrozenTrial):
            return NotImplemented
        return other.__dict__ == self.__dict__

    def __lt__(self, other: Any) -> bool:

        if not isinstance(other, FrozenTrial):
            return NotImplemented

        return self.number < other.number

    def __le__(self, other: Any) -> bool:

        if not isinstance(other, FrozenTrial):
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
            )
            + ", value=None",
        )

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:

        return self._suggest(name, FloatDistribution(low, high, log=log, step=step))

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_uniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:

        return self.suggest_float(name, low, high, log=True)

    @deprecated_func("3.0.0", "6.0.0", text=_suggest_deprecated_msg)
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:

        return self.suggest_float(name, low, high, step=q)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        return int(self._suggest(name, IntDistribution(low, high, log=log, step=step)))

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

        if self.state != TrialState.WAITING and self.datetime_start is None:
            raise ValueError(
                "`datetime_start` is supposed to be set when the trial state is not waiting."
            )

        if self.state.is_finished():
            if self.datetime_complete is None:
                raise ValueError("`datetime_complete` is supposed to be set for a finished trial.")
        else:
            if self.datetime_complete is not None:
                raise ValueError(
                    "`datetime_complete` is supposed to be None for an unfinished trial."
                )

        if self.state == TrialState.COMPLETE and self._values is None:
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
            warnings.warn(
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
    def value(self) -> Optional[float]:

        if self._values is not None:
            if len(self._values) > 1:
                raise RuntimeError(
                    "This attribute is not available during multi-objective optimization."
                )
            return self._values[0]
        return None

    @value.setter
    def value(self, v: Optional[float]) -> None:

        if self._values is not None:
            if len(self._values) > 1:
                raise RuntimeError(
                    "This attribute is not available during multi-objective optimization."
                )

        if v is not None:
            self._values = [v]
        else:
            self._values = None

    # These `_get_values`, `_set_values`, and `values = property(_get_values, _set_values)` are
    # defined to pass the mypy.
    # See https://github.com/python/mypy/issues/3004#issuecomment-726022329.
    def _get_values(self) -> Optional[List[float]]:

        return self._values

    def _set_values(self, v: Optional[Sequence[float]]) -> None:

        if v is not None:
            self._values = list(v)
        else:
            self._values = None

    values = property(_get_values, _set_values)

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
        """Return the maximum step of :attr:`intermediate_values` in the trial.

        Returns:
            The maximum step of intermediates.
        """

        if len(self.intermediate_values) == 0:
            return None
        else:
            return max(self.intermediate_values.keys())

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


def create_trial(
    *,
    state: TrialState = TrialState.COMPLETE,
    value: Optional[float] = None,
    values: Optional[Sequence[float]] = None,
    params: Optional[Dict[str, Any]] = None,
    distributions: Optional[Dict[str, BaseDistribution]] = None,
    user_attrs: Optional[Dict[str, Any]] = None,
    system_attrs: Optional[Dict[str, Any]] = None,
    intermediate_values: Optional[Dict[int, float]] = None,
) -> FrozenTrial:
    """Create a new :class:`~optuna.trial.FrozenTrial`.

    Example:

        .. testcode::

            import optuna
            from optuna.distributions import CategoricalDistribution
            from optuna.distributions import FloatDistribution

            trial = optuna.trial.create_trial(
                params={"x": 1.0, "y": 0},
                distributions={
                    "x": FloatDistribution(0, 10),
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

    .. note::
        When ``state`` is :class:`TrialState.COMPLETE`, the following parameters are
        required:

        * ``params``
        * ``distributions``
        * ``value`` or ``values``

    Args:
        state:
            Trial state.
        value:
            Trial objective value. Must be specified if ``state`` is :class:`TrialState.COMPLETE`.
            ``value`` and ``values`` must not be specified at the same time.
        values:
            Sequence of the trial objective values. The length is greater than 1 if the problem is
            multi-objective optimization.
            Must be specified if ``state`` is :class:`TrialState.COMPLETE`.
            ``value`` and ``values`` must not be specified at the same time.
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
    distributions = {
        key: _convert_old_distribution_to_new_distribution(dist)
        for key, dist in distributions.items()
    }
    user_attrs = user_attrs or {}
    system_attrs = system_attrs or {}
    intermediate_values = intermediate_values or {}

    if state == TrialState.WAITING:
        datetime_start = None
    else:
        datetime_start = datetime.datetime.now()

    if state.is_finished():
        datetime_complete: Optional[datetime.datetime] = datetime_start
    else:
        datetime_complete = None

    trial = FrozenTrial(
        number=-1,
        trial_id=-1,
        state=state,
        value=value,
        values=values,
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
