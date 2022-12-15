import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
import warnings

from optuna import distributions
from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial._base import BaseTrial


_suggest_deprecated_msg = "Use :func:`~optuna.trial.FixedTrial.suggest_float` instead."


class FixedTrial(BaseTrial):
    """A trial class which suggests a fixed value for each parameter.

    This object has the same methods as :class:`~optuna.trial.Trial`, and it suggests pre-defined
    parameter values. The parameter values can be determined at the construction of the
    :class:`~optuna.trial.FixedTrial` object. In contrast to :class:`~optuna.trial.Trial`,
    :class:`~optuna.trial.FixedTrial` does not depend on :class:`~optuna.study.Study`, and it is
    useful for deploying optimization results.

    Example:

        Evaluate an objective function with parameter values given by a user.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x**2 + y


            assert objective(optuna.trial.FixedTrial({"x": 1, "y": 0})) == 1


    .. note::
        Please refer to :class:`~optuna.trial.Trial` for details of methods and properties.

    Args:
        params:
            A dictionary containing all parameters.
        number:
            A trial number. Defaults to ``0``.

    """

    def __init__(self, params: Dict[str, Any], number: int = 0) -> None:

        self._params = params
        self._suggested_params: Dict[str, Any] = {}
        self._distributions: Dict[str, BaseDistribution] = {}
        self._user_attrs: Dict[str, Any] = {}
        self._system_attrs: Dict[str, Any] = {}
        self._datetime_start = datetime.datetime.now()
        self._number = number

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

        pass

    def should_prune(self) -> bool:

        return False

    def set_user_attr(self, key: str, value: Any) -> None:

        self._user_attrs[key] = value

    @deprecated_func("3.1.0", "6.0.0")
    def set_system_attr(self, key: str, value: Any) -> None:

        self._system_attrs[key] = value

    def _suggest(self, name: str, distribution: BaseDistribution) -> Any:

        if name not in self._params:
            raise ValueError(
                "The value of the parameter '{}' is not found. Please set it at "
                "the construction of the FixedTrial object.".format(name)
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

        self._suggested_params[name] = value
        self._distributions[name] = distribution

        return value

    @property
    def params(self) -> Dict[str, Any]:

        return self._suggested_params

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:

        return self._distributions

    @property
    def user_attrs(self) -> Dict[str, Any]:

        return self._user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:

        return self._system_attrs

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:

        return self._datetime_start

    @property
    def number(self) -> int:

        return self._number
