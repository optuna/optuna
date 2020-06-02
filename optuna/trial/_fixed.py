import datetime
from typing import Optional

from optuna import distributions
from optuna.trial._base import BaseTrial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Sequence  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.distributions import CategoricalChoiceType  # NOQA
    from optuna.study import Study  # NOQA

    FloatingPointDistributionType = Union[
        distributions.UniformDistribution, distributions.LogUniformDistribution
    ]


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
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            assert objective(optuna.trial.FixedTrial({'x': 1, 'y': 0})) == 1


    .. note::
        Please refer to :class:`~optuna.trial.Trial` for details of methods and properties.

    Args:
        params:
            A dictionary containing all parameters.
        number:
            A trial number. Defaults to ``0``.

    """

    def __init__(self, params, number=0):
        # type: (Dict[str, Any], int) -> None

        self._params = params
        self._suggested_params = {}  # type: Dict[str, Any]
        self._distributions = {}  # type: Dict[str, BaseDistribution]
        self._user_attrs = {}  # type: Dict[str, Any]
        self._system_attrs = {}  # type: Dict[str, Any]
        self._datetime_start = datetime.datetime.now()
        self._number = number

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
                raise NotImplementedError(
                    "The parameter `step` is not supported when `log` is True."
                )
            else:
                return self._suggest(
                    name, distributions.DiscreteUniformDistribution(low=low, high=high, q=step)
                )
        else:
            if log:
                return self._suggest(
                    name, distributions.LogUniformDistribution(low=low, high=high)
                )
            else:
                return self._suggest(name, distributions.UniformDistribution(low=low, high=high))

    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name, distributions.UniformDistribution(low=low, high=high))

    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        return self._suggest(name, distributions.LogUniformDistribution(low=low, high=high))

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        discrete = distributions.DiscreteUniformDistribution(low=low, high=high, q=q)
        return self._suggest(name, discrete)

    def suggest_int(self, name, low, high, step=1, log=False):
        # type: (str, int, int, int, bool) -> int
        if log:
            sample = self._suggest(
                name, distributions.IntLogUniformDistribution(low=low, high=high, step=step)
            )
        else:
            sample = self._suggest(
                name, distributions.IntUniformDistribution(low=low, high=high, step=step)
            )
        return int(sample)

    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType

        choices = tuple(choices)
        return self._suggest(name, distributions.CategoricalDistribution(choices=choices))

    def _suggest(self, name, distribution):
        # type: (str, BaseDistribution) -> Any

        if name not in self._params:
            raise ValueError(
                "The value of the parameter '{}' is not found. Please set it at "
                "the construction of the FixedTrial object.".format(name)
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

        self._suggested_params[name] = value
        self._distributions[name] = distribution

        return value

    def report(self, value, step):
        # type: (float, int) -> None

        pass

    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool

        return False

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self._user_attrs[key] = value

    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        self._system_attrs[key] = value

    @property
    def params(self):
        # type: () -> Dict[str, Any]

        return self._suggested_params

    @property
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]

        return self._distributions

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self._user_attrs

    @property
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        return self._system_attrs

    @property
    def datetime_start(self):
        # type: () -> Optional[datetime.datetime]

        return self._datetime_start

    @property
    def number(self) -> int:

        return self._number
