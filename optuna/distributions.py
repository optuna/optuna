import abc
import copy
import decimal
import json
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Union
import warnings


CategoricalChoiceType = Union[None, bool, int, float, str]


class BaseDistribution(object, metaclass=abc.ABCMeta):
    """Base class for distributions.

    Note that distribution classes are not supposed to be called by library users.
    They are used by :class:`~optuna.trial.Trial` and :class:`~optuna.samplers` internally.
    """

    def to_external_repr(self, param_value_in_internal_repr: float) -> Any:
        """Convert internal representation of a parameter value into external representation.

        Args:
            param_value_in_internal_repr:
                Optuna's internal representation of a parameter value.

        Returns:
            Optuna's external representation of a parameter value.
        """

        return param_value_in_internal_repr

    def to_internal_repr(self, param_value_in_external_repr: Any) -> float:
        """Convert external representation of a parameter value into internal representation.

        Args:
            param_value_in_external_repr:
                Optuna's external representation of a parameter value.

        Returns:
            Optuna's internal representation of a parameter value.
        """

        return param_value_in_external_repr

    @abc.abstractmethod
    def single(self) -> bool:
        """Test whether the range of this distribution contains just a single value.

        Returns:
            :obj:`True` if the range of this distribution contains just a single value,
            otherwise :obj:`False`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _contains(self, param_value_in_internal_repr: float) -> bool:
        """Test if a parameter value is contained in the range of this distribution.

        Args:
            param_value_in_internal_repr:
                Optuna's internal representation of a parameter value.

        Returns:
            :obj:`True` if the parameter value is contained in the range of this distribution,
            otherwise :obj:`False`.
        """

        raise NotImplementedError

    def _asdict(self) -> Dict:

        return self.__dict__

    def __eq__(self, other: Any) -> bool:

        if not isinstance(other, BaseDistribution):
            return NotImplemented
        if not type(self) is type(other):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:

        return hash((self.__class__,) + tuple(sorted(self.__dict__.items())))

    def __repr__(self) -> str:

        kwargs = ", ".join("{}={}".format(k, v) for k, v in sorted(self.__dict__.items()))
        return "{}({})".format(self.__class__.__name__, kwargs)


class UniformDistribution(BaseDistribution):
    """A uniform distribution in the linear domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_uniform`, and passed to
    :mod:`~optuna.samplers` in general.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is excluded from the range.

    Raises:
        ValueError:
            If ``low`` value is larger than ``high`` value.
    """

    def __init__(self, low: float, high: float) -> None:

        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )

        self.low = low
        self.high = high

    def single(self) -> bool:

        return self.low == self.high

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        value = param_value_in_internal_repr
        if self.low == self.high:
            return value == self.low
        else:
            return self.low <= value < self.high


class LogUniformDistribution(BaseDistribution):
    """A uniform distribution in the log domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_loguniform`, and passed to
    :mod:`~optuna.samplers` in general.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is excluded from the range.

    Raises:
        ValueError:
            If ``low`` value is larger than ``high`` value, or ``low`` value is smaller than or
            equal to 0.
    """

    def __init__(self, low: float, high: float) -> None:

        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )
        if low <= 0.0:
            raise ValueError(
                "The `low` value must be larger than 0 for a log distribution "
                "(low={}, high={}).".format(low, high)
            )

        self.low = low
        self.high = high

    def single(self) -> bool:

        return self.low == self.high

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        value = param_value_in_internal_repr
        if self.low == self.high:
            return value == self.low
        else:
            return self.low <= value < self.high


class DiscreteUniformDistribution(BaseDistribution):
    """A discretized uniform distribution in the linear domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_discrete_uniform`, and passed
    to :mod:`~optuna.samplers` in general.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by :math:`q`,
        :math:`\\mathsf{high}` will be replaced with the maximum of :math:`k q + \\mathsf{low}
        < \\mathsf{high}`, where :math:`k` is an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
        q:
            A discretization step.

    Raises:
        ValueError:
            If ``low`` value is larger than ``high`` value.
    """

    def __init__(self, low: float, high: float, q: float) -> None:
        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}, q={}).".format(low, high, q)
            )

        high = _adjust_discrete_uniform_high(low, high, q)

        self.low = low
        self.high = high
        self.q = q

    def single(self) -> bool:

        if self.low == self.high:
            return True
        high = decimal.Decimal(str(self.high))
        low = decimal.Decimal(str(self.low))
        q = decimal.Decimal(str(self.q))
        if (high - low) < q:
            return True
        return False

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        value = param_value_in_internal_repr
        return self.low <= value <= self.high


class IntUniformDistribution(BaseDistribution):
    """A uniform distribution on integers.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by
        :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced with the maximum of
        :math:`k \\times \\mathsf{step} + \\mathsf{low} < \\mathsf{high}`, where :math:`k` is
        an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
        step:
            A step for spacing between values.

    Raises:
        ValueError:
            If ``low`` value is larger than ``high`` value, or ``step`` value is smaller or
            equal to 0.
    """

    def __init__(self, low: int, high: int, step: int = 1) -> None:
        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )
        if step <= 0:
            raise ValueError(
                "The `step` value must be non-zero positive value, but step={}.".format(step)
            )

        high = _adjust_int_uniform_high(low, high, step)

        self.low = low
        self.high = high
        self.step = step

    def to_external_repr(self, param_value_in_internal_repr: float) -> int:

        return int(param_value_in_internal_repr)

    def to_internal_repr(self, param_value_in_external_repr: int) -> float:

        return float(param_value_in_external_repr)

    def single(self) -> bool:

        if self.low == self.high:
            return True
        return (self.high - self.low) < self.step

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        value = param_value_in_internal_repr
        return self.low <= value <= self.high


class IntLogUniformDistribution(BaseDistribution):
    """A uniform distribution on integers in the log domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
        step:
            A step for spacing between values.

            .. warning::
                Deprecated in v2.0.0. ``step`` argument will be removed in the future.
                The removal of this feature is currently scheduled for v4.0.0,
                but this schedule is subject to change.

                Samplers and other components in Optuna relying on this distribution will ignore
                this value and assume that ``step`` is always 1.
                User-defined samplers may continue to use other values besides 1 during the
                deprecation.

    Raises:
        ValueError:
            If ``low`` value is larger than ``high`` value, or ``low`` value is smaller than 1.
    """

    def __init__(self, low: int, high: int, step: int = 1) -> None:
        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )

        if low < 1.0:
            raise ValueError(
                "The `low` value must be equal to or greater than 1 for a log distribution "
                "(low={}, high={}).".format(low, high)
            )

        if step != 1:
            self._warn_step()

        self.low = low
        self.high = high
        self._step = step

    def __repr__(self) -> str:
        # TODO(hvy): `BaseDistribution.__repr__` could rely on `_asdict` instead of `__dict__`.
        # `IntLogUniformDistribution` would not have to override `__repr__`.
        kwargs = ", ".join("{}={}".format(k, v) for k, v in sorted(self._asdict().items()))
        return "{}({})".format(self.__class__.__name__, kwargs)

    def _asdict(self) -> Dict:
        d = copy.copy(self.__dict__)
        d["step"] = d.pop("_step")
        return d

    def _warn_step(self) -> None:
        warnings.warn(
            "Samplers and other components in Optuna will assume that `step` is 1. "
            "`step` argument is deprecated and will be removed in the future. "
            "The removal of this feature is currently scheduled for v4.0.0, "
            "but this schedule is subject to change.",
            FutureWarning,
        )

    def to_external_repr(self, param_value_in_internal_repr: float) -> int:
        return int(param_value_in_internal_repr)

    def to_internal_repr(self, param_value_in_external_repr: int) -> float:
        return float(param_value_in_external_repr)

    def single(self) -> bool:
        return self.low == self.high

    def _contains(self, param_value_in_internal_repr: float) -> bool:
        value = param_value_in_internal_repr
        return self.low <= value <= self.high

    @property
    def step(self) -> int:
        self._warn_step()
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._warn_step()
        self._step = value


class CategoricalDistribution(BaseDistribution):
    """A categorical distribution.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_categorical`, and
    passed to :mod:`~optuna.samplers` in general.

    Args:
        choices:
            Parameter value candidates.

    .. note::

        Not all types are guaranteed to be compatible with all storages. It is recommended to
        restrict the types of the choices to :obj:`None`, :class:`bool`, :class:`int`,
        :class:`float` and :class:`str`.

    Attributes:
        choices:
            Parameter value candidates.

    Raises:
        ValueError:
            If ``choices`` do not contain any elements.
    """

    def __init__(self, choices: Sequence[CategoricalChoiceType]) -> None:

        if len(choices) == 0:
            raise ValueError("The `choices` must contains one or more elements.")
        for choice in choices:
            if choice is not None and not isinstance(choice, (bool, int, float, str)):
                message = (
                    "Choices for a categorical distribution should be a tuple of None, bool, "
                    "int, float and str for persistent storage but contains {} which is of type "
                    "{}.".format(choice, type(choice).__name__)
                )
                warnings.warn(message)

        self.choices = tuple(choices)

    def to_external_repr(self, param_value_in_internal_repr: float) -> CategoricalChoiceType:

        return self.choices[int(param_value_in_internal_repr)]

    def to_internal_repr(self, param_value_in_external_repr: CategoricalChoiceType) -> float:

        try:
            return self.choices.index(param_value_in_external_repr)
        except ValueError as e:
            raise ValueError(
                "'{}' not in {}.".format(param_value_in_external_repr, self.choices)
            ) from e

    def single(self) -> bool:

        return len(self.choices) == 1

    def _contains(self, param_value_in_internal_repr: float) -> bool:

        index = int(param_value_in_internal_repr)
        return 0 <= index < len(self.choices)


DISTRIBUTION_CLASSES = (
    UniformDistribution,
    LogUniformDistribution,
    DiscreteUniformDistribution,
    IntUniformDistribution,
    IntLogUniformDistribution,
    CategoricalDistribution,
)


def json_to_distribution(json_str: str) -> BaseDistribution:
    """Deserialize a distribution in JSON format.

    Args:
        json_str: A JSON-serialized distribution.

    Returns:
        A deserialized distribution.

    Raises:
        ValueError:
            If the unknown class is specified.
    """

    json_dict = json.loads(json_str)

    if json_dict["name"] == CategoricalDistribution.__name__:
        json_dict["attributes"]["choices"] = tuple(json_dict["attributes"]["choices"])

    for cls in DISTRIBUTION_CLASSES:
        if json_dict["name"] == cls.__name__:
            return cls(**json_dict["attributes"])

    raise ValueError("Unknown distribution class: {}".format(json_dict["name"]))


def distribution_to_json(dist: BaseDistribution) -> str:
    """Serialize a distribution to JSON format.

    Args:
        dist: A distribution to be serialized.

    Returns:
        A JSON string of a given distribution.

    """

    return json.dumps({"name": dist.__class__.__name__, "attributes": dist._asdict()})


def check_distribution_compatibility(
    dist_old: BaseDistribution, dist_new: BaseDistribution
) -> None:
    """A function to check compatibility of two distributions.

    Note that this method is not supposed to be called by library users.

    Args:
        dist_old: A distribution previously recorded in storage.
        dist_new: A distribution newly added to storage.

    Returns:
        True denotes given distributions are compatible. Otherwise, they are not.

    Raises:
        ValueError:
            If different distribution kinds are set to ``dist_old`` and ``dist_new``,
            or ``dist_old.choices`` doesn't match ``dist_new.choices``
            for :class:`~optuna.distributions.CategoricalDistribution`.
    """

    if dist_old.__class__ != dist_new.__class__:
        raise ValueError("Cannot set different distribution kind to the same parameter name.")

    if not isinstance(dist_old, CategoricalDistribution):
        return
    if not isinstance(dist_new, CategoricalDistribution):
        return
    if dist_old.choices != dist_new.choices:
        raise ValueError(
            CategoricalDistribution.__name__ + " does not support dynamic value space."
        )


def _adjust_discrete_uniform_high(low: float, high: float, q: float) -> float:
    d_high = decimal.Decimal(str(high))
    d_low = decimal.Decimal(str(low))
    d_q = decimal.Decimal(str(q))

    d_r = d_high - d_low

    if d_r % d_q != decimal.Decimal("0"):
        old_high = high
        high = float((d_r // d_q) * d_q + d_low)
        warnings.warn(
            "The distribution is specified by [{low}, {old_high}] and q={step}, but the range "
            "is not divisible by `q`. It will be replaced by [{low}, {high}].".format(
                low=low, old_high=old_high, high=high, step=q
            )
        )

    return high


def _adjust_int_uniform_high(low: int, high: int, step: int) -> int:
    r = high - low
    if r % step != 0:
        old_high = high
        high = r // step * step + low
        warnings.warn(
            "The distribution is specified by [{low}, {old_high}] and step={step}, but the range "
            "is not divisible by `step`. It will be replaced by [{low}, {high}].".format(
                low=low, old_high=old_high, high=high, step=step
            )
        )
    return high


def _get_single_value(distribution: BaseDistribution) -> Union[int, float, CategoricalChoiceType]:

    assert distribution.single()

    if isinstance(
        distribution,
        (
            UniformDistribution,
            LogUniformDistribution,
            DiscreteUniformDistribution,
            IntUniformDistribution,
            IntLogUniformDistribution,
        ),
    ):
        return distribution.low
    elif isinstance(distribution, CategoricalDistribution):
        return distribution.choices[0]
    assert False
