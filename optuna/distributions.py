import abc
import decimal
import json
import warnings

from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Sequence  # NOQA
    from typing import Union  # NOQA

    CategoricalChoiceType = Union[None, bool, int, float, str]


class BaseDistribution(object, metaclass=abc.ABCMeta):
    """Base class for distributions.

    Note that distribution classes are not supposed to be called by library users.
    They are used by :class:`~optuna.trial.Trial` and :class:`~optuna.samplers` internally.
    """

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        """Convert internal representation of a parameter value into external representation.

        Args:
            param_value_in_internal_repr:
                Optuna's internal representation of a parameter value.

        Returns:
            Optuna's external representation of a parameter value.
        """

        return param_value_in_internal_repr

    def to_internal_repr(self, param_value_in_external_repr):
        # type: (Any) -> float
        """Convert external representation of a parameter value into internal representation.

        Args:
            param_value_in_external_repr:
                Optuna's external representation of a parameter value.

        Returns:
            Optuna's internal representation of a parameter value.
        """

        return param_value_in_external_repr

    @abc.abstractmethod
    def single(self):
        # type: () -> bool
        """Test whether the range of this distribution contains just a single value.

        When this method returns :obj:`True`, :mod:`~optuna.samplers` always sample
        the same value from the distribution.

        Returns:
            :obj:`True` if the range of this distribution contains just a single value,
            otherwise :obj:`False`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool
        """Test if a parameter value is contained in the range of this distribution.

        Args:
            param_value_in_internal_repr:
                Optuna's internal representation of a parameter value.

        Returns:
            :obj:`True` if the parameter value is contained in the range of this distribution,
            otherwise :obj:`False`.
        """

        raise NotImplementedError

    def _asdict(self):
        # type: () -> Dict

        return self.__dict__

    def __eq__(self, other):
        # type: (Any) -> bool

        if not isinstance(other, BaseDistribution):
            return NotImplemented
        if not type(self) is type(other):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):
        # type: () -> int

        return hash((self.__class__,) + tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        # type: () -> str

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
    """

    def __init__(self, low, high):
        # type: (float, float) -> None

        if low > high:
            raise ValueError(
                "The `low` value must be smaller than or equal to the `high` value "
                "(low={}, high={}).".format(low, high)
            )

        self.low = low
        self.high = high

    def single(self):
        # type: () -> bool

        return self.low == self.high

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

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
    """

    def __init__(self, low, high):
        # type: (float, float) -> None

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

    def single(self):
        # type: () -> bool

        return self.low == self.high

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

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
        \\lt \\mathsf{high}`, where :math:`k` is an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
        q:
            A discretization step.
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

    def single(self):
        # type: () -> bool

        if self.low == self.high:
            return True
        high = decimal.Decimal(str(self.high))
        low = decimal.Decimal(str(self.low))
        q = decimal.Decimal(str(self.q))
        if (high - low) < q:
            return True
        return False

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

        value = param_value_in_internal_repr
        return self.low <= value <= self.high


class IntUniformDistribution(BaseDistribution):
    """A uniform distribution on integers.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by
        :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced with the maximum of
        :math:`k \\times \\mathsf{step} + \\mathsf{low} \\lt \\mathsf{high}`, where :math:`k` is
        an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
        step:
            A step for spacing between values.
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

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> int

        return int(param_value_in_internal_repr)

    def to_internal_repr(self, param_value_in_external_repr):
        # type: (int) -> float

        return float(param_value_in_external_repr)

    def single(self):
        # type: () -> bool

        if self.low == self.high:
            return True
        return (self.high - self.low) < self.step

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

        value = param_value_in_internal_repr
        return self.low <= value <= self.high


class IntLogUniformDistribution(BaseDistribution):
    """A uniform distribution on integers in the log domain.

    This object is instantiated by :func:`~optuna.trial.Trial.suggest_int`, and passed to
    :mod:`~optuna.samplers` in general.

    .. note::
        If the range :math:`[\\mathsf{low}, \\mathsf{high}]` is not divisible by
        :math:`\\mathsf{step}`, :math:`\\mathsf{high}` will be replaced with the maximum of
        :math:`k \\times \\mathsf{step} + \\mathsf{low} \\lt \\mathsf{high}`, where :math:`k` is
        an integer.

    Attributes:
        low:
            Lower endpoint of the range of the distribution. ``low`` is included in the range.
        high:
            Upper endpoint of the range of the distribution. ``high`` is included in the range.
        step:
            A step for spacing between values.
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

        if low < 1.0:
            raise ValueError(
                "The `low` value must be equal to or greater than 1 for a log distribution "
                "(low={}, high={}).".format(low, high)
            )

        high = _adjust_int_uniform_high(low, high, step)

        self.low = low
        self.high = high
        self.step = step

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> int

        return int(param_value_in_internal_repr)

    def to_internal_repr(self, param_value_in_external_repr):
        # type: (int) -> float

        return float(param_value_in_external_repr)

    def single(self):
        # type: () -> bool

        if self.low == self.high:
            return True
        return (self.high - self.low) < self.step

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

        value = param_value_in_internal_repr
        return self.low <= value <= self.high


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
    """

    def __init__(self, choices):
        # type: (Sequence[CategoricalChoiceType]) -> None

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

        self.choices = choices

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> CategoricalChoiceType

        return self.choices[int(param_value_in_internal_repr)]

    def to_internal_repr(self, param_value_in_external_repr):
        # type: (CategoricalChoiceType) -> float

        try:
            return self.choices.index(param_value_in_external_repr)
        except ValueError as e:
            raise ValueError(
                "'{}' not in {}.".format(param_value_in_external_repr, self.choices)
            ) from e

    def single(self):
        # type: () -> bool

        return len(self.choices) == 1

    def _contains(self, param_value_in_internal_repr):
        # type: (float) -> bool

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


def json_to_distribution(json_str):
    # type: (str) -> BaseDistribution
    """Deserialize a distribution in JSON format.

    Args:
        json_str: A JSON-serialized distribution.

    Returns:
        A deserialized distribution.
    """

    json_dict = json.loads(json_str)

    if json_dict["name"] == CategoricalDistribution.__name__:
        json_dict["attributes"]["choices"] = tuple(json_dict["attributes"]["choices"])

    for cls in DISTRIBUTION_CLASSES:
        if json_dict["name"] == cls.__name__:
            return cls(**json_dict["attributes"])

    raise ValueError("Unknown distribution class: {}".format(json_dict["name"]))


def distribution_to_json(dist):
    # type: (BaseDistribution) -> str
    """Serialize a distribution to JSON format.

    Args:
        dist: A distribution to be serialized.

    Returns:
        A JSON string of a given distribution.

    """

    return json.dumps({"name": dist.__class__.__name__, "attributes": dist._asdict()})


def check_distribution_compatibility(dist_old, dist_new):
    # type: (BaseDistribution, BaseDistribution) -> None
    """A function to check compatibility of two distributions.

    Note that this method is not supposed to be called by library users.

    Args:
        dist_old: A distribution previously recorded in storage.
        dist_new: A distribution newly added to storage.

    Returns:
        True denotes given distributions are compatible. Otherwise, they are not.
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
