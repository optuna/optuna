import abc
import json
import six
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import NamedTuple
from typing import Tuple
from typing import Union


@six.add_metaclass(abc.ABCMeta)
class BaseDistribution(object):

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        return param_value_in_internal_repr

    def to_internal_repr(self, param_value_in_external_repr):
        # type: (Any) -> float
        return param_value_in_external_repr

    @abc.abstractmethod
    def _asdict(self):
        # type: () -> Dict
        raise NotImplementedError


class UniformDistribution(
    NamedTuple(
        '_BaseUniformDistribution',
        [('low', float), ('high', float)]), BaseDistribution):
    pass


class LogUniformDistribution(
    NamedTuple(
        '_BaseLogUniformDistribution',
        [('low', float), ('high', float)]), BaseDistribution):
    pass


class CategoricalDistribution(
    NamedTuple(
        '_BaseCategoricalDistribution',
        [('choices', Tuple[Union[float, str]])]), BaseDistribution):

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Union[float, str]
        return self.choices[int(param_value_in_internal_repr)]

    def to_internal_repr(self, param_value_in_external_repr):
        return self.choices.index(param_value_in_external_repr)


DISTRIBUTION_CLASSES = (UniformDistribution, LogUniformDistribution, CategoricalDistribution)


def json_to_distribution(json_str):
    # type: (str) -> BaseDistribution
    loaded = json.loads(json_str)

    if loaded['name'] == CategoricalDistribution.__name__:
        loaded['attributes']['choices'] = tuple(loaded['attributes']['choices'])

    for cls in DISTRIBUTION_CLASSES:
        if loaded['name'] == cls.__name__:
            return cls(**loaded['attributes'])

    raise ValueError('Unknown distribution class: {}'.format(loaded['name']))
