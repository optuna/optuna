from typing import Any
from typing import List
from typing import NamedTuple


class BaseDistribution(object):

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        return param_value_in_internal_repr


class UniformDistribution(
    BaseDistribution, NamedTuple(
        '_BaseUniformDistribution',
        [('low', float), ('high', float)])):
    pass


class LogUniformDistribution(
    BaseDistribution, NamedTuple(
        '_BaseLogUniformDistribution',
        [('low', float), ('high', float)])):
    pass


class CategoricalDistribution(
    BaseDistribution, NamedTuple(
        '_BaseCategoricalDistribution',
        [('choices', List[Any])])):

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        return self.choices[int(param_value_in_internal_repr)]
