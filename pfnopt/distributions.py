from typing import Any
from typing import List
from typing import NamedTuple


class _BaseDistribution(object):

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        return param_value_in_internal_repr


class UniformDistribution(
    _BaseDistribution, NamedTuple(
        '_BaseUniformDistribution',
        [('low', float), ('high', float)])):
    pass


class LogUniformDistribution(
    _BaseDistribution, NamedTuple(
        '_BaseLogUniformDistribution',
        [('low', float), ('high', float)])):
    pass


class CategoricalDistribution(
    _BaseDistribution, NamedTuple(
        '_BaseCategoricalDistribution',
        [('choices', List[Any])])):

    def to_external_repr(self, param_value_in_internal_repr):
        return self.choices[int(param_value_in_internal_repr)]
