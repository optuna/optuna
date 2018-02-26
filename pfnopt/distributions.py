import json
from typing import Any
from typing import NamedTuple
from typing import Tuple


class BaseDistribution(object):

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        return param_value_in_internal_repr

    def to_internal_repr(self, param_value_in_external_repr):
        return param_value_in_external_repr

    def to_json(self):
        # type: () -> str
        raise NotImplementedError()

    @staticmethod
    def from_json(json_str):
        # type: (str) -> BaseDistribution
        raise NotImplementedError()


class UniformDistribution(
    BaseDistribution, NamedTuple(
        '_BaseUniformDistribution',
        [('low', float), ('high', float)])):

    def to_json(self):
        # type: () -> str
        return json.dumps({'name': self.__class__.__name__, 'attributes': self._asdict()})

    @staticmethod
    def from_json(json_str):
        # type: (str) -> UniformDistribution
        loaded = json.loads(json_str)
        assert loaded['name'] == UniformDistribution.__name__
        attributes = loaded['attributes']
        return UniformDistribution(low=attributes['low'], high=attributes['high'])


class LogUniformDistribution(
    BaseDistribution, NamedTuple(
        '_BaseLogUniformDistribution',
        [('low', float), ('high', float)])):

    def to_json(self):
        # type: () -> str
        return json.dumps({'name': self.__class__.__name__, 'attributes': self._asdict()})

    @staticmethod
    def from_json(json_str):
        # type: (str) -> LogUniformDistribution
        loaded = json.loads(json_str)
        assert loaded['name'] == LogUniformDistribution.__name__
        attributes = loaded['attributes']
        return LogUniformDistribution(low=attributes['low'], high=attributes['high'])


class CategoricalDistribution(
    BaseDistribution, NamedTuple(
        '_BaseCategoricalDistribution',
        [('choices', Tuple[Any])])):

    def to_json(self):
        # type: () -> str
        attributes = {'choices': list(self.choices)}
        return json.dumps({'name': self.__class__.__name__, 'attributes': attributes})

    @staticmethod
    def from_json(json_str):
        # type: (str) -> CategoricalDistribution
        loaded = json.loads(json_str)
        assert loaded['name'] == CategoricalDistribution.__name__
        return CategoricalDistribution(choices=tuple(loaded['attributes']['choices']))

    def to_external_repr(self, param_value_in_internal_repr):
        # type: (float) -> Any
        return self.choices[int(param_value_in_internal_repr)]

    def to_internal_repr(self, param_value_in_external_repr):
        return self.choices.index(param_value_in_external_repr)


def distribution_from_json(json_str):
    valid_classes = [UniformDistribution, LogUniformDistribution, CategoricalDistribution]

    loaded = json.loads(json_str)
    for cls in valid_classes:
        if loaded['name'] == cls.__name__:
            return cls.from_json(json_str)

    raise ValueError()
