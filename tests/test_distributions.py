import json
import pytest
from typing import Any  # NOQA
from typing import Dict  # NOQA

from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import check_distribution_compatibility
from pfnopt.distributions import distribution_to_json
from pfnopt.distributions import json_to_distribution
from pfnopt.distributions import LogUniformDistribution
from pfnopt.distributions import UniformDistribution

EXAMPLE_DISTRIBUTION = {
    'u': UniformDistribution(low=1., high=2.),
    'l': LogUniformDistribution(low=0.001, high=100),
    'c1': CategoricalDistribution(choices=(2.71, -float('inf'))),
    'c2': CategoricalDistribution(choices=('Roppongi', 'Azabu'))
}  # type: Dict[str, Any]

EXAMPLE_JSON = {
    'u': '{"name": "UniformDistribution", "attributes": {"low": 1.0, "high": 2.0}}',
    'l': '{"name": "LogUniformDistribution", "attributes": {"low": 0.001, "high": 100}}',
    'c1': '{"name": "CategoricalDistribution", "attributes": {"choices": [2.71, -Infinity]}}',
    'c2': '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}'
}


def test_json_to_distribution():
    # type: () -> None

    for key in EXAMPLE_JSON.keys():
        assert json_to_distribution(EXAMPLE_JSON[key]) == EXAMPLE_DISTRIBUTION[key]

    unknown_json = '{"name": "UnknownDistribution", "attributes": {"low": 1.0, "high": 2.0}}'
    pytest.raises(ValueError, lambda: json_to_distribution(unknown_json))


def test_distribution_to_json():
    # type: () -> None

    for key in EXAMPLE_JSON.keys():
        json_actual = distribution_to_json(EXAMPLE_DISTRIBUTION[key])
        assert json.loads(json_actual) == json.loads(EXAMPLE_JSON[key])


def test_check_distribution_compatibility():
    # type: () -> None

    # test the same distribution
    for key in EXAMPLE_JSON.keys():
        check_distribution_compatibility(EXAMPLE_DISTRIBUTION[key], EXAMPLE_DISTRIBUTION[key])

    # test different distribution classes
    pytest.raises(ValueError, lambda: check_distribution_compatibility(
        EXAMPLE_DISTRIBUTION['u'],
        EXAMPLE_DISTRIBUTION['l']))

    # test dynamic value range (CategoricalDistribution)
    pytest.raises(ValueError, lambda: check_distribution_compatibility(
        EXAMPLE_DISTRIBUTION['c2'],
        EXAMPLE_DISTRIBUTION['c2']._replace(choice=('Roppongi', 'Akasaka'))))

    # test dynamic value range (CategoricalDistribution)
    check_distribution_compatibility(
        EXAMPLE_DISTRIBUTION['u'],
        EXAMPLE_DISTRIBUTION['u']._replace(low=-1.0, high=-2.0))
    check_distribution_compatibility(
        EXAMPLE_DISTRIBUTION['l'],
        EXAMPLE_DISTRIBUTION['l']._replace(low=-0.1, high=1.0))
