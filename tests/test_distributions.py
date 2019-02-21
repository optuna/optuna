import json
import pytest
from typing import Any  # NOQA
from typing import Dict  # NOQA

from optuna import distributions

EXAMPLE_DISTRIBUTIONS = {
    'u': distributions.UniformDistribution(low=1., high=2.),
    'l': distributions.LogUniformDistribution(low=0.001, high=100),
    'du': distributions.DiscreteUniformDistribution(low=1., high=10., q=2.),
    'iu': distributions.IntUniformDistribution(low=1, high=10),
    'c1': distributions.CategoricalDistribution(choices=(2.71, -float('inf'))),
    'c2': distributions.CategoricalDistribution(choices=('Roppongi', 'Azabu'))
}  # type: Dict[str, Any]

EXAMPLE_JSONS = {
    'u': '{"name": "UniformDistribution", "attributes": {"low": 1.0, "high": 2.0}}',
    'l': '{"name": "LogUniformDistribution", "attributes": {"low": 0.001, "high": 100}}',
    'du': '{"name": "DiscreteUniformDistribution",'
    '"attributes": {"low": 1.0, "high": 10.0, "q": 2.0}}',
    'iu': '{"name": "IntUniformDistribution", "attributes": {"low": 1, "high": 10}}',
    'c1': '{"name": "CategoricalDistribution", "attributes": {"choices": [2.71, -Infinity]}}',
    'c2': '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}'
}


def test_json_to_distribution():
    # type: () -> None

    for key in EXAMPLE_JSONS.keys():
        distribution_actual = distributions.json_to_distribution(EXAMPLE_JSONS[key])
        assert distribution_actual == EXAMPLE_DISTRIBUTIONS[key]

    unknown_json = '{"name": "UnknownDistribution", "attributes": {"low": 1.0, "high": 2.0}}'
    pytest.raises(ValueError, lambda: distributions.json_to_distribution(unknown_json))


def test_distribution_to_json():
    # type: () -> None

    for key in EXAMPLE_JSONS.keys():
        json_actual = distributions.distribution_to_json(EXAMPLE_DISTRIBUTIONS[key])
        assert json.loads(json_actual) == json.loads(EXAMPLE_JSONS[key])


def test_check_distribution_compatibility():
    # type: () -> None

    # test the same distribution
    for key in EXAMPLE_JSONS.keys():
        distributions.check_distribution_compatibility(EXAMPLE_DISTRIBUTIONS[key],
                                                       EXAMPLE_DISTRIBUTIONS[key])

    # test different distribution classes
    pytest.raises(
        ValueError, lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS['u'], EXAMPLE_DISTRIBUTIONS['l']))

    # test dynamic value range (CategoricalDistribution)
    pytest.raises(
        ValueError, lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS['c2'], EXAMPLE_DISTRIBUTIONS['c2']._replace(
                choice=('Roppongi', 'Akasaka'))))

    # test dynamic value range (except CategoricalDistribution)
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS['u'], EXAMPLE_DISTRIBUTIONS['u']._replace(low=-1.0, high=-2.0))
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS['l'], EXAMPLE_DISTRIBUTIONS['l']._replace(low=-0.1, high=1.0))
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS['du'], EXAMPLE_DISTRIBUTIONS['du']._replace(
            low=-1.0, high=10.0, q=3.))
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS['iu'], EXAMPLE_DISTRIBUTIONS['iu']._replace(low=-1, high=1))
