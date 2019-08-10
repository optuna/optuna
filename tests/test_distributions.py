import json
import pytest

from optuna import distributions
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA

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


def test_contains():
    # type: () -> None

    u = distributions.UniformDistribution(low=1., high=2.)
    assert not u._contains(0.9)
    assert u._contains(1)
    assert u._contains(1.5)
    assert not u._contains(2)

    lu = distributions.LogUniformDistribution(low=0.001, high=100)
    assert not lu._contains(0.0)
    assert lu._contains(0.001)
    assert lu._contains(12.3)
    assert not lu._contains(100)

    du = distributions.DiscreteUniformDistribution(low=1., high=10., q=2.)
    assert not du._contains(0.9)
    assert du._contains(1.0)
    assert du._contains(3.5)
    assert du._contains(6)
    assert du._contains(10)
    assert not du._contains(10.1)

    iu = distributions.IntUniformDistribution(low=1, high=10)
    assert not iu._contains(0.9)
    assert iu._contains(1)
    assert iu._contains(3.5)
    assert iu._contains(6)
    assert iu._contains(10)
    assert iu._contains(10.1)
    assert not iu._contains(11)

    c = distributions.CategoricalDistribution(choices=('Roppongi', 'Azabu'))
    assert not c._contains(-1)
    assert c._contains(0)
    assert c._contains(1)
    assert c._contains(1.5)
    assert not c._contains(3)


def test_empty_range_contains():
    # type: () -> None

    u = distributions.UniformDistribution(low=1.0, high=1.0)
    assert not u._contains(0.9)
    assert u._contains(1.0)
    assert not u._contains(1.1)

    lu = distributions.LogUniformDistribution(low=1.0, high=1.0)
    assert not lu._contains(0.9)
    assert lu._contains(1.0)
    assert not lu._contains(1.1)

    du = distributions.DiscreteUniformDistribution(low=1.0, high=1.0, q=2.0)
    assert not du._contains(0.9)
    assert du._contains(1.0)
    assert not du._contains(1.1)

    iu = distributions.IntUniformDistribution(low=1, high=1)
    assert not iu._contains(0)
    assert iu._contains(1)
    assert not iu._contains(2)


def test_single():
    # type: () -> None

    single_distributions = [
        distributions.UniformDistribution(low=1.0, high=1.0),
        distributions.LogUniformDistribution(low=7.3, high=7.3),
        distributions.DiscreteUniformDistribution(low=2.22, high=2.22, q=0.1),
        distributions.IntUniformDistribution(low=-123, high=-123),
        distributions.CategoricalDistribution(choices=('foo', ))
    ]  # type: List[distributions.BaseDistribution]
    for distribution in single_distributions:
        assert distribution.single()

    nonsingle_distributions = [
        distributions.UniformDistribution(low=0.0, high=-100.0),
        distributions.UniformDistribution(low=1.0, high=1.001),
        distributions.LogUniformDistribution(low=7.3, high=7.2),
        distributions.LogUniformDistribution(low=7.3, high=10),
        distributions.DiscreteUniformDistribution(low=-30, high=-40, q=3),
        distributions.DiscreteUniformDistribution(low=-30, high=-20, q=2),
        distributions.IntUniformDistribution(low=123, high=100),
        distributions.IntUniformDistribution(low=-123, high=0),
        distributions.CategoricalDistribution(choices=()),
        distributions.CategoricalDistribution(choices=('foo', 'bar'))
    ]  # type: List[distributions.BaseDistribution]
    for distribution in nonsingle_distributions:
        assert not distribution.single()
