import copy
import json
import warnings

import pytest

from optuna import distributions
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA

EXAMPLE_DISTRIBUTIONS = {
    "u": distributions.UniformDistribution(low=1.0, high=2.0),
    "l": distributions.LogUniformDistribution(low=0.001, high=100),
    "du": distributions.DiscreteUniformDistribution(low=1.0, high=9.0, q=2.0),
    "iu": distributions.IntUniformDistribution(low=1, high=9, step=2),
    "c1": distributions.CategoricalDistribution(choices=(2.71, -float("inf"))),
    "c2": distributions.CategoricalDistribution(choices=("Roppongi", "Azabu")),
    "ilu": distributions.IntLogUniformDistribution(low=2, high=12, step=2),
}  # type: Dict[str, Any]

EXAMPLE_JSONS = {
    "u": '{"name": "UniformDistribution", "attributes": {"low": 1.0, "high": 2.0}}',
    "l": '{"name": "LogUniformDistribution", "attributes": {"low": 0.001, "high": 100}}',
    "du": '{"name": "DiscreteUniformDistribution",'
    '"attributes": {"low": 1.0, "high": 9.0, "q": 2.0}}',
    "iu": '{"name": "IntUniformDistribution", "attributes": {"low": 1, "high": 9, "step": 2}}',
    "c1": '{"name": "CategoricalDistribution", "attributes": {"choices": [2.71, -Infinity]}}',
    "c2": '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}',
    "ilu": '{"name": "IntLogUniformDistribution",'
    '"attributes": {"low": 2, "high": 12, "step": 2}}',
}


def test_json_to_distribution():
    # type: () -> None

    for key in EXAMPLE_JSONS.keys():
        distribution_actual = distributions.json_to_distribution(EXAMPLE_JSONS[key])
        assert distribution_actual == EXAMPLE_DISTRIBUTIONS[key]

    unknown_json = '{"name": "UnknownDistribution", "attributes": {"low": 1.0, "high": 2.0}}'
    pytest.raises(ValueError, lambda: distributions.json_to_distribution(unknown_json))


def test_backward_compatibility_int_uniform_distribution():
    # type: () -> None

    json_str = '{"name": "IntUniformDistribution", "attributes": {"low": 1, "high": 10}}'
    actual = distributions.json_to_distribution(json_str)
    expected = distributions.IntUniformDistribution(low=1, high=10)
    assert actual == expected


def test_distribution_to_json():
    # type: () -> None

    for key in EXAMPLE_JSONS.keys():
        json_actual = distributions.distribution_to_json(EXAMPLE_DISTRIBUTIONS[key])
        assert json.loads(json_actual) == json.loads(EXAMPLE_JSONS[key])


def test_check_distribution_compatibility():
    # type: () -> None

    # test the same distribution
    for key in EXAMPLE_JSONS.keys():
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS[key], EXAMPLE_DISTRIBUTIONS[key]
        )

    # test different distribution classes
    pytest.raises(
        ValueError,
        lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["u"], EXAMPLE_DISTRIBUTIONS["l"]
        ),
    )

    # test dynamic value range (CategoricalDistribution)
    pytest.raises(
        ValueError,
        lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["c2"],
            distributions.CategoricalDistribution(choices=("Roppongi", "Akasaka")),
        ),
    )

    # test dynamic value range (except CategoricalDistribution)
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["u"], distributions.UniformDistribution(low=-3.0, high=-2.0)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["l"], distributions.LogUniformDistribution(low=0.1, high=1.0)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["du"],
        distributions.DiscreteUniformDistribution(low=-1.0, high=11.0, q=3.0),
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["iu"], distributions.IntUniformDistribution(low=-1, high=1)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["ilu"],
        distributions.IntLogUniformDistribution(low=1, high=13, step=1),
    )


def test_contains() -> None:
    u = distributions.UniformDistribution(low=1.0, high=2.0)
    assert not u._contains(0.9)
    assert u._contains(1)
    assert u._contains(1.5)
    assert not u._contains(2)

    lu = distributions.LogUniformDistribution(low=0.001, high=100)
    assert not lu._contains(0.0)
    assert lu._contains(0.001)
    assert lu._contains(12.3)
    assert not lu._contains(100)

    with warnings.catch_warnings():
        # UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [1.0, 9.0].
        warnings.simplefilter("ignore", category=UserWarning)
        du = distributions.DiscreteUniformDistribution(low=1.0, high=10.0, q=2.0)
    assert not du._contains(0.9)
    assert du._contains(1.0)
    assert du._contains(3.5)
    assert du._contains(6)
    assert du._contains(9)
    assert not du._contains(9.1)
    assert not du._contains(10)

    iu = distributions.IntUniformDistribution(low=1, high=10)
    assert not iu._contains(0.9)
    assert iu._contains(1)
    assert iu._contains(4)
    assert iu._contains(6)
    assert iu._contains(10)
    assert not iu._contains(10.1)
    assert not iu._contains(11)

    # IntUniformDistribution with a 'step' parameter.
    with warnings.catch_warnings():
        # UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [1, 9].
        warnings.simplefilter("ignore", category=UserWarning)
        iuq = distributions.IntUniformDistribution(low=1, high=10, step=2)
    assert not iuq._contains(0.9)
    assert iuq._contains(1)
    assert iuq._contains(4)
    assert iuq._contains(6)
    assert iuq._contains(9)
    assert not iuq._contains(9.1)
    assert not iuq._contains(10)

    c = distributions.CategoricalDistribution(choices=("Roppongi", "Azabu"))
    assert not c._contains(-1)
    assert c._contains(0)
    assert c._contains(1)
    assert c._contains(1.5)
    assert not c._contains(3)

    ilu = distributions.IntUniformDistribution(low=2, high=12)
    assert not ilu._contains(0.9)
    assert ilu._contains(2)
    assert ilu._contains(4)
    assert ilu._contains(6)
    assert ilu._contains(12)
    assert not ilu._contains(12.1)
    assert not ilu._contains(13)

    # IntLogUniformDistribution with a 'step' parameter.
    with warnings.catch_warnings():
        # UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [2, 6].
        warnings.simplefilter("ignore", category=UserWarning)
        iluq = distributions.IntLogUniformDistribution(low=2, high=7, step=2)
    assert not iluq._contains(0.9)
    assert iluq._contains(2)
    assert iluq._contains(4)
    assert iluq._contains(5)
    assert iluq._contains(6)
    assert not iluq._contains(6.1)
    assert not iluq._contains(7)


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

    iuq = distributions.IntUniformDistribution(low=1, high=1, step=2)
    assert not iuq._contains(0)
    assert iuq._contains(1)
    assert not iuq._contains(2)

    ilu = distributions.IntUniformDistribution(low=1, high=1)
    assert not ilu._contains(0)
    assert ilu._contains(1)
    assert not ilu._contains(2)

    iluq = distributions.IntUniformDistribution(low=1, high=1, step=2)
    assert not iluq._contains(0)
    assert iluq._contains(1)
    assert not iluq._contains(2)


def test_single():
    # type: () -> None

    with warnings.catch_warnings():
        # UserWarning will be raised since the range is not divisible by step.
        warnings.simplefilter("ignore", category=UserWarning)
        single_distributions = [
            distributions.UniformDistribution(low=1.0, high=1.0),
            distributions.LogUniformDistribution(low=7.3, high=7.3),
            distributions.DiscreteUniformDistribution(low=2.22, high=2.22, q=0.1),
            distributions.DiscreteUniformDistribution(low=2.22, high=2.24, q=0.3),
            distributions.IntUniformDistribution(low=-123, high=-123),
            distributions.IntUniformDistribution(low=-123, high=-120, step=4),
            distributions.CategoricalDistribution(choices=("foo",)),
            distributions.IntLogUniformDistribution(low=2, high=2),
            distributions.IntLogUniformDistribution(low=2, high=2, step=2),
        ]  # type: List[distributions.BaseDistribution]
    for distribution in single_distributions:
        assert distribution.single()

    nonsingle_distributions = [
        distributions.UniformDistribution(low=1.0, high=1.001),
        distributions.LogUniformDistribution(low=7.3, high=10),
        distributions.DiscreteUniformDistribution(low=-30, high=-20, q=2),
        distributions.DiscreteUniformDistribution(low=-30, high=-20, q=10),
        # In Python, "0.3 - 0.2 != 0.1" is True.
        distributions.DiscreteUniformDistribution(low=0.2, high=0.3, q=0.1),
        distributions.DiscreteUniformDistribution(low=0.7, high=0.8, q=0.1),
        distributions.IntUniformDistribution(low=-123, high=0),
        distributions.IntUniformDistribution(low=-123, high=0, step=123),
        distributions.CategoricalDistribution(choices=("foo", "bar")),
        distributions.IntLogUniformDistribution(low=2, high=4),
        distributions.IntLogUniformDistribution(low=2, high=4, step=2),
    ]  # type: List[distributions.BaseDistribution]
    for distribution in nonsingle_distributions:
        assert not distribution.single()


def test_empty_distribution():
    # type: () -> None

    # Empty distributions cannot be instantiated.
    with pytest.raises(ValueError):
        distributions.UniformDistribution(low=0.0, high=-100.0)

    with pytest.raises(ValueError):
        distributions.LogUniformDistribution(low=7.3, high=7.2)

    with pytest.raises(ValueError):
        distributions.DiscreteUniformDistribution(low=-30, high=-40, q=3)

    with pytest.raises(ValueError):
        distributions.IntUniformDistribution(low=123, high=100)

    with pytest.raises(ValueError):
        distributions.IntUniformDistribution(low=123, high=100, step=2)

    with pytest.raises(ValueError):
        distributions.CategoricalDistribution(choices=())

    with pytest.raises(ValueError):
        distributions.IntLogUniformDistribution(low=123, high=100)

    with pytest.raises(ValueError):
        distributions.IntLogUniformDistribution(low=123, high=100, step=2)


def test_invalid_distribution():
    # type: () -> None

    with pytest.warns(UserWarning):
        distributions.CategoricalDistribution(choices=({"foo": "bar"},))  # type: ignore


def test_eq_ne_hash():
    # type: () -> None

    # Two instances of a class are regarded as equivalent if the fields have the same values.
    for d in EXAMPLE_DISTRIBUTIONS.values():
        d_copy = copy.deepcopy(d)
        assert d == d_copy
        assert not d != d_copy
        assert hash(d) == hash(d_copy)

    # Different field values.
    d0 = distributions.UniformDistribution(low=1, high=2)
    d1 = distributions.UniformDistribution(low=1, high=3)
    assert d0 != d1
    assert not d0 == d1
    assert hash(d0) != hash(d1)

    # Different distribution classes.
    d2 = distributions.IntUniformDistribution(low=1, high=2)
    assert d0 != d2
    assert not d0 == d2
    assert hash(d0) != hash(d2)

    # Different types.
    assert d0 != 1
    assert not d0 == 1
    assert d0 != "foo"
    assert not d0 == "foo"


def test_repr():
    # type: () -> None

    # The following variable is needed to apply `eval` to distribution
    # instances that contain `float('inf')` as a field value.
    inf = float("inf")  # NOQA

    for d in EXAMPLE_DISTRIBUTIONS.values():
        assert d == eval("distributions." + repr(d))


def test_uniform_distribution_asdict():
    # type: () -> None

    assert EXAMPLE_DISTRIBUTIONS["u"]._asdict() == {"low": 1.0, "high": 2.0}


def test_log_uniform_distribution_asdict():
    # type: () -> None

    assert EXAMPLE_DISTRIBUTIONS["l"]._asdict() == {"low": 0.001, "high": 100}


def test_discrete_uniform_distribution_asdict():
    # type: () -> None

    assert EXAMPLE_DISTRIBUTIONS["du"]._asdict() == {"low": 1.0, "high": 9.0, "q": 2.0}


def test_int_uniform_distribution_asdict():
    # type: () -> None

    assert EXAMPLE_DISTRIBUTIONS["iu"]._asdict() == {"low": 1, "high": 9, "step": 2}


def test_int_log_uniform_distribution_asdict():
    # type: () -> None

    assert EXAMPLE_DISTRIBUTIONS["ilu"]._asdict() == {"low": 2, "high": 12, "step": 2}
