import copy
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import warnings

import pytest

from optuna import distributions


EXAMPLE_DISTRIBUTIONS: Dict[str, Any] = {
    "i": distributions.IntDistribution(low=1, high=9, log=False),
    "il": distributions.IntDistribution(low=2, high=12, log=True),
    "id": distributions.IntDistribution(low=1, high=9, log=False, step=2),
    "f": distributions.FloatDistribution(low=1.0, high=2.0, log=False),
    "fl": distributions.FloatDistribution(low=0.001, high=100.0, log=True),
    "fd": distributions.FloatDistribution(low=1.0, high=9.0, log=False, step=2.0),
    "u": distributions.UniformDistribution(low=1.0, high=2.0),
    "l": distributions.LogUniformDistribution(low=0.001, high=100),
    "du": distributions.DiscreteUniformDistribution(low=1.0, high=9.0, q=2.0),
    "iu": distributions.IntUniformDistribution(low=1, high=9),
    "iuq": distributions.IntUniformDistribution(low=1, high=9, step=2),
    "c1": distributions.CategoricalDistribution(choices=(2.71, -float("inf"))),
    "c2": distributions.CategoricalDistribution(choices=("Roppongi", "Azabu")),
    "c3": distributions.CategoricalDistribution(choices=["Roppongi", "Azabu"]),
    "ilu": distributions.IntLogUniformDistribution(low=2, high=12),
}

EXAMPLE_JSONS = {
    "i": '{"name": "IntDistribution", '
    '"attributes": {"low": 1, "high": 9, "log": false, "step": 1}}',
    "il": '{"name": "IntDistribution", '
    '"attributes": {"low": 2, "high": 12, "log": true, "step": 1}}',
    "id": '{"name": "IntDistribution", '
    '"attributes": {"low": 1, "high": 9, "log": false, "step": 2}}',
    "f": '{"name": "FloatDistribution", '
    '"attributes": {"low": 1.0, "high": 2.0, "log": false, "step": null}}',
    "fl": '{"name": "FloatDistribution", '
    '"attributes": {"low": 0.001, "high": 100.0, "log": true, "step": null}}',
    "fd": '{"name": "FloatDistribution", '
    '"attributes": {"low": 1.0, "high": 9.0, "step": 2.0, "log": false}}',
    "u": '{"name": "UniformDistribution", "attributes": {"low": 1.0, "high": 2.0}}',
    "l": '{"name": "LogUniformDistribution", "attributes": {"low": 0.001, "high": 100}}',
    "du": '{"name": "DiscreteUniformDistribution",'
    '"attributes": {"low": 1.0, "high": 9.0, "q": 2.0}}',
    "iu": '{"name": "IntUniformDistribution", "attributes": {"low": 1, "high": 9}}',
    "iuq": '{"name": "IntUniformDistribution", "attributes": {"low": 1, "high": 9, "step": 2}}',
    "c1": '{"name": "CategoricalDistribution", "attributes": {"choices": [2.71, -Infinity]}}',
    "c2": '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}',
    "c3": '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}',
    "ilu": '{"name": "IntLogUniformDistribution", "attributes": {"low": 2, "high": 12}}',
}

EXAMPLE_ABBREVIATED_JSONS = {
    "u": '{"type": "float", "low": 1.0, "high": 2.0}',
    "l": '{"type": "float", "low": 0.001, "high": 100, "log": true}',
    "du": '{"type": "float", "low": 1.0, "high": 9.0, "step": 2.0}',
    "iu": '{"type": "int", "low": 1, "high": 9}',
    "iuq": '{"type": "int", "low": 1, "high": 9, "step": 2}',
    "c1": '{"type": "categorical", "choices": [2.71, -Infinity]}',
    "c2": '{"type": "categorical", "choices": ["Roppongi", "Azabu"]}',
    "c3": '{"type": "categorical", "choices": ["Roppongi", "Azabu"]}',
    "ilu": '{"type": "int", "low": 2, "high": 12, "log": true}',
}


def test_json_to_distribution() -> None:

    for key in EXAMPLE_JSONS:
        distribution_actual = distributions.json_to_distribution(EXAMPLE_JSONS[key])
        assert distribution_actual == EXAMPLE_DISTRIBUTIONS[key]

    unknown_json = '{"name": "UnknownDistribution", "attributes": {"low": 1.0, "high": 2.0}}'
    pytest.raises(ValueError, lambda: distributions.json_to_distribution(unknown_json))


def test_abbreviated_json_to_distribution() -> None:

    for key in EXAMPLE_ABBREVIATED_JSONS:
        distribution_actual = distributions.json_to_distribution(EXAMPLE_ABBREVIATED_JSONS[key])
        assert distribution_actual == EXAMPLE_DISTRIBUTIONS[key]

    unknown_json = '{"type": "unknown", "low": 1.0, "high": 2.0}'
    pytest.raises(ValueError, lambda: distributions.json_to_distribution(unknown_json))

    invalid_distribution = (
        '{"type": "float", "low": 0.0, "high": -100.0}',
        '{"type": "float", "low": 7.3, "high": 7.2, "log": true}',
        '{"type": "float", "low": -30.0, "high": -40.0, "step": 3.0}',
        '{"type": "float", "low": 1.0, "high": 100.0, "step": 0.0}',
        '{"type": "float", "low": 1.0, "high": 100.0, "step": -1.0}',
        '{"type": "int", "low": 123, "high": 100}',
        '{"type": "int", "low": 123, "high": 100, "step": 2}',
        '{"type": "int", "low": 123, "high": 100, "log": true}',
        '{"type": "int", "low": 1, "high": 100, "step": 0}',
        '{"type": "int", "low": 1, "high": 100, "step": -1}',
        '{"type": "categorical", "choices": []}',
    )
    for distribution in invalid_distribution:
        pytest.raises(ValueError, lambda: distributions.json_to_distribution(distribution))


def test_backward_compatibility_int_uniform_distribution() -> None:

    json_str = '{"name": "IntUniformDistribution", "attributes": {"low": 1, "high": 10}}'
    actual = distributions.json_to_distribution(json_str)
    expected = distributions.IntUniformDistribution(low=1, high=10)
    assert actual == expected


def test_distribution_to_json() -> None:

    for key in EXAMPLE_JSONS:
        json_actual = json.loads(distributions.distribution_to_json(EXAMPLE_DISTRIBUTIONS[key]))
        json_expect = json.loads(EXAMPLE_JSONS[key])
        if (
            json_expect["name"] in ("IntUniformDistribution", "IntLogUniformDistribution")
            and "step" not in json_expect["attributes"]
        ):
            json_expect["attributes"]["step"] = 1
        assert json_actual == json_expect


def test_check_distribution_compatibility() -> None:

    # test the same distribution
    for key in EXAMPLE_JSONS:
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS[key], EXAMPLE_DISTRIBUTIONS[key]
        )

    # test different distribution classes
    pytest.raises(
        ValueError,
        lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["i"], EXAMPLE_DISTRIBUTIONS["fl"]
        ),
    )

    pytest.raises(
        ValueError,
        lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["u"], EXAMPLE_DISTRIBUTIONS["l"]
        ),
    )

    # test compatibility between IntDistributions.
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["id"], EXAMPLE_DISTRIBUTIONS["i"]
    )

    with pytest.raises(ValueError):
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["i"], EXAMPLE_DISTRIBUTIONS["il"]
        )

    with pytest.raises(ValueError):
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["il"], EXAMPLE_DISTRIBUTIONS["id"]
        )

    # test compatibility between FloatDistributions.
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["fd"], EXAMPLE_DISTRIBUTIONS["f"]
    )

    with pytest.raises(ValueError):
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["f"], EXAMPLE_DISTRIBUTIONS["fl"]
        )

    with pytest.raises(ValueError):
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["fl"], EXAMPLE_DISTRIBUTIONS["fd"]
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
        EXAMPLE_DISTRIBUTIONS["i"], distributions.IntDistribution(low=-3, high=2)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["il"], distributions.IntDistribution(low=1, high=13, log=True)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["id"], distributions.IntDistribution(low=-3, high=2, step=2)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["f"], distributions.FloatDistribution(low=-3.0, high=-2.0)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["fl"], distributions.FloatDistribution(low=0.1, high=1.0, log=True)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["fd"], distributions.FloatDistribution(low=-1.0, high=11.0, step=0.5)
    )
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
        EXAMPLE_DISTRIBUTIONS["iuq"], distributions.IntUniformDistribution(low=-1, high=1)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["ilu"], distributions.IntLogUniformDistribution(low=1, high=13)
    )


@pytest.mark.parametrize(
    ("expected", "value", "step"),
    [
        (False, 0.9, 1),
        (True, 1, 1),
        (False, 1.5, 1),
        (True, 4, 1),
        (True, 10, 1),
        (False, 11, 1),
        (False, 10, 2),
        (True, 1, 3),
        (False, 5, 3),
        (True, 10, 3),
    ],
)
def test_int_contains(expected: bool, value: float, step: int) -> None:
    i = distributions.IntDistribution(low=1, high=10, step=step)
    assert i._contains(value) == expected


@pytest.mark.parametrize(
    ("expected", "value", "step"),
    [
        (False, 1.99, None),
        (True, 2.0, None),
        (True, 2.5, None),
        (True, 7, None),
        (False, 7.1, None),
        (False, 0.99, 2.0),
        (True, 2.0, 2.0),
        (False, 3.0, 2.0),
        (True, 6, 2.0),
        (False, 6.1, 2.0),
    ],
)
def test_float_contains(expected: bool, value: float, step: Optional[float]) -> None:
    with warnings.catch_warnings():
        # When `step` is 2.0, UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [2.0, 6.0].
        warnings.simplefilter("ignore", category=UserWarning)
        f = distributions.FloatDistribution(low=2.0, high=7.0, step=step)
    assert f._contains(value) == expected


def test_contains() -> None:

    u = distributions.UniformDistribution(low=1.0, high=2.0)
    assert not u._contains(0.9)
    assert u._contains(1)
    assert u._contains(1.5)
    assert u._contains(2)
    assert not u._contains(2.1)

    lu = distributions.LogUniformDistribution(low=0.001, high=100)
    assert not lu._contains(0.0)
    assert lu._contains(0.001)
    assert lu._contains(12.3)
    assert lu._contains(100)
    assert not lu._contains(1000)

    with warnings.catch_warnings():
        # UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [1.0, 9.0].
        warnings.simplefilter("ignore", category=UserWarning)
        du = distributions.DiscreteUniformDistribution(low=1.0, high=10.0, q=2.0)
    assert not du._contains(0.9)
    assert du._contains(1.0)
    assert not du._contains(3.5)
    assert not du._contains(6)
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
    assert not iuq._contains(4)
    assert not iuq._contains(6)
    assert iuq._contains(9)
    assert not iuq._contains(9.1)
    assert not iuq._contains(10)

    c = distributions.CategoricalDistribution(choices=("Roppongi", "Azabu"))
    assert not c._contains(-1)
    assert c._contains(0)
    assert c._contains(1)
    assert c._contains(1.5)
    assert not c._contains(3)

    ilu = distributions.IntLogUniformDistribution(low=2, high=12)
    assert not ilu._contains(0.9)
    assert ilu._contains(2)
    assert ilu._contains(4)
    assert ilu._contains(6)
    assert ilu._contains(12)
    assert not ilu._contains(12.1)
    assert not ilu._contains(13)


def test_empty_range_contains() -> None:

    i = distributions.IntDistribution(low=1, high=1)
    assert not i._contains(0)
    assert i._contains(1)
    assert not i._contains(2)

    f = distributions.FloatDistribution(low=1.0, high=1.0)
    assert not f._contains(0.9)
    assert f._contains(1.0)
    assert not f._contains(1.1)

    fd = distributions.FloatDistribution(low=1.0, high=1.0, step=2.0)
    assert not fd._contains(0.9)
    assert fd._contains(1.0)
    assert not fd._contains(1.1)

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

    ilu = distributions.IntLogUniformDistribution(low=1, high=1)
    assert not ilu._contains(0)
    assert ilu._contains(1)
    assert not ilu._contains(2)


@pytest.mark.parametrize(
    ("expected", "low", "high", "log", "step"),
    [
        (True, 1, 1, False, 1),
        (True, 3, 3, False, 2),
        (True, 2, 2, True, 1),
        (False, -123, 0, False, 1),
        (False, -123, 0, False, 123),
        (False, 2, 4, True, 1),
    ],
)
def test_int_single(expected: bool, low: int, high: int, log: bool, step: int) -> None:
    distribution = distributions.IntDistribution(low=low, high=high, log=log, step=step)
    assert distribution.single() == expected


@pytest.mark.parametrize(
    ("expected", "low", "high", "log", "step"),
    [
        (True, 2.0, 2.0, False, None),
        (True, 2.0, 2.0, True, None),
        (True, 2.22, 2.22, False, 0.1),
        (True, 2.22, 2.24, False, 0.3),
        (False, 1.0, 1.001, False, None),
        (False, 7.3, 10.0, True, None),
        (False, -30, -20, False, 2),
        (False, -30, -20, False, 10),
        # In Python, "0.3 - 0.2 != 0.1" is True.
        (False, 0.2, 0.3, False, 0.1),
        (False, 0.7, 0.8, False, 0.1),
    ],
)
def test_float_single(
    expected: bool, low: float, high: float, log: bool, step: Optional[float]
) -> None:
    distribution = distributions.FloatDistribution(low=low, high=high, log=log, step=step)
    assert distribution.single() == expected


def test_single() -> None:

    with warnings.catch_warnings():
        # UserWarning will be raised since the range is not divisible by step.
        warnings.simplefilter("ignore", category=UserWarning)
        single_distributions: List[distributions.BaseDistribution] = [
            distributions.UniformDistribution(low=1.0, high=1.0),
            distributions.LogUniformDistribution(low=7.3, high=7.3),
            distributions.DiscreteUniformDistribution(low=2.22, high=2.22, q=0.1),
            distributions.DiscreteUniformDistribution(low=2.22, high=2.24, q=0.3),
            distributions.IntUniformDistribution(low=-123, high=-123),
            distributions.IntUniformDistribution(low=-123, high=-120, step=4),
            distributions.CategoricalDistribution(choices=("foo",)),
            distributions.IntLogUniformDistribution(low=2, high=2),
        ]
    for distribution in single_distributions:
        assert distribution.single()

    nonsingle_distributions: List[distributions.BaseDistribution] = [
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
    ]
    for distribution in nonsingle_distributions:
        assert not distribution.single()


def test_empty_distribution() -> None:

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


def test_invalid_distribution() -> None:

    with pytest.warns(UserWarning):
        distributions.CategoricalDistribution(choices=({"foo": "bar"},))  # type: ignore


def test_eq_ne_hash() -> None:

    # Two instances of a class are regarded as equivalent if the fields have the same values.
    for d in EXAMPLE_DISTRIBUTIONS.values():
        d_copy = copy.deepcopy(d)
        assert d == d_copy
        assert hash(d) == hash(d_copy)

    # Different field values.
    di0 = distributions.FloatDistribution(low=1, high=2)
    di1 = distributions.FloatDistribution(low=1, high=3)
    assert di0 != di1

    # Different distribution classes.
    di2 = distributions.IntDistribution(low=1, high=2)
    assert di0 != di2

    # Different field values.
    d0 = distributions.UniformDistribution(low=1, high=2)
    d1 = distributions.UniformDistribution(low=1, high=3)
    assert d0 != d1

    # Different distribution classes.
    d2 = distributions.IntUniformDistribution(low=1, high=2)
    assert d0 != d2


def test_repr() -> None:

    # The following variable is needed to apply `eval` to distribution
    # instances that contain `float('inf')` as a field value.
    inf = float("inf")  # NOQA

    for d in EXAMPLE_DISTRIBUTIONS.values():
        assert d == eval("distributions." + repr(d))


@pytest.mark.parametrize(
    ("key", "low", "high", "log", "step"),
    [
        ("i", 1, 9, False, 1),
        ("il", 2, 12, True, 1),
        ("id", 1, 9, False, 2),
    ],
)
def test_int_distribution_asdict(key: str, low: int, high: int, log: bool, step: int) -> None:
    expected_dict = {"low": low, "high": high, "log": log, "step": step}
    assert EXAMPLE_DISTRIBUTIONS[key]._asdict() == expected_dict


@pytest.mark.parametrize(
    ("key", "low", "high", "log", "step"),
    [
        ("f", 1.0, 2.0, False, None),
        ("fl", 0.001, 100.0, True, None),
        ("fd", 1.0, 9.0, False, 2.0),
    ],
)
def test_float_distribution_asdict(
    key: str, low: float, high: float, log: bool, step: Optional[float]
) -> None:
    expected_dict = {"low": low, "high": high, "log": log, "step": step}
    assert EXAMPLE_DISTRIBUTIONS[key]._asdict() == expected_dict


def test_uniform_distribution_asdict() -> None:

    assert EXAMPLE_DISTRIBUTIONS["u"]._asdict() == {"low": 1.0, "high": 2.0}


def test_log_uniform_distribution_asdict() -> None:

    assert EXAMPLE_DISTRIBUTIONS["l"]._asdict() == {"low": 0.001, "high": 100}


def test_discrete_uniform_distribution_asdict() -> None:

    assert EXAMPLE_DISTRIBUTIONS["du"]._asdict() == {"low": 1.0, "high": 9.0, "q": 2.0}


def test_int_uniform_distribution_asdict() -> None:

    assert EXAMPLE_DISTRIBUTIONS["iu"]._asdict() == {"low": 1, "high": 9, "step": 1}
    assert EXAMPLE_DISTRIBUTIONS["iuq"]._asdict() == {"low": 1, "high": 9, "step": 2}


def test_int_log_uniform_distribution_asdict() -> None:

    assert EXAMPLE_DISTRIBUTIONS["ilu"]._asdict() == {"low": 2, "high": 12, "step": 1}


def test_int_init_error() -> None:

    # Empty distributions cannot be instantiated.
    with pytest.raises(ValueError):
        distributions.IntDistribution(low=123, high=100)

    with pytest.raises(ValueError):
        distributions.IntDistribution(low=100, high=10, log=True)

    with pytest.raises(ValueError):
        distributions.IntDistribution(low=123, high=100, step=2)

    # 'step' must be 1 when 'log' is True.
    with pytest.raises(ValueError):
        distributions.IntDistribution(low=1, high=100, log=True, step=2)

    # 'step' should be positive.
    with pytest.raises(ValueError):
        distributions.IntDistribution(low=1, high=100, step=0)

    with pytest.raises(ValueError):
        distributions.IntDistribution(low=1, high=10, step=-1)


def test_float_init_error() -> None:

    # Empty distributions cannot be instantiated.
    with pytest.raises(ValueError):
        distributions.FloatDistribution(low=0.0, high=-100.0)

    with pytest.raises(ValueError):
        distributions.FloatDistribution(low=7.3, high=7.2, log=True)

    with pytest.raises(ValueError):
        distributions.FloatDistribution(low=-30.0, high=-40.0, step=2.5)

    # 'step' must be None when 'log' is True.
    with pytest.raises(ValueError):
        distributions.FloatDistribution(low=1.0, high=100.0, log=True, step=0.5)

    # 'step' should be positive.
    with pytest.raises(ValueError):
        distributions.FloatDistribution(low=1.0, high=10.0, step=0)

    with pytest.raises(ValueError):
        distributions.FloatDistribution(low=1.0, high=100.0, step=-1)


def test_discrete_uniform_distribution_invalid_q() -> None:

    with pytest.raises(ValueError):
        distributions.DiscreteUniformDistribution(low=1, high=100, q=0)

    with pytest.raises(ValueError):
        distributions.DiscreteUniformDistribution(low=1, high=100, q=-1)


def test_int_uniform_distribution_invalid_step() -> None:

    with pytest.raises(ValueError):
        distributions.IntUniformDistribution(low=1, high=100, step=0)

    with pytest.raises(ValueError):
        distributions.IntUniformDistribution(low=1, high=100, step=-1)


def test_categorical_distribution_different_sequence_types() -> None:

    c1 = distributions.CategoricalDistribution(choices=("Roppongi", "Azabu"))
    c2 = distributions.CategoricalDistribution(choices=["Roppongi", "Azabu"])

    assert c1 == c2
