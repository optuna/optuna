from __future__ import annotations

import copy
import json
from typing import Any
from typing import cast
import warnings

import numpy as np
import pytest

from optuna import distributions


_choices = (None, True, False, 0, 1, 0.0, 1.0, float("nan"), float("inf"), -float("inf"), "", "a")
_choices_json = '[null, true, false, 0, 1, 0.0, 1.0, NaN, Infinity, -Infinity, "", "a"]'

EXAMPLE_DISTRIBUTIONS: dict[str, Any] = {
    "i": distributions.IntDistribution(low=1, high=9, log=False),
    # i2 and i3 are identical to i, and tested for cases when `log` and `step` are omitted in json.
    "i2": distributions.IntDistribution(low=1, high=9, log=False),
    "i3": distributions.IntDistribution(low=1, high=9, log=False),
    "il": distributions.IntDistribution(low=2, high=12, log=True),
    "il2": distributions.IntDistribution(low=2, high=12, log=True),
    "id": distributions.IntDistribution(low=1, high=9, log=False, step=2),
    "id2": distributions.IntDistribution(low=1, high=9, log=False, step=2),
    "f": distributions.FloatDistribution(low=1.0, high=2.0, log=False),
    "fl": distributions.FloatDistribution(low=0.001, high=100.0, log=True),
    "fd": distributions.FloatDistribution(low=1.0, high=9.0, log=False, step=2.0),
    "c1": distributions.CategoricalDistribution(choices=_choices),
    "c2": distributions.CategoricalDistribution(choices=("Roppongi", "Azabu")),
    "c3": distributions.CategoricalDistribution(choices=["Roppongi", "Azabu"]),
}

EXAMPLE_JSONS = {
    "i": '{"name": "IntDistribution", "attributes": {"low": 1, "high": 9}}',
    "i2": '{"name": "IntDistribution", "attributes": {"low": 1, "high": 9, "log": false}}',
    "i3": '{"name": "IntDistribution", '
    '"attributes": {"low": 1, "high": 9, "log": false, "step": 1}}',
    "il": '{"name": "IntDistribution", ' '"attributes": {"low": 2, "high": 12, "log": true}}',
    "il2": '{"name": "IntDistribution", '
    '"attributes": {"low": 2, "high": 12, "log": true, "step": 1}}',
    "id": '{"name": "IntDistribution", ' '"attributes": {"low": 1, "high": 9, "step": 2}}',
    "id2": '{"name": "IntDistribution", '
    '"attributes": {"low": 1, "high": 9, "log": false, "step": 2}}',
    "f": '{"name": "FloatDistribution", '
    '"attributes": {"low": 1.0, "high": 2.0, "log": false, "step": null}}',
    "fl": '{"name": "FloatDistribution", '
    '"attributes": {"low": 0.001, "high": 100.0, "log": true, "step": null}}',
    "fd": '{"name": "FloatDistribution", '
    '"attributes": {"low": 1.0, "high": 9.0, "step": 2.0, "log": false}}',
    "c1": f'{{"name": "CategoricalDistribution", "attributes": {{"choices": {_choices_json}}}}}',
    "c2": '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}',
    "c3": '{"name": "CategoricalDistribution", "attributes": {"choices": ["Roppongi", "Azabu"]}}',
}

EXAMPLE_ABBREVIATED_JSONS = {
    "i": '{"type": "int", "low": 1, "high": 9}',
    "i2": '{"type": "int", "low": 1, "high": 9, "log": false}',
    "i3": '{"type": "int", "low": 1, "high": 9, "log": false, "step": 1}',
    "il": '{"type": "int", "low": 2, "high": 12, "log": true}',
    "il2": '{"type": "int", "low": 2, "high": 12, "log": true, "step": 1}',
    "id": '{"type": "int", "low": 1, "high": 9, "step": 2}',
    "id2": '{"type": "int", "low": 1, "high": 9, "log": false, "step": 2}',
    "f": '{"type": "float", "low": 1.0, "high": 2.0, "log": false, "step": null}',
    "fl": '{"type": "float", "low": 0.001, "high": 100, "log": true, "step": null}',
    "fd": '{"type": "float", "low": 1.0, "high": 9.0, "log": false, "step": 2.0}',
    "c1": f'{{"type": "categorical", "choices": {_choices_json}}}',
    "c2": '{"type": "categorical", "choices": ["Roppongi", "Azabu"]}',
    "c3": '{"type": "categorical", "choices": ["Roppongi", "Azabu"]}',
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


def test_distribution_to_json() -> None:
    for key in EXAMPLE_JSONS:
        json_actual = json.loads(distributions.distribution_to_json(EXAMPLE_DISTRIBUTIONS[key]))
        json_expect = json.loads(EXAMPLE_JSONS[key])
        if json_expect["name"] == "IntDistribution" and "step" not in json_expect["attributes"]:
            json_expect["attributes"]["step"] = 1
        if json_expect["name"] == "IntDistribution" and "log" not in json_expect["attributes"]:
            json_expect["attributes"]["log"] = False
        assert json_actual == json_expect


def test_check_distribution_compatibility() -> None:
    # Test the same distribution.
    for key in EXAMPLE_JSONS:
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS[key], EXAMPLE_DISTRIBUTIONS[key]
        )
        # We need to create new objects to compare NaNs.
        # See https://github.com/optuna/optuna/pull/3567#pullrequestreview-974939837.
        distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS[key], distributions.json_to_distribution(EXAMPLE_JSONS[key])
        )

    # Test different distribution classes.
    pytest.raises(
        ValueError,
        lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["i"], EXAMPLE_DISTRIBUTIONS["fl"]
        ),
    )

    # Test compatibility between IntDistributions.
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

    # Test compatibility between FloatDistributions.
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

    # Test dynamic value range (CategoricalDistribution).
    pytest.raises(
        ValueError,
        lambda: distributions.check_distribution_compatibility(
            EXAMPLE_DISTRIBUTIONS["c2"],
            distributions.CategoricalDistribution(choices=("Roppongi", "Akasaka")),
        ),
    )

    # Test dynamic value range (except CategoricalDistribution).
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["i"], distributions.IntDistribution(low=-3, high=2)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["il"], distributions.IntDistribution(low=1, high=13, log=True)
    )
    distributions.check_distribution_compatibility(
        EXAMPLE_DISTRIBUTIONS["id"], distributions.IntDistribution(low=-3, high=1, step=2)
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


@pytest.mark.parametrize("value", (0, 1, 4, 10, 11, 1.1, "1", "1.1", "-1.0", True, False))
def test_int_internal_representation(value: Any) -> None:
    i = distributions.IntDistribution(low=1, high=10)

    if isinstance(value, int):
        expected_value = value
    else:
        expected_value = int(float(value))
    assert i.to_external_repr(i.to_internal_repr(value)) == expected_value


@pytest.mark.parametrize(
    "value, kwargs",
    [
        ("foo", {}),
        ((), {}),
        ([], {}),
        ({}, {}),
        (set(), {}),
        (np.ones(2), {}),
        (np.nan, {}),
        (0, dict(log=True)),
        (-1, dict(log=True)),
    ],
)
def test_int_internal_representation_error(value: Any, kwargs: dict[str, Any]) -> None:
    i = distributions.IntDistribution(low=1, high=10, **kwargs)
    with pytest.raises(ValueError):
        i.to_internal_repr(value)


@pytest.mark.parametrize(
    "value",
    (1.99, 2.0, 4.5, 7, 7.1, 1, "1", "1.1", "-1.0", True, False),
)
def test_float_internal_representation(value: Any) -> None:
    f = distributions.FloatDistribution(low=2.0, high=7.0)

    if isinstance(value, float):
        expected_value = value
    else:
        expected_value = float(value)
    assert f.to_external_repr(f.to_internal_repr(value)) == expected_value


@pytest.mark.parametrize(
    "value, kwargs",
    [
        ("foo", {}),
        ((), {}),
        ([], {}),
        ({}, {}),
        (set(), {}),
        (np.ones(2), {}),
        (np.nan, {}),
        (0.0, dict(log=True)),
        (-1.0, dict(log=True)),
    ],
)
def test_float_internal_representation_error(value: Any, kwargs: dict[str, Any]) -> None:
    f = distributions.FloatDistribution(low=2.0, high=7.0, **kwargs)
    with pytest.raises(ValueError):
        f.to_internal_repr(value)


def test_categorical_internal_representation() -> None:
    c = EXAMPLE_DISTRIBUTIONS["c1"]
    for choice in c.choices:
        if isinstance(choice, float) and np.isnan(choice):
            assert np.isnan(c.to_external_repr(c.to_internal_repr(choice)))
        else:
            assert c.to_external_repr(c.to_internal_repr(choice)) == choice

    # We need to create new objects to compare NaNs.
    # See https://github.com/optuna/optuna/pull/3567#pullrequestreview-974939837.
    c_ = distributions.json_to_distribution(EXAMPLE_JSONS["c1"])
    for choice in cast(distributions.CategoricalDistribution, c_).choices:
        if isinstance(choice, float) and np.isnan(choice):
            assert np.isnan(c.to_external_repr(c.to_internal_repr(choice)))
        else:
            assert c.to_external_repr(c.to_internal_repr(choice)) == choice


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
    with warnings.catch_warnings():
        # When `step` is 2, UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [1, 9].
        warnings.simplefilter("ignore", category=UserWarning)
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
def test_float_contains(expected: bool, value: float, step: float | None) -> None:
    with warnings.catch_warnings():
        # When `step` is 2.0, UserWarning will be raised since the range is not divisible by 2.
        # The range will be replaced with [2.0, 6.0].
        warnings.simplefilter("ignore", category=UserWarning)
        f = distributions.FloatDistribution(low=2.0, high=7.0, step=step)
    assert f._contains(value) == expected


def test_categorical_contains() -> None:
    c = distributions.CategoricalDistribution(choices=("Roppongi", "Azabu"))
    assert not c._contains(-1)
    assert c._contains(0)
    assert c._contains(1)
    assert c._contains(1.5)
    assert not c._contains(3)


def test_empty_range_contains() -> None:
    i = distributions.IntDistribution(low=1, high=1)
    assert not i._contains(0)
    assert i._contains(1)
    assert not i._contains(2)

    iq = distributions.IntDistribution(low=1, high=1, step=2)
    assert not iq._contains(0)
    assert iq._contains(1)
    assert not iq._contains(2)

    il = distributions.IntDistribution(low=1, high=1, log=True)
    assert not il._contains(0)
    assert il._contains(1)
    assert not il._contains(2)

    f = distributions.FloatDistribution(low=1.0, high=1.0)
    assert not f._contains(0.9)
    assert f._contains(1.0)
    assert not f._contains(1.1)

    fd = distributions.FloatDistribution(low=1.0, high=1.0, step=2.0)
    assert not fd._contains(0.9)
    assert fd._contains(1.0)
    assert not fd._contains(1.1)

    fl = distributions.FloatDistribution(low=1.0, high=1.0, log=True)
    assert not fl._contains(0.9)
    assert fl._contains(1.0)
    assert not fl._contains(1.1)


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
    expected: bool, low: float, high: float, log: bool, step: float | None
) -> None:
    with warnings.catch_warnings():
        # When `step` is 0.3, UserWarning will be raised since the range is not divisible by 0.3.
        # The range will be replaced with [2.22, 2.24].
        warnings.simplefilter("ignore", category=UserWarning)
        distribution = distributions.FloatDistribution(low=low, high=high, log=log, step=step)
    assert distribution.single() == expected


def test_categorical_single() -> None:
    assert distributions.CategoricalDistribution(choices=("foo",)).single()
    assert not distributions.CategoricalDistribution(choices=("foo", "bar")).single()


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
    d0 = distributions.FloatDistribution(low=1, high=2)
    d1 = distributions.FloatDistribution(low=1, high=3)
    assert d0 != d1

    # Different distribution classes.
    d2 = distributions.IntDistribution(low=1, high=2)
    assert d0 != d2


def test_repr() -> None:
    # The following variables are needed to apply `eval` to distribution
    # instances that contain `float('nan')` or `float('inf')` as a field value.
    nan = float("nan")  # NOQA
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
    key: str, low: float, high: float, log: bool, step: float | None
) -> None:
    expected_dict = {"low": low, "high": high, "log": log, "step": step}
    assert EXAMPLE_DISTRIBUTIONS[key]._asdict() == expected_dict


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


def test_categorical_init_error() -> None:
    with pytest.raises(ValueError):
        distributions.CategoricalDistribution(choices=())


def test_categorical_distribution_different_sequence_types() -> None:
    c1 = distributions.CategoricalDistribution(choices=("Roppongi", "Azabu"))
    c2 = distributions.CategoricalDistribution(choices=["Roppongi", "Azabu"])

    assert c1 == c2


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_convert_old_distribution_to_new_distribution() -> None:
    ud = distributions.UniformDistribution(low=0, high=10)
    assert distributions._convert_old_distribution_to_new_distribution(
        ud
    ) == distributions.FloatDistribution(low=0, high=10, log=False, step=None)

    dud = distributions.DiscreteUniformDistribution(low=0, high=10, q=2)
    assert distributions._convert_old_distribution_to_new_distribution(
        dud
    ) == distributions.FloatDistribution(low=0, high=10, log=False, step=2)

    lud = distributions.LogUniformDistribution(low=1, high=10)
    assert distributions._convert_old_distribution_to_new_distribution(
        lud
    ) == distributions.FloatDistribution(low=1, high=10, log=True, step=None)

    id = distributions.IntUniformDistribution(low=0, high=10)
    assert distributions._convert_old_distribution_to_new_distribution(
        id
    ) == distributions.IntDistribution(low=0, high=10, log=False, step=1)

    idd = distributions.IntUniformDistribution(low=0, high=10, step=2)
    assert distributions._convert_old_distribution_to_new_distribution(
        idd
    ) == distributions.IntDistribution(low=0, high=10, log=False, step=2)

    ild = distributions.IntLogUniformDistribution(low=1, high=10)
    assert distributions._convert_old_distribution_to_new_distribution(
        ild
    ) == distributions.IntDistribution(low=1, high=10, log=True, step=1)


def test_convert_old_distribution_to_new_distribution_noop() -> None:
    # No conversion happens for CategoricalDistribution.
    cd = distributions.CategoricalDistribution(choices=["a", "b", "c"])
    assert distributions._convert_old_distribution_to_new_distribution(cd) == cd

    # No conversion happens for new distributions.
    fd = distributions.FloatDistribution(low=0, high=10, log=False, step=None)
    assert distributions._convert_old_distribution_to_new_distribution(fd) == fd

    dfd = distributions.FloatDistribution(low=0, high=10, log=False, step=2)
    assert distributions._convert_old_distribution_to_new_distribution(dfd) == dfd

    lfd = distributions.FloatDistribution(low=1, high=10, log=True, step=None)
    assert distributions._convert_old_distribution_to_new_distribution(lfd) == lfd

    id = distributions.IntDistribution(low=0, high=10)
    assert distributions._convert_old_distribution_to_new_distribution(id) == id

    idd = distributions.IntDistribution(low=0, high=10, step=2)
    assert distributions._convert_old_distribution_to_new_distribution(idd) == idd

    ild = distributions.IntDistribution(low=1, high=10, log=True)
    assert distributions._convert_old_distribution_to_new_distribution(ild) == ild


def test_is_distribution_log() -> None:
    lfd = distributions.FloatDistribution(low=1, high=10, log=True)
    assert distributions._is_distribution_log(lfd)

    lid = distributions.IntDistribution(low=1, high=10, log=True)
    assert distributions._is_distribution_log(lid)

    fd = distributions.FloatDistribution(low=0, high=10, log=False)
    assert not distributions._is_distribution_log(fd)

    id = distributions.IntDistribution(low=0, high=10, log=False)
    assert not distributions._is_distribution_log(id)

    cd = distributions.CategoricalDistribution(choices=["a", "b", "c"])
    assert not distributions._is_distribution_log(cd)
