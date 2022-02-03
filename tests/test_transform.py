import math
from typing import Any

import numpy
import pytest

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution


@pytest.mark.parametrize(
    "param,distribution",
    [
        (0, IntDistribution(0, 3)),
        (1, IntDistribution(1, 10, log=True)),
        (2, IntDistribution(0, 10, step=2)),
        (0.0, UniformDistribution(0, 3)),
        (1.0, LogUniformDistribution(1, 10)),
        (0.2, DiscreteUniformDistribution(0, 1, q=0.2)),
        (0.0, FloatDistribution(0, 3)),
        (1.0, FloatDistribution(1, 10, log=True)),
        (0.2, FloatDistribution(0, 1, step=0.2)),
        ("foo", CategoricalDistribution(["foo"])),
        ("bar", CategoricalDistribution(["foo", "bar", "baz"])),
    ],
)
def test_search_space_transform_shapes_dtypes(param: Any, distribution: BaseDistribution) -> None:
    trans = _SearchSpaceTransform({"x0": distribution})
    trans_params = trans.transform({"x0": param})

    if isinstance(distribution, CategoricalDistribution):
        expected_bounds_shape = (len(distribution.choices), 2)
        expected_params_shape = (len(distribution.choices),)
    else:
        expected_bounds_shape = (1, 2)
        expected_params_shape = (1,)
    assert trans.bounds.shape == expected_bounds_shape
    assert trans.bounds.dtype == numpy.float64
    assert trans_params.shape == expected_params_shape
    assert trans_params.dtype == numpy.float64


def test_search_space_transform_encoding() -> None:
    trans = _SearchSpaceTransform({"x0": IntDistribution(0, 3)})

    assert len(trans.column_to_encoded_columns) == 1
    numpy.testing.assert_equal(trans.column_to_encoded_columns[0], numpy.array([0]))
    numpy.testing.assert_equal(trans.encoded_column_to_column, numpy.array([0]))

    trans = _SearchSpaceTransform({"x0": CategoricalDistribution(["foo", "bar", "baz"])})

    assert len(trans.column_to_encoded_columns) == 1
    numpy.testing.assert_equal(trans.column_to_encoded_columns[0], numpy.array([0, 1, 2]))
    numpy.testing.assert_equal(trans.encoded_column_to_column, numpy.array([0, 0, 0]))

    trans = _SearchSpaceTransform(
        {
            "x0": UniformDistribution(0, 3),
            "x1": CategoricalDistribution(["foo", "bar", "baz"]),
            "x3": DiscreteUniformDistribution(0, 1, q=0.2),
        }
    )

    assert len(trans.column_to_encoded_columns) == 3
    numpy.testing.assert_equal(trans.column_to_encoded_columns[0], numpy.array([0]))
    numpy.testing.assert_equal(trans.column_to_encoded_columns[1], numpy.array([1, 2, 3]))
    numpy.testing.assert_equal(trans.column_to_encoded_columns[2], numpy.array([4]))
    numpy.testing.assert_equal(trans.encoded_column_to_column, numpy.array([0, 1, 1, 1, 2]))


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "param,distribution",
    [
        (0, IntDistribution(0, 3)),
        (3, IntDistribution(0, 3)),
        (1, IntDistribution(1, 10, log=True)),
        (10, IntDistribution(1, 10, log=True)),
        (2, IntDistribution(0, 10, step=2)),
        (0.0, UniformDistribution(0, 3)),
        (3.0, UniformDistribution(0, 3)),
        (1.0, LogUniformDistribution(1, 10)),
        (10.0, LogUniformDistribution(1, 10)),
        (0.2, DiscreteUniformDistribution(0, 1, q=0.2)),
        (1.0, DiscreteUniformDistribution(0, 1, q=0.2)),
        (0.0, FloatDistribution(0, 3)),
        (1.0, FloatDistribution(1, 10, log=True)),
        (0.2, FloatDistribution(0, 1, step=0.2)),
    ],
)
def test_search_space_transform_numerical(
    transform_log: bool,
    transform_step: bool,
    param: Any,
    distribution: BaseDistribution,
) -> None:
    trans = _SearchSpaceTransform({"x0": distribution}, transform_log, transform_step)

    expected_low = distribution.low  # type: ignore
    expected_high = distribution.high  # type: ignore

    if isinstance(distribution, LogUniformDistribution):
        if transform_log:
            expected_low = math.log(expected_low)
            expected_high = math.log(expected_high)
    elif isinstance(distribution, DiscreteUniformDistribution):
        if transform_step:
            half_step = 0.5 * distribution.q
            expected_low -= half_step
            expected_high += half_step
    elif isinstance(distribution, FloatDistribution):
        if transform_log and distribution.log:
            expected_low = math.log(expected_low)
            expected_high = math.log(expected_high)
        if transform_step and distribution.step is not None:
            half_step = 0.5 * distribution.step
            expected_low -= half_step
            expected_high += half_step
    elif isinstance(distribution, IntDistribution):
        if transform_step:
            half_step = 0.5 * distribution.step
            expected_low -= half_step
            expected_high += half_step
        if distribution.log and transform_log:
            expected_low = math.log(expected_low)
            expected_high = math.log(expected_high)

    for bound in trans.bounds:
        assert bound[0] == expected_low
        assert bound[1] == expected_high

    trans_params = trans.transform({"x0": param})
    assert trans_params.size == 1
    assert expected_low <= trans_params <= expected_high


@pytest.mark.parametrize(
    "param,distribution",
    [
        ("foo", CategoricalDistribution(["foo"])),
        ("bar", CategoricalDistribution(["foo", "bar", "baz"])),
    ],
)
def test_search_space_transform_values_categorical(
    param: Any, distribution: CategoricalDistribution
) -> None:
    trans = _SearchSpaceTransform({"x0": distribution})

    for bound in trans.bounds:
        assert bound[0] == 0.0
        assert bound[1] == 1.0

    trans_params = trans.transform({"x0": param})

    for trans_param in trans_params:
        assert trans_param in (0.0, 1.0)


def test_search_space_transform_untransform_params() -> None:
    search_space = {
        "x0": CategoricalDistribution(["corge"]),
        "x1": CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        "x2": CategoricalDistribution(["quux", "quuz"]),
        "x3": FloatDistribution(2, 3),
        "x4": FloatDistribution(-2, 2),
        "x5": FloatDistribution(1, 10),
        "x6": FloatDistribution(1, 1),
        "x7": FloatDistribution(0, 1, step=0.2),
        "x8": IntDistribution(2, 4),
        "x9": IntDistribution(1, 10, log=True),
        "x10": IntDistribution(1, 9, step=2),
        "x11": DiscreteUniformDistribution(0, 1, q=0.2),
        "x12": UniformDistribution(2, 3),
        "x13": LogUniformDistribution(1, 10),
        "x14": UniformDistribution(-2, -2),
        "x15": LogUniformDistribution(1, 1),
    }

    params = {
        "x0": "corge",
        "x1": "qux",
        "x2": "quux",
        "x3": 2.0,
        "x4": -2,
        "x5": 1.0,
        "x6": 1.0,
        "x7": 0.2,
        "x8": 2,
        "x9": 1,
        "x10": 3,
        "x11": 0.2,
        "x12": 2.0,
        "x13": 1.0,
        "x14": -2.0,
        "x15": 1.0,
    }

    trans = _SearchSpaceTransform(search_space)
    trans_params = trans.transform(params)
    untrans_params = trans.untransform(trans_params)

    for name in params.keys():
        assert untrans_params[name] == params[name]
