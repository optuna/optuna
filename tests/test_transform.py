import math
from typing import Any

import numpy
import pytest

from optuna._transform import _Transform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "param,distribution",
    [
        (0, IntUniformDistribution(0, 3)),
        (1, IntLogUniformDistribution(1, 10)),
        (2, IntUniformDistribution(0, 10, step=2)),
        (0.0, UniformDistribution(0, 3)),
        (1.0, LogUniformDistribution(1, 10)),
        (0.2, DiscreteUniformDistribution(0, 1, q=0.2)),
        ("foo", CategoricalDistribution(["foo"])),
        ("bar", CategoricalDistribution(["foo", "bar", "baz"])),
    ],
)
def test_transform_shapes_dtypes(
    transform_log: bool,
    transform_step: bool,
    param: Any,
    distribution: BaseDistribution,
) -> None:
    trans = _Transform({"x0": distribution}, transform_log, transform_step)
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


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "param,distribution",
    [
        (0, IntUniformDistribution(0, 3)),
        (1, IntLogUniformDistribution(1, 10)),
        (2, IntUniformDistribution(0, 10, step=2)),
        (0.0, UniformDistribution(0, 3)),
        (1.0, LogUniformDistribution(1, 10)),
        (0.2, DiscreteUniformDistribution(0, 1, q=0.2)),
    ],
)
def test_transform_numerical(
    transform_log: bool,
    transform_step: bool,
    param: Any,
    distribution: BaseDistribution,
) -> None:
    trans = _Transform({"x0": distribution}, transform_log, transform_step)

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
    elif isinstance(distribution, IntUniformDistribution):
        if transform_step:
            half_step = 0.5 * distribution.step
            expected_low -= half_step
            expected_high += half_step
    elif isinstance(distribution, IntLogUniformDistribution):
        if transform_step:
            half_step = 0.5
            expected_low -= half_step
            expected_high += half_step
        if transform_log:
            expected_low = math.log(expected_low)
            expected_high = math.log(expected_high)

    for bound in trans.bounds:
        assert bound[0] == expected_low
        assert bound[1] == expected_high

    trans_params = trans.transform({"x0": param})
    assert trans_params.size == 1

    if isinstance(distribution, (IntUniformDistribution, IntLogUniformDistribution)):
        assert expected_low <= trans_params <= expected_high
    else:
        # TODO(hvy): Change second `<=` to `<` when `suggest_float` is fixed.
        assert expected_low <= trans_params <= expected_high


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "param,distribution",
    [
        ("foo", CategoricalDistribution(["foo"])),
        ("bar", CategoricalDistribution(["foo", "bar", "baz"])),
    ],
)
def test_transform_fit_values_categorical(
    transform_log: bool,
    transform_step: bool,
    param: Any,
    distribution: CategoricalDistribution,
) -> None:
    trans = _Transform({"x0": distribution}, transform_log, transform_step)

    for bound in trans.bounds:
        assert bound[0] == 0.0
        assert bound[1] == 1.0

    trans_params = trans.transform({"x0": param})

    for trans_param in trans_params:
        assert trans_param in (0.0, 1.0)


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
def test_transform_shapes_dtypes_values_categorical_with_other_distribution(
    transform_log: bool, transform_step: bool
) -> None:
    search_space = {
        "x0": DiscreteUniformDistribution(0, 1, q=0.2),
        "x1": CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        "x2": IntLogUniformDistribution(1, 10),
        "x3": CategoricalDistribution(["quux", "quuz"]),
    }
    params = {
        "x0": 0.2,
        "x1": "qux",
        "x2": 1,
        "x3": "quux",
    }

    trans = _Transform(search_space, transform_log, transform_step)

    trans_params = trans.transform(params)

    n_tot_choices = len(search_space["x1"].choices)  # type: ignore
    n_tot_choices += len(search_space["x3"].choices)  # type: ignore
    assert trans_params.shape == (n_tot_choices + 2,)
    assert trans.bounds.shape == (n_tot_choices + 2, 2)

    for i, (low, high) in enumerate(trans.bounds):
        if i == 0:
            expected_low = search_space["x0"].low  # type: ignore
            expected_high = search_space["x0"].high  # type: ignore
            if transform_step:
                half_step = 0.5 * 0.2
                expected_low -= half_step
                expected_high += half_step
            assert low == expected_low
            assert high == expected_high
        elif i in (1, 2, 3, 4):
            assert low == 0.0
            assert high == 1.0
        elif i == 5:
            expected_low = search_space["x2"].low  # type: ignore
            expected_high = search_space["x2"].high  # type: ignore
            if transform_step:
                half_step = 0.5
                expected_low -= half_step
                expected_high += half_step
            if transform_log:
                expected_low = math.log(expected_low)
                expected_high = math.log(expected_high)
            assert low == expected_low
            assert high == expected_high
        elif i in (6, 7):
            assert low == 0.0
            assert high == 1.0
        else:
            assert False

    for i, trans_param in enumerate(trans_params):
        if i == 0:
            assert search_space["x0"].low <= trans_param <= search_space["x0"].high  # type: ignore
        elif i in (1, 2, 3, 4):
            assert 0.0 <= trans_param <= 1.0
        elif i == 5:
            expected_low = search_space["x2"].low  # type: ignore
            expected_high = search_space["x2"].high  # type: ignore
            if transform_log:
                expected_low = math.log(expected_low)
                expected_high = math.log(expected_high)
            assert expected_low <= trans_param <= expected_high
        elif i in (6, 7):
            assert 0.0 <= trans_param <= 1.0
        else:
            assert False


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
def test_transform_untransform_params(transform_log: bool, transform_step: bool) -> None:
    search_space = {
        "x0": DiscreteUniformDistribution(0, 1, q=0.2),
        "x1": CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        "x2": IntLogUniformDistribution(1, 10),
        "x3": CategoricalDistribution(["quux", "quuz"]),
        "x4": UniformDistribution(2, 3),
        "x5": LogUniformDistribution(1, 10),
        "x6": IntUniformDistribution(2, 4),
        "x7": CategoricalDistribution(["corge"]),
    }
    params = {
        "x0": 0.2,
        "x1": "qux",
        "x2": 1,
        "x3": "quux",
        "x4": 2.0,
        "x5": 1.0,
        "x6": 2,
        "x7": "corge",
    }

    trans = _Transform(search_space, transform_log, transform_step)

    trans_params = trans.transform(params)

    untrans_params = trans.untransform(trans_params)

    for name in params.keys():
        assert untrans_params[name] == params[name]
