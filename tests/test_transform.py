import math
from typing import Any

import numpy
import pytest

import optuna
from optuna._transform import _Transform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.trial import Trial


def suggest(trial: Trial, name: str, distribution: BaseDistribution) -> Any:
    if isinstance(distribution, CategoricalDistribution):
        return trial.suggest_categorical(name, distribution.choices)
    if isinstance(distribution, DiscreteUniformDistribution):
        return trial.suggest_float(name, distribution.low, distribution.high, step=distribution.q)
    if isinstance(distribution, IntLogUniformDistribution):
        return trial.suggest_int(name, distribution.low, distribution.high, log=True)
    if isinstance(distribution, IntUniformDistribution):
        return trial.suggest_int(name, distribution.low, distribution.high)
    if isinstance(distribution, LogUniformDistribution):
        return trial.suggest_float(name, distribution.low, distribution.high, log=True)
    if isinstance(distribution, UniformDistribution):
        return trial.suggest_float(name, distribution.low, distribution.high)
    else:
        assert False


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "distribution",
    [
        IntUniformDistribution(0, 3),
        IntLogUniformDistribution(1, 10),
        IntUniformDistribution(0, 10, step=2),
        UniformDistribution(0, 3),
        LogUniformDistribution(1, 10),
        DiscreteUniformDistribution(0, 1, q=0.2),
        CategoricalDistribution(["foo"]),
        CategoricalDistribution(["foo", "bar", "baz"]),
    ],
)
def test_transform_fit_shapes_dtypes(
    transform_log: bool,
    transform_step: bool,
    distribution: BaseDistribution,
) -> None:
    def objective(trial: Trial) -> float:
        value = suggest(trial, "x0", distribution)
        if isinstance(distribution, CategoricalDistribution):
            value = float(distribution.choices.index(value))
        return value

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    trans = _Transform({"x0": distribution}, transform_log, transform_step)
    trans_params, trans_values = trans.transform(study.trials)

    if isinstance(distribution, CategoricalDistribution):
        expected_bounds_shape = (len(distribution.choices), 2)
        expected_params_shape = (3, len(distribution.choices))
    else:
        expected_bounds_shape = (1, 2)
        expected_params_shape = (3, 1)
    assert trans.bounds.shape == expected_bounds_shape
    assert trans.bounds.dtype == numpy.float64
    assert trans_params.shape == expected_params_shape
    assert trans_params.dtype == numpy.float64
    assert trans_values.shape == (3,)
    assert trans_values.dtype == numpy.float64


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "distribution",
    [
        IntUniformDistribution(0, 3),
        IntLogUniformDistribution(1, 10),
        IntUniformDistribution(0, 10, step=2),
        UniformDistribution(0, 3),
        LogUniformDistribution(1, 10),
        DiscreteUniformDistribution(0, 1, q=0.2),
    ],
)
def test_transform_fit_values_numerical(
    transform_log: bool,
    transform_step: bool,
    distribution: BaseDistribution,
) -> None:
    def objective(trial: Trial) -> float:
        return float(suggest(trial, "x0", distribution))

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

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

    trans_params, trans_values = trans.transform(study.trials)

    for params in trans_params:
        for param in params:
            if isinstance(distribution, (IntUniformDistribution, IntLogUniformDistribution)):
                assert expected_low <= param <= expected_high
            else:
                # TODO(hvy): Change second `<=` to `<` when `suggest_float` is fixed.
                assert expected_low <= param <= expected_high

    for value, trial in zip(trans_values, study.trials):
        assert value == trial.value


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
@pytest.mark.parametrize(
    "distribution",
    [
        CategoricalDistribution(["foo"]),
        CategoricalDistribution(["foo", "bar", "baz"]),
    ],
)
def test_transform_fit_values_categorical(
    transform_log: bool,
    transform_step: bool,
    distribution: CategoricalDistribution,
) -> None:
    def objective(trial: Trial) -> float:
        return float(distribution.choices.index(suggest(trial, "x0", distribution)))

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    trans = _Transform({"x0": distribution}, transform_log, transform_step)

    for bound in trans.bounds:
        assert bound[0] == 0.0
        assert bound[1] == 1.0

    trans_params, trans_values = trans.transform(study.trials)

    for params in trans_params:
        for param in params:
            assert 0.0 <= param <= 1.0

    for value, trial in zip(trans_values, study.trials):
        assert value == trial.value


@pytest.mark.parametrize("transform_log", [True, False])
@pytest.mark.parametrize("transform_step", [True, False])
def test_transform_fit_shapes_dtypes_values_categorical_with_other_distribution(
    transform_log: bool, transform_step: bool
) -> None:
    search_space = {
        "x0": DiscreteUniformDistribution(0, 1, q=0.2),
        "x1": CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        "x2": IntLogUniformDistribution(1, 10),
        "x3": CategoricalDistribution(["quux", "quuz"]),
    }

    def objective(trial: Trial) -> float:
        x0 = suggest(trial, "x0", search_space["x0"])
        x1 = suggest(trial, "x1", search_space["x1"])
        x1 = float(search_space["x1"].choices.index(x1))  # type: ignore
        x2 = suggest(trial, "x2", search_space["x2"])
        x3 = suggest(trial, "x3", search_space["x3"])
        x3 = float(search_space["x3"].choices.index(x3))  # type: ignore
        return x0 + x1 + x2 + x3

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    trans = _Transform(search_space, transform_log, transform_step)

    trans_params, trans_values = trans.transform(study.trials)

    n_tot_choices = len(search_space["x1"].choices)  # type: ignore
    n_tot_choices += len(search_space["x3"].choices)  # type: ignore
    assert trans_params.shape == (3, n_tot_choices + 2)
    assert trans_values.shape == (3,)
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

    for params in trans_params:
        for i, param in enumerate(params):
            if i == 0:
                assert search_space["x0"].low <= param <= search_space["x0"].high  # type: ignore
            elif i in (1, 2, 3, 4):
                assert 0.0 <= param <= 1.0
            elif i == 5:
                expected_low = search_space["x2"].low  # type: ignore
                expected_high = search_space["x2"].high  # type: ignore
                if transform_log:
                    expected_low = math.log(expected_low)
                    expected_high = math.log(expected_high)
                assert expected_low <= param <= expected_high
            elif i in (6, 7):
                assert 0.0 <= param <= 1.0
            else:
                assert False

    for value, trial in zip(trans_values, study.trials):
        assert value == trial.value


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

    def objective(trial: Trial) -> float:
        x0 = suggest(trial, "x0", search_space["x0"])
        x1 = suggest(trial, "x1", search_space["x1"])
        x1 = float(search_space["x1"].choices.index(x1))  # type: ignore
        x2 = suggest(trial, "x2", search_space["x2"])
        x3 = suggest(trial, "x3", search_space["x3"])
        x3 = float(search_space["x3"].choices.index(x3))  # type: ignore
        x4 = suggest(trial, "x4", search_space["x4"])
        x5 = suggest(trial, "x5", search_space["x5"])
        x6 = suggest(trial, "x6", search_space["x6"])
        x7 = suggest(trial, "x7", search_space["x7"])
        x7 = float(search_space["x7"].choices.index(x7))  # type: ignore
        return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    trans = _Transform(search_space, transform_log)
    params, values = trans.transform(study.trials)

    trial_number = values.argmin()
    next_trans_params = params[trial_number]
    params = trans.untransform_single_params(next_trans_params)

    expected_params = study.best_params
    for name in search_space.keys():
        assert params[name] == expected_params[name]
