from __future__ import annotations

import datetime
import math
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
from optuna import create_study
from optuna import distributions
from optuna import load_study
from optuna import samplers
from optuna import storages
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.testing.pruners import DeterministicPruner
from optuna.testing.samplers import DeterministicSampler
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.testing.tempfile_pool import NamedTemporaryFilePool
from optuna.trial import Trial
from optuna.trial._trial import _LazyTrialSystemAttrs


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_check_distribution_suggest_float(storage_mode: str) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        x1 = trial.suggest_float("x1", 10, 20)
        x2 = trial.suggest_uniform("x1", 10, 20)

        assert x1 == x2

        x3 = trial.suggest_float("x2", 1e-5, 1e-3, log=True)
        x4 = trial.suggest_loguniform("x2", 1e-5, 1e-3)

        assert x3 == x4

        x5 = trial.suggest_float("x3", 10, 20, step=1.0)
        x6 = trial.suggest_discrete_uniform("x3", 10, 20, 1.0)

        assert x5 == x6
        with pytest.raises(ValueError):
            trial.suggest_float("x4", 1e-5, 1e-2, step=1e-5, log=True)

        with pytest.raises(ValueError):
            trial.suggest_int("x1", 10, 20)

        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_int("x1", 10, 20)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_check_distribution_suggest_uniform(storage_mode: str) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns() as record:
            trial.suggest_uniform("x", 10, 20)
            trial.suggest_uniform("x", 10, 20)
            trial.suggest_uniform("x", 10, 30)

        # we expect exactly one warning (not counting ones caused by deprecation)
        warning_records = [r for r in record if r.category is not FutureWarning]
        assert len(warning_records) == 1, [warning.message for warning in warning_records]

        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20)

        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_check_distribution_suggest_loguniform(storage_mode: str) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns() as record:
            trial.suggest_loguniform("x", 10, 20)
            trial.suggest_loguniform("x", 10, 20)
            trial.suggest_loguniform("x", 10, 30)

        # We expect exactly one warning (not counting ones caused by deprecation).
        warning_records = [r for r in record if r.category is not FutureWarning]
        assert len(warning_records) == 1, [warning.message for warning in warning_records]

        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20)

        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_check_distribution_suggest_discrete_uniform(storage_mode: str) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns() as record:
            trial.suggest_discrete_uniform("x", 10, 20, 2)
            trial.suggest_discrete_uniform("x", 10, 20, 2)
            trial.suggest_discrete_uniform("x", 10, 22, 2)

        # We expect exactly one warning (not counting ones caused by deprecation).
        warning_records = [r for r in record if r.category is not FutureWarning]
        assert len(warning_records) == 1, [warning.message for warning in warning_records]

        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20, step=2)

        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20, step=2)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("enable_log", [False, True])
def test_check_distribution_suggest_int(storage_mode: str, enable_log: bool) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns() as record:
            trial.suggest_int("x", 10, 20, log=enable_log)
            trial.suggest_int("x", 10, 20, log=enable_log)
            trial.suggest_int("x", 10, 22, log=enable_log)

        # We expect exactly one warning (not counting ones caused by deprecation).
        warning_records = [r for r in record if r.category is not FutureWarning]
        assert len(warning_records) == 1, [warning.message for warning in warning_records]

        with pytest.raises(ValueError):
            trial.suggest_float("x", 10, 20, log=enable_log)

        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_float("x", 10, 20, log=enable_log)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_check_distribution_suggest_categorical(storage_mode: str) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        trial.suggest_categorical("x", [10, 20, 30])

        with pytest.raises(ValueError):
            trial.suggest_categorical("x", [10, 20])

        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20)

        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_int("x", 10, 20)


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_uniform(storage_mode: str) -> None:
    sampler = DeterministicSampler({"x": 1.0, "y": 2.0})

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        assert trial.suggest_uniform("x", 0.0, 3.0) == 1.0  # Test suggesting a param.
        assert trial.suggest_uniform("x", 0.0, 3.0) == 1.0  # Test suggesting the same param.
        assert trial.suggest_uniform("y", 0.0, 3.0) == 2.0  # Test suggesting a different param.
        assert trial.params == {"x": 1.0, "y": 2.0}


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_loguniform(storage_mode: str) -> None:
    with pytest.raises(ValueError):
        FloatDistribution(low=1.0, high=0.9, log=True)

    with pytest.raises(ValueError):
        FloatDistribution(low=0.0, high=0.9, log=True)

    sampler = DeterministicSampler({"x": 1.0, "y": 2.0})

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        assert trial.suggest_loguniform("x", 0.1, 4.0) == 1.0  # Test suggesting a param.
        assert trial.suggest_loguniform("x", 0.1, 4.0) == 1.0  # Test suggesting the same param.
        assert trial.suggest_loguniform("y", 0.1, 4.0) == 2.0  # Test suggesting a different param.
        assert trial.params == {"x": 1.0, "y": 2.0}


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_discrete_uniform(storage_mode: str) -> None:
    sampler = DeterministicSampler({"x": 1.0, "y": 2.0})

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        assert (
            trial.suggest_discrete_uniform("x", 0.0, 3.0, 1.0) == 1.0
        )  # Test suggesting a param.
        assert (
            trial.suggest_discrete_uniform("x", 0.0, 3.0, 1.0) == 1.0
        )  # Test suggesting the same param.
        assert (
            trial.suggest_discrete_uniform("y", 0.0, 3.0, 1.0) == 2.0
        )  # Test suggesting a different param.
        assert trial.params == {"x": 1.0, "y": 2.0}


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_low_equals_high(storage_mode: str) -> None:
    with patch.object(
        distributions, "_get_single_value", wraps=distributions._get_single_value
    ) as mock_object, StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=samplers.TPESampler(n_startup_trials=0))

        trial = study.ask()

        assert trial.suggest_uniform("a", 1.0, 1.0) == 1.0  # Suggesting a param.
        assert mock_object.call_count == 1
        assert trial.suggest_uniform("a", 1.0, 1.0) == 1.0  # Suggesting the same param.
        assert mock_object.call_count == 1

        assert trial.suggest_loguniform("b", 1.0, 1.0) == 1.0  # Suggesting a param.
        assert mock_object.call_count == 2
        assert trial.suggest_loguniform("b", 1.0, 1.0) == 1.0  # Suggesting the same param.
        assert mock_object.call_count == 2

        assert trial.suggest_discrete_uniform("c", 1.0, 1.0, 1.0) == 1.0  # Suggesting a param.
        assert mock_object.call_count == 3
        assert (
            trial.suggest_discrete_uniform("c", 1.0, 1.0, 1.0) == 1.0
        )  # Suggesting the same param.
        assert mock_object.call_count == 3

        assert trial.suggest_int("d", 1, 1) == 1  # Suggesting a param.
        assert mock_object.call_count == 4
        assert trial.suggest_int("d", 1, 1) == 1  # Suggesting the same param.
        assert mock_object.call_count == 4

        assert trial.suggest_float("e", 1.0, 1.0) == 1.0  # Suggesting a param.
        assert mock_object.call_count == 5
        assert trial.suggest_float("e", 1.0, 1.0) == 1.0  # Suggesting the same param.
        assert mock_object.call_count == 5

        assert trial.suggest_float("f", 0.5, 0.5, log=True) == 0.5  # Suggesting a param.
        assert mock_object.call_count == 6
        assert trial.suggest_float("f", 0.5, 0.5, log=True) == 0.5  # Suggesting the same param.
        assert mock_object.call_count == 6

        assert trial.suggest_float("g", 0.5, 0.5, log=False) == 0.5  # Suggesting a param.
        assert mock_object.call_count == 7
        assert trial.suggest_float("g", 0.5, 0.5, log=False) == 0.5  # Suggesting the same param.
        assert mock_object.call_count == 7

        assert trial.suggest_float("h", 0.5, 0.5, step=1.0) == 0.5  # Suggesting a param.
        assert mock_object.call_count == 8
        assert trial.suggest_float("h", 0.5, 0.5, step=1.0) == 0.5  # Suggesting the same param.
        assert mock_object.call_count == 8

        assert trial.suggest_int("i", 1, 1, log=True) == 1  # Suggesting a param.
        assert mock_object.call_count == 9
        assert trial.suggest_int("i", 1, 1, log=True) == 1  # Suggesting the same param.
        assert mock_object.call_count == 9


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "range_config",
    [
        {"low": 0.0, "high": 10.0, "q": 3.0, "mod_high": 9.0},
        {"low": 1.0, "high": 11.0, "q": 3.0, "mod_high": 10.0},
        {"low": 64.0, "high": 1312.0, "q": 160.0, "mod_high": 1184.0},
        {"low": 0.0, "high": 10.0, "q": math.pi, "mod_high": 3 * math.pi},
        {"low": 0.0, "high": 3.45, "q": 0.1, "mod_high": 3.4},
    ],
)
def test_suggest_discrete_uniform_range(storage_mode: str, range_config: dict[str, float]) -> None:
    sampler = samplers.RandomSampler()

    # Check upper endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.high
    with patch.object(sampler, "sample_independent", mock) as mock_object, StorageSupplier(
        storage_mode
    ) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns(UserWarning):
            x = trial.suggest_discrete_uniform(
                "x", range_config["low"], range_config["high"], range_config["q"]
            )
        assert x == range_config["mod_high"]
        assert mock_object.call_count == 1

    # Check lower endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.low
    with patch.object(sampler, "sample_independent", mock) as mock_object, StorageSupplier(
        storage_mode
    ) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns(UserWarning):
            x = trial.suggest_discrete_uniform(
                "x", range_config["low"], range_config["high"], range_config["q"]
            )
        assert x == range_config["low"]
        assert mock_object.call_count == 1


def test_suggest_float_invalid_step() -> None:
    study = create_study()
    trial = study.ask()

    with pytest.raises(ValueError):
        trial.suggest_float("x1", 10, 20, step=0)

    with pytest.raises(ValueError):
        trial.suggest_float("x2", 10, 20, step=-1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_int(storage_mode: str) -> None:
    sampler = DeterministicSampler({"x": 1, "y": 2})

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        assert trial.suggest_int("x", 0, 3) == 1  # Test suggesting a param.
        assert trial.suggest_int("x", 0, 3) == 1  # Test suggesting the same param.
        assert trial.suggest_int("y", 0, 3) == 2  # Test suggesting a different param.
        assert trial.params == {"x": 1, "y": 2}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "range_config",
    [
        {"low": 0, "high": 10, "step": 3, "mod_high": 9},
        {"low": 1, "high": 11, "step": 3, "mod_high": 10},
        {"low": 64, "high": 1312, "step": 160, "mod_high": 1184},
    ],
)
def test_suggest_int_range(storage_mode: str, range_config: dict[str, int]) -> None:
    sampler = samplers.RandomSampler()

    # Check upper endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.high
    with patch.object(sampler, "sample_independent", mock) as mock_object, StorageSupplier(
        storage_mode
    ) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns(UserWarning):
            x = trial.suggest_int(
                "x", range_config["low"], range_config["high"], step=range_config["step"]
            )
        assert x == range_config["mod_high"]
        assert mock_object.call_count == 1

    # Check lower endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.low
    with patch.object(sampler, "sample_independent", mock) as mock_object, StorageSupplier(
        storage_mode
    ) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        with pytest.warns(UserWarning):
            x = trial.suggest_int(
                "x", range_config["low"], range_config["high"], step=range_config["step"]
            )
        assert x == range_config["low"]
        assert mock_object.call_count == 1


def test_suggest_int_invalid_step() -> None:
    study = create_study()
    trial = study.ask()

    with pytest.raises(ValueError):
        trial.suggest_int("x1", 10, 20, step=0)

    with pytest.raises(ValueError):
        trial.suggest_int("x2", 10, 20, step=-1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_int_log(storage_mode: str) -> None:
    sampler = DeterministicSampler({"x": 1, "y": 2})

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()

        assert trial.suggest_int("x", 1, 3, log=True) == 1  # Test suggesting a param.
        assert trial.suggest_int("x", 1, 3, log=True) == 1  # Test suggesting the same param.
        assert trial.suggest_int("y", 1, 3, log=True) == 2  # Test suggesting a different param.
        assert trial.params == {"x": 1, "y": 2}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_suggest_int_log_invalid_range(storage_mode: str) -> None:
    sampler = samplers.RandomSampler()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()
        with warnings.catch_warnings():
            # UserWarning will be raised since [0.5, 10] is not divisible by 1.
            warnings.simplefilter("ignore", category=UserWarning)
            with pytest.raises(ValueError):
                trial.suggest_int("z", 0.5, 10, log=True)  # type: ignore

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)
        trial = study.ask()
        with pytest.raises(ValueError):
            trial.suggest_int("w", 1, 3, step=2, log=True)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_distributions(storage_mode: str) -> None:
    def objective(trial: Trial) -> float:
        trial.suggest_float("a", 0, 10)
        trial.suggest_float("b", 0.1, 10, log=True)
        trial.suggest_float("c", 0, 10, step=1)
        trial.suggest_int("d", 0, 10)
        trial.suggest_categorical("e", ["foo", "bar", "baz"])
        trial.suggest_int("f", 1, 10, log=True)

        return 1.0

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(objective, n_trials=1)

        assert study.best_trial.distributions == {
            "a": FloatDistribution(low=0, high=10),
            "b": FloatDistribution(low=0.1, high=10, log=True),
            "c": FloatDistribution(low=0, high=10, step=1),
            "d": IntDistribution(low=0, high=10),
            "e": CategoricalDistribution(choices=("foo", "bar", "baz")),
            "f": IntDistribution(low=1, high=10, log=True),
        }


def test_should_prune() -> None:
    pruner = DeterministicPruner(True)
    study = create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)
    assert trial.should_prune()


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_relative_parameters(storage_mode: str) -> None:
    class SamplerStubForTestRelativeParameters(samplers.BaseSampler):
        def infer_relative_search_space(
            self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
        ) -> dict[str, distributions.BaseDistribution]:
            return {
                "x": FloatDistribution(low=5, high=6),
                "y": FloatDistribution(low=5, high=6),
            }

        def sample_relative(
            self,
            study: "optuna.study.Study",
            trial: "optuna.trial.FrozenTrial",
            search_space: dict[str, distributions.BaseDistribution],
        ) -> dict[str, Any]:
            return {"x": 5.5, "y": 5.5, "z": 5.5}

        def sample_independent(
            self,
            study: "optuna.study.Study",
            trial: "optuna.trial.FrozenTrial",
            param_name: str,
            param_distribution: distributions.BaseDistribution,
        ) -> Any:
            return 5.0

    sampler = SamplerStubForTestRelativeParameters()
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=sampler)

        def create_trial() -> Trial:
            return study.ask()

        # Suggested by `sample_relative`.
        trial0 = create_trial()
        distribution0 = FloatDistribution(low=0, high=100)
        assert trial0._suggest("x", distribution0) == 5.5

        # Not suggested by `sample_relative` (due to unknown parameter name).
        trial1 = create_trial()
        distribution1 = distribution0
        assert trial1._suggest("w", distribution1) != 5.5

        # Not suggested by `sample_relative` (due to incompatible value range).
        trial2 = create_trial()
        distribution2 = FloatDistribution(low=0, high=5)
        assert trial2._suggest("x", distribution2) != 5.5

        # Error (due to incompatible distribution class).
        trial3 = create_trial()
        distribution3 = IntDistribution(low=1, high=100)
        with pytest.raises(ValueError):
            trial3._suggest("y", distribution3)

        # Error ('z' is included in `sample_relative` but not in `infer_relative_search_space`).
        trial4 = create_trial()
        distribution4 = FloatDistribution(low=0, high=10)
        with pytest.raises(ValueError):
            trial4._suggest("z", distribution4)

        # Error (due to incompatible distribution class).
        trial5 = create_trial()
        distribution5 = IntDistribution(low=1, high=100, log=True)
        with pytest.raises(ValueError):
            trial5._suggest("y", distribution5)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_datetime_start(storage_mode: str) -> None:
    trial_datetime_start: list[datetime.datetime | None] = [None]

    def objective(trial: Trial) -> float:
        trial_datetime_start[0] = trial.datetime_start
        return 1.0

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(objective, n_trials=1)

        assert study.trials[0].datetime_start == trial_datetime_start[0]


def test_report_value() -> None:
    study = create_study()
    trial = study.ask()

    # Report values that can be cast to `float` (OK).
    trial.report(1.23, 1)
    trial.report(float("nan"), 2)
    trial.report("1.23", 3)  # type: ignore
    trial.report("inf", 4)  # type: ignore
    trial.report(1, 5)
    trial.report(np.array([1], dtype=np.float32)[0], 6)

    # Report values that cannot be cast to `float`.
    with pytest.raises(TypeError):
        trial.report(None, 7)  # type: ignore

    with pytest.raises(TypeError):
        trial.report("foo", 7)  # type: ignore

    with pytest.raises(TypeError):
        trial.report([1, 2, 3], 7)  # type: ignore

    with pytest.raises(TypeError):
        trial.report("foo", -1)  # type: ignore


def test_report_step() -> None:
    study = create_study()
    trial = study.ask()
    value = 1.0

    # Report values whose steps can be cast to `int` (OK).
    trial.report(value, 0)
    trial.report(value, 1.0)  # type: ignore
    trial.report(value, np.int64(2))  # type: ignore

    # Report values whose steps cannot be cast to `int` (Error).
    with pytest.raises(TypeError):
        trial.report(value, None)  # type: ignore

    with pytest.raises(TypeError):
        trial.report(value, "foo")  # type: ignore

    with pytest.raises(TypeError):
        trial.report(value, [1, 2, 3])  # type: ignore

    # Report a value whose step is negative (Error).
    with pytest.raises(ValueError):
        trial.report(value, -1)


def test_report_warning() -> None:
    study = create_study()
    trial = study.ask()

    trial.report(1.23, 1)

    # Warn if multiple times call report method at the same step
    with pytest.warns(UserWarning):
        trial.report(1, 1)


def test_suggest_with_multi_objectives() -> None:
    study = create_study(directions=["maximize", "maximize"])

    def objective(trial: Trial) -> tuple[float, float]:
        p0 = trial.suggest_float("p0", -10, 10)
        p1 = trial.suggest_float("p1", 3, 5)
        p2 = trial.suggest_float("p2", 0.00001, 0.1, log=True)
        p3 = trial.suggest_float("p3", 100, 200, step=5)
        p4 = trial.suggest_int("p4", -20, -15)
        p5 = trial.suggest_categorical("p5", [7, 1, 100])
        p6 = trial.suggest_float("p6", -10, 10, step=1.0)
        p7 = trial.suggest_int("p7", 1, 7, log=True)
        return (
            p0 + p1 + p2,
            p3 + p4 + p5 + p6 + p7,
        )

    study.optimize(objective, n_trials=10)


def test_raise_error_for_report_with_multi_objectives() -> None:
    study = create_study(directions=["maximize", "maximize"])

    def objective(trial: Trial) -> tuple[float, float]:
        with pytest.raises(NotImplementedError):
            trial.report(1.0, 0)
        return 1.0, 1.0

    study.optimize(objective, n_trials=1)


def test_raise_error_for_should_prune_multi_objectives() -> None:
    study = create_study(directions=["maximize", "maximize"])

    def objective(trial: Trial) -> tuple[float, float]:
        with pytest.raises(NotImplementedError):
            trial.should_prune()
        return 1.0, 1.0

    study.optimize(objective, n_trials=1)


def test_persisted_param() -> None:
    study_name = "my_study"

    with NamedTemporaryFilePool() as fp:
        storage = f"sqlite:///{fp.name}"
        study = create_study(storage=storage, study_name=study_name)
        assert isinstance(study._storage, storages._CachedStorage), "Pre-condition."

        # Test more than one trial. The `_CachedStorage` does a cache miss for the first trial and
        # thus behaves differently for the first trial in comparisons to the following.
        for _ in range(3):
            trial = study.ask()
            trial.suggest_float("x", 0, 1)

        study = load_study(storage=storage, study_name=study_name)

        assert all("x" in t.params for t in study.trials)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_lazy_trial_system_attrs(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        trial = study.ask()
        storage.set_trial_system_attr(trial._trial_id, "int", 0)
        storage.set_trial_system_attr(trial._trial_id, "str", "A")

        # _LazyTrialSystemAttrs gets attrs the first time it is needed.
        # Then, we create the instance for each method, and test the first and second use.

        system_attrs = _LazyTrialSystemAttrs(trial._trial_id, storage)
        assert system_attrs == {"int": 0, "str": "A"}
        assert system_attrs == {"int": 0, "str": "A"}

        system_attrs = _LazyTrialSystemAttrs(trial._trial_id, storage)
        assert len(system_attrs) == 2
        assert len(system_attrs) == 2

        system_attrs = _LazyTrialSystemAttrs(trial._trial_id, storage)
        assert set(system_attrs.keys()) == {"int", "str"}
        assert set(system_attrs.keys()) == {"int", "str"}

        system_attrs = _LazyTrialSystemAttrs(trial._trial_id, storage)
        assert set(system_attrs.values()) == {0, "A"}
        assert set(system_attrs.values()) == {0, "A"}

        system_attrs = _LazyTrialSystemAttrs(trial._trial_id, storage)
        assert set(system_attrs.items()) == {("int", 0), ("str", "A")}
        assert set(system_attrs.items()) == {("int", 0), ("str", "A")}


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("positional_args_names", [[], ["step"], ["step", "log"]])
def test_suggest_int_positional_args(positional_args_names: list[str]) -> None:
    # If log is specified as positional, step must also be provided as positional.
    study = optuna.create_study()
    trial = study.ask()
    kwargs = dict(step=1, log=False)
    args = [kwargs[name] for name in positional_args_names]
    # No error should not be raised even if the coding style is old.
    trial.suggest_int("x", -1, 1, *args)
