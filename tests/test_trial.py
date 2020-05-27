import copy
import datetime
import math
from typing import Callable
from typing import Dict
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna import samplers
from optuna import storages
from optuna.study import create_study
from optuna.testing.integration import DeterministicPruner
from optuna.testing.sampler import DeterministicRelativeSampler
from optuna.trial import FixedTrial
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA

parametrize_storage = pytest.mark.parametrize(
    "storage_init_func",
    [storages.InMemoryStorage, lambda: storages.RDBStorage("sqlite:///:memory:")],
)


@parametrize_storage
def test_check_distribution_suggest_float(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    sampler = samplers.RandomSampler()
    study = create_study(storage_init_func(), sampler=sampler)
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    x1 = trial.suggest_float("x1", 10, 20)
    x2 = trial.suggest_uniform("x1", 10, 20)

    assert x1 == x2

    x3 = trial.suggest_float("x2", 1e-5, 1e-3, log=True)
    x4 = trial.suggest_loguniform("x2", 1e-5, 1e-3)

    assert x3 == x4

    x5 = trial.suggest_float("x3", 10, 20, step=1.0)
    x6 = trial.suggest_discrete_uniform("x3", 10, 20, 1.0)

    assert x5 == x6
    with pytest.raises(NotImplementedError):
        trial.suggest_float("x4", 1e-5, 1e-2, step=1e-5, log=True)


@parametrize_storage
def test_check_distribution_suggest_uniform(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    sampler = samplers.RandomSampler()
    study = create_study(storage_init_func(), sampler=sampler)
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    with pytest.warns(None) as record:
        trial.suggest_uniform("x", 10, 20)
        trial.suggest_uniform("x", 10, 20)
        trial.suggest_uniform("x", 10, 30)

    # we expect exactly one warning
    assert len(record) == 1


@parametrize_storage
def test_check_distribution_suggest_loguniform(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    sampler = samplers.RandomSampler()
    study = create_study(storage_init_func(), sampler=sampler)
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    with pytest.warns(None) as record:
        trial.suggest_loguniform("x", 10, 20)
        trial.suggest_loguniform("x", 10, 20)
        trial.suggest_loguniform("x", 10, 30)

    # we expect exactly one warning
    assert len(record) == 1


@parametrize_storage
def test_check_distribution_suggest_discrete_uniform(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    sampler = samplers.RandomSampler()
    study = create_study(storage_init_func(), sampler=sampler)
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    with pytest.warns(None) as record:
        trial.suggest_discrete_uniform("x", 10, 20, 2)
        trial.suggest_discrete_uniform("x", 10, 20, 2)
        trial.suggest_discrete_uniform("x", 10, 22, 2)

    # we expect exactly one warning
    assert len(record) == 1


@parametrize_storage
@pytest.mark.parametrize("enable_log", [False, True])
def test_check_distribution_suggest_int(
    storage_init_func: Callable[[], storages.BaseStorage], enable_log: bool
) -> None:

    sampler = samplers.RandomSampler()
    study = create_study(storage_init_func(), sampler=sampler)
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    with pytest.warns(None) as record:
        trial.suggest_int("x", 10, 20, log=enable_log)
        trial.suggest_int("x", 10, 20, log=enable_log)
        trial.suggest_int("x", 10, 22, log=enable_log)

    # We expect exactly one warning.
    assert len(record) == 1


@parametrize_storage
def test_suggest_uniform(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1.0, 2.0, 3.0]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))
        distribution = UniformDistribution(low=0.0, high=3.0)

        assert trial._suggest("x", distribution) == 1.0  # Test suggesting a param.
        assert trial._suggest("x", distribution) == 1.0  # Test suggesting the same param.
        assert trial._suggest("y", distribution) == 3.0  # Test suggesting a different param.
        assert trial.params == {"x": 1.0, "y": 3.0}
        assert mock_object.call_count == 3


@parametrize_storage
def test_suggest_loguniform(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    with pytest.raises(ValueError):
        LogUniformDistribution(low=1.0, high=0.9)

    with pytest.raises(ValueError):
        LogUniformDistribution(low=0.0, high=0.9)

    mock = Mock()
    mock.side_effect = [1.0, 2.0, 3.0]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))
        distribution = LogUniformDistribution(low=0.1, high=4.0)

        assert trial._suggest("x", distribution) == 1.0  # Test suggesting a param.
        assert trial._suggest("x", distribution) == 1.0  # Test suggesting the same param.
        assert trial._suggest("y", distribution) == 3.0  # Test suggesting a different param.
        assert trial.params == {"x": 1.0, "y": 3.0}
        assert mock_object.call_count == 3


@parametrize_storage
def test_suggest_discrete_uniform(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1.0, 2.0, 3.0]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))
        distribution = DiscreteUniformDistribution(low=0.0, high=3.0, q=1.0)

        assert trial._suggest("x", distribution) == 1.0  # Test suggesting a param.
        assert trial._suggest("x", distribution) == 1.0  # Test suggesting the same param.
        assert trial._suggest("y", distribution) == 3.0  # Test suggesting a different param.
        assert trial.params == {"x": 1.0, "y": 3.0}
        assert mock_object.call_count == 3


@parametrize_storage
def test_suggest_low_equals_high(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    study = create_study(storage_init_func(), sampler=samplers.TPESampler(n_startup_trials=0))
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    # Parameter values are determined without suggestion when low == high.
    with patch.object(trial, "_suggest", wraps=trial._suggest) as mock_object:
        assert trial.suggest_uniform("a", 1.0, 1.0) == 1.0  # Suggesting a param.
        assert trial.suggest_uniform("a", 1.0, 1.0) == 1.0  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_loguniform("b", 1.0, 1.0) == 1.0  # Suggesting a param.
        assert trial.suggest_loguniform("b", 1.0, 1.0) == 1.0  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_discrete_uniform("c", 1.0, 1.0, 1.0) == 1.0  # Suggesting a param.
        assert (
            trial.suggest_discrete_uniform("c", 1.0, 1.0, 1.0) == 1.0
        )  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_int("d", 1, 1) == 1  # Suggesting a param.
        assert trial.suggest_int("d", 1, 1) == 1  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_float("e", 1.0, 1.0) == 1.0  # Suggesting a param.
        assert trial.suggest_float("e", 1.0, 1.0) == 1.0  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_float("f", 0.5, 0.5, log=True) == 0.5  # Suggesting a param.
        assert trial.suggest_float("f", 0.5, 0.5, log=True) == 0.5  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_float("g", 0.5, 0.5, log=False) == 0.5  # Suggesting a param.
        assert trial.suggest_float("g", 0.5, 0.5, log=False) == 0.5  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_float("h", 0.5, 0.5, step=1.0) == 0.5  # Suggesting a param.
        assert trial.suggest_float("h", 0.5, 0.5, step=1.0) == 0.5  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_int("i", 1, 1, log=True) == 1  # Suggesting a param.
        assert trial.suggest_int("i", 1, 1, log=True) == 1  # Suggesting the same param.
        assert mock_object.call_count == 0


@parametrize_storage
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
def test_suggest_discrete_uniform_range(storage_init_func, range_config):
    # type: (Callable[[], storages.BaseStorage], Dict[str, float]) -> None

    sampler = samplers.RandomSampler()

    # Check upper endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.high
    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))

        with pytest.warns(UserWarning):
            x = trial.suggest_discrete_uniform(
                "x", range_config["low"], range_config["high"], range_config["q"]
            )
        assert x == range_config["mod_high"]
        assert mock_object.call_count == 1

    # Check lower endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.low
    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))

        with pytest.warns(UserWarning):
            x = trial.suggest_discrete_uniform(
                "x", range_config["low"], range_config["high"], range_config["q"]
            )
        assert x == range_config["low"]
        assert mock_object.call_count == 1


@parametrize_storage
def test_suggest_int(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1, 2, 3]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))
        distribution = IntUniformDistribution(low=0, high=3)

        assert trial._suggest("x", distribution) == 1  # Test suggesting a param.
        assert trial._suggest("x", distribution) == 1  # Test suggesting the same param.
        assert trial._suggest("y", distribution) == 3  # Test suggesting a different param.
        assert trial.params == {"x": 1, "y": 3}
        assert mock_object.call_count == 3


@parametrize_storage
@pytest.mark.parametrize(
    "range_config",
    [
        {"low": 0, "high": 10, "step": 3, "mod_high": 9},
        {"low": 1, "high": 11, "step": 3, "mod_high": 10},
        {"low": 64, "high": 1312, "step": 160, "mod_high": 1184},
    ],
)
def test_suggest_int_range(
    storage_init_func: Callable[[], storages.BaseStorage], range_config: Dict[str, int]
) -> None:

    sampler = samplers.RandomSampler()

    # Check upper endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.high
    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))

        with pytest.warns(UserWarning):
            x = trial.suggest_int(
                "x", range_config["low"], range_config["high"], step=range_config["step"]
            )
        assert x == range_config["mod_high"]
        assert mock_object.call_count == 1

    # Check lower endpoints.
    mock = Mock()
    mock.side_effect = lambda study, trial, param_name, distribution: distribution.low
    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))

        with pytest.warns(UserWarning):
            x = trial.suggest_int(
                "x", range_config["low"], range_config["high"], step=range_config["step"]
            )
        assert x == range_config["low"]
        assert mock_object.call_count == 1


@parametrize_storage
def test_suggest_int_log(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1, 2, 3]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, "sample_independent", mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))
        distribution = IntLogUniformDistribution(low=1, high=3)

        assert trial._suggest("x", distribution) == 1  # Test suggesting a param.
        assert trial._suggest("x", distribution) == 1  # Test suggesting the same param.
        assert trial._suggest("y", distribution) == 3  # Test suggesting a different param.
        assert trial.params == {"x": 1, "y": 3}
        assert mock_object.call_count == 3

    with pytest.raises(ValueError):
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study._storage.create_new_trial(study._study_id))
        with warnings.catch_warnings():
            # UserWarning will be raised since [0.5, 10] is not divisible by 1.
            warnings.simplefilter("ignore", category=UserWarning)
            trial.suggest_int("z", 0.5, 10, log=True)  # type: ignore


@parametrize_storage
def test_distributions(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    def objective(trial):
        # type: (Trial) -> float

        trial.suggest_uniform("a", 0, 10)
        trial.suggest_loguniform("b", 0.1, 10)
        trial.suggest_discrete_uniform("c", 0, 10, 1)
        trial.suggest_int("d", 0, 10)
        trial.suggest_categorical("e", ["foo", "bar", "baz"])
        trial.suggest_int("f", 1, 10, log=True)

        return 1.0

    study = create_study(storage_init_func())
    study.optimize(objective, n_trials=1)

    assert study.best_trial.distributions == {
        "a": UniformDistribution(low=0, high=10),
        "b": LogUniformDistribution(low=0.1, high=10),
        "c": DiscreteUniformDistribution(low=0, high=10, q=1),
        "d": IntUniformDistribution(low=0, high=10),
        "e": CategoricalDistribution(choices=("foo", "bar", "baz")),
        "f": IntLogUniformDistribution(low=1, high=10),
    }


def test_trial_should_prune():
    # type: () -> None

    pruner = DeterministicPruner(True)
    study = create_study(pruner=pruner)
    trial = Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 1)
    assert trial.should_prune()


def test_fixed_trial_suggest_float():
    # type: () -> None

    trial = FixedTrial({"x": 1.0})
    assert trial.suggest_float("x", -100.0, 100.0) == 1.0

    with pytest.raises(ValueError):
        trial.suggest_uniform("y", -100.0, 100.0)


def test_fixed_trial_suggest_uniform():
    # type: () -> None

    trial = FixedTrial({"x": 1.0})
    assert trial.suggest_uniform("x", -100.0, 100.0) == 1.0

    with pytest.raises(ValueError):
        trial.suggest_uniform("y", -100.0, 100.0)


def test_fixed_trial_suggest_loguniform():
    # type: () -> None

    trial = FixedTrial({"x": 0.99})
    assert trial.suggest_loguniform("x", 0.1, 1.0) == 0.99

    with pytest.raises(ValueError):
        trial.suggest_loguniform("y", 0.0, 1.0)


def test_fixed_trial_suggest_discrete_uniform():
    # type: () -> None

    trial = FixedTrial({"x": 0.9})
    assert trial.suggest_discrete_uniform("x", 0.0, 1.0, 0.1) == 0.9

    with pytest.raises(ValueError):
        trial.suggest_discrete_uniform("y", 0.0, 1.0, 0.1)


def test_fixed_trial_suggest_int():
    # type: () -> None

    trial = FixedTrial({"x": 1})
    assert trial.suggest_int("x", 0, 10) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("y", 0, 10)


def test_fixed_trial_suggest_int_log():
    # type: () -> None

    trial = FixedTrial({"x": 1})
    assert trial.suggest_int("x", 1, 10, log=True) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("y", 1, 10, log=True)


def test_fixed_trial_suggest_categorical():
    # type: () -> None

    # Integer categories.
    trial = FixedTrial({"x": 1})
    assert trial.suggest_categorical("x", [0, 1, 2, 3]) == 1

    with pytest.raises(ValueError):
        trial.suggest_categorical("y", [0, 1, 2, 3])

    # String categories.
    trial = FixedTrial({"x": "baz"})
    assert trial.suggest_categorical("x", ["foo", "bar", "baz"]) == "baz"

    # Unknown parameter.
    with pytest.raises(ValueError):
        trial.suggest_categorical("y", ["foo", "bar", "baz"])

    # Not in choices.
    with pytest.raises(ValueError):
        trial.suggest_categorical("x", ["foo", "bar"])

    # Unkown parameter and bad category type.
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):  # Must come after `pytest.warns` to catch failures.
            trial.suggest_categorical("x", [{"foo": "bar"}])  # type: ignore


def test_fixed_trial_user_attrs():
    # type: () -> None

    trial = FixedTrial({"x": 1})
    trial.set_user_attr("data", "MNIST")
    assert trial.user_attrs["data"] == "MNIST"


def test_fixed_trial_system_attrs():
    # type: () -> None

    trial = FixedTrial({"x": 1})
    trial.set_system_attr("system_message", "test")
    assert trial.system_attrs["system_message"] == "test"


def test_fixed_trial_params():
    # type: () -> None

    params = {"x": 1}
    trial = FixedTrial(params)
    assert trial.params == {}

    assert trial.suggest_uniform("x", 0, 10) == 1
    assert trial.params == params


def test_fixed_trial_report():
    # type: () -> None

    # FixedTrial ignores reported values.
    trial = FixedTrial({})
    trial.report(1.0, 1)
    trial.report(2.0, 2)


def test_fixed_trial_should_prune():
    # type: () -> None

    # FixedTrial never prunes trials.
    assert FixedTrial({}).should_prune() is False
    assert FixedTrial({}).should_prune(1) is False


def test_fixed_trial_datetime_start():
    # type: () -> None

    params = {"x": 1}
    trial = FixedTrial(params)
    assert trial.datetime_start is not None


def test_fixed_trial_number() -> None:

    params = {"x": 1}
    trial = FixedTrial(params, 2)
    assert trial.number == 2

    trial = FixedTrial(params)
    assert trial.number == 0


@parametrize_storage
def test_relative_parameters(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    relative_search_space = {
        "x": UniformDistribution(low=5, high=6),
        "y": UniformDistribution(low=5, high=6),
    }
    relative_params = {"x": 5.5, "y": 5.5, "z": 5.5}

    sampler = DeterministicRelativeSampler(relative_search_space, relative_params)  # type: ignore
    study = create_study(storage=storage_init_func(), sampler=sampler)

    def create_trial():
        # type: () -> Trial

        return Trial(study, study._storage.create_new_trial(study._study_id))

    # Suggested from `relative_params`.
    trial0 = create_trial()
    distribution0 = UniformDistribution(low=0, high=100)
    assert trial0._suggest("x", distribution0) == 5.5

    # Not suggested from `relative_params` (due to unknown parameter name).
    trial1 = create_trial()
    distribution1 = distribution0
    assert trial1._suggest("w", distribution1) != 5.5

    # Not suggested from `relative_params` (due to incompatible value range).
    trial2 = create_trial()
    distribution2 = UniformDistribution(low=0, high=5)
    assert trial2._suggest("x", distribution2) != 5.5

    # Error (due to incompatible distribution class).
    trial3 = create_trial()
    distribution3 = IntUniformDistribution(low=1, high=100)
    with pytest.raises(ValueError):
        trial3._suggest("y", distribution3)

    # Error ('z' is included in `relative_params` but not in `relative_search_space`).
    trial4 = create_trial()
    distribution4 = UniformDistribution(low=0, high=10)
    with pytest.raises(ValueError):
        trial4._suggest("z", distribution4)

    # Error (due to incompatible distribution class).
    trial5 = create_trial()
    distribution5 = IntLogUniformDistribution(low=1, high=100)
    with pytest.raises(ValueError):
        trial5._suggest("y", distribution5)


@parametrize_storage
def test_datetime_start(storage_init_func):
    # type: (Callable[[], storages.BaseStorage]) -> None

    trial_datetime_start = [None]  # type: List[Optional[datetime.datetime]]

    def objective(trial):
        # type: (Trial) -> float

        trial_datetime_start[0] = trial.datetime_start
        return 1.0

    study = create_study(storage_init_func())
    study.optimize(objective, n_trials=1)

    assert study.trials[0].datetime_start == trial_datetime_start[0]


def test_trial_report():
    # type: () -> None

    study = create_study()
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    # Report values that can be cast to `float` (OK).
    trial.report(1.23, 1)
    trial.report(float("nan"), 2)
    trial.report("1.23", 3)  # type: ignore
    trial.report("inf", 4)  # type: ignore
    trial.report(1, 5)
    trial.report(np.array([1], dtype=np.float32)[0], 6)

    # Report values that cannot be cast to `float` or steps that are negative (Error).
    with pytest.raises(TypeError):
        trial.report(None, 7)  # type: ignore

    with pytest.raises(TypeError):
        trial.report("foo", 7)  # type: ignore

    with pytest.raises(TypeError):
        trial.report([1, 2, 3], 7)  # type: ignore

    with pytest.raises(TypeError):
        trial.report("foo", -1)  # type: ignore

    with pytest.raises(ValueError):
        trial.report(1.23, -1)


def test_study_id():
    # type: () -> None

    study = create_study()
    trial = Trial(study, study._storage.create_new_trial(study._study_id))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        assert trial.study_id == trial.study._study_id

    with pytest.warns(DeprecationWarning):
        trial.study_id


def test_frozen_trial_validate():
    # type: () -> None

    # Valid.
    valid_trial = _create_frozen_trial()
    valid_trial._validate()

    # Invalid: `datetime_start` is not set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.datetime_start = None
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is `RUNNING` and `datetime_complete` is set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.state = TrialState.RUNNING
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is not `RUNNING` and `datetime_complete` is not set.
    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.state = state
        invalid_trial.datetime_complete = None
        with pytest.raises(ValueError):
            invalid_trial._validate()

    # Invalid: `state` is `COMPLETE` and `value` is not set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.value = None
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: Inconsistent `params` and `distributions`
    inconsistent_pairs = [
        # `params` has an extra element.
        ({"x": 0.1, "y": 0.5}, {"x": UniformDistribution(0, 1)}),
        # `distributions` has an extra element.
        ({"x": 0.1}, {"x": UniformDistribution(0, 1), "y": LogUniformDistribution(0.1, 1.0)}),
        # The value of `x` isn't contained in the distribution.
        ({"x": -0.5}, {"x": UniformDistribution(0, 1)}),
    ]  # type: List[Tuple[Dict[str, Any], Dict[str, BaseDistribution]]]

    for params, distributions in inconsistent_pairs:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.params = params
        invalid_trial.distributions = distributions
        with pytest.raises(ValueError):
            invalid_trial._validate()


def test_frozen_trial_eq_ne():
    # type: () -> None

    trial = _create_frozen_trial()

    trial_other = copy.copy(trial)
    assert trial == trial_other

    trial_other.value = 0.3
    assert trial != trial_other


def test_frozen_trial_lt():
    # type: () -> None

    trial = _create_frozen_trial()

    trial_other = copy.copy(trial)
    assert not trial < trial_other

    trial_other.number = trial.number + 1
    assert trial < trial_other
    assert not trial_other < trial

    with pytest.raises(TypeError):
        trial < 1

    assert trial <= trial_other
    assert not trial_other <= trial

    with pytest.raises(TypeError):
        trial <= 1

    # A list of FrozenTrials is sortable.
    trials = [trial_other, trial]
    trials.sort()
    assert trials[0] is trial
    assert trials[1] is trial_other


def _create_frozen_trial():
    # type: () -> FrozenTrial

    return FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": UniformDistribution(5, 12)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )


def test_frozen_trial_repr():
    # type: () -> None

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": UniformDistribution(5, 12)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )

    assert trial == eval(repr(trial))
