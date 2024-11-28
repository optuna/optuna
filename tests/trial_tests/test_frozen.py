from __future__ import annotations

import copy
import datetime
from typing import Any

import pytest

from optuna import create_study
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
import optuna.trial
from optuna.trial import BaseTrial
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _create_trial(
    *,
    value: float = 0.2,
    params: dict[str, Any] = {"x": 10},
    distributions: dict[str, BaseDistribution] = {"x": FloatDistribution(5, 12)},
) -> FrozenTrial:
    trial = optuna.trial.create_trial(value=value, params=params, distributions=distributions)
    trial.number = 0
    return trial


def test_eq_ne() -> None:
    trial = _create_trial()

    trial_other = copy.copy(trial)
    assert trial == trial_other

    trial_other.value = 0.3
    assert trial != trial_other


def test_lt() -> None:
    trial = _create_trial()

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


def test_repr() -> None:
    trial = _create_trial()

    assert trial == eval(repr(trial))


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_sampling(storage_mode: str) -> None:
    def objective(trial: BaseTrial) -> float:
        a = trial.suggest_float("a", 0.0, 10.0)
        b = trial.suggest_float("b", 0.1, 10.0, log=True)
        c = trial.suggest_float("c", 0.0, 10.0, step=1.0)
        d = trial.suggest_int("d", 0, 10)
        e = trial.suggest_categorical("e", [0, 1, 2])
        f = trial.suggest_int("f", 1, 10, log=True)

        return a + b + c + d + e + f

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(objective, n_trials=1)

        best_trial = study.best_trial

        # re-evaluate objective with the best hyperparameters
        v = objective(best_trial)

        assert v == best_trial.value


def test_set_value() -> None:
    trial = _create_trial()
    trial.value = 0.1
    assert trial.value == 0.1


def test_set_values() -> None:
    trial = _create_trial()
    trial.values = (0.1, 0.2)
    assert trial.values == [0.1, 0.2]  # type: ignore[comparison-overlap]

    trial = _create_trial()
    trial.values = [0.1, 0.2]
    assert trial.values == [0.1, 0.2]


def test_validate() -> None:
    # Valid.
    valid_trial = _create_trial()
    valid_trial._validate()

    # Invalid: `datetime_start` is not set when the trial is not in the waiting state.
    for state in [
        TrialState.RUNNING,
        TrialState.COMPLETE,
        TrialState.PRUNED,
        TrialState.FAIL,
    ]:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.state = state
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

    # Invalid: `state` is `FAIL`, and `value` is set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.state = TrialState.FAIL
    invalid_trial.value = 1.0
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is `COMPLETE` and `value` is not set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.value = None
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is `COMPLETE` and `value` is NaN.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.value = float("nan")
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: `state` is `COMPLETE` and `values` includes NaN.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.values = [0.0, float("nan")]
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: Inconsistent `params` and `distributions`
    inconsistent_pairs: list[tuple[dict[str, Any], dict[str, BaseDistribution]]] = [
        # `params` has an extra element.
        ({"x": 0.1, "y": 0.5}, {"x": FloatDistribution(0, 1)}),
        # `distributions` has an extra element.
        ({"x": 0.1}, {"x": FloatDistribution(0, 1), "y": FloatDistribution(0.1, 1.0, log=True)}),
        # The value of `x` isn't contained in the distribution.
        ({"x": -0.5}, {"x": FloatDistribution(0, 1)}),
    ]

    for params, distributions in inconsistent_pairs:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.params = params
        invalid_trial.distributions = distributions
        with pytest.raises(ValueError):
            invalid_trial._validate()


def test_number() -> None:
    trial = _create_trial()
    assert trial.number == 0

    trial.number = 2
    assert trial.number == 2


def test_params() -> None:
    params = {"x": 1}
    trial = _create_trial(
        value=0.2,
        params=params,
        distributions={"x": FloatDistribution(0, 10)},
    )

    assert trial.suggest_float("x", 0, 10) == 1
    assert trial.params == params

    params = {"x": 2}
    trial.params = params
    assert trial.suggest_float("x", 0, 10) == 2
    assert trial.params == params


def test_distributions() -> None:
    distributions = {"x": FloatDistribution(0, 10)}
    trial = _create_trial(
        value=0.2,
        params={"x": 1},
        distributions=dict(distributions),
    )
    assert trial.distributions == distributions

    distributions = {"x": FloatDistribution(1, 9)}
    trial.distributions = dict(distributions)
    assert trial.distributions == distributions


def test_user_attrs() -> None:
    trial = _create_trial()
    assert trial.user_attrs == {}

    user_attrs = {"data": "MNIST"}
    trial.user_attrs = user_attrs
    assert trial.user_attrs == user_attrs


def test_system_attrs() -> None:
    trial = _create_trial()
    assert trial.system_attrs == {}

    system_attrs = {"system_message": "test"}
    trial.system_attrs = system_attrs
    assert trial.system_attrs == system_attrs


def test_called_single_methods_when_multi() -> None:
    state = TrialState.COMPLETE
    values = (0.2, 0.3)
    params = {"x": 10}
    distributions: dict[str, BaseDistribution] = {"x": FloatDistribution(5, 12)}
    user_attrs = {"foo": "bar"}
    system_attrs = {"baz": "qux"}
    intermediate_values = {0: 0.0, 1: 0.1, 2: 0.1}

    trial = optuna.trial.create_trial(
        state=state,
        values=values,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        intermediate_values=intermediate_values,
    )

    with pytest.raises(RuntimeError):
        trial.value

    with pytest.raises(RuntimeError):
        trial.value = 0.1

    with pytest.raises(RuntimeError):
        trial.value = [0.1]  # type: ignore


def test_init() -> None:
    def _create_trial(value: float | None, values: list[float] | None) -> FrozenTrial:
        return FrozenTrial(
            number=0,
            trial_id=0,
            state=TrialState.COMPLETE,
            value=value,
            values=values,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={},
            distributions={"x": FloatDistribution(0, 10)},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
        )

    _ = _create_trial(0.2, None)

    _ = _create_trial(None, [0.2])

    with pytest.raises(ValueError):
        _ = _create_trial(0.2, [0.2])

    with pytest.raises(ValueError):
        _ = _create_trial(0.2, [])


# TODO(hvy): Write exhaustive test include invalid combinations when feature is no longer
# experimental.
@pytest.mark.parametrize("state", [TrialState.COMPLETE, TrialState.FAIL])
def test_create_trial(state: TrialState) -> None:
    value: float | None = 0.2
    params = {"x": 10}
    distributions: dict[str, BaseDistribution] = {"x": FloatDistribution(5, 12)}
    user_attrs = {"foo": "bar"}
    system_attrs = {"baz": "qux"}
    intermediate_values = {0: 0.0, 1: 0.1, 2: 0.1}

    if state == TrialState.FAIL:
        value = None

    trial = create_trial(
        state=state,
        value=value if state == TrialState.COMPLETE else None,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        intermediate_values=intermediate_values,
    )

    assert isinstance(trial, FrozenTrial)
    assert trial.state == state
    assert trial.value == value
    assert trial.params == params
    assert trial.distributions == distributions
    assert trial.user_attrs == user_attrs
    assert trial.system_attrs == system_attrs
    assert trial.intermediate_values == intermediate_values
    assert trial.datetime_start is not None
    assert (trial.datetime_complete is not None) == state.is_finished()

    with pytest.raises(ValueError):
        create_trial(
            state=state,
            value=0.2 if state != TrialState.COMPLETE else None,
            params=params,
            distributions=distributions,
            user_attrs=user_attrs,
            system_attrs=system_attrs,
            intermediate_values=intermediate_values,
        )


# Deprecated distributions are internally converted to corresponding distributions.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_create_trial_distribution_conversion() -> None:
    fixed_params = {
        "ud": 0,
        "dud": 2,
        "lud": 1,
        "id": 0,
        "idd": 2,
        "ild": 1,
    }

    fixed_distributions = {
        "ud": optuna.distributions.UniformDistribution(low=0, high=10),
        "dud": optuna.distributions.DiscreteUniformDistribution(low=0, high=10, q=2),
        "lud": optuna.distributions.LogUniformDistribution(low=1, high=10),
        "id": optuna.distributions.IntUniformDistribution(low=0, high=10),
        "idd": optuna.distributions.IntUniformDistribution(low=0, high=10, step=2),
        "ild": optuna.distributions.IntLogUniformDistribution(low=1, high=10),
    }

    with pytest.warns(
        FutureWarning,
        match="See https://github.com/optuna/optuna/issues/2941",
    ) as record:
        trial = create_trial(params=fixed_params, distributions=fixed_distributions, value=1)
        assert len(record) == 6

    expected_distributions = {
        "ud": optuna.distributions.FloatDistribution(low=0, high=10, log=False, step=None),
        "dud": optuna.distributions.FloatDistribution(low=0, high=10, log=False, step=2),
        "lud": optuna.distributions.FloatDistribution(low=1, high=10, log=True, step=None),
        "id": optuna.distributions.IntDistribution(low=0, high=10, log=False, step=1),
        "idd": optuna.distributions.IntDistribution(low=0, high=10, log=False, step=2),
        "ild": optuna.distributions.IntDistribution(low=1, high=10, log=True, step=1),
    }

    assert trial.distributions == expected_distributions


# It confirms that ask doesn't convert non-deprecated distributions.
def test_create_trial_distribution_conversion_noop() -> None:
    fixed_params = {
        "ud": 0,
        "dud": 2,
        "lud": 1,
        "id": 0,
        "idd": 2,
        "ild": 1,
        "cd": "a",
    }

    fixed_distributions = {
        "ud": optuna.distributions.FloatDistribution(low=0, high=10, log=False, step=None),
        "dud": optuna.distributions.FloatDistribution(low=0, high=10, log=False, step=2),
        "lud": optuna.distributions.FloatDistribution(low=1, high=10, log=True, step=None),
        "id": optuna.distributions.IntDistribution(low=0, high=10, log=False, step=1),
        "idd": optuna.distributions.IntDistribution(low=0, high=10, log=False, step=2),
        "ild": optuna.distributions.IntDistribution(low=1, high=10, log=True, step=1),
        "cd": optuna.distributions.CategoricalDistribution(choices=["a", "b", "c"]),
    }

    trial = create_trial(params=fixed_params, distributions=fixed_distributions, value=1)

    # Check fixed_distributions doesn't change.
    assert trial.distributions == fixed_distributions


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize("positional_args_names", [[], ["step"], ["step", "log"]])
def test_suggest_int_positional_args(positional_args_names: list[str]) -> None:
    # If log is specified as positional, step must also be provided as positional.
    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.0,
        values=None,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 1},
        distributions={"x": IntDistribution(-1, 1)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    kwargs = dict(step=1, log=False)
    args = [kwargs[name] for name in positional_args_names]
    # No error should not be raised even if the coding style is old.
    trial.suggest_int("x", -1, 1, *args)
