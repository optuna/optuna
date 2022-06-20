import datetime
from typing import Any
from typing import Dict

import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import BaseTrial
from optuna.trial import create_trial
from optuna.trial import FixedTrial
from optuna.trial import FrozenTrial


parametrize_trial_type = pytest.mark.parametrize("trial_type", [FixedTrial, FrozenTrial])


def _create_trial(
    trial_type: type,
    params: Dict[str, Any] = {"x": 10},
    distributions: Dict[str, BaseDistribution] = {"x": FloatDistribution(5, 12)},
) -> BaseTrial:
    if trial_type == FixedTrial:
        return FixedTrial(params)
    elif trial_type == FrozenTrial:
        trial = create_trial(value=0.2, params=params, distributions=distributions)
        trial.number = 0
        return trial
    else:
        assert False


@parametrize_trial_type
def test_suggest_float(trial_type: type) -> None:

    trial = _create_trial(
        trial_type=trial_type, params={"x": 0.2}, distributions={"x": FloatDistribution(0.0, 1.0)}
    )

    assert trial.suggest_float("x", 0.0, 1.0) == 0.2

    with pytest.raises(ValueError):
        trial.suggest_float("x", 0.0, 1.0, step=10, log=True)

    with pytest.raises(ValueError):
        trial.suggest_float("y", 0.0, 1.0)


@parametrize_trial_type
def test_suggest_uniform(trial_type: type) -> None:

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 0.2},
        distributions={"x": FloatDistribution(0.0, 1.0)},
    )

    assert trial.suggest_uniform("x", 0.0, 1.0) == 0.2

    with pytest.raises(ValueError):
        trial.suggest_uniform("y", 0.0, 1.0)


@parametrize_trial_type
def test_suggest_loguniform(trial_type: type) -> None:

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 0.99},
        distributions={"x": FloatDistribution(0.1, 1.0, log=True)},
    )
    assert trial.suggest_loguniform("x", 0.1, 1.0) == 0.99

    with pytest.raises(ValueError):
        trial.suggest_loguniform("y", 0.0, 1.0)


@parametrize_trial_type
def test_suggest_discrete_uniform(trial_type: type) -> None:

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 0.9},
        distributions={"x": FloatDistribution(0.0, 1.0, step=0.1)},
    )
    assert trial.suggest_discrete_uniform("x", 0.0, 1.0, 0.1) == 0.9

    with pytest.raises(ValueError):
        trial.suggest_discrete_uniform("y", 0.0, 1.0, 0.1)


@parametrize_trial_type
def test_suggest_int(trial_type: type) -> None:

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1},
        distributions={"x": IntDistribution(0, 10)},
    )

    assert trial.suggest_int("x", 0, 10) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("y", 0, 10)


@parametrize_trial_type
def test_suggest_int_log(trial_type: type) -> None:

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1},
        distributions={"x": IntDistribution(1, 10, log=True)},
    )

    assert trial.suggest_int("x", 1, 10, log=True) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("x", 1, 10, step=2, log=True)

    with pytest.raises(ValueError):
        trial.suggest_int("y", 1, 10, log=True)


@parametrize_trial_type
def test_suggest_categorical(trial_type: type) -> None:

    # Integer categories.
    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1},
        distributions={"x": CategoricalDistribution((0, 1, 2, 3))},
    )
    assert trial.suggest_categorical("x", (0, 1, 2, 3)) == 1

    with pytest.raises(ValueError):
        trial.suggest_categorical("y", [0, 1, 2, 3])

    # String categories.
    trial = _create_trial(
        trial_type=trial_type,
        params={"x": "baz"},
        distributions={"x": CategoricalDistribution(("foo", "bar", "baz"))},
    )
    assert trial.suggest_categorical("x", ("foo", "bar", "baz")) == "baz"

    # Unknown parameter.
    with pytest.raises(ValueError):
        trial.suggest_categorical("y", ["foo", "bar", "baz"])

    # Not in choices.
    with pytest.raises(ValueError):
        trial.suggest_categorical("x", ["foo", "bar"])

    # Unknown parameter and bad category type.
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):  # Must come after `pytest.warns` to catch failures.
            trial.suggest_categorical("x", [{"foo": "bar"}])  # type: ignore


@parametrize_trial_type
def test_not_contained_param(trial_type: type) -> None:
    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1.0},
        distributions={"x": FloatDistribution(1.0, 10.0)},
    )
    with pytest.warns(UserWarning):
        assert trial.suggest_float("x", 10.0, 100.0) == 1.0

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1.0},
        distributions={"x": FloatDistribution(1.0, 10.0, log=True)},
    )
    with pytest.warns(UserWarning):
        assert trial.suggest_float("x", 10.0, 100.0, log=True) == 1.0

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1.0},
        distributions={"x": FloatDistribution(1.0, 10.0, step=1.0)},
    )
    with pytest.warns(UserWarning):
        assert trial.suggest_float("x", 10.0, 100.0, step=1.0) == 1.0

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1.0},
        distributions={"x": IntDistribution(1, 10)},
    )
    with pytest.warns(UserWarning):
        assert trial.suggest_int("x", 10, 100) == 1

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1},
        distributions={"x": IntDistribution(1, 10)},
    )
    with pytest.warns(UserWarning):
        assert trial.suggest_int("x", 10, 100, 1) == 1

    trial = _create_trial(
        trial_type=trial_type,
        params={"x": 1},
        distributions={"x": IntDistribution(1, 10, log=True)},
    )
    with pytest.warns(UserWarning):
        assert trial.suggest_int("x", 10, 100, log=True) == 1


@parametrize_trial_type
def test_set_user_attrs(trial_type: type) -> None:

    trial = _create_trial(trial_type)
    trial.set_user_attr("data", "MNIST")
    assert trial.user_attrs["data"] == "MNIST"


@parametrize_trial_type
def test_set_system_attrs(trial_type: type) -> None:

    trial = _create_trial(trial_type)
    trial.set_system_attr("system_message", "test")
    assert trial.system_attrs["system_message"] == "test"


@parametrize_trial_type
def test_report(trial_type: type) -> None:

    # FrozenTrial ignores reported values.
    trial = _create_trial(trial_type)
    trial.report(1.0, 1)
    trial.report(2.0, 2)


@parametrize_trial_type
def test_should_prune(trial_type: type) -> None:

    # FrozenTrial never prunes trials.
    assert _create_trial(trial_type).should_prune() is False


@parametrize_trial_type
def test_datetime_start(trial_type: type) -> None:

    trial = _create_trial(trial_type)
    assert trial.datetime_start is not None
    old_date_time_start = trial.datetime_start
    trial.datetime_complete = datetime.datetime.now()  # type: ignore
    assert trial.datetime_complete != old_date_time_start  # type: ignore
