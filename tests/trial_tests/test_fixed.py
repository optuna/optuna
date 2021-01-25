import pytest

from optuna.trial import FixedTrial


def test_suggest_float() -> None:

    trial = FixedTrial({"x": 1.0})
    assert trial.suggest_float("x", -100.0, 100.0) == 1.0

    with pytest.raises(ValueError):
        trial.suggest_float("x", -100, 100, step=10, log=True)

    with pytest.raises(ValueError):
        trial.suggest_uniform("y", -100.0, 100.0)


def test_suggest_uniform() -> None:

    trial = FixedTrial({"x": 1.0})
    assert trial.suggest_uniform("x", -100.0, 100.0) == 1.0

    with pytest.raises(ValueError):
        trial.suggest_uniform("y", -100.0, 100.0)


def test_suggest_loguniform() -> None:

    trial = FixedTrial({"x": 0.99})
    assert trial.suggest_loguniform("x", 0.1, 1.0) == 0.99

    with pytest.raises(ValueError):
        trial.suggest_loguniform("y", 0.0, 1.0)


def test_suggest_discrete_uniform() -> None:

    trial = FixedTrial({"x": 0.9})
    assert trial.suggest_discrete_uniform("x", 0.0, 1.0, 0.1) == 0.9

    with pytest.raises(ValueError):
        trial.suggest_discrete_uniform("y", 0.0, 1.0, 0.1)


def test_suggest_int() -> None:

    trial = FixedTrial({"x": 1})
    assert trial.suggest_int("x", 0, 10) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("y", 0, 10)


def test_suggest_int_log() -> None:

    trial = FixedTrial({"x": 1})
    assert trial.suggest_int("x", 1, 10, log=True) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("x", 1, 10, step=2, log=True)

    with pytest.raises(ValueError):
        trial.suggest_int("y", 1, 10, log=True)


def test_suggest_categorical() -> None:

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

    # Unknown parameter and bad category type.
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):  # Must come after `pytest.warns` to catch failures.
            trial.suggest_categorical("x", [{"foo": "bar"}])  # type: ignore


def test_user_attrs() -> None:

    trial = FixedTrial({"x": 1})
    trial.set_user_attr("data", "MNIST")
    assert trial.user_attrs["data"] == "MNIST"


def test_system_attrs() -> None:

    trial = FixedTrial({"x": 1})
    trial.set_system_attr("system_message", "test")
    assert trial.system_attrs["system_message"] == "test"


def test_params() -> None:

    params = {"x": 1}
    trial = FixedTrial(params)
    assert trial.params == {}

    assert trial.suggest_uniform("x", 0, 10) == 1
    assert trial.params == params


def test_report() -> None:

    # FixedTrial ignores reported values.
    trial = FixedTrial({})
    trial.report(1.0, 1)
    trial.report(2.0, 2)


def test_should_prune() -> None:

    # FixedTrial never prunes trials.
    assert FixedTrial({}).should_prune() is False


def test_datetime_start() -> None:

    params = {"x": 1}
    trial = FixedTrial(params)
    assert trial.datetime_start is not None


def test_number() -> None:

    params = {"x": 1}
    trial = FixedTrial(params, 2)
    assert trial.number == 2

    trial = FixedTrial(params)
    assert trial.number == 0
