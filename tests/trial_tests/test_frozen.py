import copy
import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pytest

from optuna import create_study
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import BaseTrial
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def test_eq_ne() -> None:

    trial = _create_frozen_trial()

    trial_other = copy.copy(trial)
    assert trial == trial_other

    trial_other.value = 0.3
    assert trial != trial_other


def test_lt() -> None:

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


def _create_frozen_trial() -> FrozenTrial:

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


def test_repr() -> None:

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


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_sampling(storage_mode: str) -> None:
    def objective(trial: BaseTrial) -> float:

        a = trial.suggest_uniform("a", 0.0, 10.0)
        b = trial.suggest_loguniform("b", 0.1, 10.0)
        c = trial.suggest_discrete_uniform("c", 0.0, 10.0, 1.0)
        d = trial.suggest_int("d", 0, 10)
        e = trial.suggest_categorical("e", [0, 1, 2])
        f = trial.suggest_int("f", 1, 10, log=True)

        assert isinstance(e, int)
        return a + b + c + d + e + f

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(objective, n_trials=1)

        best_trial = study.best_trial

        # re-evaluate objective with the best hyperparameters
        v = objective(best_trial)

        assert v == best_trial.value


def test_suggest_float() -> None:

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 0.2},
        distributions={"x": UniformDistribution(0.0, 1.0)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )

    assert trial.suggest_float("x", 0.0, 1.0) == 0.2

    with pytest.raises(ValueError):
        trial.suggest_float("x", 0.0, 1.0, step=10, log=True)

    with pytest.raises(ValueError):
        trial.suggest_float("y", 0.0, 1.0)


def test_suggest_uniform() -> None:

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 0.2},
        distributions={"x": UniformDistribution(0.0, 1.0)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )

    assert trial.suggest_uniform("x", 0.0, 1.0) == 0.2

    with pytest.raises(ValueError):
        trial.suggest_uniform("y", 0.0, 1.0)


def test_suggest_loguniform() -> None:

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 0.99},
        distributions={"x": LogUniformDistribution(0.1, 1.0)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    assert trial.suggest_loguniform("x", 0.1, 1.0) == 0.99

    with pytest.raises(ValueError):
        trial.suggest_loguniform("y", 0.0, 1.0)


def test_suggest_discrete_uniform() -> None:

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 0.9},
        distributions={"x": DiscreteUniformDistribution(0.0, 1.0, q=0.1)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    assert trial.suggest_discrete_uniform("x", 0.0, 1.0, 0.1) == 0.9

    with pytest.raises(ValueError):
        trial.suggest_discrete_uniform("y", 0.0, 1.0, 0.1)


def test_suggest_int() -> None:

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 1},
        distributions={"x": IntUniformDistribution(0, 10)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )

    assert trial.suggest_int("x", 0, 10) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("y", 0, 10)


def test_suggest_int_log() -> None:

    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 1},
        distributions={"x": IntLogUniformDistribution(1, 10)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )

    assert trial.suggest_int("x", 1, 10, log=True) == 1

    with pytest.raises(ValueError):
        trial.suggest_int("x", 1, 10, step=2, log=True)

    with pytest.raises(ValueError):
        trial.suggest_int("y", 1, 10, log=True)


def test_suggest_categorical() -> None:

    # Integer categories.
    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 1},
        distributions={"x": CategoricalDistribution((0, 1, 2, 3))},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    assert trial.suggest_categorical("x", (0, 1, 2, 3)) == 1

    with pytest.raises(ValueError):
        trial.suggest_categorical("y", [0, 1, 2, 3])

    # String categories.
    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": "baz"},
        distributions={"x": CategoricalDistribution(("foo", "bar", "baz"))},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
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


def test_report() -> None:

    # FrozenTrial ignores reported values.
    trial = _create_frozen_trial()
    trial.report(1.0, 1)
    trial.report(2.0, 2)


def test_should_prune() -> None:

    # FrozenTrial never prunes trials.
    assert _create_frozen_trial().should_prune() is False


def test_set_user_attrs() -> None:

    trial = _create_frozen_trial()
    trial.set_user_attr("data", "MNIST")
    assert trial.user_attrs["data"] == "MNIST"


def test_set_system_attrs() -> None:

    trial = _create_frozen_trial()
    trial.set_system_attr("system_message", "test")
    assert trial.system_attrs["system_message"] == "test"


def test_set_value() -> None:

    trial = _create_frozen_trial()
    trial.value = 0.1
    assert trial.value == 0.1


def test_set_values() -> None:

    trial = _create_frozen_trial()
    trial.values = (0.1, 0.2)
    assert trial.values == [0.1, 0.2]

    trial = _create_frozen_trial()
    trial.values = [0.1, 0.2]
    assert trial.values == [0.1, 0.2]


def test_validate() -> None:

    # Valid.
    valid_trial = _create_frozen_trial()
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

    # Invalid: `state` is `COMPLETE` and `value` is not set.
    invalid_trial = copy.copy(valid_trial)
    invalid_trial.value = None
    with pytest.raises(ValueError):
        invalid_trial._validate()

    # Invalid: Inconsistent `params` and `distributions`
    inconsistent_pairs: List[Tuple[Dict[str, Any], Dict[str, BaseDistribution]]] = [
        # `params` has an extra element.
        ({"x": 0.1, "y": 0.5}, {"x": UniformDistribution(0, 1)}),
        # `distributions` has an extra element.
        ({"x": 0.1}, {"x": UniformDistribution(0, 1), "y": LogUniformDistribution(0.1, 1.0)}),
        # The value of `x` isn't contained in the distribution.
        ({"x": -0.5}, {"x": UniformDistribution(0, 1)}),
    ]

    for params, distributions in inconsistent_pairs:
        invalid_trial = copy.copy(valid_trial)
        invalid_trial.params = params
        invalid_trial.distributions = distributions
        with pytest.raises(ValueError):
            invalid_trial._validate()


def test_number() -> None:

    trial = _create_frozen_trial()
    assert trial.number == 0

    trial.number = 2
    assert trial.number == 2


def test_datetime_start() -> None:

    trial = _create_frozen_trial()
    assert trial.datetime_start is not None
    old_date_time_start = trial.datetime_start
    trial.datetime_complete = datetime.datetime.now()
    assert trial.datetime_complete != old_date_time_start


def test_params() -> None:

    params = {"x": 1}
    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params=params,
        distributions={"x": UniformDistribution(0, 10)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )

    assert trial.suggest_uniform("x", 0, 10) == 1
    assert trial.params == params

    params = {"x": 2}
    trial.params = params
    assert trial.suggest_uniform("x", 0, 10) == 2
    assert trial.params == params


def test_distributions() -> None:

    distributions = {"x": UniformDistribution(0, 10)}
    trial = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 1},
        distributions=dict(distributions),
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    assert trial.distributions == distributions

    distributions = {"x": UniformDistribution(1, 9)}
    trial.distributions = dict(distributions)
    assert trial.distributions == distributions


def test_user_attrs() -> None:

    trial = _create_frozen_trial()
    assert trial.user_attrs == {}

    user_attrs = {"data": "MNIST"}
    trial.user_attrs = user_attrs
    assert trial.user_attrs == user_attrs


def test_system_attrs() -> None:

    trial = _create_frozen_trial()
    assert trial.system_attrs == {}

    system_attrs = {"system_message": "test"}
    trial.system_attrs = system_attrs
    assert trial.system_attrs == system_attrs


def test_called_single_methods_when_multi() -> None:

    state = TrialState.COMPLETE
    values = (0.2, 0.3)
    params = {"x": 10}
    distributions = {"x": UniformDistribution(5, 12)}
    user_attrs = {"foo": "bar"}
    system_attrs = {"baz": "qux"}
    intermediate_values = {0: 0.0, 1: 0.1, 2: 0.1}

    trial = create_trial(
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
        trial.value = [0.1]


def test_init() -> None:
    def _create_trial(value: Optional[float], values: Optional[List[float]]) -> FrozenTrial:

        return FrozenTrial(
            number=0,
            trial_id=0,
            state=TrialState.COMPLETE,
            value=value,
            values=values,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={},
            distributions={"x": UniformDistribution(0, 10)},
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
