import datetime
from typing import cast

import numpy as np

import optuna


def test_suggest() -> None:
    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        p0 = trial.suggest_float("p0", -10, 10)
        p1 = trial.suggest_uniform("p1", 3, 5)
        p2 = trial.suggest_loguniform("p2", 0.00001, 0.1)
        p3 = trial.suggest_discrete_uniform("p3", 100, 200, q=5)
        p4 = trial.suggest_int("p4", -20, -15)
        p5 = cast(int, trial.suggest_categorical("p5", [7, 1, 100]))
        p6 = trial.suggest_float("p6", -10, 10, step=1.0)
        p7 = trial.suggest_int("p7", 1, 7, log=True)
        return (
            p0 + p1 + p2,
            p3 + p4 + p5 + p6 + p7,
        )

    study = optuna.batch.create_study(batch_size=4)
    study.optimize(objective, n_batches=10)


def test_report() -> None:
    batch_size = 3

    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        if trial._trials[0].number == 0:
            trial.report(np.ones(batch_size), 1)
            trial.report(10 * np.ones(batch_size), 2)
        return 100 * np.ones(batch_size)

    study = optuna.batch.create_study(batch_size=batch_size)
    study.optimize(objective, n_batches=2)

    for i in range(0, batch_size):
        trial = study.trials[i]
        assert trial.intermediate_values == {1: 1, 2: 10}
        assert trial.value == 100
        assert trial.last_step == 2

    for i in range(batch_size, 2 * batch_size):
        trial = study.trials[i]
        assert trial.intermediate_values == {}
        assert trial.value == 100
        assert trial.last_step is None


def test_number() -> None:
    batch_size = 4

    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        assert all(trial.number == np.array(range(batch_size)))
        return np.zeros(batch_size)

    study = optuna.batch.create_study(batch_size=batch_size)
    study.optimize(objective, n_batches=1)
    for i, trial in enumerate(study.trials):
        assert trial.number == i


def test_user_attrs() -> None:
    batch_size = 4

    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        trial.set_user_attr("foo", "bar")
        assert all(attr == {"foo": "bar"} for attr in trial.user_attrs)

        trial.set_user_attr("baz", "qux")
        assert all(attr == {"foo": "bar", "baz": "qux"} for attr in trial.user_attrs)

        trial.set_user_attr("foo", "quux")
        assert all(attr == {"foo": "quux", "baz": "qux"} for attr in trial.user_attrs)

        return np.ones(batch_size)

    study = optuna.batch.create_study(batch_size=batch_size)
    study.optimize(objective, n_batches=1)

    assert all(t.user_attrs == {"foo": "quux", "baz": "qux"} for t in study.trials)


def test_system_attrs() -> None:
    batch_size = 4

    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        trial.set_system_attr("foo", "bar")
        assert all(attr == {"foo": "bar"} for attr in trial.system_attrs)

        trial.set_system_attr("baz", "qux")
        assert all(attr == {"foo": "bar", "baz": "qux"} for attr in trial.system_attrs)

        trial.set_system_attr("foo", "quux")
        assert all(attr == {"foo": "quux", "baz": "qux"} for attr in trial.system_attrs)

        return np.ones(batch_size)

    study = optuna.batch.create_study(batch_size=batch_size)
    study.optimize(objective, n_batches=1)

    assert all(t.system_attrs == {"foo": "quux", "baz": "qux"} for t in study.trials)


def test_params_and_distributions() -> None:
    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        x = trial.suggest_float("x", -10, 10)
        assert all(set(params.keys()) == {"x"} for params in trial.params)
        assert all(set(distributions.keys()) == {"x"} for distributions in trial.distributions)
        return x

    study = optuna.batch.create_study(batch_size=4)
    study.optimize(objective, n_batches=1)
    assert all(set(trial.params.keys()) == {"x"} for trial in study.trials)
    assert all(set(trial.distributions.keys()) == {"x"} for trial in study.trials)
    assert all(
        isinstance(trial.distributions["x"], optuna.distributions.UniformDistribution)
        for trial in study.trials
    )


def test_datetime() -> None:
    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        x = trial.suggest_float("x", -10, 10)
        assert all(isinstance(d, datetime.datetime) for d in trial.datetime_start)
        return x

    study = optuna.batch.create_study(batch_size=4)
    study.optimize(objective, n_batches=1)
    assert all(isinstance(trial.datetime_start, datetime.datetime) for trial in study.trials)
    assert all(isinstance(trial.datetime_complete, datetime.datetime) for trial in study.trials)
