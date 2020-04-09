import datetime
from typing import List

import optuna


def test_suggest() -> None:
    study = optuna.multi_objective.create_study(["maximize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        p0 = trial.suggest_float("p0", -10, 10)
        p1 = trial.suggest_uniform("p1", 3, 5)
        p2 = trial.suggest_loguniform("p2", 0.00001, 0.1)
        p3 = trial.suggest_discrete_uniform("p3", 100, 200, q=5)
        p4 = trial.suggest_int("p4", -20, -15)
        p5 = trial.suggest_categorical("p5", [7, 1, 100])
        return [p0 + p1 + p2, p3 + p4 + p5]

    study.optimize(objective, n_trials=10)


def test_report() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        trial.report([1, 2, 3], 1)
        trial.report([10, 20, 30], 2)
        return [100, 200, 300]

    study.optimize(objective, n_trials=1)
    trial = study.trials[0]

    assert trial.intermediate_values == {1: [1, 2, 3], 2: [10, 20, 30]}
    assert trial.values == [100, 200, 300]


def test_number() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(
        trial: optuna.multi_objective.trial.MultiObjectiveTrial, number: int
    ) -> List[float]:
        assert trial.number == number
        return [0, 0, 0]

    for i in range(10):
        study.optimize(lambda t: objective(t, i), n_trials=1)

    for i, trial in enumerate(study.trials):
        assert trial.number == i


def test_user_attrs() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        trial.set_user_attr("foo", "bar")
        assert trial.user_attrs == {"foo": "bar"}
        return [0, 0, 0]

    study.optimize(objective, n_trials=1)

    assert study.trials[0].user_attrs == {"foo": "bar"}


def test_system_attrs() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        trial.set_system_attr("foo", "bar")
        assert trial.system_attrs == {"foo": "bar"}
        return [0, 0, 0]

    study.optimize(objective, n_trials=1)

    assert study.trials[0].system_attrs == {"foo": "bar"}


def test_params_and_distributions() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        x = trial.suggest_uniform("x", 0, 10)

        assert set(trial.params.keys()) == {"x"}
        assert set(trial.distributions.keys()) == {"x"}
        assert isinstance(trial.distributions["x"], optuna.distributions.UniformDistribution)

        return [x, x, x]

    study.optimize(objective, n_trials=1)

    trial = study.trials[0]
    assert set(trial.params.keys()) == {"x"}
    assert set(trial.distributions.keys()) == {"x"}
    assert isinstance(trial.distributions["x"], optuna.distributions.UniformDistribution)


def test_datetime() -> None:
    study = optuna.multi_objective.create_study(["maximize", "minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        assert isinstance(trial.datetime_start, datetime.datetime)

        return [0, 0, 0]

    study.optimize(objective, n_trials=1)

    assert isinstance(study.trials[0].datetime_start, datetime.datetime)
    assert isinstance(study.trials[0].datetime_complete, datetime.datetime)
