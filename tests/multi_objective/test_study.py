from typing import List
import uuid

import pytest

import optuna
from optuna.structs import StudyDirection
from optuna.testing.storage import StorageSupplier


def test_create_study() -> None:
    study = optuna.multi_objective.create_study(["maximize"])
    assert study.n_objectives == 1
    assert study.directions == [StudyDirection.MAXIMIZE]

    study = optuna.multi_objective.create_study(["maximize", "minimize"])
    assert study.n_objectives == 2
    assert study.directions == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]


def test_load_study() -> None:
    with StorageSupplier("new") as storage:
        study_name = str(uuid.uuid4())

        with pytest.raises(ValueError):
            # Test loading an unexisting study.
            optuna.multi_objective.study.load_study(study_name=study_name, storage=storage)

        # Create a new study.
        created_study = optuna.multi_objective.study.create_study(
            ["minimize"], study_name=study_name, storage=storage
        )

        # Test loading an existing study.
        loaded_study = optuna.multi_objective.study.load_study(
            study_name=study_name, storage=storage
        )
        assert created_study._study._study_id == loaded_study._study._study_id


@pytest.mark.parametrize("n_objectives", [1, 2, 3])
def test_optimize(n_objectives: int) -> None:
    directions = ["minimize" for _ in range(n_objectives)]
    study = optuna.multi_objective.create_study(directions)

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        return [trial.suggest_uniform("v{}".format(i), 0, 5) for i in range(n_objectives)]

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10

    for trial in study.trials:
        assert len(trial.values) == n_objectives


def test_pareto_front() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == set()

    study.optimize(lambda t: [2, 2], n_trials=1)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(2, 2)}

    study.optimize(lambda t: [1, 1], n_trials=1)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 1), (2, 2)}

    study.optimize(lambda t: [3, 1], n_trials=1)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 1), (2, 2)}

    study.optimize(lambda t: [1, 3], n_trials=1)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 3)}


def test_study_user_attrs() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])
    assert study.user_attrs == {}

    study.set_user_attr("foo", "bar")
    assert study.user_attrs == {"foo": "bar"}


def test_study_system_attrs() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])
    assert study.system_attrs == {"multi_objective.study.directions": ["minimize", "maximize"]}

    study.set_system_attr("foo", "bar")
    assert study.system_attrs == {
        "multi_objective.study.directions": ["minimize", "maximize"],
        "foo": "bar",
    }


def test_enqueue_trial() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])

    study.enqueue_trial({"x": 2})
    study.enqueue_trial({"x": 3})

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        if trial.number == 0:
            assert trial.suggest_uniform("x", 0, 100) == 2
        elif trial.number == 1:
            assert trial.suggest_uniform("x", 0, 100) == 3

        return [0, 0]

    study.optimize(objective, n_trials=2)
