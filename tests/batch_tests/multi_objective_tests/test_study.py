from typing import List
import uuid

import numpy as np
import pytest

import optuna
from optuna.batch.multi_objective.trial import BatchMultiObjectiveTrial
from optuna.study import StudyDirection
from optuna.testing.storage import StorageSupplier


def test_create_study() -> None:
    study = optuna.batch.multi_objective.create_study(["maximize"])
    assert study.n_objectives == 1
    assert study.directions == [StudyDirection.MAXIMIZE]

    study = optuna.batch.multi_objective.create_study(
        ["maximize", "minimize"]
    )
    assert study.n_objectives == 2
    assert study.directions == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]

    with pytest.raises(ValueError):
        # Empty `directions` is not allowed.
        study = optuna.batch.multi_objective.create_study([])


def test_load_study() -> None:
    with StorageSupplier("sqlite") as storage:
        study_name = str(uuid.uuid4)

        with pytest.raises(KeyError):
            # Test loading an unexisting study.
            optuna.batch.multi_objective.study.load_study(study_name=study_name, storage=storage)

        created_study = optuna.batch.multi_objective.study.create_study(
            ["minimize"], study_name=study_name, storage=storage
        )

        # Test loading an existing study.
        loaded_study = optuna.multi_objective.study.load_study(
            study_name=study_name, storage=storage
        )
        assert created_study._study_id == loaded_study._study_id


@pytest.mark.parametrize("n_objectives", [1, 2, 3])
def test_optimize(n_objectives: int) -> None:
    directions = ["minimize" for _ in range(n_objectives)]
    study = optuna.batch.multi_objective.create_study(directions)

    def objective(trial: BatchMultiObjectiveTrial) -> List[np.ndarray]:
        return [trial.suggest_float("v{}".format(i), 0, 5) for i in range(n_objectives)]

    study.optimize(objective, n_batches=3, batch_size=2)

    assert len(study.trials) == 6

    for trial in study.trials:
        assert len(trial.values) == n_objectives


def test_pareto_front() -> None:
    study = optuna.batch.multi_objective.create_study(["minimize", "maximize"])

    study.optimize(lambda t: [np.array([2, 1]), np.array([2, 1])], n_batches=1, batch_size=2)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 1), (2, 2)}

    study.optimize(lambda t: [np.array([3, 1]), np.array([1, 3])], n_batches=1, batch_size=2)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 3)}
    assert len(study.get_pareto_front_trials()) == 1

    # Add the same trial results as the above ones.
    study.optimize(lambda t: [np.array([3, 1]), np.array([1, 3])], n_batches=1, batch_size=2)
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 3)}
    assert len(study.get_pareto_front_trials()) == 2
