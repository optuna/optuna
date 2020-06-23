from typing import List
from typing import Tuple
from unittest.mock import patch
import uuid

import _pytest.capture
import pytest

import optuna
from optuna.study import StudyDirection
from optuna.testing.storage import StorageSupplier


def test_create_study() -> None:
    study = optuna.multi_objective.create_study(["maximize"])
    assert study.n_objectives == 1
    assert study.directions == [StudyDirection.MAXIMIZE]

    study = optuna.multi_objective.create_study(["maximize", "minimize"])
    assert study.n_objectives == 2
    assert study.directions == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]

    with pytest.raises(ValueError):
        # Empty `directions` isn't allowed.
        study = optuna.multi_objective.create_study([])


def test_load_study() -> None:
    with StorageSupplier("sqlite") as storage:
        study_name = str(uuid.uuid4())

        with pytest.raises(KeyError):
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
    assert len(study.get_pareto_front_trials()) == 1

    study.optimize(lambda t: [1, 3], n_trials=1)  # The trial result is the same as the above one.
    assert {tuple(t.values) for t in study.get_pareto_front_trials()} == {(1, 3)}
    assert len(study.get_pareto_front_trials()) == 2


def test_study_user_attrs() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])
    assert study.user_attrs == {}

    study.set_user_attr("foo", "bar")
    assert study.user_attrs == {"foo": "bar"}

    study.set_user_attr("baz", "qux")
    assert study.user_attrs == {"foo": "bar", "baz": "qux"}

    study.set_user_attr("foo", "quux")
    assert study.user_attrs == {"foo": "quux", "baz": "qux"}


def test_study_system_attrs() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])
    assert study.system_attrs == {"multi_objective:study:directions": ["minimize", "maximize"]}

    study.set_system_attr("foo", "bar")
    assert study.system_attrs == {
        "multi_objective:study:directions": ["minimize", "maximize"],
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


def test_callbacks() -> None:
    study = optuna.multi_objective.create_study(["minimize", "maximize"])

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> Tuple[float, float]:
        x = trial.suggest_float("x", 0, 10)
        y = trial.suggest_float("y", 0, 10)
        return x, y

    list0 = []
    list1 = []
    callbacks = [
        lambda study, trial: list0.append(trial.number),
        lambda study, trial: list1.append(trial.number),
    ]
    study.optimize(objective, n_trials=2, callbacks=callbacks)

    assert list0 == [0, 1]
    assert list1 == [0, 1]


def test_log_completed_trial(capsys: _pytest.capture.CaptureFixture) -> None:

    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.multi_objective.create_study(["minimize", "maximize"])
    study.optimize(lambda t: (1.0, 1.0), n_trials=1)
    _, err = capsys.readouterr()
    assert "Trial 0" in err

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda t: (1.0, 1.0), n_trials=1)
    _, err = capsys.readouterr()
    assert "Trial 1" not in err

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study.optimize(lambda t: (1.0, 1.0), n_trials=1)
    _, err = capsys.readouterr()
    assert "Trial 2" in err


def test_log_completed_trial_skip_storage_access() -> None:

    study = optuna.multi_objective.create_study(["minimize", "maximize"])

    new_trial_id = study._study._storage.create_new_trial(study._study._study_id)
    trial = optuna.Trial(study._study, new_trial_id)
    storage = study._storage

    with patch.object(storage, "get_trial", wraps=storage.get_trial) as mock_object:
        optuna.multi_objective.study._log_completed_trial(study, trial, 1.0)
        # Trial.params and MultiObjectiveTrial._get_values access storage.
        assert mock_object.call_count == 2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with patch.object(storage, "get_trial", wraps=storage.get_trial) as mock_object:
        optuna.multi_objective.study._log_completed_trial(study, trial, 1.0)
        assert mock_object.call_count == 0

    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    with patch.object(storage, "get_trial", wraps=storage.get_trial) as mock_object:
        optuna.multi_objective.study._log_completed_trial(study, trial, 1.0)
        assert mock_object.call_count == 2
