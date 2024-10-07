import os

import pytest

import optuna
from optuna.storages.journal import JournalFileBackend


all_journal_files = [f"{os.path.dirname(__file__)}/assets/4.0.0.dev.log"]


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_empty_study(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study = optuna.load_study(study_name="single_empty", storage=storage)

    assert study.directions == [optuna.study.StudyDirection.MINIMIZE]
    assert len(study.user_attrs) == 0
    assert len(study.trials) == 0


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_create_and_delete_study(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    assert len(storage.get_all_studies()) == 4
    assert storage.get_study_id_from_name("single_empty") is not None
    with pytest.raises(KeyError):
        storage.get_study_id_from_name("single_to_be_deleted")


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_set_study_user_and_system_attrs(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study_id = storage.get_study_id_from_name("single_user_attr")
    user_attrs = storage.get_study_user_attrs(study_id)
    assert user_attrs["a"] == 1
    assert len(user_attrs) == 3

    study_id = storage.get_study_id_from_name("single_system_attr")
    system_attrs = storage.get_study_system_attrs(study_id)
    assert system_attrs["A"] == 1
    assert len(system_attrs) == 3


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_create_trial(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study_id = storage.get_study_id_from_name("single_optimization")
    trials = storage.get_all_trials(study_id)
    assert len(trials) == 10


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_set_trial_param(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study_id = storage.get_study_id_from_name("single_optimization")
    trials = storage.get_all_trials(study_id)

    for trial in trials:
        assert -5 <= trial.params["x"] <= 5
        assert 0 <= trial.params["y"] <= 10
        assert trial.params["z"] in [-5, 0, 5]


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_set_trial_state_values(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study_id = storage.get_study_id_from_name("single_optimization")
    trials = storage.get_all_trials(study_id)

    for trial in trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert trial.value is not None and 0 <= trial.value <= 150


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_set_trial_intermediate_value(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study_id = storage.get_study_id_from_name("single_optimization")
    trials = storage.get_all_trials(study_id)

    for trial in trials:
        assert len(trial.intermediate_values) == 1
        assert trial.intermediate_values[0] == 0.5


@pytest.mark.parametrize("journal_file", all_journal_files)
def test_set_trial_user_and_system_attrs(journal_file: str) -> None:
    storage = optuna.storages.JournalStorage(JournalFileBackend(journal_file))

    study_id = storage.get_study_id_from_name("single_optimization")
    trials = storage.get_all_trials(study_id)

    for trial in trials:
        assert trial.user_attrs[f"a_{trial.number}"] == 0
        assert trial.system_attrs[f"b_{trial.number}"] == 1
