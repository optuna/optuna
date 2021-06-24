import pytest

import optuna


def test_warning() -> None:
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    with pytest.warns(FutureWarning):
        optuna.integration.CatalystPruningCallback("abc", "def", False, trial=trial)
