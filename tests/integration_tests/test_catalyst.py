import pytest

import optuna


pytestmark = pytest.mark.integration


def test_warning() -> None:
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    with pytest.warns(FutureWarning):
        optuna.integration.CatalystPruningCallback(
            trial=trial,
            loader_key="abc",
            metric_key="def",
            minimize=False,
        )
