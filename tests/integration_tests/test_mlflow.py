import pytest

import optuna
from optuna.integration.mlflow import MlflowCallback


def test_experiment_or_study_name():
    # type: () -> None

    mlflc = MlflowCallback()
    study = optuna.create_study()
    with pytest.raises(ValueError):
        mlflc(study, None)
