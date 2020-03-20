import pytest
from unittest.mock import patch

import optuna
from optuna.integration.mlflow import MlflowCallback
from ..test_structs import _create_frozen_trial


def test_experiment_or_study_name():
    # type: () -> None

    mlflc = MlflowCallback()
    study = optuna.create_study()
    ft = _create_frozen_trial()
    with pytest.raises(ValueError):
        mlflc(study, ft)
