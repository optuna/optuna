import py

import optuna
from optuna.integration.mlflow import MlflowCallback
from ..test_structs import _create_frozen_trial


def test_happy_case(tmpdir):
    # type: (py.path.local) -> None

    db_file_name = "sqlite:///{}/example.db".format(tmpdir)

    mlflc = MlflowCallback(
        tracking_uri=db_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)
