import py
from unittest.mock import patch

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


@patch('mlflow.set_tags')
def test_set_tags(set_tags, tmpdir):
    # type: (unittest.mock.MagicMock, py.path.local) -> None

    db_file_name = "sqlite:///{}/example.db".format(tmpdir)

    mlflc = MlflowCallback(
        tracking_uri=db_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)

    assert set_tags.called
    assert set_tags.call_count == 1
    call_arg = set_tags.call_args_list[0][0][0]
    assert call_arg['trial_state'] == 'TrialState.COMPLETE'
    assert call_arg['x'] == 'UniformDistribution(high=12, low=5)'
    assert call_arg['trial_number'] == '0'
