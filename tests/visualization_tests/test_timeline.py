from io import BytesIO
from typing import List

import pytest

import optuna
from optuna.study.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import go
import optuna.visualization._timeline
from optuna.visualization._timeline import _get_timeline_info


@pytest.mark.parametrize(
    "n",
    [0, 3],
)
def test_get_timeline_info(n: int) -> None:
    study = optuna.create_study()
    for _ in range(n):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 0.5},
                distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
                value=0,
                state=optuna.trial.TrialState.COMPLETE,
            )
        )
    assert len(_get_timeline_info(study).bars) == n
    if n > 0:
        assert type(_get_timeline_info(study).bars[0].frozen_trial) is FrozenTrial


def _create_study(xs: List[TrialState]) -> Study:
    study = optuna.create_study()
    for i, x in enumerate(xs):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(i)},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, float(len(xs)))},
                value=0.0,
                state=x,
            )
        )
    return study


@pytest.mark.parametrize(
    "xs",
    [
        [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL],
        [TrialState.FAIL, TrialState.PRUNED, TrialState.COMPLETE],
    ],
)
def test_get_timeline_plot(xs: List[TrialState]) -> None:
    study = _create_study(xs)
    info = _get_timeline_info(study)

    fig = optuna.visualization._timeline._get_timeline_plot(info)
    assert type(fig) is go.Figure
    fig.write_image(BytesIO())
