from io import BytesIO
from typing import Any
from typing import Callable

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.trial import create_trial
from optuna.visualization._plotly_imports import go
import optuna.visualization._slice
from optuna.visualization._slice import _get_slice_plot_info
from optuna.visualization._slice import _SlicePlotInfo
from optuna.visualization._slice import _SliceSubplotInfo
from optuna.visualization.matplotlib._matplotlib_imports import plt
import optuna.visualization.matplotlib._slice


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_slice_plot_info(study, None, None, "target_name")


def test_get_slice_plot_info_empty() -> None:
    study = create_study(direction="minimize")
    assert _get_slice_plot_info(study, None, None, "target_name") == _SlicePlotInfo(
        target_name="target_name", subplots=[]
    )


def test_get_slice_plot_info() -> None:
    study = create_study(direction="minimize")
    study.add_trial(
        create_trial(
            value=1.0,
            params={"x": 1.0, "y": 2.0},
            distributions={
                "x": FloatDistribution(0.0, 2.0),
                "y": FloatDistribution(1.0, 2.0, log=True),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=2.0,
            params={"x": 3.0, "z": "a", "w": 3},
            distributions={
                "x": FloatDistribution(0.0, 5.0),
                "z": CategoricalDistribution(["a", "b"]),
                "w": CategoricalDistribution([2, 3, 5]),
            },
        )
    )

    # Default arguments
    assert _get_slice_plot_info(study, None, None, "target_name") == _SlicePlotInfo(
        target_name="target_name",
        subplots=[
            _SliceSubplotInfo(
                param_name="w", x=[3], y=[2.0], trial_numbers=[1], is_log=False, is_numerical=True
            ),
            _SliceSubplotInfo(
                param_name="x",
                x=[1.0, 3.0],
                y=[1.0, 2.0],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=True,
            ),
            _SliceSubplotInfo(
                param_name="y", x=[2.0], y=[1.0], trial_numbers=[0], is_log=True, is_numerical=True
            ),
            _SliceSubplotInfo(
                param_name="z",
                x=["a"],
                y=[2.0],
                trial_numbers=[1],
                is_log=False,
                is_numerical=False,
            ),
        ],
    )

    # Specify params
    assert _get_slice_plot_info(study, ["x"], None, "target_name") == _SlicePlotInfo(
        target_name="target_name",
        subplots=[
            _SliceSubplotInfo(
                param_name="x",
                x=[1.0, 3.0],
                y=[1.0, 2.0],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=True,
            ),
        ],
    )

    # Specify target
    assert _get_slice_plot_info(study, None, lambda _: 0.0, "target_name") == _SlicePlotInfo(
        target_name="target_name",
        subplots=[
            _SliceSubplotInfo(
                param_name="w", x=[3], y=[0.0], trial_numbers=[1], is_log=False, is_numerical=True
            ),
            _SliceSubplotInfo(
                param_name="x",
                x=[1.0, 3.0],
                y=[0.0, 0.0],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=True,
            ),
            _SliceSubplotInfo(
                param_name="y", x=[2.0], y=[0.0], trial_numbers=[0], is_log=True, is_numerical=True
            ),
            _SliceSubplotInfo(
                param_name="z",
                x=["a"],
                y=[0.0],
                trial_numbers=[1],
                is_log=False,
                is_numerical=False,
            ),
        ],
    )

    # Nonexistent parameter
    with pytest.raises(ValueError):
        _get_slice_plot_info(study, ["nonexistent"], None, "target_name")


@pytest.mark.parametrize(
    "plotter",
    [
        optuna.visualization._slice._get_slice_plot,
        optuna.visualization.matplotlib._slice._get_slice_plot,
    ],
)
@pytest.mark.parametrize(
    "info",
    [
        _SlicePlotInfo(target_name="target_name", subplots=[]),
        _SlicePlotInfo(
            target_name="target_name",
            subplots=[
                _SliceSubplotInfo(
                    param_name="x",
                    x=[1.0, 3.0],
                    y=[1.0, 2.0],
                    trial_numbers=[0, 1],
                    is_log=False,
                    is_numerical=True,
                ),
            ],
        ),
        _SlicePlotInfo(
            target_name="target_name",
            subplots=[
                _SliceSubplotInfo(
                    param_name="x",
                    x=[1.0, 3.0],
                    y=[0.0, 0.0],
                    trial_numbers=[0, 1],
                    is_log=False,
                    is_numerical=True,
                ),
                _SliceSubplotInfo(
                    param_name="y",
                    x=[2.0],
                    y=[0.0],
                    trial_numbers=[0],
                    is_log=True,
                    is_numerical=True,
                ),
                _SliceSubplotInfo(
                    param_name="z",
                    x=["a"],
                    y=[2.0],
                    trial_numbers=[1],
                    is_log=False,
                    is_numerical=False,
                ),
            ],
        ),
    ],
)
def test_get_slice_plot(plotter: Callable[[_SlicePlotInfo], Any], info: _SlicePlotInfo) -> None:
    figure = plotter(info)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
