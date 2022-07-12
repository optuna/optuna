from typing import Any
from typing import Callable
from typing import List
from typing import Optional

import pytest

from optuna.study import create_study
from optuna.study import Study
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.visualization import plot_slice as plotly_plot_slice
from optuna.visualization._plotly_imports import go
from optuna.visualization._slice import _get_slice_plot_info
from optuna.visualization._utils import COLOR_SCALE
from optuna.visualization.matplotlib import plot_slice as plt_plot_slice
from optuna.visualization.matplotlib._matplotlib_imports import Axes


parametrize_plot_slice = pytest.mark.parametrize("plot_slice", [plotly_plot_slice, plt_plot_slice])


def _create_study_with_failed_trial() -> Study:

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    return study


@parametrize_plot_slice
def test_plot_slice_customized_target_name(plot_slice: Callable[..., Any]) -> None:

    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = plot_slice(study, params=params, target_name="Target Name")
    if isinstance(figure, go.Figure):
        figure.layout.yaxis.title.text == "Target Name"
    elif isinstance(figure, Axes):
        assert figure[0].yaxis.label.get_text() == "Target Name"


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_slice_plot_info(study, None, None, "Objective Value")


@pytest.mark.parametrize(
    "specific_create_study",
    [create_study, _create_study_with_failed_trial],
)
@pytest.mark.parametrize(
    "params",
    [
        [],
        ["param_a"],
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_get_slice_plot_info_empty(
    specific_create_study: Callable[[], Study], params: Optional[List[str]]
) -> None:

    study = specific_create_study()
    info = _get_slice_plot_info(study, params=params, target=None, target_name="Objective Value")
    assert len(info.subplots) == 0


def test_get_slice_plot_into_non_exist_param() -> None:
    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        _get_slice_plot_info(
            study, params=["not-exist-param"], target=None, target_name="Objective Value"
        )


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_get_slice_plot_info_customized_target(params: List[str]) -> None:

    study = prepare_study_with_trials()
    info = _get_slice_plot_info(
        study, params=params, target=lambda t: t.params["param_d"], target_name="Objective Value"
    )
    n_params = len(params)
    assert len(info.subplots) == n_params


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = prepare_study_with_trials(with_c_d=False, direction=direction)

    # Since `plot_slice`'s colormap depends on only trial.number, `reversecale` is not in the plot.
    marker = plotly_plot_slice(study).data[0]["marker"]
    assert COLOR_SCALE == [v[1] for v in marker["colorscale"]]
    assert "reversecale" not in marker
