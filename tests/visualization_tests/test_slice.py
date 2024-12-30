from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.study import Study
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.visualization import plot_slice as plotly_plot_slice
from optuna.visualization._plotly_imports import go
from optuna.visualization._slice import _get_slice_plot_info
from optuna.visualization._slice import _SlicePlotInfo
from optuna.visualization._slice import _SliceSubplotInfo
from optuna.visualization._utils import COLOR_SCALE
from optuna.visualization.matplotlib import plot_slice as plt_plot_slice
from optuna.visualization.matplotlib._matplotlib_imports import Axes
from optuna.visualization.matplotlib._matplotlib_imports import plt


parametrize_plot_slice = pytest.mark.parametrize("plot_slice", [plotly_plot_slice, plt_plot_slice])


def _create_study_with_failed_trial() -> Study:
    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    return study


def _create_study_with_log_scale_and_str_category_2d() -> Study:
    study = create_study()
    distributions = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": CategoricalDistribution(["100", "101"]),
    }
    study.add_trial(
        create_trial(
            value=0.0, params={"param_a": 1e-6, "param_b": "101"}, distributions=distributions
        )
    )
    study.add_trial(
        create_trial(
            value=1.0, params={"param_a": 1e-5, "param_b": "100"}, distributions=distributions
        )
    )
    return study


def _create_study_mixture_category_types() -> Study:
    study = create_study()
    distributions: dict[str, BaseDistribution] = {
        "param_a": CategoricalDistribution([None, "100"]),
        "param_b": CategoricalDistribution([101, 102.0]),
    }
    study.add_trial(
        create_trial(
            value=0.0, params={"param_a": None, "param_b": 101}, distributions=distributions
        )
    )
    study.add_trial(
        create_trial(
            value=0.5, params={"param_a": "100", "param_b": 102.0}, distributions=distributions
        )
    )
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


@parametrize_plot_slice
@pytest.mark.parametrize(
    "specific_create_study, params",
    [
        [create_study, []],
        [create_study, ["param_a"]],
        [create_study, None],
        [prepare_study_with_trials, []],
        [prepare_study_with_trials, ["param_a"]],
        [prepare_study_with_trials, None],
        [_create_study_with_log_scale_and_str_category_2d, None],
        [_create_study_mixture_category_types, None],
    ],
)
def test_plot_slice(
    plot_slice: Callable[..., Any],
    specific_create_study: Callable[[], Study],
    params: list[str] | None,
) -> None:
    study = specific_create_study()
    figure = plot_slice(study, params=params)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_slice_plot_info(study, None, target=None, target_name="Objective Value")


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
    specific_create_study: Callable[[], Study], params: list[str] | None
) -> None:
    study = specific_create_study()
    info = _get_slice_plot_info(study, params=params, target=None, target_name="Objective Value")
    assert len(info.subplots) == 0


def test_get_slice_plot_info_non_exist_param_error() -> None:
    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        _get_slice_plot_info(study, params=["optuna"], target=None, target_name="Objective Value")


@pytest.mark.parametrize(
    "params",
    [
        ["param_a"],
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_get_slice_plot_info_params(params: list[str] | None) -> None:
    study = prepare_study_with_trials()
    params = ["param_a", "param_b", "param_c", "param_d"] if params is None else params
    expected_subplot_infos = {
        "param_a": _SliceSubplotInfo(
            param_name="param_a",
            x=[1.0, 2.5],
            y=[0.0, 1.0],
            trial_numbers=[0, 2],
            is_log=False,
            is_numerical=True,
            x_labels=None,
            constraints=[True, True],
        ),
        "param_b": _SliceSubplotInfo(
            param_name="param_b",
            x=[2.0, 0.0, 1.0],
            y=[0.0, 2.0, 1.0],
            trial_numbers=[0, 1, 2],
            is_log=False,
            is_numerical=True,
            x_labels=None,
            constraints=[True, True, True],
        ),
        "param_c": _SliceSubplotInfo(
            param_name="param_c",
            x=[3.0, 4.5],
            y=[0.0, 1.0],
            trial_numbers=[0, 2],
            is_log=False,
            is_numerical=True,
            x_labels=None,
            constraints=[True, True],
        ),
        "param_d": _SliceSubplotInfo(
            param_name="param_d",
            x=[4.0, 4.0, 2.0],
            y=[0.0, 2.0, 1.0],
            trial_numbers=[0, 1, 2],
            is_log=False,
            is_numerical=True,
            x_labels=None,
            constraints=[True, True, True],
        ),
    }

    info = _get_slice_plot_info(study, params=params, target=None, target_name="Objective Value")
    assert info == _SlicePlotInfo(
        target_name="Objective Value",
        subplots=[expected_subplot_infos[p] for p in params],
    )


def test_get_slice_plot_info_customized_target() -> None:
    params = ["param_a"]
    study = prepare_study_with_trials()
    info = _get_slice_plot_info(
        study,
        params=params,
        target=lambda t: t.params["param_d"],
        target_name="param_d",
    )
    assert info == _SlicePlotInfo(
        target_name="param_d",
        subplots=[
            _SliceSubplotInfo(
                param_name="param_a",
                x=[1.0, 2.5],
                y=[4.0, 2.0],
                trial_numbers=[0, 2],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
        ],
    )


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # First column has 1 observation.
        ["param_b", "param_a"],  # Second column has 1 observation
    ],
)
def test_get_slice_plot_info_for_few_observations(params: list[str]) -> None:
    study = create_study(direction="minimize")
    study.add_trial(
        create_trial(
            values=[0.0],
            params={"param_a": 1.0, "param_b": 2.0},
            distributions={
                "param_a": FloatDistribution(0.0, 3.0),
                "param_b": FloatDistribution(0.0, 3.0),
            },
        )
    )
    study.add_trial(
        create_trial(
            values=[2.0],
            params={"param_b": 0.0},
            distributions={"param_b": FloatDistribution(0.0, 3.0)},
        )
    )
    info = _get_slice_plot_info(study, params, None, "Objective Value")

    assert info == _SlicePlotInfo(
        target_name="Objective Value",
        subplots=[
            _SliceSubplotInfo(
                param_name="param_a",
                x=[1.0],
                y=[0.0],
                trial_numbers=[0],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True],
            ),
            _SliceSubplotInfo(
                param_name="param_b",
                x=[2.0, 0.0],
                y=[0.0, 2.0],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
        ],
    )


def test_get_slice_plot_info_log_scale_and_str_category_2_params() -> None:
    study = _create_study_with_log_scale_and_str_category_2d()
    info = _get_slice_plot_info(study, None, None, "Objective Value")
    distribution_b = study.trials[0].distributions["param_b"]
    assert isinstance(distribution_b, CategoricalDistribution)
    assert info == _SlicePlotInfo(
        target_name="Objective Value",
        subplots=[
            _SliceSubplotInfo(
                param_name="param_a",
                x=[1e-6, 1e-5],
                y=[0.0, 1.0],
                trial_numbers=[0, 1],
                is_log=True,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
            _SliceSubplotInfo(
                param_name="param_b",
                x=["101", "100"],
                y=[0.0, 1.0],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=False,
                x_labels=distribution_b.choices,
                constraints=[True, True],
            ),
        ],
    )


def test_get_slice_plot_info_mixture_category_types() -> None:
    study = _create_study_mixture_category_types()
    info = _get_slice_plot_info(study, None, None, "Objective Value")
    distribution_a = study.trials[0].distributions["param_a"]
    distribution_b = study.trials[0].distributions["param_b"]
    assert isinstance(distribution_a, CategoricalDistribution)
    assert isinstance(distribution_b, CategoricalDistribution)
    assert info == _SlicePlotInfo(
        target_name="Objective Value",
        subplots=[
            _SliceSubplotInfo(
                param_name="param_a",
                x=[None, "100"],
                y=[0.0, 0.5],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=False,
                x_labels=distribution_a.choices,
                constraints=[True, True],
            ),
            _SliceSubplotInfo(
                param_name="param_b",
                x=[101, 102.0],
                y=[0.0, 0.5],
                trial_numbers=[0, 1],
                is_log=False,
                is_numerical=False,
                x_labels=distribution_b.choices,
                constraints=[True, True],
            ),
        ],
    )


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_get_slice_plot_info_nonfinite_removed(value: float) -> None:
    study = prepare_study_with_trials(value_for_first_trial=value)
    info = _get_slice_plot_info(
        study, params=["param_b", "param_d"], target=None, target_name="Objective Value"
    )
    assert info == _SlicePlotInfo(
        target_name="Objective Value",
        subplots=[
            _SliceSubplotInfo(
                param_name="param_b",
                x=[0.0, 1.0],
                y=[2.0, 1.0],
                trial_numbers=[1, 2],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
            _SliceSubplotInfo(
                param_name="param_d",
                x=[4.0, 2.0],
                y=[2.0, 1.0],
                trial_numbers=[1, 2],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
        ],
    )


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_get_slice_plot_info_nonfinite_multiobjective(objective: int, value: float) -> None:
    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    info = _get_slice_plot_info(
        study,
        params=["param_b", "param_d"],
        target=lambda t: t.values[objective],
        target_name="Target Name",
    )
    assert info == _SlicePlotInfo(
        target_name="Target Name",
        subplots=[
            _SliceSubplotInfo(
                param_name="param_b",
                x=[0.0, 1.0],
                y=[2.0, 1.0],
                trial_numbers=[1, 2],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
            _SliceSubplotInfo(
                param_name="param_d",
                x=[4.0, 2.0],
                y=[2.0, 1.0],
                trial_numbers=[1, 2],
                is_log=False,
                is_numerical=True,
                x_labels=None,
                constraints=[True, True],
            ),
        ],
    )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = create_study(direction=direction)
    for i in range(3):
        study.add_trial(
            create_trial(
                value=float(i),
                params={"param_a": float(i), "param_b": float(i)},
                distributions={
                    "param_a": FloatDistribution(0.0, 3.0),
                    "param_b": FloatDistribution(0.0, 3.0),
                },
            )
        )

    # Since `plot_slice`'s colormap depends on only trial.number, `reversecale` is not in the plot.
    marker = plotly_plot_slice(study).data[0]["marker"]
    assert COLOR_SCALE == [v[1] for v in marker["colorscale"]]
    assert "reversecale" not in marker
