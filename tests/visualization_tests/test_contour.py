from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
import math
from typing import Any

import numpy as np
import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.study import Study
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.visualization import plot_contour as plotly_plot_contour
from optuna.visualization._contour import _AxisInfo
from optuna.visualization._contour import _ContourInfo
from optuna.visualization._contour import _get_contour_info
from optuna.visualization._contour import _SubContourInfo
from optuna.visualization._plotly_imports import go
from optuna.visualization._utils import COLOR_SCALE
from optuna.visualization.matplotlib import plot_contour as plt_plot_contour
from optuna.visualization.matplotlib._matplotlib_imports import Axes
from optuna.visualization.matplotlib._matplotlib_imports import plt


parametrize_plot_contour = pytest.mark.parametrize(
    "plot_contour", [plotly_plot_contour, plt_plot_contour]
)


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


def _create_study_with_log_scale_and_str_category_3d() -> Study:
    study = create_study()
    distributions = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": CategoricalDistribution(["100", "101"]),
        "param_c": CategoricalDistribution(["one", "two"]),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "101", "param_c": "one"},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "100", "param_c": "two"},
            distributions=distributions,
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


def _create_study_with_overlapping_params(direction: str) -> Study:
    study = create_study(direction=direction)
    distributions = {
        "param_a": FloatDistribution(1.0, 2.0),
        "param_b": CategoricalDistribution(["100", "101"]),
        "param_c": CategoricalDistribution(["foo", "bar"]),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1.0, "param_b": "101", "param_c": "foo"},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1.0, "param_b": "101", "param_c": "bar"},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2.0, "param_b": "100", "param_c": "foo"},
            distributions=distributions,
        )
    )
    return study


@parametrize_plot_contour
def test_plot_contour_customized_target_name(plot_contour: Callable[..., Any]) -> None:
    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params, target_name="Target Name")
    if isinstance(figure, go.Figure):
        assert figure.data[0]["colorbar"].title.text == "Target Name"
    elif isinstance(figure, Axes):
        assert figure.figure.axes[-1].get_ylabel() == "Target Name"


@parametrize_plot_contour
@pytest.mark.parametrize(
    "specific_create_study, params",
    [
        [create_study, []],
        [create_study, ["param_a"]],
        [create_study, ["param_a", "param_b"]],
        [create_study, ["param_a", "param_b", "param_c"]],
        [create_study, ["param_a", "param_b", "param_c", "param_d"]],
        [create_study, None],
        [_create_study_with_failed_trial, []],
        [_create_study_with_failed_trial, ["param_a"]],
        [_create_study_with_failed_trial, ["param_a", "param_b"]],
        [_create_study_with_failed_trial, ["param_a", "param_b", "param_c"]],
        [_create_study_with_failed_trial, ["param_a", "param_b", "param_c", "param_d"]],
        [_create_study_with_failed_trial, None],
        [prepare_study_with_trials, []],
        [prepare_study_with_trials, ["param_a"]],
        [prepare_study_with_trials, ["param_a", "param_b"]],
        [prepare_study_with_trials, ["param_a", "param_b", "param_c"]],
        [prepare_study_with_trials, ["param_a", "param_b", "param_c", "param_d"]],
        [prepare_study_with_trials, None],
        [_create_study_with_log_scale_and_str_category_2d, None],
        [_create_study_with_log_scale_and_str_category_3d, None],
        [_create_study_mixture_category_types, None],
    ],
)
def test_plot_contour(
    plot_contour: Callable[..., Any],
    specific_create_study: Callable[[], Study],
    params: list[str] | None,
) -> None:
    study = specific_create_study()
    figure = plot_contour(study, params=params)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_contour_info(study)


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
def test_get_contour_info_empty(
    specific_create_study: Callable[[], Study], params: list[str] | None
) -> None:
    study = specific_create_study()
    info = _get_contour_info(study, params=params)
    assert len(info.sorted_params) == 0
    assert len(info.sub_plot_infos) == 0


def test_get_contour_info_non_exist_param_error() -> None:
    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        _get_contour_info(study, ["optuna"])


@pytest.mark.parametrize("params", [[], ["param_a"]])
def test_get_contour_info_too_short_params(params: list[str]) -> None:
    study = prepare_study_with_trials()
    info = _get_contour_info(study, params=params)
    assert len(info.sorted_params) == len(params)
    assert len(info.sub_plot_infos) == len(params)


def test_get_contour_info_2_params() -> None:
    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    info = _get_contour_info(study, params=params)
    assert info == _ContourInfo(
        sorted_params=params,
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name="param_a",
                        range=(0.925, 2.575),
                        is_log=False,
                        is_cat=False,
                        indices=[0.925, 1.0, 2.5, 2.575],
                        values=[1.0, None, 2.5],
                    ),
                    yaxis=_AxisInfo(
                        name="param_b",
                        range=(-0.1, 2.1),
                        is_log=False,
                        is_cat=False,
                        indices=[-0.1, 0.0, 1.0, 2.0, 2.1],
                        values=[2.0, 0.0, 1.0],
                    ),
                    z_values={(1, 3): 0.0, (2, 2): 1.0},
                    constraints=[True, True, True],
                )
            ]
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_get_contour_info_more_than_2_params(params: list[str] | None) -> None:
    study = prepare_study_with_trials()
    n_params = len(params) if params is not None else 4
    info = _get_contour_info(study, params=params)
    assert len(info.sorted_params) == n_params
    assert np.shape(np.asarray(info.sub_plot_infos, dtype=object)) == (n_params, n_params, 4)


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_get_contour_info_customized_target(params: list[str]) -> None:
    study = prepare_study_with_trials()
    info = _get_contour_info(
        study, params=params, target=lambda t: t.params["param_d"], target_name="param_d"
    )
    n_params = len(params)
    assert len(info.sorted_params) == n_params
    plot_shape = (1, 1, 4) if n_params == 2 else (n_params, n_params, 4)
    assert np.shape(np.asarray(info.sub_plot_infos, dtype=object)) == plot_shape


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # `x_axis` has one observation.
        ["param_b", "param_a"],  # `y_axis` has one observation.
    ],
)
def test_generate_contour_plot_for_few_observations(params: list[str]) -> None:
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

    info = _get_contour_info(study, params=params)
    assert info == _ContourInfo(
        sorted_params=sorted(params),
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name=sorted(params)[0],
                        range=(1.0, 1.0),
                        is_log=False,
                        is_cat=False,
                        indices=[1.0],
                        values=[1.0, None],
                    ),
                    yaxis=_AxisInfo(
                        name=sorted(params)[1],
                        range=(-0.1, 2.1),
                        is_log=False,
                        is_cat=False,
                        indices=[-0.1, 0.0, 2.0, 2.1],
                        values=[2.0, 0.0],
                    ),
                    z_values={},
                    constraints=[],
                )
            ]
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_contour_info_log_scale_and_str_category_2_params() -> None:
    # If the search space has two parameters, plot_contour generates a single plot.
    study = _create_study_with_log_scale_and_str_category_2d()
    info = _get_contour_info(study)
    assert info == _ContourInfo(
        sorted_params=["param_a", "param_b"],
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name="param_a",
                        range=(math.pow(10, -6.05), math.pow(10, -4.95)),
                        is_log=True,
                        is_cat=False,
                        indices=[math.pow(10, -6.05), 1e-6, 1e-5, math.pow(10, -4.95)],
                        values=[1e-6, 1e-5],
                    ),
                    yaxis=_AxisInfo(
                        name="param_b",
                        range=(-0.05, 1.05),
                        is_log=False,
                        is_cat=True,
                        indices=["100", "101"],
                        values=["101", "100"],
                    ),
                    z_values={(1, 1): 0.0, (2, 0): 1.0},
                    constraints=[True, True],
                )
            ]
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_contour_info_log_scale_and_str_category_more_than_2_params() -> None:
    # If the search space has three parameters, plot_contour generates nine plots.
    study = _create_study_with_log_scale_and_str_category_3d()
    info = _get_contour_info(study)
    params = ["param_a", "param_b", "param_c"]
    assert info.sorted_params == params
    assert np.shape(np.asarray(info.sub_plot_infos, dtype=object)) == (3, 3, 4)
    ranges = {
        "param_a": (math.pow(10, -6.05), math.pow(10, -4.95)),
        "param_b": (-0.05, 1.05),
        "param_c": (-0.05, 1.05),
    }
    is_log = {"param_a": True, "param_b": False, "param_c": False}
    is_cat = {"param_a": False, "param_b": True, "param_c": True}
    indices: dict[str, list[str | float]] = {
        "param_a": [math.pow(10, -6.05), 1e-6, 1e-5, math.pow(10, -4.95)],
        "param_b": ["100", "101"],
        "param_c": ["one", "two"],
    }
    values = {"param_a": [1e-6, 1e-5], "param_b": ["101", "100"], "param_c": ["one", "two"]}

    def _check_axis(axis: _AxisInfo, name: str) -> None:
        assert axis.name == name
        assert axis.range == ranges[name]
        assert axis.is_log == is_log[name]
        assert axis.is_cat == is_cat[name]
        assert axis.indices == indices[name]
        assert axis.values == values[name]

    for yi in range(3):
        for xi in range(3):
            xaxis = info.sub_plot_infos[yi][xi].xaxis
            yaxis = info.sub_plot_infos[yi][xi].yaxis
            x_param = params[xi]
            y_param = params[yi]
            _check_axis(xaxis, x_param)
            _check_axis(yaxis, y_param)
            z_values = info.sub_plot_infos[yi][xi].z_values
            if xi == yi:
                assert z_values == {}
            else:
                for i, v in enumerate([0.0, 1.0]):
                    x_value = xaxis.values[i]
                    y_value = yaxis.values[i]
                    assert x_value is not None
                    assert y_value is not None
                    xi = xaxis.indices.index(x_value)
                    yi = yaxis.indices.index(y_value)
                    assert z_values[(xi, yi)] == v


def test_get_contour_info_mixture_category_types() -> None:
    study = _create_study_mixture_category_types()
    info = _get_contour_info(study)
    assert info == _ContourInfo(
        sorted_params=["param_a", "param_b"],
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name="param_a",
                        range=(-0.05, 1.05),
                        is_log=False,
                        is_cat=True,
                        indices=["100", "None"],
                        values=["None", "100"],
                    ),
                    yaxis=_AxisInfo(
                        name="param_b",
                        range=(100.95, 102.05),
                        is_log=False,
                        is_cat=False,
                        indices=[100.95, 101, 102, 102.05],
                        values=[101.0, 102.0],
                    ),
                    z_values={(0, 2): 0.5, (1, 1): 0.0},
                    constraints=[True, True],
                )
            ]
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_get_contour_info_nonfinite_removed(value: float) -> None:
    study = prepare_study_with_trials(value_for_first_trial=value)
    info = _get_contour_info(study, params=["param_b", "param_d"])
    assert info == _ContourInfo(
        sorted_params=["param_b", "param_d"],
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name="param_b",
                        range=(-0.05, 1.05),
                        is_log=False,
                        is_cat=False,
                        indices=[-0.05, 0.0, 1.0, 1.05],
                        values=[0.0, 1.0],
                    ),
                    yaxis=_AxisInfo(
                        name="param_d",
                        range=(1.9, 4.1),
                        is_log=False,
                        is_cat=False,
                        indices=[1.9, 2.0, 4.0, 4.1],
                        values=[4.0, 2.0],
                    ),
                    z_values={(1, 2): 2.0, (2, 1): 1.0},
                    constraints=[True, True],
                )
            ]
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_get_contour_info_nonfinite_multiobjective(objective: int, value: float) -> None:
    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    info = _get_contour_info(
        study,
        params=["param_b", "param_d"],
        target=lambda t: t.values[objective],
        target_name="Target Name",
    )
    assert info == _ContourInfo(
        sorted_params=["param_b", "param_d"],
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name="param_b",
                        range=(-0.05, 1.05),
                        is_log=False,
                        is_cat=False,
                        indices=[-0.05, 0.0, 1.0, 1.05],
                        values=[0.0, 1.0],
                    ),
                    yaxis=_AxisInfo(
                        name="param_d",
                        range=(1.9, 4.1),
                        is_log=False,
                        is_cat=False,
                        indices=[1.9, 2.0, 4.0, 4.1],
                        values=[4.0, 2.0],
                    ),
                    z_values={(1, 2): 2.0, (2, 1): 1.0},
                    constraints=[True, True],
                )
            ]
        ],
        reverse_scale=True,
        target_name="Target Name",
    )


@pytest.mark.parametrize("direction,expected", (("minimize", 0.0), ("maximize", 1.0)))
def test_get_contour_info_overlapping_params(direction: str, expected: float) -> None:
    study = _create_study_with_overlapping_params(direction)
    info = _get_contour_info(study, params=["param_a", "param_b"])
    assert info == _ContourInfo(
        sorted_params=["param_a", "param_b"],
        sub_plot_infos=[
            [
                _SubContourInfo(
                    xaxis=_AxisInfo(
                        name="param_a",
                        range=(0.95, 2.05),
                        is_log=False,
                        is_cat=False,
                        indices=[0.95, 1.0, 2.0, 2.05],
                        values=[1.0, 1.0, 2.0],
                    ),
                    yaxis=_AxisInfo(
                        name="param_b",
                        range=(-0.05, 1.05),
                        is_log=False,
                        is_cat=True,
                        indices=["100", "101"],
                        values=["101", "101", "100"],
                    ),
                    z_values={(1, 1): expected, (2, 0): 1.0},
                    constraints=[True, True, True],
                )
            ]
        ],
        reverse_scale=False if direction == "maximize" else True,
        target_name="Objective Value",
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

    # `target` is `None`.
    contour = plotly_plot_contour(study).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    if direction == "minimize":
        assert contour["reversescale"]
    else:
        assert not contour["reversescale"]

    # When `target` is not `None`, `reversescale` is always `True`.
    contour = plotly_plot_contour(study, target=lambda t: t.number, target_name="Number").data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    assert contour["reversescale"]

    # Multi-objective optimization.
    study = create_study(directions=[direction, direction])
    for i in range(3):
        study.add_trial(
            create_trial(
                values=[float(i), float(i)],
                params={"param_a": float(i), "param_b": float(i)},
                distributions={
                    "param_a": FloatDistribution(0.0, 3.0),
                    "param_b": FloatDistribution(0.0, 3.0),
                },
            )
        )
    contour = plotly_plot_contour(study, target=lambda t: t.number, target_name="Number").data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    assert contour["reversescale"]
