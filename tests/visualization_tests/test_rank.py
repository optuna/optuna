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
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import create_study
from optuna.study import Study
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.visualization import plot_rank as plotly_plot_rank
from optuna.visualization._plotly_imports import go
from optuna.visualization._rank import _AxisInfo
from optuna.visualization._rank import _convert_color_idxs_to_scaled_rgb_colors
from optuna.visualization._rank import _get_axis_info
from optuna.visualization._rank import _get_order_with_same_order_averaging
from optuna.visualization._rank import _get_rank_info
from optuna.visualization._rank import _RankPlotInfo
from optuna.visualization._rank import _RankSubplotInfo


parametrize_plot_rank = pytest.mark.parametrize("plot_rank", [plotly_plot_rank])


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


def _create_study_with_constraints() -> Study:
    study = create_study()
    distributions: dict[str, BaseDistribution] = {
        "param_a": FloatDistribution(0.1, 0.2),
        "param_b": FloatDistribution(0.3, 0.4),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 0.11, "param_b": 0.31},
            distributions=distributions,
            system_attrs={_CONSTRAINTS_KEY: [-0.1, 0.0]},
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 0.19, "param_b": 0.34},
            distributions=distributions,
            system_attrs={_CONSTRAINTS_KEY: [0.1, 0.0]},
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


def _named_tuple_equal(t1: Any, t2: Any) -> bool:
    if isinstance(t1, np.ndarray):
        return bool(np.all(t1 == t2))
    elif isinstance(t1, tuple) or isinstance(t1, list):
        if len(t1) != len(t2):
            return False
        for x, y in zip(t1, t2):
            if not _named_tuple_equal(x, y):
                return False
        return True
    else:
        return t1 == t2


def _get_nested_list_shape(nested_list: list[list[Any]]) -> tuple[int, int]:
    assert all(len(nested_list[0]) == len(row) for row in nested_list)
    return len(nested_list), len(nested_list[0])


@parametrize_plot_rank
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
        [_create_study_with_constraints, None],
    ],
)
def test_plot_rank(
    plot_rank: Callable[..., Any],
    specific_create_study: Callable[[], Study],
    params: list[str] | None,
) -> None:
    study = specific_create_study()
    figure = plot_rank(study, params=params)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_rank_info(study, params=None, target=None, target_name="Objective Value")


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
def test_get_rank_info_empty(
    specific_create_study: Callable[[], Study], params: list[str] | None
) -> None:
    study = specific_create_study()
    info = _get_rank_info(study, params=params, target=None, target_name="Objective Value")
    assert len(info.params) == 0
    assert len(info.sub_plot_infos) == 0


def test_get_rank_info_non_exist_param_error() -> None:
    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        _get_rank_info(study, ["optuna"], target=None, target_name="Objective Value")


@pytest.mark.parametrize("params", [[], ["param_a"]])
def test_get_rank_info_too_short_params(params: list[str]) -> None:
    study = prepare_study_with_trials()
    info = _get_rank_info(study, params=params, target=None, target_name="Objective Value")
    assert len(info.params) == len(params)
    assert len(info.sub_plot_infos) == len(params)


def test_get_rank_info_2_params() -> None:
    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    info = _get_rank_info(study, params=params, target=None, target_name="Objective Value")
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=params,
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=_AxisInfo(
                            name="param_a",
                            range=(0.925, 2.575),
                            is_log=False,
                            is_cat=False,
                        ),
                        yaxis=_AxisInfo(
                            name="param_b",
                            range=(-0.1, 2.1),
                            is_log=False,
                            is_cat=False,
                        ),
                        xs=[1.0, 2.5],
                        ys=[2.0, 1.0],
                        trials=[study.trials[0], study.trials[2]],
                        zs=np.array([0.0, 1.0]),
                        colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 0.5])),
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([0.0, 2.0, 1.0]),
            colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0, 0.5])),
            has_custom_target=False,
        ),
    )


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_get_rank_info_more_than_2_params(params: list[str] | None) -> None:
    study = prepare_study_with_trials()
    n_params = len(params) if params is not None else 4
    info = _get_rank_info(study, params=params, target=None, target_name="Objective Value")
    assert len(info.params) == n_params
    assert _get_nested_list_shape(info.sub_plot_infos) == (n_params, n_params)


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_get_rank_info_customized_target(params: list[str]) -> None:
    study = prepare_study_with_trials()
    info = _get_rank_info(
        study, params=params, target=lambda t: t.params["param_d"], target_name="param_d"
    )
    n_params = len(params)
    assert len(info.params) == n_params
    plot_shape = (1, 1) if n_params == 2 else (n_params, n_params)
    assert _get_nested_list_shape(info.sub_plot_infos) == plot_shape


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # `x_axis` has one observation.
        ["param_b", "param_a"],  # `y_axis` has one observation.
    ],
)
def test_generate_rank_plot_for_no_plots(params: list[str]) -> None:
    study = create_study(direction="minimize")
    study.add_trial(
        create_trial(
            values=[0.0],
            params={"param_a": 1.0},
            distributions={
                "param_a": FloatDistribution(0.0, 3.0),
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

    info = _get_rank_info(study, params=params, target=None, target_name="Objective Value")
    axis_infos = {
        "param_a": _AxisInfo(
            name="param_a",
            range=(1.0, 1.0),
            is_log=False,
            is_cat=False,
        ),
        "param_b": _AxisInfo(
            name="param_b",
            range=(0.0, 0.0),
            is_log=False,
            is_cat=False,
        ),
    }
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=params,
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=axis_infos[params[0]],
                        yaxis=axis_infos[params[1]],
                        xs=[],
                        ys=[],
                        trials=[],
                        zs=np.array([]),
                        colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([])).reshape(
                            -1, 3
                        ),
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([0.0, 2.0]),
            colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0])),
            has_custom_target=False,
        ),
    )


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # `x_axis` has one observation.
        ["param_b", "param_a"],  # `y_axis` has one observation.
    ],
)
def test_generate_rank_plot_for_few_observations(params: list[str]) -> None:
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

    info = _get_rank_info(study, params=params, target=None, target_name="Objective Value")
    axis_infos = {
        "param_a": _AxisInfo(
            name="param_a",
            range=(1.0, 1.0),
            is_log=False,
            is_cat=False,
        ),
        "param_b": _AxisInfo(
            name="param_b",
            range=(-0.1, 2.1),
            is_log=False,
            is_cat=False,
        ),
    }
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=params,
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=axis_infos[params[0]],
                        yaxis=axis_infos[params[1]],
                        xs=[study.get_trials()[0].params[params[0]]],
                        ys=[study.get_trials()[0].params[params[1]]],
                        trials=[study.get_trials()[0]],
                        zs=np.array([0.0]),
                        colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0])),
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([0.0, 2.0]),
            colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0])),
            has_custom_target=False,
        ),
    )


def test_get_rank_info_log_scale_and_str_category_2_params() -> None:
    # If the search space has two parameters, plot_rank generates a single plot.
    study = _create_study_with_log_scale_and_str_category_2d()
    info = _get_rank_info(study, params=None, target=None, target_name="Objective Value")
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=["param_a", "param_b"],
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=_AxisInfo(
                            name="param_a",
                            range=(math.pow(10, -6.05), math.pow(10, -4.95)),
                            is_log=True,
                            is_cat=False,
                        ),
                        yaxis=_AxisInfo(
                            name="param_b",
                            range=(-0.05, 1.05),
                            is_log=False,
                            is_cat=True,
                        ),
                        xs=[1e-6, 1e-5],
                        ys=["101", "100"],
                        trials=[study.trials[0], study.trials[1]],
                        zs=np.array([0.0, 1.0]),
                        colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0])),
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([0.0, 1.0]),
            colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0])),
            has_custom_target=False,
        ),
    )


def test_get_rank_info_log_scale_and_str_category_more_than_2_params() -> None:
    # If the search space has three parameters, plot_rank generates nine plots.
    study = _create_study_with_log_scale_and_str_category_3d()
    info = _get_rank_info(study, params=None, target=None, target_name="Objective Value")
    params = ["param_a", "param_b", "param_c"]
    assert info.params == params
    assert _get_nested_list_shape(info.sub_plot_infos) == (3, 3)
    ranges = {
        "param_a": (math.pow(10, -6.05), math.pow(10, -4.95)),
        "param_b": (-0.05, 1.05),
        "param_c": (-0.05, 1.05),
    }
    is_log = {"param_a": True, "param_b": False, "param_c": False}
    is_cat = {"param_a": False, "param_b": True, "param_c": True}

    param_values = {"param_a": [1e-6, 1e-5], "param_b": ["101", "100"], "param_c": ["one", "two"]}
    zs = np.array([0.0, 1.0])
    colors = _convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0]))

    def _check_axis(axis: _AxisInfo, name: str) -> None:
        assert axis.name == name
        assert axis.range == ranges[name]
        assert axis.is_log == is_log[name]
        assert axis.is_cat == is_cat[name]

    for yi in range(3):
        for xi in range(3):
            xaxis = info.sub_plot_infos[yi][xi].xaxis
            yaxis = info.sub_plot_infos[yi][xi].yaxis
            x_param = params[xi]
            y_param = params[yi]
            _check_axis(xaxis, x_param)
            _check_axis(yaxis, y_param)
            assert info.sub_plot_infos[yi][xi].xs == param_values[x_param]
            assert info.sub_plot_infos[yi][xi].ys == param_values[y_param]
            assert info.sub_plot_infos[yi][xi].trials == study.trials
            assert np.all(info.sub_plot_infos[yi][xi].zs == zs)
            assert np.all(info.sub_plot_infos[yi][xi].colors == colors)

    info.target_name == "Objective Value"
    assert np.all(info.zs == zs)
    assert np.all(info.colors == colors)
    assert not info.has_custom_target


def test_get_rank_info_mixture_category_types() -> None:
    study = _create_study_mixture_category_types()
    info = _get_rank_info(study, params=None, target=None, target_name="Objective Value")
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=["param_a", "param_b"],
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=_AxisInfo(
                            name="param_a",
                            range=(-0.05, 1.05),
                            is_log=False,
                            is_cat=True,
                        ),
                        yaxis=_AxisInfo(
                            name="param_b",
                            range=(100.95, 102.05),
                            is_log=False,
                            is_cat=False,
                        ),
                        xs=[None, "100"],
                        ys=[101, 102.0],
                        trials=study.trials,
                        zs=np.array([0.0, 0.5]),
                        colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0])),
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([0.0, 0.5]),
            colors=_convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0])),
            has_custom_target=False,
        ),
    )


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_get_rank_info_nonfinite(value: float) -> None:
    study = prepare_study_with_trials(value_for_first_trial=value)
    info = _get_rank_info(
        study, params=["param_b", "param_d"], target=None, target_name="Objective Value"
    )

    colors = (
        _convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0, 0.5]))
        if value == float("-inf")
        else _convert_color_idxs_to_scaled_rgb_colors(np.array([1.0, 0.5, 0.0]))
    )
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=["param_b", "param_d"],
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=_AxisInfo(
                            name="param_b",
                            range=(-0.1, 2.1),
                            is_log=False,
                            is_cat=False,
                        ),
                        yaxis=_AxisInfo(
                            name="param_d",
                            range=(1.9, 4.1),
                            is_log=False,
                            is_cat=False,
                        ),
                        xs=[2.0, 0.0, 1.0],
                        ys=[4.0, 4.0, 2.0],
                        trials=study.trials,
                        zs=np.array([value, 2.0, 1.0]),
                        colors=colors,
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([value, 2.0, 1.0]),
            colors=colors,
            has_custom_target=False,
        ),
    )


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), float("-inf")))
def test_get_rank_info_nonfinite_multiobjective(objective: int, value: float) -> None:
    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    info = _get_rank_info(
        study,
        params=["param_b", "param_d"],
        target=lambda t: t.values[objective],
        target_name="Target Name",
    )
    colors = (
        _convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0, 0.5]))
        if value == float("-inf")
        else _convert_color_idxs_to_scaled_rgb_colors(np.array([1.0, 0.5, 0.0]))
    )
    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=["param_b", "param_d"],
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=_AxisInfo(
                            name="param_b",
                            range=(-0.1, 2.1),
                            is_log=False,
                            is_cat=False,
                        ),
                        yaxis=_AxisInfo(
                            name="param_d",
                            range=(1.9, 4.1),
                            is_log=False,
                            is_cat=False,
                        ),
                        xs=[2.0, 0.0, 1.0],
                        ys=[4.0, 4.0, 2.0],
                        trials=study.trials,
                        zs=np.array([value, 2.0, 1.0]),
                        colors=colors,
                    )
                ]
            ],
            target_name="Target Name",
            zs=np.array([value, 2.0, 1.0]),
            colors=colors,
            has_custom_target=True,
        ),
    )


def test_generate_rank_info_with_constraints() -> None:
    study = _create_study_with_constraints()
    info = _get_rank_info(study, params=None, target=None, target_name="Objective Value")
    expected_color = _convert_color_idxs_to_scaled_rgb_colors(np.array([0.0, 1.0]))
    expected_color[1] = [204, 204, 204]

    assert _named_tuple_equal(
        info,
        _RankPlotInfo(
            params=["param_a", "param_b"],
            sub_plot_infos=[
                [
                    _RankSubplotInfo(
                        xaxis=_get_axis_info(study.trials, "param_a"),
                        yaxis=_get_axis_info(study.trials, "param_b"),
                        xs=[0.11, 0.19],
                        ys=[0.31, 0.34],
                        trials=study.trials,
                        zs=np.array([0.0, 1.0]),
                        colors=expected_color,
                    )
                ]
            ],
            target_name="Objective Value",
            zs=np.array([0.0, 1.0]),
            colors=expected_color,
            has_custom_target=False,
        ),
    )


def test_get_order_with_same_order_averaging() -> None:
    x = np.array([6.0, 2.0, 3.0, 1.0, 4.5, 4.5, 8.0, 8.0, 0.0, 8.0])
    assert np.all(x == _get_order_with_same_order_averaging(x))


def test_convert_color_idxs_to_scaled_rgb_colors() -> None:
    x1 = np.array([0.1, 0.2])
    result1 = _convert_color_idxs_to_scaled_rgb_colors(x1)
    np.testing.assert_array_equal(result1, [[69, 117, 180], [116, 173, 209]])

    x2 = np.array([])
    result2 = _convert_color_idxs_to_scaled_rgb_colors(x2)
    np.testing.assert_array_equal(result2, [])
