from io import BytesIO
import math
import tempfile
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_contour as plotly_plot_contour
from optuna.visualization._contour import _AxisInfo
from optuna.visualization._contour import _get_contour_info
from optuna.visualization._plotly_imports import _imports as plotly_imports
from optuna.visualization._utils import COLOR_SCALE
from optuna.visualization.matplotlib import plot_contour as plt_plot_contour
from optuna.visualization.matplotlib._matplotlib_imports import _imports as plt_imports


if plotly_imports.is_successful():
    from optuna.visualization._plotly_imports import go


if plt_imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


RANGE_TYPE = Union[Tuple[str, str], Tuple[float, float]]

parametrize_plot_contour = pytest.mark.parametrize(
    "plot_contour", [plotly_plot_contour, plt_plot_contour]
)


def save_static_image(figure: Union[go.Figure, Axes, np.ndarray]) -> None:
    if isinstance(figure, go.Figure):
        with tempfile.TemporaryDirectory() as td:
            figure.write_image(td + "tmp.png")
    else:
        plt.savefig(BytesIO())


@parametrize_plot_contour
def test_target_is_none_and_study_is_multi_obj(plot_contour: Callable[..., Any]) -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_contour(study)


@parametrize_plot_contour
def test_target_is_not_none_and_study_is_multi_obj(plot_contour: Callable[..., Any]) -> None:

    # Multiple sub-figures.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=True)
    figure = plot_contour(study, target=lambda t: t.values[0])
    save_static_image(figure)

    # Single figure.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=False)
    figure = plot_contour(study, target=lambda t: t.values[0])
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_customized_target_name(plot_contour: Callable[..., Any]) -> None:

    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params, target_name="Target Name")
    if isinstance(figure, go.Figure):
        assert figure.data[0]["colorbar"].title.text == "Target Name"
    elif isinstance(figure, Axes):
        assert figure.figure.axes[-1].get_ylabel() == "Target Name"
    save_static_image(figure)


@parametrize_plot_contour
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
def test_plot_contour(plot_contour: Callable[..., Any], params: Optional[List[str]]) -> None:

    # Test with no trial.
    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = plot_contour(study_without_trials, params=params)
    save_static_image(figure)

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    save_static_image(figure)

    # Test with some trials.
    study = prepare_study_with_trials()

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        plot_contour(study, ["optuna", "Optuna"])

    figure = plot_contour(study, params=params)
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_log_scale_and_str_category(plot_contour: Callable[..., Any]) -> None:

    # If the search space has two parameters, plot_contour generates a single plot.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "100"},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": CategoricalDistribution(["100", "101"]),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "101"},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": CategoricalDistribution(["100", "101"]),
            },
        )
    )

    figure = plot_contour(study)
    save_static_image(figure)

    # If the search space has three parameters, plot_contour generates nine plots.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "100", "param_c": "one"},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": CategoricalDistribution(["100", "101"]),
                "param_c": CategoricalDistribution(["one", "two"]),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "101", "param_c": "two"},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": CategoricalDistribution(["100", "101"]),
                "param_c": CategoricalDistribution(["one", "two"]),
            },
        )
    )

    figure = plot_contour(study)
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_mixture_category_types(plot_contour: Callable[..., Any]) -> None:
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": None, "param_b": 101},
            distributions={
                "param_a": CategoricalDistribution([None, "100"]),
                "param_b": CategoricalDistribution([101, 102.0]),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=0.5,
            params={"param_a": "100", "param_b": 102.0},
            distributions={
                "param_a": CategoricalDistribution([None, "100"]),
                "param_b": CategoricalDistribution([101, 102.0]),
            },
        )
    )
    figure = plot_contour(study)
    save_static_image(figure)


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
def test_get_contour_info_no_trial(params: Optional[List[str]]) -> None:

    study_without_trials = prepare_study_with_trials(no_trials=True)
    info = _get_contour_info(study_without_trials, params=params, target=None)
    assert len(info.sorted_params) == 0
    assert len(info.sub_plot_infos) == 0


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
def test_get_contour_info_ignored_error(params: Optional[List[str]]) -> None:
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    info = _get_contour_info(study, params=params, target=None)
    assert len(info.sorted_params) == 0
    assert len(info.sub_plot_infos) == 0


def test_get_contour_info_error() -> None:
    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        _get_contour_info(study, ["optuna", "Optuna"])


@pytest.mark.parametrize("params", [[], ["param_a"]])
def test_get_contour_info_empty(params: Optional[List[str]]) -> None:
    study = prepare_study_with_trials()
    info = _get_contour_info(study, params=params)
    assert len(info.sorted_params) <= 1
    assert len(info.sub_plot_infos) <= 1


def test_get_contour_info_2_params() -> None:
    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    info = _get_contour_info(study, params=params)
    assert info.sorted_params == params
    xaxis = info.sub_plot_infos[0][0].xaxis
    yaxis = info.sub_plot_infos[0][0].yaxis
    assert xaxis.name == "param_a"
    assert yaxis.name == "param_b"
    assert xaxis.range == (0.925, 2.575)
    assert yaxis.range == (-0.1, 2.1)
    assert not xaxis.is_log
    assert not yaxis.is_log
    assert not xaxis.is_cat
    assert not yaxis.is_cat
    assert xaxis.indices == [0.925, 1.0, 2.5, 2.575]
    assert yaxis.indices == [-0.1, 0.0, 1.0, 2.0, 2.1]
    assert xaxis.values == [1.0, None, 2.5]
    assert yaxis.values == [2.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_get_contour_info_more_than_2_params(params: Optional[List[str]]) -> None:

    study = prepare_study_with_trials()
    n_params = len(params) if params is not None else 4
    info = _get_contour_info(study, params=params)
    assert len(info.sorted_params) == n_params
    assert np.shape(np.asarray(info.sub_plot_infos)) == (n_params, n_params, 3)


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_get_contour_info_customized_target(params: List[str]) -> None:

    study = prepare_study_with_trials()
    info = _get_contour_info(study, params=params, target=lambda t: t.params["param_d"])
    n_params = len(params)
    assert len(info.sorted_params) == n_params
    plot_shape = (1, 1, 3) if n_params == 2 else (n_params, n_params, 3)
    assert np.shape(np.asarray(info.sub_plot_infos)) == plot_shape


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # `x_axis` has one observation.
        ["param_b", "param_a"],  # `y_axis` has one observation.
    ],
)
def test_generate_contour_plot_for_few_observations(params: List[str]) -> None:

    study = prepare_study_with_trials(less_than_two=True)
    info = _get_contour_info(study, params=params)
    assert info.sorted_params == sorted(params)
    assert np.shape(np.asarray(info.sub_plot_infos)) == (1, 1, 3)
    xaxis = info.sub_plot_infos[0][0].xaxis
    yaxis = info.sub_plot_infos[0][0].yaxis
    assert xaxis.name == sorted(params)[0]
    assert yaxis.name == sorted(params)[1]
    assert xaxis.range == (1.0, 1.0)
    assert yaxis.range == (-0.1, 2.1)
    assert not xaxis.is_log
    assert not yaxis.is_log
    assert not xaxis.is_cat
    assert not yaxis.is_cat
    assert xaxis.indices == [1.0]
    assert yaxis.indices == [-0.1, 0.0, 2.0, 2.1]
    assert xaxis.values == [1.0, None]
    assert yaxis.values == [2.0, 0.0]
    z_values = np.asarray(info.sub_plot_infos[0][0].z_values)
    assert np.all([np.isnan(v) for v in z_values.flatten()])


def test_get_contour_info_log_scale_and_str_category_2_params() -> None:

    # If the search space has two parameters, plot_contour generates a single plot.
    study = create_study()
    distributions = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": CategoricalDistribution(["100", "101"]),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "101"},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "100"},
            distributions=distributions,
        )
    )

    info = _get_contour_info(study)
    assert info.sorted_params == ["param_a", "param_b"]
    assert np.shape(np.asarray(info.sub_plot_infos)) == (1, 1, 3)
    xaxis = info.sub_plot_infos[0][0].xaxis
    yaxis = info.sub_plot_infos[0][0].yaxis
    assert xaxis.name == "param_a"
    assert yaxis.name == "param_b"
    assert xaxis.range == (math.pow(10, -6.05), math.pow(10, -4.95))
    assert yaxis.range == (-0.05, 1.05)
    assert xaxis.is_log
    assert not yaxis.is_log
    assert not xaxis.is_cat
    assert yaxis.is_cat
    assert xaxis.indices == [math.pow(10, -6.05), 1e-6, 1e-5, math.pow(10, -4.95)]
    assert yaxis.indices == ["100", "101"]
    assert xaxis.values == [1e-6, 1e-5]
    assert yaxis.values == ["101", "100"]
    z_values = np.asarray(info.sub_plot_infos[0][0].z_values)
    assert np.shape(z_values) == (2, 4)
    assert z_values[1][1] == 0.0
    assert z_values[0][2] == 1.0


def test_get_contour_info_log_scale_and_str_category_more_than_2_params() -> None:
    # If the search space has three parameters, plot_contour generates nine plots.
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

    info = _get_contour_info(study)
    params = ["param_a", "param_b", "param_c"]
    assert info.sorted_params == params
    assert np.shape(np.asarray(info.sub_plot_infos)) == (3, 3, 3)
    ranges = {
        "param_a": (math.pow(10, -6.05), math.pow(10, -4.95)),
        "param_b": (-0.05, 1.05),
        "param_c": (-0.05, 1.05),
    }
    is_log = {"param_a": True, "param_b": False, "param_c": False}
    is_cat = {"param_a": False, "param_b": True, "param_c": True}
    indices: Dict[str, List[Union[str, float]]] = {
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
            z_values = np.asarray(info.sub_plot_infos[yi][xi].z_values)
            if xi == yi:
                assert np.shape(z_values) == (1, 0)
            else:
                assert np.shape(z_values) == (len(indices[y_param]), len(indices[x_param]))
                for i, v in enumerate([0.0, 1.0]):
                    x_value = xaxis.values[i]
                    y_value = yaxis.values[i]
                    assert x_value is not None
                    assert y_value is not None
                    xi = xaxis.indices.index(x_value)
                    yi = yaxis.indices.index(y_value)
                    assert z_values[yi][xi] == v


def test_get_contour_info_mixture_category_types() -> None:
    study = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "param_a": CategoricalDistribution([None, "100"]),
        "param_b": CategoricalDistribution([101, 102.0]),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": None, "param_b": 101},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=0.5,
            params={"param_a": "100", "param_b": 102.0},
            distributions=distributions,
        )
    )

    info = _get_contour_info(study)
    assert info.sorted_params == ["param_a", "param_b"]
    assert np.shape(np.asarray(info.sub_plot_infos)) == (1, 1, 3)
    xaxis = info.sub_plot_infos[0][0].xaxis
    yaxis = info.sub_plot_infos[0][0].yaxis
    assert xaxis.name == "param_a"
    assert yaxis.name == "param_b"
    assert xaxis.range == (-0.05, 1.05)
    assert yaxis.range == (100.95, 102.05)
    assert not xaxis.is_log
    assert not yaxis.is_log
    assert xaxis.is_cat
    assert not yaxis.is_cat
    assert xaxis.indices == ["100", "None"]
    assert yaxis.indices == [100.95, 101, 102, 102.05]
    assert xaxis.values == ["None", "100"]
    assert yaxis.values == [101.0, 102.0]
    z_values = np.asarray(info.sub_plot_infos[0][0].z_values)
    assert np.shape(z_values) == (4, 2)
    assert z_values[1][1] == 0.0
    assert z_values[2][0] == 0.5


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_test_get_contour_info_nonfinite_removed(value: float) -> None:

    study = prepare_study_with_trials(with_c_d=True, value_for_first_trial=value)
    info = _get_contour_info(study, params=["param_b", "param_d"])
    assert info.sorted_params == ["param_b", "param_d"]
    assert np.shape(np.asarray(info.sub_plot_infos)) == (1, 1, 3)
    xaxis = info.sub_plot_infos[0][0].xaxis
    yaxis = info.sub_plot_infos[0][0].yaxis
    assert xaxis.name == "param_b"
    assert yaxis.name == "param_d"
    assert xaxis.range == (-0.05, 1.05)
    assert yaxis.range == (1.9, 4.1)
    assert not xaxis.is_log
    assert not yaxis.is_log
    assert not xaxis.is_cat
    assert not yaxis.is_cat
    assert xaxis.indices == [-0.05, 0.0, 1.0, 1.05]
    assert yaxis.indices == [1.9, 2.0, 4.0, 4.1]
    assert xaxis.values == [0.0, 1.0]
    assert yaxis.values == [4.0, 2.0]
    z_values = np.asarray(info.sub_plot_infos[0][0].z_values)
    assert np.shape(z_values) == (4, 4)
    assert value not in z_values.flatten()


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_test_get_contour_info_nonfinite_multiobjective(objective: int, value: float) -> None:

    study = prepare_study_with_trials(with_c_d=True, n_objectives=2, value_for_first_trial=value)
    info = _get_contour_info(
        study, params=["param_b", "param_d"], target=lambda t: t.values[objective]
    )
    assert info.sorted_params == ["param_b", "param_d"]
    assert np.shape(np.asarray(info.sub_plot_infos)) == (1, 1, 3)
    xaxis = info.sub_plot_infos[0][0].xaxis
    yaxis = info.sub_plot_infos[0][0].yaxis
    assert xaxis.name == "param_b"
    assert yaxis.name == "param_d"
    assert xaxis.range == (-0.05, 1.05)
    assert yaxis.range == (1.9, 4.1)
    assert not xaxis.is_log
    assert not yaxis.is_log
    assert not xaxis.is_cat
    assert not yaxis.is_cat
    assert xaxis.indices == [-0.05, 0.0, 1.0, 1.05]
    assert yaxis.indices == [1.9, 2.0, 4.0, 4.1]
    assert xaxis.values == [0.0, 1.0]
    assert yaxis.values == [4.0, 2.0]
    z_values = np.asarray(info.sub_plot_infos[0][0].z_values)
    assert np.shape(z_values) == (4, 4)
    assert value not in z_values.flatten()


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = prepare_study_with_trials(with_c_d=False, direction=direction)

    # `target` is `None`.
    contour = plotly_plot_contour(study).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    if direction == "minimize":
        assert contour["reversescale"]
    else:
        assert not contour["reversescale"]

    # When `target` is not `None`, `reversescale` is always `True`.
    contour = plotly_plot_contour(study, target=lambda t: t.number).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    assert contour["reversescale"]

    # Multi-objective optimization.
    study = prepare_study_with_trials(with_c_d=False, n_objectives=2, direction=direction)
    contour = plotly_plot_contour(study, target=lambda t: t.number).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    assert contour["reversescale"]
