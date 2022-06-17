from io import BytesIO
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
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_contour as plotly_plot_contour
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


def is_empty(figure: Union[go.Figure, Axes, np.ndarray]) -> bool:
    if isinstance(figure, go.Figure):
        return len(figure.data) == 0
    elif isinstance(figure, Axes):
        return not figure.has_data()
    else:
        return not any([f.has_data() for f in figure.flatten()])


def save_static_image(figure: Union[go.Figure, Axes, np.ndarray]) -> None:
    if isinstance(figure, go.Figure):
        with tempfile.TemporaryDirectory() as td:
            figure.write_image(td + "tmp.png")
    else:
        plt.savefig(BytesIO())


def get_n_plots(figure: Union[go.Figure, Axes, np.ndarray]) -> int:
    if isinstance(figure, go.Figure):
        if len(figure.data) <= 2:
            return 1
        return (len(figure.data) - int((1 + np.sqrt(1 + 8 * len(figure.data))) / 4)) // 2
    elif isinstance(figure, Axes):
        return 1
    else:
        assert len(figure.shape) == 2
        assert figure.shape[0] == figure.shape[1]
        return figure.shape[0] * (figure.shape[0] - 1)


def get_x_points(figure: Union[go.Figure, Axes]) -> Optional[List[float]]:
    if isinstance(figure, go.Figure):
        return list(figure.data[1]["x"]) if figure.data[1]["x"] is not None else None
    elif isinstance(figure, Axes):
        if len(figure.collections) == 0:
            return None
        return list(figure.collections[-1].get_offsets()[:, 0])
    else:
        assert False


def get_y_points(figure: Union[go.Figure, Axes]) -> Optional[List[float]]:
    if isinstance(figure, go.Figure):
        return list(figure.data[1]["y"]) if figure.data[1]["y"] is not None else None
    elif isinstance(figure, Axes):
        if len(figure.collections) == 0:
            return None
        return list(figure.collections[-1].get_offsets()[:, 1])
    else:
        assert False


def get_x_range(figure: Union[go.Figure, Axes]) -> Tuple[float, float]:
    if isinstance(figure, go.Figure):
        return figure.layout["xaxis"].range
    elif isinstance(figure, Axes):
        if get_x_is_log(figure):
            l, r = figure.get_xlim()
            return np.log10(l), np.log10(r)
        return figure.get_xlim()
    else:
        assert False


def get_y_range(figure: Union[go.Figure, Axes]) -> Tuple[float, float]:
    if isinstance(figure, go.Figure):
        return figure.layout["yaxis"].range
    elif isinstance(figure, Axes):
        if get_y_is_log(figure):
            l, r = figure.get_ylim()
            return np.log10(l), np.log10(r)
        return figure.get_ylim()
    else:
        assert False


def get_x_name(figure: Union[go.Figure, Axes]) -> str:
    if isinstance(figure, go.Figure):
        return figure.layout["xaxis"].title.text
    elif isinstance(figure, Axes):
        return figure.get_xlabel()
    else:
        assert False


def get_y_name(figure: Union[go.Figure, Axes]) -> str:
    if isinstance(figure, go.Figure):
        return figure.layout["yaxis"].title.text
    elif isinstance(figure, Axes):
        return figure.get_ylabel()
    else:
        assert False


def get_x_ticks(figure: Union[go.Figure, Axes]) -> List[CategoricalChoiceType]:
    if isinstance(figure, go.Figure):
        return list(figure.data[0]["x"])
    elif isinstance(figure, Axes):
        return list(map(lambda t: t.get_text(), figure.get_xticklabels()))
    else:
        assert False


def get_y_ticks(figure: Union[go.Figure, Axes]) -> List[CategoricalChoiceType]:
    if isinstance(figure, go.Figure):
        return list(figure.data[0]["y"])
    elif isinstance(figure, Axes):
        return list(map(lambda t: t.get_text(), figure.get_yticklabels()))
    else:
        assert False


def get_x_is_log(figure: Union[go.Figure, Axes]) -> bool:
    if isinstance(figure, go.Figure):
        return figure.layout["xaxis"].type == "log"
    elif isinstance(figure, Axes):
        return figure.get_xscale() == "log"
    else:
        assert False


def get_y_is_log(figure: Union[go.Figure, Axes]) -> bool:
    if isinstance(figure, go.Figure):
        return figure.layout["yaxis"].type == "log"
    elif isinstance(figure, Axes):
        return figure.get_yscale() == "log"
    else:
        assert False


def get_target_name(figure: Union[go.Figure, Axes]) -> Optional[str]:
    if isinstance(figure, go.Figure):
        return figure.data[0]["colorbar"].title.text
    elif isinstance(figure, Axes):
        target_name = figure.figure.axes[-1].get_ylabel()
        if target_name == "":
            return None
        return target_name
    else:
        assert False


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
def test_plot_contour_no_trial(
    plot_contour: Callable[..., Any], params: Optional[List[str]]
) -> None:

    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = plot_contour(study_without_trials, params=params)
    assert is_empty(figure)
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
def test_plot_contour_ignored_error(
    plot_contour: Callable[..., Any], params: Optional[List[str]]
) -> None:
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    assert is_empty(figure)
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_error(plot_contour: Callable[..., Any]) -> None:
    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        plot_contour(study, ["optuna", "Optuna"])


@parametrize_plot_contour
@pytest.mark.parametrize("params", [[], ["param_a"]])
def test_plot_contour_empty(plot_contour: Callable[..., Any], params: Optional[List[str]]) -> None:
    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params)
    assert is_empty(figure)
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_2_params(plot_contour: Callable[..., Any]) -> None:
    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params)
    assert get_n_plots(figure) == 1
    assert not is_empty(figure)
    assert get_target_name(figure) == "Objective Value"
    assert get_x_points(figure) == [1.0, 2.5]
    assert get_y_points(figure) == [2.0, 1.0]
    assert get_x_range(figure) == (0.925, 2.575)
    assert get_y_range(figure) == (-0.1, 2.1)
    assert get_x_name(figure) == "param_a"
    assert get_y_name(figure) == "param_b"
    save_static_image(figure)


@parametrize_plot_contour
@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_plot_contour_more_than_2_params(
    plot_contour: Callable[..., Any], params: Optional[List[str]]
) -> None:

    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params)
    n_params = len(params) if params is not None else 4
    assert get_n_plots(figure) == n_params * (n_params - 1)
    assert not is_empty(figure)
    save_static_image(figure)


@parametrize_plot_contour
@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target(
    plot_contour: Callable[..., Any], params: List[str]
) -> None:

    study = prepare_study_with_trials()
    with pytest.warns(UserWarning):
        figure = plot_contour(study, params=params, target=lambda t: t.params["param_d"])
    assert get_n_plots(figure) == 1 if len(params) == 2 else len(params) * (len(params) - 1)
    assert not is_empty(figure)
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_customized_target_name(plot_contour: Callable[..., Any]) -> None:

    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params, target_name="Target Name")
    assert get_n_plots(figure) == 1
    assert not is_empty(figure)
    assert get_target_name(figure) == "Target Name"
    save_static_image(figure)


@parametrize_plot_contour
@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # `x_axis` has one observation.
        ["param_b", "param_a"],  # `y_axis` has one observation.
    ],
)
def test_generate_contour_plot_for_few_observations(
    plot_contour: Callable[..., Any],
    params: List[str],
) -> None:

    study = prepare_study_with_trials(less_than_two=True)

    figure = plot_contour(study, params=params)
    assert get_x_points(figure) is None
    assert get_y_points(figure) is None
    assert get_target_name(figure) is None
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_log_scale_and_str_category_2_params(
    plot_contour: Callable[..., Any],
) -> None:

    # If the search space has two parameters, plot_contour generates a single plot.
    study = create_study()
    distributions = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": CategoricalDistribution(["100", "101"]),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "100"},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "101"},
            distributions=distributions,
        )
    )

    figure = plot_contour(study)
    assert not is_empty(figure)
    assert get_n_plots(figure) == 1
    assert get_x_range(figure) == (-6.05, -4.95)
    assert get_y_range(figure) == (-0.05, 1.05)
    assert get_x_name(figure) == "param_a"
    assert get_y_name(figure) == "param_b"
    assert get_x_is_log(figure)
    assert not get_y_is_log(figure)
    assert get_y_ticks(figure) == ["100", "101"]
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_log_scale_and_str_category_more_than_2_params(
    plot_contour: Callable[..., Any],
) -> None:
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
            params={"param_a": 1e-6, "param_b": "100", "param_c": "one"},
            distributions=distributions,
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "101", "param_c": "two"},
            distributions=distributions,
        )
    )

    figure = plot_contour(study)
    assert get_n_plots(figure) == 6
    assert not is_empty(figure)
    save_static_image(figure)


@parametrize_plot_contour
def test_plot_contour_mixture_category_types(
    plot_contour: Callable[..., Any],
) -> None:
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
    figure = plot_contour(study)

    # yaxis is treated as non-categorical
    assert get_x_range(figure) == (-0.05, 1.05)
    assert get_y_range(figure) == (100.95, 102.05)
    assert get_x_ticks(figure) == ["100", "None"]
    assert get_y_ticks(figure) != [101, 102.0]
    save_static_image(figure)


@parametrize_plot_contour
@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_nonfinite_removed(plot_contour: Callable[..., Any], value: float) -> None:

    study = prepare_study_with_trials(with_c_d=True, value_for_first_trial=value)
    figure = plot_contour(study, params=["param_b", "param_d"])
    save_static_image(figure)


@parametrize_plot_contour
@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_nonfinite_multiobjective(
    plot_contour: Callable[..., Any], objective: int, value: float
) -> None:

    study = prepare_study_with_trials(with_c_d=True, n_objectives=2, value_for_first_trial=value)
    figure = plot_contour(
        study, params=["param_b", "param_d"], target=lambda t: t.values[objective]
    )
    save_static_image(figure)


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
