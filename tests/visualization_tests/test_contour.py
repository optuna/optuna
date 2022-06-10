from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.testing.visualization.contour import BaseTestableContourFigure
from optuna.testing.visualization.contour import MatplotlibContourFigure
from optuna.testing.visualization.contour import PlotlyContourFigure
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_contour
from optuna.visualization._utils import COLOR_SCALE


RANGE_TYPE = Union[Tuple[str, str], Tuple[float, float]]
parametrize_figure = pytest.mark.parametrize(
    "create_figure", [PlotlyContourFigure, MatplotlibContourFigure]
)


@parametrize_figure
def test_target_is_none_and_study_is_multi_obj(
    create_figure: Callable[..., BaseTestableContourFigure],
) -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _ = create_figure(study)


@parametrize_figure
def test_target_is_not_none_and_study_is_multi_obj(
    create_figure: Callable[..., BaseTestableContourFigure],
) -> None:

    # Multiple sub-figures.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=True)
    figure = create_figure(study, target=lambda t: t.values[0])
    figure.save_static_image()

    # Single figure.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=False)
    figure = create_figure(study, target=lambda t: t.values[0])
    figure.save_static_image()


@parametrize_figure
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
    create_figure: Callable[..., BaseTestableContourFigure], params: Optional[List[str]]
) -> None:

    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = create_figure(study_without_trials, params=params)
    assert figure.is_empty()
    figure.save_static_image()


@parametrize_figure
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
    create_figure: Callable[..., BaseTestableContourFigure], params: Optional[List[str]]
) -> None:
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = create_figure(study, params=params)
    assert figure.is_empty()
    figure.save_static_image()


@parametrize_figure
def test_plot_contour_error(create_figure: Callable[..., BaseTestableContourFigure]) -> None:

    study = prepare_study_with_trials()

    with pytest.raises(ValueError):
        _ = create_figure(study, ["optuna", "Optuna"])


@parametrize_figure
@pytest.mark.parametrize("params", [[], ["param_a"]])
def test_plot_contour_empty(
    create_figure: Callable[..., BaseTestableContourFigure], params: Optional[List[str]]
) -> None:

    study = prepare_study_with_trials()
    figure = create_figure(study, params=params)
    assert figure.is_empty()
    figure.save_static_image()


@parametrize_figure
def test_plot_contour_2_params(create_figure: Callable[..., BaseTestableContourFigure]) -> None:

    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = create_figure(study, params=params)
    assert figure.get_n_params() == 2
    assert figure.get_n_plots() == 1
    assert not figure.is_empty()
    assert figure.get_target_name() == "Objective Value"
    assert figure.get_x_points() == [1.0, 2.5]
    assert figure.get_y_points() == [2.0, 1.0]
    assert figure.get_x_range() == (0.925, 2.575)
    assert figure.get_y_range() == (-0.1, 2.1)
    assert figure.get_x_name() == "param_a"
    assert figure.get_y_name() == "param_b"
    figure.save_static_image()


@parametrize_figure
@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_plot_contour_more_than_2_params(
    create_figure: Callable[..., BaseTestableContourFigure], params: Optional[List[str]]
) -> None:

    study = prepare_study_with_trials()
    figure = create_figure(study, params=params)
    n_params = len(params) if params is not None else 4
    assert figure.get_n_plots() == n_params * (n_params - 1)
    assert figure.get_n_params() == n_params
    assert not figure.is_empty()
    figure.save_static_image()


@parametrize_figure
@pytest.mark.parametrize("params", [["param_a", "param_b"], ["param_a", "param_b", "param_c"]])
def test_plot_contour_customized_target(
    create_figure: Callable[..., BaseTestableContourFigure],
    params: List[str],
) -> None:

    study = prepare_study_with_trials()
    with pytest.warns(UserWarning):
        figure = create_figure(study, params=params, target=lambda t: t.params["param_d"])
    assert figure.get_n_params() == len(params)
    assert figure.get_n_plots() == 1 if len(params) == 2 else len(params) * (len(params) - 1)
    assert not figure.is_empty()
    figure.save_static_image()


@parametrize_figure
def test_plot_contour_customized_target_name(
    create_figure: Callable[..., BaseTestableContourFigure],
) -> None:

    params = ["param_a", "param_b"]
    study = prepare_study_with_trials()
    figure = create_figure(study, params=params, target_name="Target Name")
    assert figure.get_n_params() == 2
    assert figure.get_n_plots() == 1
    assert not figure.is_empty()
    assert figure.get_target_name() == "Target Name"
    figure.save_static_image()


@parametrize_figure
@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],  # `x_axis` has one observation.
        ["param_b", "param_a"],  # `y_axis` has one observation.
    ],
)
def test_generate_contour_plot_for_few_observations(
    create_figure: Callable[..., BaseTestableContourFigure],
    params: List[str],
) -> None:

    study = prepare_study_with_trials(less_than_two=True)

    figure = create_figure(study, params=params)
    assert figure.get_x_points() is None
    assert figure.get_y_points() is None
    assert figure.get_target_name() is None
    figure.save_static_image()


@parametrize_figure
def test_plot_contour_log_scale_and_str_category_2_params(
    create_figure: Callable[..., BaseTestableContourFigure],
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

    figure = create_figure(study)
    assert not figure.is_empty()
    assert figure.get_n_params() == 2
    assert figure.get_n_plots() == 1
    assert figure.get_x_range() == (-6.05, -4.95)
    assert figure.get_y_range() == (-0.05, 1.05)
    assert figure.get_x_name() == "param_a"
    assert figure.get_y_name() == "param_b"
    assert figure.get_x_is_log()
    assert not figure.get_y_is_log()
    assert figure.get_y_ticks() == ["100", "101"]
    figure.save_static_image()


@parametrize_figure
def test_plot_contour_log_scale_and_str_category_more_than_2_params(
    create_figure: Callable[..., BaseTestableContourFigure],
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

    figure = create_figure(study)
    assert figure.get_n_plots() == 6
    assert figure.get_n_params() == 3
    assert not figure.is_empty()
    figure.save_static_image()


@parametrize_figure
def test_plot_contour_mixture_category_types(
    create_figure: Callable[..., BaseTestableContourFigure],
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
    figure = create_figure(study)

    # yaxis is treated as non-categorical
    assert figure.get_x_range() == (-0.05, 1.05)
    assert figure.get_y_range() == (100.95, 102.05)
    assert figure.get_x_ticks() == ["100", "None"]
    assert figure.get_y_ticks() != [101, 102.0]
    figure.save_static_image()


@parametrize_figure
@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_nonfinite_removed(
    create_figure: Callable[..., BaseTestableContourFigure], value: float
) -> None:

    study = prepare_study_with_trials(with_c_d=True, value_for_first_trial=value)
    figure = create_figure(study, params=["param_b", "param_d"])
    figure.save_static_image()


@parametrize_figure
@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_nonfinite_multiobjective(
    create_figure: Callable[..., BaseTestableContourFigure], objective: int, value: float
) -> None:

    study = prepare_study_with_trials(with_c_d=True, n_objectives=2, value_for_first_trial=value)
    figure = create_figure(
        study, params=["param_b", "param_d"], target=lambda t: t.values[objective]
    )
    figure.save_static_image()


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = prepare_study_with_trials(with_c_d=False, direction=direction)

    # `target` is `None`.
    contour = plot_contour(study).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    if direction == "minimize":
        assert contour["reversescale"]
    else:
        assert not contour["reversescale"]

    # When `target` is not `None`, `reversescale` is always `True`.
    contour = plot_contour(study, target=lambda t: t.number).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    assert contour["reversescale"]

    # Multi-objective optimization.
    study = prepare_study_with_trials(with_c_d=False, n_objectives=2, direction=direction)
    contour = plot_contour(study, target=lambda t: t.number).data[0]
    assert COLOR_SCALE == [v[1] for v in contour["colorscale"]]
    assert contour["reversescale"]
