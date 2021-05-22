import itertools
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from packaging import version
import plotly
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
from optuna.study import create_study
from optuna.study import StudyDirection
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_contour
from optuna.visualization._contour import _generate_contour_subplot


RANGE_TYPE = Union[Tuple[str, str], Tuple[float, float]]


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_contour(study)


def test_target_is_not_none_and_study_is_multi_obj() -> None:

    # Multiple sub-figures.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=True)
    plot_contour(study, target=lambda t: t.values[0])

    # Single figure.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=False)
    plot_contour(study, target=lambda t: t.values[0])


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
def test_plot_contour(params: Optional[List[str]]) -> None:

    # Test with no trial.
    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = plot_contour(study_without_trials, params=params)
    assert len(figure.data) == 0

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    assert len(figure.data) == 0

    # Test with some trials.
    study = prepare_study_with_trials()

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        plot_contour(study, ["optuna", "Optuna"])

    figure = plot_contour(study, params=params)
    if params is not None and len(params) < 3:
        if len(params) <= 1:
            assert not figure.data
        elif len(params) == 2:
            assert figure.data[0]["x"] == (0.925, 1.0, 2.5, 2.575)
            assert figure.data[0]["y"] == (-0.1, 0.0, 1.0, 2.0, 2.1)
            assert figure.data[0]["z"][3][1] == 0.0
            assert figure.data[0]["z"][2][2] == 1.0
            assert figure.data[0]["colorbar"]["title"]["text"] == "Objective Value"
            assert figure.data[1]["x"] == (1.0, 2.5)
            assert figure.data[1]["y"] == (2.0, 1.0)
            assert figure.layout["xaxis"]["range"] == (0.925, 2.575)
            assert figure.layout["yaxis"]["range"] == (-0.1, 2.1)
    else:
        # TODO(crcrpar): Add more checks. Currently this only checks the number of data.
        n_params = len(params) if params is not None else 4
        assert len(figure.data) == n_params ** 2 + n_params * (n_params - 1)


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target(params: List[str]) -> None:

    study = prepare_study_with_trials()
    with pytest.warns(UserWarning):
        figure = plot_contour(study, params=params, target=lambda t: t.params["param_d"])
    for data in figure.data:
        if "z" in data:
            assert 4.0 in itertools.chain.from_iterable(data["z"])
            assert 2.0 in itertools.chain.from_iterable(data["z"])
    if len(params) == 2:
        assert figure.data[0]["z"][3][1] == 4.0
        assert figure.data[0]["z"][2][2] == 2.0


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target_name(params: List[str]) -> None:

    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params, target_name="Target Name")
    for data in figure.data:
        if "colorbar" in data:
            assert data["colorbar"]["title"]["text"] == "Target Name"


def test_generate_contour_plot_for_few_observations() -> None:

    study = prepare_study_with_trials(less_than_two=True)
    trials = study.trials

    # `x_axis` has one observation.
    params = ["param_a", "param_b"]
    contour, scatter = _generate_contour_subplot(
        trials, params[0], params[1], StudyDirection.MINIMIZE
    )
    assert contour.x is None and contour.y is None and scatter.x is None and scatter.y is None

    # `y_axis` has one observation.
    params = ["param_b", "param_a"]
    contour, scatter = _generate_contour_subplot(
        trials, params[0], params[1], StudyDirection.MINIMIZE
    )
    assert contour.x is None and contour.y is None and scatter.x is None and scatter.y is None


def test_plot_contour_log_scale_and_str_category() -> None:

    # If the search space has two parameters, plot_contour generates a single plot.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "100"},
            distributions={
                "param_a": LogUniformDistribution(1e-7, 1e-2),
                "param_b": CategoricalDistribution(["100", "101"]),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "101"},
            distributions={
                "param_a": LogUniformDistribution(1e-7, 1e-2),
                "param_b": CategoricalDistribution(["100", "101"]),
            },
        )
    )

    figure = plot_contour(study)
    assert figure.layout["xaxis"]["range"] == (-6.05, -4.95)
    if version.parse(plotly.__version__) >= version.parse("4.12.0"):
        assert figure.layout["yaxis"]["range"] == (-0.05, 1.05)
    else:
        assert figure.layout["yaxis"]["range"] == ("100", "101")
    assert figure.layout["xaxis_type"] == "log"
    assert figure.layout["yaxis_type"] == "category"

    # If the search space has three parameters, plot_contour generates nine plots.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "100", "param_c": "one"},
            distributions={
                "param_a": LogUniformDistribution(1e-7, 1e-2),
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
                "param_a": LogUniformDistribution(1e-7, 1e-2),
                "param_b": CategoricalDistribution(["100", "101"]),
                "param_c": CategoricalDistribution(["one", "two"]),
            },
        )
    )

    figure = plot_contour(study)
    param_a_range = (-6.05, -4.95)
    if version.parse(plotly.__version__) >= version.parse("4.12.0"):
        param_b_range: RANGE_TYPE = (-0.05, 1.05)
        param_c_range: RANGE_TYPE = (-0.05, 1.05)
    else:
        param_b_range = ("100", "101")
        param_c_range = ("one", "two")
    param_a_type = "log"
    param_b_type = "category"
    param_c_type = "category"
    axis_to_range_and_type = {
        "xaxis": (param_a_range, param_a_type),
        "xaxis2": (param_b_range, param_b_type),
        "xaxis3": (param_c_range, param_c_type),
        "xaxis4": (param_a_range, param_a_type),
        "xaxis5": (param_b_range, param_b_type),
        "xaxis6": (param_c_range, param_c_type),
        "xaxis7": (param_a_range, param_a_type),
        "xaxis8": (param_b_range, param_b_type),
        "xaxis9": (param_c_range, param_c_type),
        "yaxis": (param_a_range, param_a_type),
        "yaxis2": (param_a_range, param_a_type),
        "yaxis3": (param_a_range, param_a_type),
        "yaxis4": (param_b_range, param_b_type),
        "yaxis5": (param_b_range, param_b_type),
        "yaxis6": (param_b_range, param_b_type),
        "yaxis7": (param_c_range, param_c_type),
        "yaxis8": (param_c_range, param_c_type),
        "yaxis9": (param_c_range, param_c_type),
    }

    for axis, (param_range, param_type) in axis_to_range_and_type.items():
        assert figure.layout[axis]["range"] == param_range
        assert figure.layout[axis]["type"] == param_type


def test_plot_contour_mixture_category_types() -> None:
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

    # yaxis is treated as non-categorical
    if version.parse(plotly.__version__) >= version.parse("4.12.0"):
        assert figure.layout["xaxis"]["range"] == (-0.05, 1.05)
        assert figure.layout["yaxis"]["range"] == (100.95, 102.05)
    else:
        assert figure.layout["xaxis"]["range"] == ("100", "None")
        assert figure.layout["yaxis"]["range"] == (100.95, 102.05)
    assert figure.layout["xaxis"]["type"] == "category"
    assert figure.layout["yaxis"]["type"] != "category"
