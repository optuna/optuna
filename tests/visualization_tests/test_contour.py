from typing import List
from typing import Optional

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
            assert figure.data[1]["x"] == (1.0, 2.5)
            assert figure.data[1]["y"] == (2.0, 1.0)
            assert figure.layout["xaxis"]["range"] == (0.925, 2.575)
            assert figure.layout["yaxis"]["range"] == (-0.1, 2.1)
    else:
        # TODO(crcrpar): Add more checks. Currently this only checks the number of data.
        n_params = len(params) if params is not None else 4
        assert len(figure.data) == n_params ** 2 + n_params * (n_params - 1)


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
    param_b_range = ("100", "101")
    param_c_range = ("one", "two")
    param_a_type = "log"
    param_b_type = "category"
    param_c_type = None
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
