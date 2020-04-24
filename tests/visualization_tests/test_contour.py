import pytest

from optuna.distributions import LogUniformDistribution
from optuna.study import create_study
from optuna.study import StudyDirection
from optuna.testing.visualization import prepare_study_with_trials
from optuna import type_checking
from optuna.visualization.contour import _generate_contour_subplot
from optuna.visualization.contour import plot_contour

if type_checking.TYPE_CHECKING:
    from typing import List, Optional  # NOQA

    from optuna.trial import Trial  # NOQA


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
def test_plot_contour(params):
    # type: (Optional[List[str]]) -> None

    # Test with no trial.
    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = plot_contour(study_without_trials, params=params)
    assert len(figure.data) == 0

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_):
        # type: (Trial) -> float

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
            assert figure.data[0]["x"] == (1.0, 2.5)
            assert figure.data[0]["y"] == (0.0, 1.0, 2.0)
            assert figure.data[1]["x"] == (1.0, 2.5)
            assert figure.data[1]["y"] == (2.0, 1.0)
            assert figure.layout["xaxis"]["range"] == (1.0, 2.5)
            assert figure.layout["yaxis"]["range"] == (0.0, 2.0)
    else:
        # TODO(crcrpar): Add more checks. Currently this only checks the number of data.
        n_params = len(params) if params is not None else 4
        assert len(figure.data) == n_params ** 2 + n_params * (n_params - 1)


def test_generate_contour_plot_for_few_observations():
    # type: () -> None

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


def test_plot_contour_log_scale():
    # type: () -> None

    # If the search space has two parameters, plot_contour generates a single plot.
    study = create_study()
    study._append_trial(
        value=0.0,
        params={"param_a": 1e-6, "param_b": 1e-4,},
        distributions={
            "param_a": LogUniformDistribution(1e-7, 1e-2),
            "param_b": LogUniformDistribution(1e-5, 1e-1),
        },
    )
    study._append_trial(
        value=1.0,
        params={"param_a": 1e-5, "param_b": 1e-3,},
        distributions={
            "param_a": LogUniformDistribution(1e-7, 1e-2),
            "param_b": LogUniformDistribution(1e-5, 1e-1),
        },
    )

    figure = plot_contour(study)
    assert figure.layout["xaxis"]["range"] == (-6, -5)
    assert figure.layout["yaxis"]["range"] == (-4, -3)
    assert figure.layout["xaxis_type"] == "log"
    assert figure.layout["yaxis_type"] == "log"

    # If the search space has three parameters, plot_contour generates nine plots.
    study = create_study()
    study._append_trial(
        value=0.0,
        params={"param_a": 1e-6, "param_b": 1e-4, "param_c": 1e-2,},
        distributions={
            "param_a": LogUniformDistribution(1e-7, 1e-2),
            "param_b": LogUniformDistribution(1e-5, 1e-1),
            "param_c": LogUniformDistribution(1e-3, 10),
        },
    )
    study._append_trial(
        value=1.0,
        params={"param_a": 1e-5, "param_b": 1e-3, "param_c": 1e-1,},
        distributions={
            "param_a": LogUniformDistribution(1e-7, 1e-2),
            "param_b": LogUniformDistribution(1e-5, 1e-1),
            "param_c": LogUniformDistribution(1e-3, 10),
        },
    )

    figure = plot_contour(study)
    param_a_range = (-6, -5)
    param_b_range = (-4, -3)
    param_c_range = (-2, -1)
    axis_to_range = {
        "xaxis": param_a_range,
        "xaxis2": param_b_range,
        "xaxis3": param_c_range,
        "xaxis4": param_a_range,
        "xaxis5": param_b_range,
        "xaxis6": param_c_range,
        "xaxis7": param_a_range,
        "xaxis8": param_b_range,
        "xaxis9": param_c_range,
        "yaxis": param_a_range,
        "yaxis2": param_a_range,
        "yaxis3": param_a_range,
        "yaxis4": param_b_range,
        "yaxis5": param_b_range,
        "yaxis6": param_b_range,
        "yaxis7": param_c_range,
        "yaxis8": param_c_range,
        "yaxis9": param_c_range,
    }

    for axis, param_range in axis_to_range.items():
        assert figure.layout[axis]["range"] == param_range
        assert figure.layout[axis]["type"] == "log"
