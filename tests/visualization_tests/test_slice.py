import pytest

from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna import type_checking
from optuna.visualization.slice import plot_slice

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


def test_plot_slice():
    # type: () -> None

    # Test with no trial.
    study = prepare_study_with_trials(no_trials=True)
    figure = plot_slice(study)
    assert len(figure.data) == 0

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_slice(study)
    assert len(figure.data) == 2
    assert figure.data[0]["x"] == (1.0, 2.5)
    assert figure.data[0]["y"] == (0.0, 1.0)
    assert figure.data[1]["x"] == (2.0, 0.0, 1.0)
    assert figure.data[1]["y"] == (0.0, 2.0, 1.0)

    # Test with a trial to select parameter.
    figure = plot_slice(study, params=["param_a"])
    assert len(figure.data) == 1
    assert figure.data[0]["x"] == (1.0, 2.5)
    assert figure.data[0]["y"] == (0.0, 1.0)

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_slice(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_slice(study)
    assert len(figure.data) == 0


def test_plot_slice_log_scale():
    # type: () -> None

    study = create_study()
    study._append_trial(
        value=0.0,
        params={"x_linear": 1.0, "y_log": 1e-3,},
        distributions={
            "x_linear": UniformDistribution(0.0, 3.0),
            "y_log": LogUniformDistribution(1e-5, 1.0),
        },
    )

    # Plot a parameter.
    figure = plot_slice(study, params=["y_log"])
    assert figure.layout["xaxis_type"] == "log"
    figure = plot_slice(study, params=["x_linear"])
    assert figure.layout["xaxis_type"] is None

    # Plot multiple parameters.
    figure = plot_slice(study)
    assert figure.layout["xaxis_type"] is None
    assert figure.layout["xaxis2_type"] == "log"
