import pytest

from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_slice


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_slice(study)


def test_plot_slice() -> None:

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
    assert figure.layout.yaxis.title.text == "Objective Value"

    # Test with a trial to select parameter.
    figure = plot_slice(study, params=["param_a"])
    assert len(figure.data) == 1
    assert figure.data[0]["x"] == (1.0, 2.5)
    assert figure.data[0]["y"] == (0.0, 1.0)

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_slice(study, params=["param_a"], target=lambda t: t.params["param_b"])
    assert len(figure.data) == 1
    assert figure.data[0]["x"] == (1.0, 2.5)
    assert figure.data[0]["y"] == (2.0, 1.0)
    assert figure.layout.yaxis.title.text == "Objective Value"

    # Test with a customized target name.
    figure = plot_slice(study, target_name="Target Name")
    assert figure.layout.yaxis.title.text == "Target Name"

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_slice(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_slice(study)
    assert len(figure.data) == 0


def test_plot_slice_log_scale() -> None:

    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"x_linear": 1.0, "y_log": 1e-3},
            distributions={
                "x_linear": FloatDistribution(0.0, 3.0),
                "y_log": FloatDistribution(1e-5, 1.0, log=True),
            },
        )
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
