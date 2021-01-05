import pytest

from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_slice


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_slice(study)


def test_plot_slice() -> None:

    # Test with no trial.
    study = prepare_study_with_trials(no_trials=True)
    figure = plot_slice(study)
    assert not figure.has_data()

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    # TODO(ytknzw): Add more specific assertion with the test case.
    figure = plot_slice(study)
    assert len(figure) == 2
    assert figure[0].has_data()
    assert figure[1].has_data()

    # Test with a trial to select parameter.
    # TODO(ytknzw): Add more specific assertion with the test case.
    figure = plot_slice(study, params=["param_a"])
    assert figure.has_data()

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_slice(study, params=["param_a"], target=lambda t: t.params["param_b"])
    assert figure.has_data()

    # Test with a customized target name.
    figure = plot_slice(study, target_name="Target Name")
    assert len(figure) == 2
    assert figure[0].has_data()
    assert figure[1].has_data()

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_slice(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_slice(study)
    assert not figure.has_data()


def test_plot_slice_log_scale() -> None:

    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"x_linear": 1.0, "y_log": 1e-3},
            distributions={
                "x_linear": UniformDistribution(0.0, 3.0),
                "y_log": LogUniformDistribution(1e-5, 1.0),
            },
        )
    )

    # Plot a parameter.
    # TODO(ytknzw): Add more specific assertion with the test case.
    figure = plot_slice(study, params=["y_log"])
    assert figure.has_data()
    figure = plot_slice(study, params=["x_linear"])
    assert figure.has_data()

    # Plot multiple parameters.
    # TODO(ytknzw): Add more specific assertion with the test case.
    figure = plot_slice(study)
    assert len(figure) == 2
    assert figure[0].has_data()
    assert figure[1].has_data()
