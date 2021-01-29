import pytest

from optuna.study import create_study
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_optimization_history


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_optimization_history(study)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history(direction: str) -> None:
    # Test with no trial.
    study = create_study(direction=direction)
    figure = plot_optimization_history(study)
    assert not figure.has_data()

    def objective(trial: Trial) -> float:

        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with a trial.
    # TODO(ytknzw): Add more specific assertion with the test case.
    study = create_study(direction=direction)
    study.optimize(objective, n_trials=3)
    figure = plot_optimization_history(study)
    assert figure.has_data()

    # Test customized target.
    with pytest.warns(UserWarning):
        figure = plot_optimization_history(study, target=lambda t: t.number)
    assert figure.has_data()

    # Test customized target name.
    figure = plot_optimization_history(study, target_name="Target Name")
    assert figure.has_data()

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:
        raise ValueError

    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    figure = plot_optimization_history(study)
    assert not figure.has_data()
