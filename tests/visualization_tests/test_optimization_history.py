import numpy as np
import pytest

from optuna.study import create_study
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_optimization_history(study)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history(direction: str) -> None:
    # Test with no trial.
    study = create_study(direction=direction)
    figure = plot_optimization_history(study)
    assert len(figure.data) == 0

    def objective(trial: Trial) -> float:

        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with a trial.
    study = create_study(direction=direction)
    study.optimize(objective, n_trials=3)
    figure = plot_optimization_history(study)
    assert len(figure.data) == 2
    assert figure.data[0].x == (0, 1, 2)
    assert figure.data[0].y == (1.0, 2.0, 0.0)
    assert figure.data[1].x == (0, 1, 2)
    if direction == "minimize":
        assert np.array_equal(figure.data[1].y, np.array([1.0, 1.0, 0.0], dtype=np.float64))
    else:
        assert np.array_equal(figure.data[1].y, np.array([1.0, 2.0, 2.0], dtype=np.float64))
    assert figure.data[0].name == "Objective Value"
    assert figure.layout.yaxis.title.text == "Objective Value"

    # Test customized target.
    with pytest.warns(UserWarning):
        figure = plot_optimization_history(study, target=lambda t: t.number)
    assert len(figure.data) == 1
    assert np.array_equal(figure.data[0].x, np.array([0, 1, 2], dtype=float))
    assert np.array_equal(figure.data[0].y, np.array([0, 1, 2], dtype=float))

    # Test customized target name.
    figure = plot_optimization_history(study, target_name="Target Name")
    assert figure.data[0].name == "Target Name"
    assert figure.layout.yaxis.title.text == "Target Name"

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:
        raise ValueError

    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    figure = plot_optimization_history(study)
    assert len(figure.data) == 0
