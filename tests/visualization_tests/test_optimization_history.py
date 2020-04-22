import pytest

from optuna.study import create_study
from optuna import type_checking
from optuna.visualization.optimization_history import plot_optimization_history

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history(direction):
    # type: (str) -> None

    # Test with no trial.
    study = create_study(direction=direction)
    figure = plot_optimization_history(study)
    assert len(figure.data) == 0

    def objective(trial):
        # type: (Trial) -> float

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
        assert figure.data[1].y == (1.0, 1.0, 0.0)
    else:
        assert figure.data[1].y == (1.0, 2.0, 2.0)

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    figure = plot_optimization_history(study)
    assert len(figure.data) == 0
