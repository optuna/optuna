import pytest

from optuna.study import create_study
from optuna.visualization.matplotlib import plot_edf


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history(direction: str) -> None:
    # Test with no studies.
    figure = plot_edf([])
    assert not figure.has_data()

    # Test with no trials.
    figure = plot_edf(create_study(direction=direction))
    assert not figure.has_data()

    figure = plot_edf([create_study(direction=direction), create_study(direction=direction)])
    assert not figure.has_data()

    # Test with a study.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf(study0)
    assert figure.has_data()

    # Test with two studies.
    # TODO(ytknzw): Add more specific assertion with the numbers of the studies.
    study1 = create_study(direction=direction)
    study1.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf([study0, study1])
    assert figure.has_data()
    figure = plot_edf((study0, study1))
    assert figure.has_data()
