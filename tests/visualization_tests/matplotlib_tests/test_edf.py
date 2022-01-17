from typing import Sequence

import numpy as np
import pytest

from optuna.study import create_study
from optuna.visualization.matplotlib import plot_edf


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_edf(study)


def _validate_edf_values(edf_values: Sequence[float]) -> None:
    np_values = np.array(edf_values)
    # This line confirms that the values are monotonically non-decreasing.
    assert np.all(np_values[1:] - np_values[:-1] >= 0)
    # This line confirms that the values are in [0,1].
    assert np.all((0 <= np_values) * (np_values <= 1))


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history(direction: str) -> None:
    # Test with no studies.
    figure = plot_edf([])
    assert len(figure.get_lines()) == 0

    # Test with no trials.
    figure = plot_edf(create_study(direction=direction))
    assert len(figure.get_lines()) == 0

    figure = plot_edf([create_study(direction=direction), create_study(direction=direction)])
    assert len(figure.get_lines()) == 0

    # Test with a study.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf(study0)
    lines = figure.get_lines()
    _validate_edf_values(lines[0].get_ydata())
    assert len(lines) == 1
    assert figure.xaxis.label.get_text() == "Objective Value"

    # Test with two studies.
    study1 = create_study(direction=direction)
    study1.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf([study0, study1])
    lines = figure.get_lines()
    for line in lines:
        _validate_edf_values(line.get_ydata())
    assert len(lines) == 2

    figure = plot_edf((study0, study1))
    lines = figure.get_lines()
    for line in lines:
        _validate_edf_values(line.get_ydata())
    assert len(lines) == 2

    # Test with a customized target value.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    with pytest.warns(UserWarning):
        figure = plot_edf(study0, target=lambda t: t.params["x"])
    lines = figure.get_lines()
    _validate_edf_values(lines[0].get_ydata())
    assert len(lines) == 1

    # Test with a customized target name.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf(study0, target_name="Target Name")
    lines = figure.get_lines()
    _validate_edf_values(lines[0].get_ydata())
    assert len(figure.get_lines()) == 1
    assert figure.xaxis.label.get_text() == "Target Name"
