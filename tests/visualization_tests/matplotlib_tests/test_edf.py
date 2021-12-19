import pytest

from optuna.study import create_study
from optuna.visualization.matplotlib import plot_edf
from typing import List
import matplotlib.pyplot as plt

def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_edf(study)

confirm_0_1 = lambda y: 0<=y and y<=1

def confirm_monotonous_increase_and_0_1(lst:List)-> bool:
    last_value=lst[0]
    if not confirm_0_1(last_value):
        return False
    for i in lst[1:]:
        if not confirm_0_1(i):
            return False
        if last_value<=i:
            last_value=i
        else:
            return False
    return True

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
    lines  = figure.get_lines()
    assert confirm_monotonous_increase_and_0_1(lines[0].get_ydata())
    assert len(lines) == 1
    assert figure.xaxis.label.get_text() == "Objective Value"

    # Test with two studies.
    study1 = create_study(direction=direction)
    study1.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf([study0, study1])
    lines  = figure.get_lines()
    for line in lines:
        y = line.get_ydata()
        assert confirm_monotonous_increase_and_0_1(y)
    assert len(lines) == 2

    figure = plot_edf((study0, study1))
    lines  = figure.get_lines()
    for line in lines:
        y = line.get_ydata()
        assert confirm_monotonous_increase_and_0_1(y)
    assert len(lines) == 2

    # Test with a customized target value.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    with pytest.warns(UserWarning):
        figure = plot_edf(study0, target=lambda t: t.params["x"])
    lines  = figure.get_lines()
    assert confirm_monotonous_increase_and_0_1(lines[0].get_ydata())
    assert len(lines) == 1

    # Test with a customized target name.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf(study0, target_name="Target Name")
    lines  = figure.get_lines()
    assert confirm_monotonous_increase_and_0_1(lines[0].get_ydata())
    assert len(figure.get_lines()) == 1
    assert figure.xaxis.label.get_text() == "Target Name"