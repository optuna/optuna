from io import BytesIO
from typing import Any
from typing import Callable
from typing import List
from typing import Union

import numpy as np
import pytest

import optuna
from optuna import Study
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.visualization import plot_edf as plotly_plot_edf
from optuna.visualization._edf import _get_edf_info
from optuna.visualization._plotly_imports import _imports as plotly_imports
from optuna.visualization.matplotlib import plot_edf as plt_plot_edf
from optuna.visualization.matplotlib._matplotlib_imports import _imports as plt_imports


if plotly_imports.is_successful():
    from optuna.visualization._plotly_imports import go

if plt_imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


NUM_SAMPLES_X_AXIS = 100

parametrized_plot_contour = pytest.mark.parametrize("plot_edf", [plotly_plot_edf, plt_plot_edf])


def save_static_image(figure: Union[go.Figure, Axes, np.ndarray]) -> None:
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())


def _assert_target_label(
    figure: Union[go.Figure, Axes, np.ndarray], expected: str = "Objective Value"
) -> None:
    if isinstance(figure, go.Figure):
        assert figure.layout.xaxis.title.text == expected
    elif isinstance(figure, Axes):
        assert figure.xaxis.label.get_text() == expected


@parametrized_plot_contour
def test_target_is_none_and_study_is_multi_obj(plot_edf: Callable[..., Any]) -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_edf(study)


@parametrized_plot_contour
def _validate_edf_values(edf_values: np.ndarray) -> None:
    np_values = np.array(edf_values)
    # Confirms that the values are monotonically non-decreasing.
    assert np.all(np.diff(np_values) >= 0.0)

    # Confirms that the values are in [0,1].
    assert np.all((0 <= np_values) & (np_values <= 1))


@parametrized_plot_contour
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_edf_plot_target_label(plot_edf: Callable[..., Any], direction: str) -> None:
    # Test with no studies.
    figure = plot_edf([])
    save_static_image(figure)

    # Test with no trials.
    figure = plot_edf(create_study(direction=direction))
    save_static_image(figure)

    figure = plot_edf([create_study(direction=direction), create_study(direction=direction)])
    save_static_image(figure)

    # Test with a study.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf(study0)
    _assert_target_label(figure)

    # Test with two studies.
    study1 = create_study(direction=direction)
    study1.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf([study0, study1])
    _assert_target_label(figure)

    # Test with a customized target value.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    with pytest.warns(UserWarning):
        figure = plot_edf(study0, target=lambda t: t.params["x"])
    _assert_target_label(figure)

    # Test with a customized target name.
    study0 = create_study(direction=direction)
    study0.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    figure = plot_edf(study0, target_name="Target Name")
    _assert_target_label(figure, expected="Target Name")


def test_empty_edf_info() -> None:

    edf_info = _get_edf_info([])
    assert len(edf_info.lines) == 0
    assert len(edf_info.x_values) == 0

    study = create_study()
    edf_info = _get_edf_info(study)
    assert len(edf_info.lines) == 0
    assert len(edf_info.x_values) == 0

    trial = study.ask()
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    edf_info = _get_edf_info(study)
    assert len(edf_info.lines) == 0
    assert len(edf_info.x_values) == 0


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_removed(value: float) -> None:

    study = prepare_study_with_trials(value_for_first_trial=value)
    edf_info = _get_edf_info(study)

    x_values = edf_info.x_values
    min_objective = 1.0
    max_objective = 2.0
    assert np.min(x_values) == min_objective
    assert np.max(x_values) == max_objective
    assert len(x_values) == NUM_SAMPLES_X_AXIS

    lines = edf_info.lines
    assert len(lines) == 1
    assert lines[0].study_name == study.study_name
    _validate_edf_values(lines[0].y_values)


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_nonfinite_multiobjective(objective: int, value: float) -> None:

    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    edf_info = _get_edf_info(study, target=lambda t: t.values[objective])

    x_values = edf_info.x_values
    min_objective = 1.0
    max_objective = 2.0
    assert np.min(x_values) == min_objective
    assert np.max(x_values) == max_objective
    assert len(x_values) == NUM_SAMPLES_X_AXIS

    lines = edf_info.lines
    assert len(lines) == 1
    assert lines[0].study_name == study.study_name
    _validate_edf_values(lines[0].y_values)


def test_inconsistent_number_of_trial_values() -> None:

    studies: List[Study] = []
    n_studies = 5

    for i in range(n_studies):
        study = prepare_study_with_trials()
        if i % 2 == 0:
            study.add_trial(create_trial(value=1.0))
        studies.append(study)

    edf_info = _get_edf_info(studies)

    x_values = edf_info.x_values
    min_objective = 0.0
    max_objective = 2.0
    assert np.min(x_values) == min_objective
    assert np.max(x_values) == max_objective
    assert len(x_values) == NUM_SAMPLES_X_AXIS

    lines = edf_info.lines
    assert len(lines) == n_studies
    for line, study in zip(lines, studies):
        assert line.study_name == study.study_name
        _validate_edf_values(line.y_values)
