from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

import numpy as np
import pytest

import optuna
from optuna import Study
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.visualization import plot_edf as plotly_plot_edf
from optuna.visualization._edf import _EDFInfo
from optuna.visualization._edf import _get_edf_info
from optuna.visualization._edf import NUM_SAMPLES_X_AXIS
from optuna.visualization._plotly_imports import _imports as plotly_imports
from optuna.visualization.matplotlib import plot_edf as plt_plot_edf
from optuna.visualization.matplotlib._matplotlib_imports import _imports as plt_imports


if plotly_imports.is_successful():
    from optuna.visualization._plotly_imports import go

if plt_imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


parametrized_plot_edf = pytest.mark.parametrize("plot_edf", [plotly_plot_edf, plt_plot_edf])


def save_static_image(figure: go.Figure | Axes | np.ndarray) -> None:
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()


@parametrized_plot_edf
def _validate_edf_values(edf_values: np.ndarray) -> None:
    np_values = np.array(edf_values)
    # Confirms that the values are monotonically non-decreasing.
    assert np.all(np.diff(np_values) >= 0.0)

    # Confirms that the values are in [0,1].
    assert np.all((0 <= np_values) & (np_values <= 1))


@parametrized_plot_edf
def test_target_is_none_and_study_is_multi_obj(plot_edf: Callable[..., Any]) -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_edf(study)


@parametrized_plot_edf
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_edf_plot_no_trials(plot_edf: Callable[..., Any], direction: str) -> None:
    figure = plot_edf(create_study(direction=direction))
    save_static_image(figure)


@parametrized_plot_edf
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("num_studies", [0, 1, 2])
def test_edf_plot_no_trials_studies(
    plot_edf: Callable[..., Any], direction: str, num_studies: int
) -> None:
    studies = [create_study(direction=direction) for _ in range(num_studies)]
    figure = plot_edf(studies)
    save_static_image(figure)


@parametrized_plot_edf
@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("num_studies", [0, 1, 2])
def test_plot_edf_with_multiple_studies(
    plot_edf: Callable[..., Any], direction: str, num_studies: int
) -> None:
    studies = []
    for _ in range(num_studies):
        study = create_study(direction=direction)
        study.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
        studies.append(study)
    figure = plot_edf(studies)
    save_static_image(figure)


@parametrized_plot_edf
def test_plot_edf_with_target(plot_edf: Callable[..., Any]) -> None:
    study = create_study()
    study.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    with pytest.warns(UserWarning):
        figure = plot_edf(study, target=lambda t: t.params["x"])
    save_static_image(figure)


@parametrized_plot_edf
@pytest.mark.parametrize("target_name", [None, "Target Name"])
def test_plot_edf_with_target_name(plot_edf: Callable[..., Any], target_name: str | None) -> None:
    study = create_study()
    study.optimize(lambda t: t.suggest_float("x", 0, 5), n_trials=10)
    if target_name is None:
        figure = plot_edf(study)
    else:
        figure = plot_edf(study, target_name=target_name)

    expected = target_name if target_name is not None else "Objective Value"
    if isinstance(figure, go.Figure):
        assert figure.layout.xaxis.title.text == expected
    elif isinstance(figure, Axes):
        assert figure.xaxis.label.get_text() == expected

    save_static_image(figure)


def test_empty_edf_info() -> None:
    def _assert_empty(info: _EDFInfo) -> None:
        assert info.lines == []
        np.testing.assert_array_equal(info.x_values, np.array([]))

    edf_info = _get_edf_info([])
    _assert_empty(edf_info)

    study = create_study()
    edf_info = _get_edf_info(study)
    _assert_empty(edf_info)

    trial = study.ask()
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    edf_info = _get_edf_info(study)
    _assert_empty(edf_info)


@pytest.mark.parametrize("n_studies", [1, 2, 3])
@pytest.mark.parametrize("target", [None, lambda t: t.params["x"]])
def test_get_edf_info(n_studies: int, target: Callable[[optuna.trial.FrozenTrial], float]) -> None:
    studies = []
    n_trials = 3
    max_target = 5
    for i in range(n_studies):
        study = create_study(study_name=str(i))
        study.optimize(lambda t: t.suggest_float("x", 0, max_target), n_trials=n_trials)
        studies.append(study)

    info = _get_edf_info(studies, target=target, target_name="Target Name")
    assert info.x_values.shape == (NUM_SAMPLES_X_AXIS,)
    assert all([0 <= x <= max_target for x in info.x_values])
    assert len(info.lines) == n_studies
    for i, line in enumerate(info.lines):
        assert str(i) == line.study_name
        assert line.y_values.shape == (NUM_SAMPLES_X_AXIS,)
        _validate_edf_values(line.y_values)


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_nonfinite_removed(value: float) -> None:
    study = prepare_study_with_trials(value_for_first_trial=value)
    edf_info = _get_edf_info(study)
    assert all(np.isfinite(edf_info.x_values))


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_nonfinite_multiobjective(objective: int, value: float) -> None:
    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    edf_info = _get_edf_info(
        study, target=lambda t: t.values[objective], target_name="Target Name"
    )
    assert all(np.isfinite(edf_info.x_values))


def test_inconsistent_number_of_trial_values() -> None:
    studies: list[Study] = []
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
