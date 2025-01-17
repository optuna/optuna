from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

import pytest

from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.study import Study
from optuna.terminator import BaseErrorEvaluator
from optuna.terminator import BaseImprovementEvaluator
from optuna.terminator import CrossValidationErrorEvaluator
from optuna.terminator import RegretBoundEvaluator
from optuna.terminator import report_cross_validation_scores
from optuna.terminator import StaticErrorEvaluator
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import TrialState
from optuna.visualization import plot_terminator_improvement as plotly_plot_terminator_improvement
from optuna.visualization._terminator_improvement import _get_improvement_info
from optuna.visualization._terminator_improvement import _get_y_range
from optuna.visualization._terminator_improvement import _ImprovementInfo


parametrize_plot_terminator_improvement = pytest.mark.parametrize(
    "plot_terminator_improvement", [plotly_plot_terminator_improvement]
)


def _create_study_with_failed_trial() -> Study:
    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    return study


def _prepare_study_with_cross_validation_scores() -> Study:
    study = create_study()
    for _ in range(3):
        trial = study.ask({"x": FloatDistribution(0, 1)})
        report_cross_validation_scores(trial, [1.0, 2.0])
        study.tell(trial, 0)
    return study


def test_study_is_multi_objective() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_improvement_info(study=study)


@parametrize_plot_terminator_improvement
@pytest.mark.parametrize(
    "specific_create_study, plot_error",
    [
        (create_study, False),
        (_create_study_with_failed_trial, False),
        (prepare_study_with_trials, False),
        (_prepare_study_with_cross_validation_scores, False),
        (_prepare_study_with_cross_validation_scores, True),
    ],
)
def test_plot_terminator_improvement(
    plot_terminator_improvement: Callable[..., Any],
    specific_create_study: Callable[[], Study],
    plot_error: bool,
) -> None:
    study = specific_create_study()
    figure = plot_terminator_improvement(study, plot_error)
    figure.write_image(BytesIO())


@pytest.mark.parametrize(
    "specific_create_study",
    [create_study, _create_study_with_failed_trial],
)
@pytest.mark.parametrize("plot_error", [False, True])
def test_get_terminator_improvement_info_empty(
    specific_create_study: Callable[[], Study], plot_error: bool
) -> None:
    study = specific_create_study()
    info = _get_improvement_info(study, plot_error)
    assert info == _ImprovementInfo(trial_numbers=[], improvements=[], errors=None)


@pytest.mark.parametrize("get_error", [False, True])
@pytest.mark.parametrize(
    "improvement_evaluator_class", [lambda: RegretBoundEvaluator(), lambda: None]
)
@pytest.mark.parametrize(
    "error_evaluator_class",
    [
        lambda: CrossValidationErrorEvaluator(),
        lambda: StaticErrorEvaluator(0),
        lambda: None,
    ],
)
def test_get_improvement_info(
    get_error: bool,
    improvement_evaluator_class: Callable[[], BaseImprovementEvaluator | None],
    error_evaluator_class: Callable[[], BaseErrorEvaluator | None],
) -> None:
    study = _prepare_study_with_cross_validation_scores()

    info = _get_improvement_info(
        study, get_error, improvement_evaluator_class(), error_evaluator_class()
    )
    assert info.trial_numbers == [0, 1, 2]
    assert len(info.improvements) == 3
    if get_error:
        assert info.errors is not None
        assert len(info.errors) == 3
        assert info.errors[0] == info.errors[1] == info.errors[2]
    else:
        assert info.errors is None


def test_get_improvement_info_started_with_failed_trials() -> None:
    study = create_study()
    for _ in range(3):
        study.add_trial(create_trial(state=TrialState.FAIL))
    trial = study.ask({"x": FloatDistribution(0, 1)})
    study.tell(trial, 0)

    info = _get_improvement_info(study)
    assert info.trial_numbers == [3]
    assert len(info.improvements) == 1
    assert info.errors is None


@pytest.mark.parametrize(
    "info",
    [
        _ImprovementInfo(trial_numbers=[0], improvements=[0], errors=None),
        _ImprovementInfo(trial_numbers=[0], improvements=[0], errors=[0]),
        _ImprovementInfo(trial_numbers=[0, 1], improvements=[0, 1], errors=[0, 1]),
    ],
)
@pytest.mark.parametrize("min_n_trials", [1, 2])
def test_get_y_range(info: _ImprovementInfo, min_n_trials: int) -> None:
    y_range = _get_y_range(info, min_n_trials)
    assert len(y_range) == 2
    assert y_range[0] <= y_range[1]
