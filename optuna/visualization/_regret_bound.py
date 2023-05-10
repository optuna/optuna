from __future__ import annotations

from typing import NamedTuple

import tqdm

import optuna
from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study.study import Study
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


PADDING_RATIO_Y = 0.05
OPACITY = 0.25


class _RegretBoundInfo(NamedTuple):
    trial_numbers: list[int]
    regret_bounds: list[float]
    errors: list[float] | None


@experimental_func("3.2.0")
def plot_regret_bound(
    study: Study,
    plot_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> "go.Figure":
    _imports.check()

    info = _get_regret_bound_info(study, plot_error, improvement_evaluator, error_evaluator)
    return _get_regret_bound_plot(info, min_n_trials)


def _get_regret_bound_info(
    study: Study,
    get_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
) -> _RegretBoundInfo:
    if improvement_evaluator is None:
        improvement_evaluator = RegretBoundEvaluator()
    if error_evaluator is None:
        error_evaluator = CrossValidationErrorEvaluator()

    trial_numbers = []
    regret_bounds = []
    errors = []

    for i, trial in enumerate(tqdm.tqdm(study.trials)):
        trial_numbers.append(trial.number)
        trials = study.trials[: i + 1]
        trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        regret_bound = improvement_evaluator.evaluate(
            trials=trials, study_direction=study.direction
        )
        regret_bounds.append(regret_bound)

        if get_error:
            error = error_evaluator.evaluate(trials=trials, study_direction=study.direction)
            errors.append(error)

    if len(errors) == 0:
        return _RegretBoundInfo(
            trial_numbers=trial_numbers, regret_bounds=regret_bounds, errors=None
        )
    else:
        return _RegretBoundInfo(
            trial_numbers=trial_numbers, regret_bounds=regret_bounds, errors=errors
        )


def _get_regret_bound_scatter(
    trial_numbers: list[int],
    regret_bounds: list[float],
    opacity: float = 1.0,
    showlegend: bool = True,
) -> "go.Scatter":
    plotly_blue_with_opacity = f"rgba(99, 110, 250, {opacity})"
    return go.Scatter(
        x=trial_numbers,
        y=regret_bounds,
        mode="markers+lines",
        marker=dict(color=plotly_blue_with_opacity),
        line=dict(color=plotly_blue_with_opacity),
        name="Regret Bound",
        showlegend=showlegend,
        legendgroup="regret_bound",
    )


def _get_error_scatter(
    trial_numbers: list[int],
    errors: list[float] | None,
) -> "go.Scatter":
    if errors is None:
        return go.Scatter()

    plotly_red = "rgb(239, 85, 59)"
    return go.Scatter(
        x=trial_numbers,
        y=errors,
        mode="markers+lines",
        name="Error",
        marker=dict(color=plotly_red),
        line=dict(color=plotly_red),
    )


def _get_y_range(info: _RegretBoundInfo, min_n_trials: int) -> tuple[float, float]:
    min_value = min(info.regret_bounds)
    if info.errors is not None:
        min_value = min(min_value, min(info.errors))

    # Determine the display range based on trials after min_n_trials.
    if len(info.trial_numbers) > min_n_trials:
        max_value = max(info.regret_bounds[min_n_trials:])
    # If there are no trials after min_trials, determine the display range based on all trials.
    else:
        max_value = max(info.regret_bounds)

    if info.errors is not None:
        max_value = max(max_value, max(info.errors))

    padding = (max_value - min_value) * PADDING_RATIO_Y
    return (min_value - padding, max_value + padding)


def _get_regret_bound_plot(info: _RegretBoundInfo, min_n_trials: int) -> "go.Figure":
    n_trials = len(info.trial_numbers)

    fig = go.Figure(
        layout=go.Layout(
            title="Regret Bound Plot", xaxis=dict(title="Trial"), yaxis=dict(title="Regret Bound")
        )
    )
    if n_trials == 0:
        _logger.warning("There are no complete trials.")
        return fig

    fig.add_trace(
        _get_regret_bound_scatter(
            info.trial_numbers[: min_n_trials + 1],
            info.regret_bounds[: min_n_trials + 1],
            OPACITY,  # Plot line for values below min_n_trials by light color.
            n_trials <= min_n_trials,  # Avoid showing legend twice.
        )
    )

    if n_trials > min_n_trials:
        fig.add_trace(
            _get_regret_bound_scatter(
                info.trial_numbers[min_n_trials:],
                info.regret_bounds[min_n_trials:],
            )
        )

    fig.add_trace(_get_error_scatter(info.trial_numbers, info.errors))

    fig.update_yaxes(range=_get_y_range(info, min_n_trials))
    return fig
