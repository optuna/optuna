from __future__ import annotations

from typing import NamedTuple

import plotly.graph_objects as go
import tqdm

import optuna
from optuna._experimental import experimental_func
from optuna.study.study import Study
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator


PADDING_RATIO = 0.05


class _RegretBoundsInfo(NamedTuple):
    trial_numbers: list[int]
    regret_bounds: list[float]
    errors: list[float] | None


@experimental_func("3.2.0")
def plot_regret_bounds(
    study: Study,
    plot_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
    min_n_trials: int = DEFAULT_MIN_N_TRIALS,
) -> "go.Figure":
    info = _get_regret_bounds_info(study, plot_error, improvement_evaluator, error_evaluator)
    return _get_regret_bounds_plot(info, min_n_trials)


def _get_regret_bounds_info(
    study: Study,
    get_error: bool = False,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
) -> _RegretBoundsInfo:
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
        return _RegretBoundsInfo(
            trial_numbers=trial_numbers, regret_bounds=regret_bounds, errors=None
        )
    else:
        return _RegretBoundsInfo(
            trial_numbers=trial_numbers, regret_bounds=regret_bounds, errors=errors
        )


def _get_regret_bounds_plot(info: _RegretBoundsInfo, min_n_trials: int) -> "go.Figure":
    n_trials = len(info.trial_numbers)

    fig = go.Figure()
    plotly_blue_with_opacity = "rgba(99, 110, 250, 0.5)"
    fig.add_trace(
        go.Scatter(
            x=info.trial_numbers[: min_n_trials + 1],
            y=info.regret_bounds[: min_n_trials + 1],
            mode="markers+lines",
            name="Regret Bound",
            showlegend=n_trials <= min_n_trials,
            legendgroup="regret_bounds",
            marker=dict(color=plotly_blue_with_opacity),
            line=dict(color=plotly_blue_with_opacity),
        )
    )
    if n_trials > min_n_trials:
        plotly_blue = "rgb(99, 110, 250)"
        fig.add_trace(
            go.Scatter(
                x=info.trial_numbers[min_n_trials:],
                y=info.regret_bounds[min_n_trials:],
                mode="markers+lines",
                name="Regret Bound",
                showlegend=True,
                legendgroup="regret_bounds",
                marker=dict(color=plotly_blue),
                line=dict(color=plotly_blue),
            )
        )
    if info.errors is not None:
        plotly_red = "rgb(239, 85, 59)"
        fig.add_trace(
            go.Scatter(
                x=info.trial_numbers,
                y=info.errors,
                mode="markers+lines",
                name="Error",
                marker=dict(color=plotly_red),
                line=dict(color=plotly_red),
            )
        )

    min_value = min(info.regret_bounds)
    if info.errors is not None:
        min_value = min(min_value, min(info.errors))

    if n_trials > min_n_trials:
        max_value = max(info.regret_bounds[min_n_trials:])
    else:
        max_value = max(info.regret_bounds)
    if info.errors is not None:
        max_value = max(max_value, max(info.errors))

    padding = (max_value - min_value) * PADDING_RATIO

    fig.update_layout(
        title="Regret Bounds Plot",
        xaxis=dict(title="Trial"),
        yaxis=dict(title="Regret Bound", range=(min_value - padding, max_value + padding)),
    )

    return fig
