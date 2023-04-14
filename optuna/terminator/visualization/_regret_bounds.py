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
    max_y_value = max(info.regret_bounds)
    if info.errors is not None:
        max_y_value = max(max_y_value, max(info.errors))

    traces = []
    traces.append(
        go.Scatter(x=info.trial_numbers, y=info.regret_bounds, mode="markers", showlegend=False)
    )
    traces.append(
        go.Scatter(x=info.trial_numbers, y=info.regret_bounds, mode="lines", name="Regret Bound")
    )
    if info.errors is not None:
        traces.append(
            go.Scatter(x=info.trial_numbers, y=info.errors, mode="markers", showlegend=False)
        )
        traces.append(go.Scatter(x=info.trial_numbers, y=info.errors, mode="lines", name="Error"))

    fig = go.Figure(data=traces)

    x1_filled = min(info.trial_numbers[-1], min_n_trials)
    min_trials_area = go.layout.Shape(
        type="rect",
        x0=0,
        x1=x1_filled,
        y0=0,
        y1=max_y_value,
        fillcolor="gray",
        opacity=0.2,
        layer="below",
        line=dict(width=0),
    )

    fig.update_layout(
        title="Regret Bounds Plot",
        xaxis=dict(title="Trial"),
        yaxis=dict(title="Regret Bound"),
        shapes=(min_trials_area,),
    )
    fig.add_annotation(
        go.layout.Annotation(
            x=x1_filled,
            y=max_y_value * 1.1,
            text="min_n_trials",
            showarrow=False,
            font=dict(size=14),
        )
    )
    return fig
