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
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator


class _RegretBoundsInfo(NamedTuple):
    trial_numbers: list[int]
    regret_bounds: list[float]
    errors: list[float] | None


@experimental_func("3.2.0")
def plot_regret_bouds(
    study: Study,
    improvement_evaluator: BaseImprovementEvaluator | None = None,
    error_evaluator: BaseErrorEvaluator | None = None,
) -> "go.Figure":
    info = _get_regret_bounds_info(study, improvement_evaluator, error_evaluator)
    return _get_regret_bounds_plot(info)


def _get_regret_bounds_info(
    study: Study,
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

        error = error_evaluator.evaluate(trials=trials, study_direction=study.direction)
        errors.append(error)

    return _RegretBoundsInfo(
        trial_numbers=trial_numbers, regret_bounds=regret_bounds, errors=errors
    )


def _get_regret_bounds_plot(info: _RegretBoundsInfo) -> "go.Figure":
    pass
