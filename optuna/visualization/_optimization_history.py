from typing import Sequence

from optuna._study_direction import StudyDirection
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_optimization_history(study: Study) -> "go.Figure":
    """Plot optimization history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_optimization_history(study)

        .. raw:: html

            <iframe src="../../../_static/plot_optimization_history.html"
             width="100%" height="500px" frameborder="0">
            </iframe>

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()
    return _get_optimization_history_plot(study)


def _get_optimization_history_plot(study: Study) -> "go.Figure":

    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "#Trials"},
        yaxis={"title": "Objective Value"},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return go.Figure(data=[], layout=layout)

    best_values = [float("inf")] if study.direction == StudyDirection.MINIMIZE else [-float("inf")]
    comp = min if study.direction == StudyDirection.MINIMIZE else max
    for trial in trials:
        trial_value = trial.value
        assert trial_value is not None and not isinstance(trial_value, Sequence)  # For mypy
        best_values.append(comp(best_values[-1], trial_value))
    best_values.pop(0)
    traces = [
        go.Scatter(
            x=[t.number for t in trials],
            y=[t.value for t in trials],
            mode="markers",
            name="Objective Value",
        ),
        go.Scatter(x=[t.number for t in trials], y=best_values, name="Best Value"),
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure
