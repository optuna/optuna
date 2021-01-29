from typing import Callable
from typing import Optional

from optuna._study_direction import StudyDirection
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_optimization_history(
    study: Study,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot optimization history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_optimization_history(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_optimization_history_plot(study, target, target_name)


def _get_optimization_history_plot(
    study: Study,
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
) -> "go.Figure":

    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "#Trials"},
        yaxis={"title": target_name},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return go.Figure(data=[], layout=layout)

    if target is None:
        if study.direction == StudyDirection.MINIMIZE:
            best_values = [float("inf")]
        else:
            best_values = [-float("inf")]
        comp = min if study.direction == StudyDirection.MINIMIZE else max
        for trial in trials:
            trial_value = trial.value
            assert trial_value is not None  # For mypy
            best_values.append(comp(best_values[-1], trial_value))
        best_values.pop(0)
        traces = [
            go.Scatter(
                x=[t.number for t in trials],
                y=[t.value for t in trials],
                mode="markers",
                name=target_name,
            ),
            go.Scatter(x=[t.number for t in trials], y=best_values, name="Best Value"),
        ]
    else:
        traces = [
            go.Scatter(
                x=[t.number for t in trials],
                y=[target(t) for t in trials],
                mode="markers",
                name=target_name,
            ),
        ]

    figure = go.Figure(data=traces, layout=layout)

    return figure
