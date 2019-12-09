from optuna.logging import get_logger
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna import type_checking
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

if type_checking.TYPE_CHECKING:
    from optuna.visualization.plotly_imports import Figure  # NOQA

if is_available():
    from optuna.visualization.plotly_imports import go

logger = get_logger(__name__)


def plot_optimization_history(study):
    # type: (Study) -> None
    """Plot optimization history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. code::

            import optuna

            def objective(trial):
                ...

            study = optuna.create_study()
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_optimization_history(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
    """

    _check_plotly_availability()
    figure = _get_optimization_history_plot(study)
    figure.show()


def _get_optimization_history_plot(study):
    # type: (Study) -> Figure

    layout = go.Layout(
        title='Optimization History Plot',
        xaxis={'title': '#Trials'},
        yaxis={'title': 'Objective Value'},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning('Study instance does not contain trials.')
        return go.Figure(data=[], layout=layout)

    best_values = [float('inf')] if study.direction == StudyDirection.MINIMIZE else [-float('inf')]
    comp = min if study.direction == StudyDirection.MINIMIZE else max
    for trial in trials:
        trial_value = trial.value
        assert trial_value is not None  # For mypy
        best_values.append(comp(best_values[-1], trial_value))
    best_values.pop(0)
    traces = [
        go.Scatter(x=[t.number for t in trials], y=[t.value for t in trials],
                   mode='markers', name='Objective Value'),
        go.Scatter(x=[t.number for t in trials], y=best_values, name='Best Value')
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure
