from optuna.logging import get_logger
from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna.visualization._imports import *  # NOQA
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

logger = get_logger(__name__)

if is_available():
    from optuna.visualization._imports import go


def plot_intermediate_values(study):
    # type: (Study) -> None
    """Plot intermediate values of all trials in a study.

    Example:

        The following code snippet shows how to plot intermediate values.

        .. code::

            import optuna

            def objective(trial):
                # Intermediate values are supposed to be reported inside the objective function.
                ...

            study = optuna.create_study()
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_intermediate_values(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.
    """

    _check_plotly_availability()
    figure = _get_intermediate_plot(study)
    figure.show()


def _get_intermediate_plot(study):
    # type: (Study) -> Figure

    layout = go.Layout(
        title='Intermediate Values Plot',
        xaxis={'title': 'Step'},
        yaxis={'title': 'Intermediate Value'},
        showlegend=False
    )

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in study.trials if trial.state in target_state]

    if len(trials) == 0:
        logger.warning('Study instance does not contain trials.')
        return go.Figure(data=[], layout=layout)

    traces = []
    for trial in trials:
        if trial.intermediate_values:
            trace = go.Scatter(
                x=tuple(trial.intermediate_values.keys()),
                y=tuple(trial.intermediate_values.values()),
                mode='lines+markers',
                marker={
                    'maxdisplayed': 10
                },
                name='Trial{}'.format(trial.number)
            )
            traces.append(trace)

    if not traces:
        logger.warning(
            'You need to set up the pruning feature to utilize `plot_intermediate_values()`')
        return go.Figure(data=[], layout=layout)

    figure = go.Figure(data=traces, layout=layout)

    return figure
