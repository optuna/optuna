from optuna.logging import get_logger
from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna import type_checking

logger = get_logger(__name__)

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

try:
    import plotly.graph_objs as go
    from plotly.graph_objs._figure import Figure  # NOQA
    from plotly.offline import init_notebook_mode
    _available = True
except ImportError as e:
    _import_error = e
    # Visualization features are disabled because plotly is not available.
    _available = False


def plot_intermediate_values(study):
    # type: (Study) -> None
    """Inside Jupyter notebook, plot intermediate values of all trials in a study.

    Example:

        The following code snippet shows how to plot intermediate values inside Jupyter Notebook.

        .. code::

            import optuna

            def objective(trial):
                # Intermediate values are supposed to be reported inside the objective function.
                ...

            study = optuna.create_study()
            study.optimize(n_trials=100)

            optuna.visualization.plot_intermediate_values(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.
    """

    _check_plotly_availability()
    init_notebook_mode(connected=True)
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

    trials = study.trials

    if len(trials) == 0:
        logger.warning('Study instance does not contain trials.')
        return go.Figure(data=[], layout=layout)
    if hasattr(trials[0], 'intermediate_values') is False:
        logger.warning(
            'You need to set up the pruning feature to utilize plot_intermediate_values()')
        return go.Figure(data=[], layout=layout)

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in trials if trial.state in target_state]
    traces = []
    for trial in trials:
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

    figure = go.Figure(data=traces, layout=layout)

    return figure


def _check_plotly_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Plotly is not available. Please install plotly to use this feature. '
            'Plotly can be installed by executing `$ pip install plotly`. '
            'For further information, please refer to the installation guide of plotly. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')

    from distutils.version import StrictVersion
    from plotly import __version__ as plotly_version
    if StrictVersion(plotly_version) < StrictVersion('4.0.0'):
        raise ImportError(
            'Your version of Plotly is ' + plotly_version + ' . '
            'Please install plotly version 4.0.0 or higher. '
            'Plotly can be installed by executing `$ pip install -U plotly>=4.0.0`. '
            'For further information, please refer to the installation guide of plotly. ')
