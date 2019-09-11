from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

try:
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode
    from IPython.display import display, HTML
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
    # type: (Study) -> List[go.Scatter]

    layout = go.Layout(
        title='Intermediate Values Plot',
        xaxis={'title': 'Step'},
        yaxis={'title': 'Intermediate Value'},
        showlegend=False
    )

    try:
        df = study.trials_dataframe()
        df = df.rename(columns={'number': 'trial_id'})
    except Exception:  # empty study
        return go.Figure(data=[], layout=layout)

    if 'intermediate_values' not in df:
        return go.Figure(data=[], layout=layout)

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    dst_df = df[df['state'].isin(target_state)]
    dst_df = dst_df[dst_df['intermediate_values'].isnull().all(axis=1) == False]
    traces = []
    for __, row in dst_df.iterrows():
        trace = go.Scatter(
            x=row['intermediate_values'].index,
            y=row['intermediate_values'],
            mode='lines+markers',
            marker={
                'maxdisplayed': 10
            },
            name='Trial{}'.format(row['trial_id'][0]))
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
            'For further information, please refer to the installation guide of plotly. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
