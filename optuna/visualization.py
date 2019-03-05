from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from typing import List  # NOQA

try:
    import plotly.graph_objs as go
    from plotly.offline import init_notebook_mode
    from plotly.offline import iplot
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
    layout = go.Layout(showlegend=False)
    iplot(go.Figure(_get_intermediate_values_data(study), layout))


def _get_intermediate_values_data(study):
    # type: (Study) -> List[go.Scatter]

    target_status = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [t for t in study.trials if t.state in target_status]

    intermediate_values = [t.intermediate_values for t in trials]
    intermediate_values = [iv for iv in intermediate_values if len(iv) != 0]

    data = [go.Scatter(x=list(iv.keys()), y=list(iv.values())) for iv in intermediate_values]

    return data


def _check_plotly_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Plotly is not available. Please install plotly to use this feature. '
            'Plotly can be installed by executing `$ pip install plotly`. '
            'For further information, please refer to the installation guide of plotly. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
