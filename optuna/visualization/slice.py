from optuna.logging import get_logger
from optuna.trial import TrialState
from optuna import type_checking
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import _is_log_scale
from optuna.visualization.utils import is_available

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA
    from optuna.visualization.plotly_imports import Scatter  # NOQA

if is_available():
    from optuna.visualization.plotly_imports import go
    from optuna.visualization.plotly_imports import make_subplots

logger = get_logger(__name__)


def plot_slice(study, params=None):
    # type: (Study, Optional[List[str]]) -> go.Figure
    """Plot the parameter relationship as slice plot in a study.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as slice plot.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_slice(study, params=['x', 'y'])

        .. raw:: html

            <iframe src="../_static/plot_slice.html" width="100%" height="500px" frameborder="0">
            </iframe>

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _check_plotly_availability()
    return _get_slice_plot(study, params)


def _get_slice_plot(study, params=None):
    # type: (Study, Optional[List[str]]) -> go.Figure

    layout = go.Layout(title="Slice Plot",)

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(list(set(params)))

    n_params = len(sorted_params)

    if n_params == 1:
        figure = go.Figure(
            data=[_generate_slice_subplot(study, trials, sorted_params[0])], layout=layout
        )
        figure.update_xaxes(title_text=sorted_params[0])
        figure.update_yaxes(title_text="Objective Value")
        if _is_log_scale(trials, sorted_params[0]):
            figure.update_xaxes(type="log")
    else:
        figure = make_subplots(rows=1, cols=len(sorted_params), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for i, param in enumerate(sorted_params):
            trace = _generate_slice_subplot(study, trials, param)
            trace.update(marker={"showscale": showscale})  # showscale's default is True.
            if showscale:
                showscale = False
            figure.add_trace(trace, row=1, col=i + 1)
            figure.update_xaxes(title_text=param, row=1, col=i + 1)
            if i == 0:
                figure.update_yaxes(title_text="Objective Value", row=1, col=1)
            if _is_log_scale(trials, param):
                figure.update_xaxes(type="log", row=1, col=i + 1)
        if n_params > 3:
            # Ensure that each subplot has a minimum width without relying on autusizing.
            figure.update_layout(width=300 * n_params)

    return figure


def _generate_slice_subplot(study, trials, param):
    # type: (Study, List[FrozenTrial], str) -> Scatter

    return go.Scatter(
        x=[t.params[param] for t in trials if param in t.params],
        y=[t.value for t in trials if param in t.params],
        mode="markers",
        marker={
            "line": {"width": 0.5, "color": "Grey",},
            "color": [t.number for t in trials if param in t.params],
            "colorscale": "Blues",
            "colorbar": {
                "title": "#Trials",
                "x": 1.0,  # Offset the colorbar position with a fixed width `xpad`.
                "xpad": 40,
            },
        },
        showlegend=False,
    )
