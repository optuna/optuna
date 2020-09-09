import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _is_log_scale

if _imports.is_successful():
    from optuna.visualization._plotly_imports import Contour
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import plotly
    from optuna.visualization._plotly_imports import Scatter

_logger = get_logger(__name__)


def plot_contour(study: Study, params: Optional[List[str]] = None) -> "go.Figure":
    """Plot the parameter relationship as contour plot in a study.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as contour plot.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            study = optuna.create_study()
            study.optimize(objective, n_trials=30)

            optuna.visualization.plot_contour(study, params=['x', 'y'])

        .. raw:: html

            <iframe src="../../_static/plot_contour.html"
                width="100%" height="500px" frameborder="0">
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

    _imports.check()
    return _get_contour_plot(study, params)


def _get_contour_plot(study: Study, params: Optional[List[str]] = None) -> "go.Figure":

    layout = go.Layout(title="Contour Plot")

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    elif len(params) <= 1:
        _logger.warning("The length of params must be greater than 1.")
        return go.Figure(data=[], layout=layout)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(list(set(params)))

    padding_ratio = 0.05
    param_values_range = {}
    for p_name in sorted_params:
        values = [t.params[p_name] for t in trials if p_name in t.params]
        max_value = max(values)
        min_value = min(values)
        if _is_log_scale(trials, p_name):
            padding = (math.log10(max_value) - math.log10(min_value)) * padding_ratio
            min_value = math.pow(10, math.log10(min_value) - padding)
            max_value = math.pow(10, math.log10(max_value) + padding)
        else:
            padding = (max_value - min_value) * padding_ratio
            min_value = min_value - padding
            max_value = max_value + padding
        param_values_range[p_name] = (min_value, max_value)

    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plots = _generate_contour_subplot(
            trials, x_param, y_param, study.direction, param_values_range
        )
        figure = go.Figure(data=sub_plots, layout=layout)
        figure.update_xaxes(title_text=x_param, range=param_values_range[x_param])
        figure.update_yaxes(title_text=y_param, range=param_values_range[y_param])
        if _is_log_scale(trials, x_param):
            log_range = [math.log10(p) for p in param_values_range[x_param]]
            figure.update_xaxes(range=log_range, type="log")
        if _is_log_scale(trials, y_param):
            log_range = [math.log10(p) for p in param_values_range[y_param]]
            figure.update_yaxes(range=log_range, type="log")
    else:
        figure = make_subplots(
            rows=len(sorted_params), cols=len(sorted_params), shared_xaxes=True, shared_yaxes=True
        )
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once
        for x_i, x_param in enumerate(sorted_params):
            for y_i, y_param in enumerate(sorted_params):
                if x_param == y_param:
                    figure.add_trace(go.Scatter(), row=y_i + 1, col=x_i + 1)
                else:
                    sub_plots = _generate_contour_subplot(
                        trials, x_param, y_param, study.direction, param_values_range
                    )
                    contour = sub_plots[0]
                    scatter = sub_plots[1]
                    contour.update(showscale=showscale)  # showscale's default is True
                    if showscale:
                        showscale = False
                    figure.add_trace(contour, row=y_i + 1, col=x_i + 1)
                    figure.add_trace(scatter, row=y_i + 1, col=x_i + 1)
                figure.update_xaxes(range=param_values_range[x_param], row=y_i + 1, col=x_i + 1)
                figure.update_yaxes(range=param_values_range[y_param], row=y_i + 1, col=x_i + 1)
                if _is_log_scale(trials, x_param):
                    log_range = [math.log10(p) for p in param_values_range[x_param]]
                    figure.update_xaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)
                if _is_log_scale(trials, y_param):
                    log_range = [math.log10(p) for p in param_values_range[y_param]]
                    figure.update_yaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)
                if x_i == 0:
                    figure.update_yaxes(title_text=y_param, row=y_i + 1, col=x_i + 1)
                if y_i == len(sorted_params) - 1:
                    figure.update_xaxes(title_text=x_param, row=y_i + 1, col=x_i + 1)

    return figure


def _generate_contour_subplot(
    trials: List[FrozenTrial],
    x_param: str,
    y_param: str,
    direction: StudyDirection,
    param_values_range: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple["Contour", "Scatter"]:

    if param_values_range is None:
        param_values_range = {}

    x_indices = sorted(list({t.params[x_param] for t in trials if x_param in t.params}))
    y_indices = sorted(list({t.params[y_param] for t in trials if y_param in t.params}))
    if len(x_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(x_param))
        return go.Contour(), go.Scatter()
    if len(y_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(y_param))
        return go.Contour(), go.Scatter()

    # Padding to the plot.
    x_range = param_values_range[x_param]
    x_indices = [x_range[0]] + x_indices + [x_range[1]]
    y_range = param_values_range[y_param]
    y_indices = [y_range[0]] + y_indices + [y_range[1]]

    z = [[float("nan") for _ in range(len(x_indices))] for _ in range(len(y_indices))]

    x_values = []
    y_values = []
    for trial in trials:
        if x_param not in trial.params or y_param not in trial.params:
            continue
        x_values.append(trial.params[x_param])
        y_values.append(trial.params[y_param])
        x_i = x_indices.index(trial.params[x_param])
        y_i = y_indices.index(trial.params[y_param])
        if isinstance(trial.value, int):
            value = float(trial.value)
        elif isinstance(trial.value, float):
            value = trial.value
        else:
            raise ValueError(
                "Trial{} has COMPLETE state, but its value is non-numeric.".format(trial.number)
            )
        z[y_i][x_i] = value

    # TODO(Yanase): Use reversescale argument to reverse colorscale if Plotly's bug is fixed.
    # If contours_coloring='heatmap' is specified, reversesecale argument of go.Contour does not
    # work correctly. See https://github.com/pfnet/optuna/issues/606.
    colorscale = plotly.colors.PLOTLY_SCALES["Blues"]
    if direction == StudyDirection.MINIMIZE:
        colorscale = [[1 - t[0], t[1]] for t in colorscale]
        colorscale.reverse()

    contour = go.Contour(
        x=x_indices,
        y=y_indices,
        z=z,
        colorbar={"title": "Objective Value"},
        colorscale=colorscale,
        connectgaps=True,
        contours_coloring="heatmap",
        hoverinfo="none",
        line_smoothing=1.3,
    )

    scatter = go.Scatter(
        x=x_values,
        y=y_values,
        marker={"line": {"width": 0.5, "color": "Grey"}, "color": "black"},
        mode="markers",
        showlegend=False,
    )

    return (contour, scatter)
