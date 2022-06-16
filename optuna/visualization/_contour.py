import math
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _get_param_values
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical
from optuna.visualization._utils import _is_reverse_scale


if _imports.is_successful():
    from optuna.visualization._plotly_imports import Contour
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import Scatter
    from optuna.visualization._utils import COLOR_SCALE

_logger = get_logger(__name__)


class _SubContourInfo(NamedTuple):
    x_indices: List[Union[str, int, float]]
    y_indices: List[Union[str, int, float]]
    x_values: List[Union[str, float]]
    y_values: List[Union[str, float]]
    z_values: List[List[float]]


class _ContourInfo(NamedTuple):
    sorted_params: List[str]
    param_values_range: Dict[str, Tuple[float, float]]
    param_is_log: Dict[str, bool]
    param_is_cat: Dict[str, bool]
    sub_plot_infos: List[List[_SubContourInfo]]


def plot_contour(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the parameter relationship as contour plot in a study.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as contour plot.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=30)

            fig = optuna.visualization.plot_contour(study, params=["x", "y"])
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the color bar.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_contour_plot(study, params, target, target_name)


def _get_contour_plot(
    study: Study,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":

    layout = go.Layout(title="Contour Plot")

    info = _get_contour_info(study, params, target)
    sorted_params = info.sorted_params
    param_values_range = info.param_values_range
    param_is_log = info.param_is_log
    param_is_cat = info.param_is_cat
    sub_plot_infos = info.sub_plot_infos

    if len(sorted_params) <= 1:
        return go.Figure(data=[], layout=layout)

    reverse_scale = _is_reverse_scale(study, target)

    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plots = _generate_contour_subplot(sub_plot_infos[0][0], reverse_scale, target_name)
        figure = go.Figure(data=sub_plots, layout=layout)
        figure.update_xaxes(title_text=x_param, range=param_values_range[x_param])
        figure.update_yaxes(title_text=y_param, range=param_values_range[y_param])

        if param_is_cat[x_param]:
            figure.update_xaxes(type="category")
        if param_is_cat[y_param]:
            figure.update_yaxes(type="category")

        if param_is_log[x_param]:
            log_range = [math.log10(p) for p in param_values_range[x_param]]
            figure.update_xaxes(range=log_range, type="log")
        if param_is_log[y_param]:
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
                        sub_plot_infos[y_i][x_i], reverse_scale, target_name
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

                if param_is_cat[x_param]:
                    figure.update_xaxes(type="category", row=y_i + 1, col=x_i + 1)
                if param_is_cat[y_param]:
                    figure.update_yaxes(type="category", row=y_i + 1, col=x_i + 1)

                if param_is_log[x_param]:
                    log_range = [math.log10(p) for p in param_values_range[x_param]]
                    figure.update_xaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)
                if param_is_log[y_param]:
                    log_range = [math.log10(p) for p in param_values_range[y_param]]
                    figure.update_yaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)

                if x_i == 0:
                    figure.update_yaxes(title_text=y_param, row=y_i + 1, col=x_i + 1)
                if y_i == len(sorted_params) - 1:
                    figure.update_xaxes(title_text=x_param, row=y_i + 1, col=x_i + 1)

    return figure


def _generate_contour_subplot(
    info: _SubContourInfo,
    reverse_scale: bool,
    target_name: str = "Objective Value",
) -> Tuple["Contour", "Scatter"]:

    x_indices = info.x_indices
    y_indices = info.y_indices
    x_values = info.x_values
    y_values = info.y_values
    z_values = info.z_values

    if len(x_indices) < 2:
        return go.Contour(), go.Scatter()
    if len(y_indices) < 2:
        return go.Contour(), go.Scatter()

    contour = go.Contour(
        x=x_indices,
        y=y_indices,
        z=z_values,
        colorbar={"title": target_name},
        colorscale=COLOR_SCALE,
        connectgaps=True,
        contours_coloring="heatmap",
        hoverinfo="none",
        line_smoothing=1.3,
        reversescale=reverse_scale,
    )

    scatter = go.Scatter(
        x=x_values,
        y=y_values,
        marker={"line": {"width": 2.0, "color": "Grey"}, "color": "black"},
        mode="markers",
        showlegend=False,
    )

    return contour, scatter


def _get_contour_info(
    study: Study,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
) -> _ContourInfo:

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        sorted_params = []
    elif params is None:
        sorted_params = sorted(all_params)
    elif len(params) <= 1:
        _logger.warning("The length of params must be greater than 1.")
        sorted_params = list(params)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(set(params))

    padding_ratio = 0.05
    param_values_range = {}
    param_is_log = {}
    param_is_cat = {}
    for p_name in sorted_params:
        values = _get_param_values(trials, p_name)

        min_value = min(values)
        max_value = max(values)

        if _is_log_scale(trials, p_name):
            padding = (math.log10(max_value) - math.log10(min_value)) * padding_ratio
            min_value = math.pow(10, math.log10(min_value) - padding)
            max_value = math.pow(10, math.log10(max_value) + padding)
            param_is_log[p_name] = True
            param_is_cat[p_name] = False

        elif _is_numerical(trials, p_name):
            padding = (max_value - min_value) * padding_ratio
            min_value = min_value - padding
            max_value = max_value + padding
            param_is_log[p_name] = False
            param_is_cat[p_name] = False

        else:
            param_is_log[p_name] = False
            param_is_cat[p_name] = True
            span = len(set(values)) - 1
            padding = span * padding_ratio
            min_value = -padding
            max_value = span + padding

        param_values_range[p_name] = (min_value, max_value)

    sub_plot_infos: List[List[_SubContourInfo]]
    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plot_info = _generate_contour_subplot_info(
            trials, x_param, y_param, param_values_range, target
        )
        sub_plot_infos = [[sub_plot_info]]
    else:
        sub_plot_infos = [
            [
                _SubContourInfo(
                    x_indices=[], y_indices=[], x_values=[], y_values=[], z_values=[[]]
                )
                for _ in range(len(sorted_params))
            ]
            for _ in range(len(sorted_params))
        ]
        for x_i, x_param in enumerate(sorted_params):
            for y_i, y_param in enumerate(sorted_params):
                sub_plot_info = _generate_contour_subplot_info(
                    trials,
                    x_param,
                    y_param,
                    param_values_range,
                    target,
                )
                sub_plot_infos[y_i][x_i] = sub_plot_info

    return _ContourInfo(
        sorted_params=sorted_params,
        param_values_range=param_values_range,
        param_is_log=param_is_log,
        param_is_cat=param_is_cat,
        sub_plot_infos=sub_plot_infos,
    )


def _generate_contour_subplot_info(
    trials: List[FrozenTrial],
    x_param: str,
    y_param: str,
    param_values_range: Dict[str, Tuple[float, float]],
    target: Optional[Callable[[FrozenTrial], float]],
) -> _SubContourInfo:

    if x_param == y_param:
        return _SubContourInfo(x_indices=[], y_indices=[], x_values=[], y_values=[], z_values=[[]])

    x_indices = sorted(set(_get_param_values(trials, x_param)))
    y_indices = sorted(set(_get_param_values(trials, y_param)))
    if len(x_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(x_param))
        return _SubContourInfo(x_indices=[], y_indices=[], x_values=[], y_values=[], z_values=[[]])
    if len(y_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(y_param))
        return _SubContourInfo(x_indices=[], y_indices=[], x_values=[], y_values=[], z_values=[[]])

    # Padding to the plot for non-categorical params.
    x_range = param_values_range[x_param]
    if _is_numerical(trials, x_param):
        x_indices = [x_range[0]] + x_indices + [x_range[1]]

    y_range = param_values_range[y_param]
    if _is_numerical(trials, y_param):
        y_indices = [y_range[0]] + y_indices + [y_range[1]]

    z_values = [[float("nan") for _ in range(len(x_indices))] for _ in range(len(y_indices))]

    x_values = []
    y_values = []
    for trial in trials:
        if x_param not in trial.params or y_param not in trial.params:
            continue
        x_value = trial.params[x_param]
        y_value = trial.params[y_param]
        if not _is_numerical(trials, x_param):
            x_value = str(x_value)
        if not _is_numerical(trials, y_param):
            y_value = str(y_value)
        x_values.append(x_value)
        y_values.append(y_value)
        x_i = x_indices.index(x_value)
        y_i = y_indices.index(y_value)

        if target is None:
            value = trial.value
        else:
            value = target(trial)

        if isinstance(value, int):
            value = float(value)
        elif not isinstance(value, float):
            raise ValueError(
                f"Trial{trial.number} has COMPLETE state, but its target value is non-numeric."
            )
        z_values[y_i][x_i] = value

    return _SubContourInfo(
        x_indices=x_indices,
        y_indices=y_indices,
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )
