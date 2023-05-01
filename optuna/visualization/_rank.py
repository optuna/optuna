import math
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import plotly
    from optuna.visualization._plotly_imports import Scatter

_logger = get_logger(__name__)


PADDING_RATIO = 0.05


class _AxisInfo(NamedTuple):
    name: str
    range: Tuple[float, float]
    is_log: bool
    is_cat: bool


class _RankSubplotInfo(NamedTuple):
    xaxis: _AxisInfo
    yaxis: _AxisInfo
    xs: List[Any]
    ys: List[Any]
    trials: List[FrozenTrial]
    zs: np.ndarray
    color_idxs: np.ndarray


class _RankPlotInfo(NamedTuple):
    params: List[str]
    sub_plot_infos: List[List[_RankSubplotInfo]]
    target_name: str
    zs: np.ndarray
    color_idxs: np.ndarray
    has_custom_target: bool


@experimental_func("3.2.0")
def plot_rank(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot parameter relations as scatter plots with colors indicating ranks of objective value.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as a rank plot.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=30)

            fig = optuna.visualization.plot_rank(study, params=["x", "y"])
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

    .. note::
        This function requires plotly >= 5.0.0.
    """

    _imports.check()
    info = _get_rank_info(study, params, target, target_name)
    return _get_rank_plot(info)


def _get_order_with_same_order_averaging(data: np.ndarray) -> np.ndarray:
    order = np.zeros_like(data, dtype=float)
    data_sorted = np.sort(data)
    for i, d in enumerate(data):
        indices = np.where(data_sorted == d)[0]
        order[i] = sum(indices) / len(indices)
    return order


def _get_rank_info(
    study: Study,
    params: Optional[List[str]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
) -> _RankPlotInfo:
    _check_plot_args(study, target, target_name)

    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        params = []
    elif params is None:
        params = sorted(all_params)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))

    if len(params) == 0:
        _logger.warning("params is an empty list.")

    has_custom_target = True
    if target is None:

        def target(trial: FrozenTrial) -> float:
            return typing.cast(float, trial.value)

        has_custom_target = False
    target_values = np.array([target(trial) for trial in trials])
    raw_ranks = _get_order_with_same_order_averaging(target_values)
    color_idxs = raw_ranks / (len(trials) - 1) if len(trials) >= 2 else np.array([0.5])
    sub_plot_infos: List[List[_RankSubplotInfo]]
    if len(params) == 2:
        x_param = params[0]
        y_param = params[1]
        sub_plot_info = _get_rank_subplot_info(trials, target_values, color_idxs, x_param, y_param)
        sub_plot_infos = [[sub_plot_info]]
    else:
        sub_plot_infos = [
            [
                _get_rank_subplot_info(trials, target_values, color_idxs, x_param, y_param)
                for x_param in params
            ]
            for y_param in params
        ]

    return _RankPlotInfo(
        params=params,
        sub_plot_infos=sub_plot_infos,
        target_name=target_name,
        zs=target_values,
        color_idxs=color_idxs,
        has_custom_target=has_custom_target,
    )


def _get_rank_subplot_info(
    trials: List[FrozenTrial],
    target_values: np.ndarray,
    color_idxs: np.ndarray,
    x_param: str,
    y_param: str,
) -> _RankSubplotInfo:
    xaxis = _get_axis_info(trials, x_param)
    yaxis = _get_axis_info(trials, y_param)

    filtered_ids = np.array(
        [
            i
            for i in range(len(trials))
            if x_param in trials[i].params and y_param in trials[i].params
        ]
    )
    filtered_trials = [trials[i] for i in filtered_ids]
    xs = [trial.params[x_param] for trial in filtered_trials]
    ys = [trial.params[y_param] for trial in filtered_trials]
    zs = target_values[filtered_ids]
    color_idxs = color_idxs[filtered_ids]
    return _RankSubplotInfo(
        xaxis=xaxis,
        yaxis=yaxis,
        xs=xs,
        ys=ys,
        trials=filtered_trials,
        zs=np.array(zs),
        color_idxs=np.array(color_idxs),
    )


def _get_axis_info(trials: List[FrozenTrial], param_name: str) -> _AxisInfo:
    values: List[Union[str, float, None]]
    is_numerical = _is_numerical(trials, param_name)
    if is_numerical:
        values = [t.params.get(param_name) for t in trials]
    else:
        values = [
            str(t.params.get(param_name)) if param_name in t.params else None for t in trials
        ]

    min_value = min([v for v in values if v is not None])
    max_value = max([v for v in values if v is not None])

    if _is_log_scale(trials, param_name):
        min_value = float(min_value)
        max_value = float(max_value)
        padding = (math.log10(max_value) - math.log10(min_value)) * PADDING_RATIO
        min_value = math.pow(10, math.log10(min_value) - padding)
        max_value = math.pow(10, math.log10(max_value) + padding)
        is_log = True
        is_cat = False

    elif is_numerical:
        min_value = float(min_value)
        max_value = float(max_value)
        padding = (max_value - min_value) * PADDING_RATIO
        min_value = min_value - padding
        max_value = max_value + padding
        is_log = False
        is_cat = False

    else:
        unique_values = set(values)
        span = len(unique_values) - 1
        if None in unique_values:
            span -= 1
        padding = span * PADDING_RATIO
        min_value = -padding
        max_value = span + padding
        is_log = False
        is_cat = True

    return _AxisInfo(
        name=param_name,
        range=(min_value, max_value),
        is_log=is_log,
        is_cat=is_cat,
    )


def _get_rank_subplot(
    info: _RankSubplotInfo, target_name: str, print_raw_objectives: bool
) -> "Scatter":
    colormap = "RdYlBu_r"
    # sample_colorscale requires plotly >= 5.0.0.
    colors = plotly.colors.sample_colorscale(colormap, info.color_idxs)

    def get_hover_text(trial: FrozenTrial, target_value: float) -> str:
        lines = [f"Trial #{trial.number}"]
        lines += [f"{k}: {v}" for k, v in trial.params.items()]
        lines += [f"<b>{target_name}: {target_value}</b>"]
        if print_raw_objectives:
            lines += [f"Objective #{i}: {v}" for i, v in enumerate(trial.values)]
        return "<br>".join(lines)

    scatter = go.Scatter(
        x=info.xs,
        y=info.ys,
        marker={
            "color": colors,
            "line": {"width": 0.5, "color": "Grey"},
        },
        mode="markers",
        showlegend=False,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=[
            get_hover_text(trial, target_value)
            for trial, target_value in zip(info.trials, info.zs)
        ],
    )
    return scatter


def _get_rank_plot(
    info: _RankPlotInfo,
) -> "go.Figure":
    params = info.params
    sub_plot_infos = info.sub_plot_infos

    layout = go.Layout(title=f"Rank ({info.target_name})")

    if len(params) == 0:
        return go.Figure(data=[], layout=layout)
    if len(params) == 2:
        x_param = params[0]
        y_param = params[1]
        sub_plot_info = sub_plot_infos[0][0]
        sub_plots = _get_rank_subplot(sub_plot_info, info.target_name, info.has_custom_target)

        figure = go.Figure(data=sub_plots, layout=layout)
        figure.update_xaxes(title_text=x_param, range=sub_plot_info.xaxis.range)
        figure.update_yaxes(title_text=y_param, range=sub_plot_info.yaxis.range)

        if sub_plot_info.xaxis.is_cat:
            figure.update_xaxes(type="category")
        if sub_plot_info.yaxis.is_cat:
            figure.update_yaxes(type="category")

        if sub_plot_info.xaxis.is_log:
            log_range = [math.log10(p) for p in sub_plot_info.xaxis.range]
            figure.update_xaxes(range=log_range, type="log")
        if sub_plot_info.yaxis.is_log:
            log_range = [math.log10(p) for p in sub_plot_info.yaxis.range]
            figure.update_yaxes(range=log_range, type="log")
    else:
        figure = make_subplots(
            rows=len(params),
            cols=len(params),
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.08 / len(params),
            vertical_spacing=0.08 / len(params),
        )

        figure.update_layout(layout)
        for x_i, x_param in enumerate(params):
            for y_i, y_param in enumerate(params):
                scatter = _get_rank_subplot(
                    sub_plot_infos[y_i][x_i], info.target_name, info.has_custom_target
                )
                figure.add_trace(scatter, row=y_i + 1, col=x_i + 1)

                xaxis = sub_plot_infos[y_i][x_i].xaxis
                yaxis = sub_plot_infos[y_i][x_i].yaxis
                figure.update_xaxes(range=xaxis.range, row=y_i + 1, col=x_i + 1)
                figure.update_yaxes(range=yaxis.range, row=y_i + 1, col=x_i + 1)

                if xaxis.is_cat:
                    figure.update_xaxes(type="category", row=y_i + 1, col=x_i + 1)
                if yaxis.is_cat:
                    figure.update_yaxes(type="category", row=y_i + 1, col=x_i + 1)

                if xaxis.is_log:
                    log_range = [math.log10(p) for p in xaxis.range]
                    figure.update_xaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)
                if yaxis.is_log:
                    log_range = [math.log10(p) for p in yaxis.range]
                    figure.update_yaxes(range=log_range, type="log", row=y_i + 1, col=x_i + 1)

                if x_i == 0:
                    figure.update_yaxes(title_text=y_param, row=y_i + 1, col=x_i + 1)
                if y_i == len(params) - 1:
                    figure.update_xaxes(title_text=x_param, row=y_i + 1, col=x_i + 1)
    sorted_zs = np.sort(info.zs)
    tick_coloridxs = [0, 0.25, 0.5, 0.75, 1]
    tick_values = np.quantile(sorted_zs, tick_coloridxs)
    rank_text = ["min.", "25%", "50%", "75%", "max."]
    ticktext = [f"{rank_text[i]} ({tick_values[i]:3g})" for i in range(len(tick_values))]

    colormap = "RdYlBu_r"
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=colormap,
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(thickness=10, tickvals=tick_coloridxs, ticktext=ticktext),
        ),
        hoverinfo="none",
        showlegend=False,
    )
    figure.add_trace(colorbar_trace)
    return figure
