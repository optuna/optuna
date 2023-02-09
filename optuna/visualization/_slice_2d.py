import math
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
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
    # range: Union[Tuple[float, float], List[Any]]
    range: Tuple[float, float]
    is_log: bool
    is_cat: bool


class _SubplotInfo(NamedTuple):
    xaxis: _AxisInfo
    yaxis: _AxisInfo
    xs: List[Any]
    ys: List[Any]
    trials: List[FrozenTrial]
    zs: np.ndarray
    transformed_zs: np.ndarray


class _Slice2DInfo(NamedTuple):
    params: List[str]
    sub_plot_infos: List[List[_SubplotInfo]]
    target_name: str
    zs: np.ndarray
    transformed_zs: np.ndarray
    has_custom_target: bool


def plot_slice_2d(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
    target_transform: Literal["rank", "none"] = "rank",
    size: Optional[Tuple[int, int]] = None,
    colormap: Any = "RdYlBu_r",
    n_ticks: int = 5,
    tick_format: str = ".3g",
) -> "go.Figure":
    """Plot the parameter relationship as 2D slice plot in a study.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as 2D slice plot.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=30)

            fig = optuna.visualization.plot_slice_2d(study, params=["x", "y"])
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
        target_transform:
            A string to specify how ``target`` value is mapped to the color scale. Available values
            are ``rank`` (default) and ``none``. If ``rank`` is specified, the colors for each point is
            determined by the rank of the corresponding ``target`` value. If ``none`` is specified,
            the colors are determined by the ``target`` value itself.
        size:
            Figure size, specified by a tuple ``(width, height)``. Defaults to :obj:`None`, where
            the default size of plotly is used.
        colormap:
            A colormap for the 2D slice plot. Defaults to ``RdYlBu_r``.
        n_ticks:
            Number of ticks on the color bar. Defaults to 5.
        tick_format:
            Format string for tick labels on the color bar. Defaults to ``".3g"``.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()
    info = _get_slice_2d_info(study, params, target, target_name, target_transform)
    return _get_slice_2d_plot(info, size, colormap, n_ticks, tick_format)


def _rankdata(data):
    # Copyright 2002 Gary Strangman.  All rights reserved
    # Copyright 2002-2016 The SciPy Developers
    #
    # The original code from Gary Strangman was heavily adapted for
    # use in SciPy by Travis Oliphant.  The original code came with the
    # following disclaimer:
    #
    # This software is provided "as-is".  There are no expressed or implied
    # warranties of any kind, including, but not limited to, the warranties
    # of merchantability and fitness for a given application.  In no event
    # shall Gary Strangman be liable for any direct, indirect, incidental,
    # special, exemplary or consequential damages (including, but not limited
    # to, loss of use, data or profits, or business interruption) however
    # caused and on any theory of liability, whether in contract, strict
    # liability or tort (including negligence or otherwise) arising in any way
    # out of the use of this software, even if advised of the possibility of
    # such damage.

    arr = np.ravel(np.asarray(data))
    sorter = np.argsort(arr)
    inv = np.empty(sorter.size, dtype=int)
    inv[sorter] = np.arange(sorter.size, dtype=int)
    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    result = 0.5 * (count[dense] + count[dense - 1] + 1)
    return result


def _get_slice_2d_info(
    study: Study,
    params: Optional[List[str]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    target_transform: Literal["rank", "none"],
) -> _Slice2DInfo:
    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        params = []
    elif params is None:
        params = sorted(all_params)
    else:
        if len(params) <= 1:
            _logger.warning("The length of params must be greater than 1.")

        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))

    has_custom_target = True
    if target is None:

        def target(trial: FrozenTrial) -> float:
            return typing.cast(float, trial.value)

        has_custom_target = False
    target_values = np.array([target(trial) for trial in trials])

    if target_transform == "rank":
        transformed_values = _rankdata(target_values)
    else:
        assert target_transform == "none"
        transformed_values = target_values

    finite_mask = np.isfinite(transformed_values)
    if np.count_nonzero(finite_mask) == 0:
        _logger.warning("All values are infinite.")
        min_v = 0
        max_v = 1
    else:
        min_v = np.min(transformed_values[finite_mask])
        max_v = np.max(transformed_values[finite_mask])

    transformed_values = np.interp(transformed_values, [min_v, max_v], [0, 1], left=0, right=1)

    sub_plot_infos: List[List[_SubplotInfo]]
    if len(params) == 2:
        x_param = params[0]
        y_param = params[1]
        sub_plot_info = _get_slice_2d_subplot_info(
            trials, target_values, transformed_values, x_param, y_param
        )
        sub_plot_infos = [[sub_plot_info]]
    else:
        sub_plot_infos = [
            [
                _get_slice_2d_subplot_info(
                    trials, target_values, transformed_values, x_param, y_param
                )
                for x_param in params
            ]
            for y_param in params
        ]

    return _Slice2DInfo(
        params=params,
        sub_plot_infos=sub_plot_infos,
        target_name=target_name,
        zs=target_values,
        transformed_zs=transformed_values,
        has_custom_target=has_custom_target,
    )


def _get_slice_2d_subplot_info(
    trials: List[FrozenTrial],
    target_values: np.ndarray,
    transformed_values: np.ndarray,
    x_param: str,
    y_param: str,
) -> _SubplotInfo:
    xaxis = _get_axis_info(trials, x_param)
    yaxis = _get_axis_info(trials, y_param)

    filtered_ids = [
        i
        for i in range(len(trials))
        if x_param in trials[i].params and y_param in trials[i].params
    ]
    filtered_trials = [trials[i] for i in filtered_ids]
    xs = [trial.params[x_param] for trial in filtered_trials]
    ys = [trial.params[y_param] for trial in filtered_trials]
    zs = [target_values[i] for i in filtered_ids]
    transformed_zs = [transformed_values[i] for i in filtered_ids]
    return _SubplotInfo(
        xaxis=xaxis,
        yaxis=yaxis,
        xs=xs,
        ys=ys,
        trials=filtered_trials,
        zs=np.array(zs),
        transformed_zs=np.array(transformed_zs),
    )


def _get_axis_info(trials: List[FrozenTrial], param_name: str) -> _AxisInfo:
    values: List[Union[str, float, None]]
    if _is_numerical(trials, param_name):
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

    elif _is_numerical(trials, param_name):
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


def _get_slice_2d_subplot(
    info: _SubplotInfo, target_name: str, print_all_objectives: bool, colormap: Any
) -> "Scatter":
    colors = plotly.colors.sample_colorscale(colormap, info.transformed_zs)

    def get_hover_text(trial: FrozenTrial, target_value: float) -> str:
        lines = [f"Trial #{trial.number}"]
        lines += [f"{k}: {v}" for k, v in trial.params.items()]
        lines += [f"<b>{target_name}: {target_value}</b>"]
        if print_all_objectives:
            lines += [f"Objective #{i}: {v}" for i, v in enumerate(trial.values)]
        return "<br>".join(lines)

    scatter = go.Scatter(
        x=info.xs,
        y=info.ys,
        marker={"color": colors},
        mode="markers",
        showlegend=False,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=[
            get_hover_text(trial, target_value)
            for trial, target_value in zip(info.trials, info.zs)
        ],
    )
    return scatter


def _get_slice_2d_plot(
    info: _Slice2DInfo,
    size: Optional[Tuple[int, int]],
    colormap: Any,
    n_ticks: int,
    tick_format: str,
) -> "go.Figure":
    params = info.params
    sub_plot_infos = info.sub_plot_infos

    layout = go.Layout(title=f"Slice 2D Plot: {info.target_name}")
    if size is not None:
        layout.width, layout.height = size

    if len(params) <= 1:
        return go.Figure(data=[], layout=layout)

    if len(params) == 2:
        x_param = params[0]
        y_param = params[1]
        sub_plot_info = sub_plot_infos[0][0]
        sub_plots = _get_slice_2d_subplot(
            sub_plot_info, info.target_name, info.has_custom_target, colormap
        )

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
                scatter = _get_slice_2d_subplot(
                    sub_plot_infos[y_i][x_i], info.target_name, info.has_custom_target, colormap
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

    tickvals = np.linspace(0, 1, n_ticks + 1)
    sorter = np.argsort(info.transformed_zs)
    tick_show = np.interp(tickvals, info.transformed_zs[sorter], info.zs[sorter])
    ticktext = [format(t, tick_format) for t in tick_show]
    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=colormap,
            showscale=True,
            cmin=0,
            cmax=1,
            colorbar=dict(thickness=10, tickvals=tickvals, ticktext=ticktext),
        ),
        hoverinfo="none",
        showlegend=False,
    )
    figure.add_trace(colorbar_trace)
    return figure
