from __future__ import annotations

from collections import defaultdict
import math
from typing import cast
from typing import NamedTuple
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import CategoricalDistribution
from optuna.logging import get_logger
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from optuna.study import Study
    from optuna.trial import FrozenTrial

from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical
from optuna.visualization._utils import _is_reverse_scale


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import plotly
    from optuna.visualization._utils import COLOR_SCALE

_logger = get_logger(__name__)


class _DimensionInfo(NamedTuple):
    label: str
    values: tuple[float, ...]
    range: tuple[float, float]
    is_log: bool
    is_cat: bool
    tickvals: list[int | float]
    ticktext: list[str]
    has_missing: bool = False


class _ParallelCoordinateInfo(NamedTuple):
    dim_objective: _DimensionInfo
    dims_params: list[_DimensionInfo]
    reverse_scale: bool
    colorbar_title: str
    color_values: tuple[float, ...] | None = None
    is_rank_color: bool = False
    draw_order: tuple[int, ...] | None = None
    feasibility: tuple[bool, ...] | None = None


_MISSING_LABEL = "None"
_MISSING_VALUE = object()


def plot_parallel_coordinate(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the high-dimensional parameter relationships in a study.

    If a trial does not contain a parameter, the line is connected to a special ``None`` tick.
    If at least one completed trial contains constraint values, feasible trials are drawn with
    solid lines and infeasible trials with dotted lines. In this case, trials without constraint
    values are treated as infeasible.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted. For a
            multi-objective study, all objective values are plotted and the lines are colored by
            Pareto rank. Pareto ranks are computed from the objective values without considering
            constraints.
        target_name:
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.

    .. note::
        The colormap is reversed when the ``target`` argument isn't :obj:`None` or ``direction``
        of :class:`~optuna.study.Study` is ``minimize``. The Pareto-rank colormap is always
        reversed for multi-objective studies.
    """

    _imports.check()
    info = _get_parallel_coordinate_info(study, params, target, target_name)
    return _get_parallel_coordinate_plot(info)


def _get_parallel_coordinate_plot(info: _ParallelCoordinateInfo) -> "go.Figure":
    layout = go.Layout(title="Parallel Coordinate Plot")

    if len(info.dims_params) == 0 or len(info.dim_objective.values) == 0:
        return go.Figure(data=[], layout=layout)

    dimensions = [info.dim_objective] + info.dims_params
    draw_order = (
        info.draw_order
        if info.draw_order is not None
        else tuple(range(len(info.dim_objective.values)))
    )
    color_values = (
        info.color_values if info.color_values is not None else info.dim_objective.values
    )
    color_min, color_max = _get_color_range(color_values, info.is_rank_color)

    normalized_dimensions = [_normalize_dimension(dim) for dim in dimensions]
    colors = _get_line_colors(color_values, color_min, color_max, info.reverse_scale)
    x_values = list(range(len(dimensions)))
    traces: list[go.Scatter] = []
    for trial_id in draw_order:
        is_feasible = info.feasibility is None or info.feasibility[trial_id]
        traces.append(
            go.Scatter(
                x=x_values,
                y=[values[trial_id] for values in normalized_dimensions],
                customdata=[
                    [dim.label, _format_dimension_value(dim, dim.values[trial_id])]
                    for dim in dimensions
                ],
                mode="lines",
                line={
                    "color": colors[trial_id],
                    "dash": "solid" if is_feasible else "dot",
                    "width": 1.2,
                },
                opacity=0.5,
                showlegend=False,
                hovertemplate="%{customdata[0]}: %{customdata[1]}<extra></extra>",
            )
        )

    colorbar: dict[str, Any] = {"title": info.colorbar_title}
    if info.is_rank_color:
        colorbar["tickvals"] = list(range(int(max(color_values)) + 1))
    if info.feasibility is not None:
        traces.extend(_get_feasibility_legend_traces())
    traces.append(
        go.Scatter(
            x=[None, None],
            y=[None, None],
            mode="markers",
            marker={
                "color": [color_min, color_max],
                "cmin": color_min,
                "cmax": color_max,
                "colorscale": COLOR_SCALE,
                "reversescale": info.reverse_scale,
                "showscale": True,
                "colorbar": colorbar,
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )

    annotations = _get_axis_annotations(dimensions)
    shapes = [
        {
            "type": "line",
            "x0": axis_id,
            "x1": axis_id,
            "y0": 0,
            "y1": 1,
            "line": {"color": "#888", "width": 1},
            "layer": "below",
        }
        for axis_id in x_values
    ]
    layout.update(
        xaxis={
            "tickmode": "array",
            "tickvals": x_values,
            "ticktext": [dim.label for dim in dimensions],
            "range": (-0.2, len(dimensions) - 0.8),
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={"range": (-0.08, 1.08), "visible": False, "fixedrange": True},
        annotations=annotations,
        shapes=shapes,
        hovermode="closest",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.08,
            "yanchor": "bottom",
        },
    )

    figure = go.Figure(data=traces, layout=layout)

    return figure


def _get_feasibility_legend_traces() -> list["go.Scatter"]:
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#2F3B52", "dash": "solid", "width": 2},
            name="Feasible",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#2F3B52", "dash": "dot", "width": 2},
            name="Infeasible",
            showlegend=True,
            hoverinfo="skip",
        ),
    ]


def _get_parallel_coordinate_info(
    study: Study,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> _ParallelCoordinateInfo:
    has_custom_target = target is not None
    is_multi_objective = target is None and study._is_multi_objective()
    if not is_multi_objective:
        _check_plot_args(study, target, target_name)

    trials = list(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if is_multi_objective:
        trials = [
            trial
            for trial in trials
            if trial.values is not None
            and len(trial.values) == len(study.directions)
            and all(math.isfinite(value) for value in trial.values)
        ]
    else:
        trials = _filter_nonfinite(trials, target=target)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError(f"Parameter {input_p_name} does not exist in your study.")
        all_params = set(params)
    sorted_params = sorted(all_params)

    if is_multi_objective:
        metric_names = study.metric_names
        objective_labels = metric_names or [
            f"Objective {objective_id}" for objective_id in range(len(study.directions))
        ]
        objective_dims = [
            _build_numerical_dimension(
                label,
                [float(cast(tuple[float, ...], trial.values)[objective_id]) for trial in trials],
            )
            for objective_id, label in enumerate(objective_labels)
        ]
        loss_values = np.asarray(
            [
                [
                    -value if direction == StudyDirection.MAXIMIZE else value
                    for value, direction in zip(
                        cast(tuple[float, ...], trial.values), study.directions
                    )
                ]
                for trial in trials
            ],
            dtype=np.float64,
        )
        ranks = _fast_non_domination_rank(loss_values)
        color_values: tuple[float, ...] | None = tuple(float(rank) for rank in ranks)
        draw_order: tuple[int, ...] | None = tuple(np.argsort(ranks)[::-1].tolist())
        colorbar_title = "Pareto Rank"
        reverse_scale = True
    else:
        if target is None:

            def _target(t: FrozenTrial) -> float:
                return cast(float, t.value)

            target = _target

        objectives = [target(t) for t in trials]
        objective_dims = [_build_numerical_dimension(target_name, objectives)]
        color_values = None
        draw_order = None
        colorbar_title = target_name
        reverse_scale = _is_reverse_scale(study, target if has_custom_target else None)

    # The value of (0, 0) is a dummy range. It is ignored when we plot.
    dim_objective = (
        objective_dims[0] if objective_dims else _build_numerical_dimension(target_name, [])
    )

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return _ParallelCoordinateInfo(
            dim_objective=dim_objective,
            dims_params=[],
            reverse_scale=reverse_scale,
            colorbar_title=colorbar_title,
            color_values=color_values,
            is_rank_color=is_multi_objective,
            draw_order=draw_order,
        )

    feasibility = _get_feasibility(trials)

    numeric_cat_params_indices: list[int] = []
    dims = []
    for dim_index, p_name in enumerate(sorted_params, start=len(objective_dims)):
        values = [t.params.get(p_name, _MISSING_VALUE) for t in trials]
        is_categorical = False
        for t in trials:
            if p_name in t.params:
                is_categorical |= isinstance(t.distributions[p_name], CategoricalDistribution)
        has_missing = any(value is _MISSING_VALUE for value in values)
        if _is_log_scale(trials, p_name):
            valid_values = [math.log10(cast(float, v)) for v in values if v is not _MISSING_VALUE]
            min_value = min(valid_values)
            max_value = max(valid_values)
            tickvals: list[int | float] = list(
                range(math.ceil(min_value), math.floor(max_value) + 1)
            )
            if min_value not in tickvals:
                tickvals = [min_value] + tickvals
            if max_value not in tickvals:
                tickvals = tickvals + [max_value]
            missing_value = _get_missing_value(min_value, max_value) if has_missing else None
            transformed_values = tuple(
                cast(float, missing_value) if v is _MISSING_VALUE else math.log10(cast(float, v))
                for v in values
            )
            if missing_value is not None:
                tickvals.insert(0, missing_value)
            dim = _DimensionInfo(
                label=_truncate_label(p_name),
                values=transformed_values,
                range=(missing_value if missing_value is not None else min_value, max_value),
                is_log=True,
                is_cat=False,
                tickvals=tickvals,
                ticktext=([_MISSING_LABEL] if has_missing else [])
                + [f"{math.pow(10, x):.3g}" for x in tickvals[bool(has_missing) :]],
                has_missing=has_missing,
            )
        elif is_categorical:
            valid_values = [v for v in values if v is not _MISSING_VALUE]
            vocab: defaultdict[Any, int] = defaultdict(lambda: len(vocab))

            ticktext: list[str]
            if _is_numerical(trials, p_name):
                _ = [vocab[v] for v in sorted(valid_values)]
                ticktext = [str(v) for v in sorted(vocab.keys())]
                numeric_cat_params_indices.append(dim_index)
            else:
                _ = [vocab[v] for v in valid_values]
                ticktext = [str(v) for v in sorted(vocab.keys(), key=lambda x: vocab[x])]
            offset = 1 if has_missing else 0
            categorical_values = tuple(
                0 if v is _MISSING_VALUE else vocab[cast(int | str | None, v)] + offset
                for v in values
            )
            if has_missing:
                ticktext = [
                    f"{text} (value)" if text == _MISSING_LABEL else text for text in ticktext
                ]
            dim = _DimensionInfo(
                label=_truncate_label(p_name),
                values=categorical_values,
                range=(min(categorical_values), max(categorical_values)),
                is_log=False,
                is_cat=True,
                tickvals=list(range(len(vocab) + offset)),
                ticktext=([_MISSING_LABEL] if has_missing else []) + ticktext,
                has_missing=has_missing,
            )
        else:
            valid_values = [cast(float, v) for v in values if v is not _MISSING_VALUE]
            min_value = min(valid_values)
            max_value = max(valid_values)
            missing_value = _get_missing_value(min_value, max_value) if has_missing else None
            numerical_values = tuple(
                cast(float, missing_value) if v is _MISSING_VALUE else cast(float, v)
                for v in values
            )
            valid_ticks = (
                [min_value]
                if min_value == max_value
                else [float(value) for value in np.linspace(min_value, max_value, num=5)]
            )
            dim = _DimensionInfo(
                label=_truncate_label(p_name),
                values=numerical_values,
                range=(missing_value if missing_value is not None else min_value, max_value),
                is_log=False,
                is_cat=False,
                tickvals=([missing_value] + valid_ticks if missing_value is not None else []),
                ticktext=(
                    [_MISSING_LABEL] + [f"{value:.3g}" for value in valid_ticks]
                    if has_missing
                    else []
                ),
                has_missing=has_missing,
            )

        dims.append(dim)

    plot_dims = dims
    if numeric_cat_params_indices:
        dims = objective_dims + plot_dims
        # np.lexsort consumes the sort keys the order from back to front.
        # So the values of parameters have to be reversed the order.
        idx = np.lexsort([dims[index].values for index in numeric_cat_params_indices][::-1])
        updated_dims = []
        for dim in dims:
            # Since the values are mapped to other categories by the index,
            # the index will be swapped according to the sorted index of numeric params.
            updated_dims.append(
                _DimensionInfo(
                    label=dim.label,
                    values=tuple(np.array(dim.values)[idx]),
                    range=dim.range,
                    is_log=dim.is_log,
                    is_cat=dim.is_cat,
                    tickvals=dim.tickvals,
                    ticktext=dim.ticktext,
                    has_missing=dim.has_missing,
                )
            )
        objective_dims = updated_dims[: len(objective_dims)]
        dim_objective = objective_dims[0]
        plot_dims = updated_dims[len(objective_dims) :]
        if color_values is not None:
            color_values = tuple(np.asarray(color_values)[idx])
        if feasibility is not None:
            feasibility = tuple(bool(value) for value in np.asarray(feasibility)[idx])
        if draw_order is not None:
            assert color_values is not None
            draw_order = tuple(
                sorted(range(len(trials)), key=color_values.__getitem__, reverse=True)
            )

    return _ParallelCoordinateInfo(
        dim_objective=dim_objective,
        dims_params=objective_dims[1:] + plot_dims,
        reverse_scale=reverse_scale,
        colorbar_title=colorbar_title,
        color_values=color_values,
        is_rank_color=is_multi_objective,
        draw_order=draw_order,
        feasibility=feasibility,
    )


def _build_numerical_dimension(label: str, values: list[float]) -> _DimensionInfo:
    value_range = (min(values), max(values)) if values else (0, 0)
    return _DimensionInfo(label, tuple(values), value_range, False, False, [], [])


def _get_feasibility(trials: list[FrozenTrial]) -> tuple[bool, ...] | None:
    if not any(len(trial.constraints) > 0 for trial in trials):
        return None

    return tuple(
        len(trial.constraints) > 0 and all(value <= 0.0 for value in trial.constraints.values())
        for trial in trials
    )


def _get_missing_value(min_value: float, max_value: float) -> float:
    if min_value == max_value:
        return min_value - 1.0
    return min_value - (max_value - min_value) * 0.12 / 0.88


def _normalize_dimension(dim: _DimensionInfo) -> tuple[float, ...]:
    min_value, max_value = dim.range
    if min_value == max_value:
        return tuple(0.5 for _ in dim.values)
    return tuple((value - min_value) / (max_value - min_value) for value in dim.values)


def _get_line_colors(
    values: tuple[float, ...], min_value: float, max_value: float, reverse_scale: bool
) -> list[str]:
    if min_value == max_value:
        normalized = [0.5] * len(values)
    else:
        normalized = [(value - min_value) / (max_value - min_value) for value in values]
    if reverse_scale:
        normalized = [1.0 - value for value in normalized]
    return [_sample_colorscale(COLOR_SCALE, value) for value in normalized]


def _sample_colorscale(colorscale: list[str], value: float) -> str:
    scaled_value = value * (len(colorscale) - 1)
    lower_index = min(int(scaled_value), len(colorscale) - 2)
    return cast(
        str,
        plotly.colors.find_intermediate_color(
            colorscale[lower_index],
            colorscale[lower_index + 1],
            scaled_value - lower_index,
            colortype="rgb",
        ),
    )


def _get_color_range(values: tuple[float, ...], is_rank_color: bool) -> tuple[float, float]:
    if is_rank_color:
        return -0.5, max(values) + 0.5

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return min_value - 0.5, max_value + 0.5
    return min_value, max_value


def _format_dimension_value(dim: _DimensionInfo, value: float) -> str:
    for tick_value, tick_text in zip(dim.tickvals, dim.ticktext):
        if math.isclose(value, tick_value):
            return tick_text
    if dim.is_log:
        return f"{math.pow(10, value):.3g}"
    return f"{value:.3g}"


def _get_axis_annotations(dimensions: list[_DimensionInfo]) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for axis_id, dim in enumerate(dimensions):
        if dim.tickvals:
            tick_values = dim.tickvals
            tick_text = dim.ticktext or [f"{value:.3g}" for value in tick_values]
        elif dim.range[0] == dim.range[1]:
            tick_values = [dim.range[0]]
            tick_text = [f"{dim.range[0]:.3g}"]
        else:
            tick_values = [float(value) for value in np.linspace(*dim.range, num=5)]
            tick_text = [f"{value:.3g}" for value in tick_values]

        normalized_ticks = _normalize_dimension(
            dim._replace(values=tuple(float(value) for value in tick_values))
        )
        for value, text in zip(normalized_ticks, tick_text):
            annotations.append(
                {
                    "x": axis_id,
                    "y": value,
                    "text": text,
                    "showarrow": False,
                    "xanchor": "right",
                    "xshift": -4,
                    "font": {"size": 10, "color": "#555"},
                }
            )
    return annotations


def _truncate_label(label: str) -> str:
    return label if len(label) < 20 else f"{label[:17]}..."
