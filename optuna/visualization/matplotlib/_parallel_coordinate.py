from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_func
from optuna.visualization._parallel_coordinate import _get_parallel_coordinate_info
from optuna.visualization._parallel_coordinate import _ParallelCoordinateInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.study import Study
    from optuna.trial import FrozenTrial


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import BoundaryNorm
    from optuna.visualization.matplotlib._matplotlib_imports import Line2D
    from optuna.visualization.matplotlib._matplotlib_imports import LineCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("2.2.0")
def plot_parallel_coordinate(
    study: Study,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the high-dimensional parameter relationships in a study with Matplotlib.

    If a trial does not contain a parameter, the line is connected to a special ``None`` tick.
    If at least one completed trial contains constraint values, feasible trials are drawn with
    solid lines and infeasible trials with dotted lines. In this case, trials without constraint
    values are treated as infeasible.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_parallel_coordinate` for an example.

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
        A :class:`matplotlib.axes.Axes` object.

    .. note::
        The colormap is reversed when the ``target`` argument isn't :obj:`None` or ``direction``
        of :class:`~optuna.study.Study` is ``minimize``. The Pareto-rank colormap is always
        reversed for multi-objective studies.
    """

    _imports.check()
    info = _get_parallel_coordinate_info(study, params, target, target_name)
    return _get_parallel_coordinate_plot(info)


def _get_parallel_coordinate_plot(info: _ParallelCoordinateInfo) -> "Axes":
    reversescale = info.reverse_scale

    # Set up the graph style.
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues_r" if reversescale else "Blues")
    ax.set_title("Parallel Coordinate Plot", y=1.03)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Prepare data for plotting.
    if len(info.dims_params) == 0 or len(info.dim_objective.values) == 0:
        return ax

    obj_min = info.dim_objective.range[0]
    obj_max = info.dim_objective.range[1]
    objective_is_constant = obj_min == obj_max
    if objective_is_constant:
        padding = abs(obj_min) * 0.05 if obj_min != 0.0 else 0.5
        obj_min -= padding
        obj_max += padding
    obj_w = obj_max - obj_min
    draw_order = (
        info.draw_order
        if info.draw_order is not None
        else tuple(range(len(info.dim_objective.values)))
    )
    dims_obj_base = [[info.dim_objective.values[i]] for i in draw_order]
    for dim in info.dims_params:
        p_min = dim.range[0]
        p_max = dim.range[1]
        p_w = p_max - p_min

        if p_w == 0.0:
            center = obj_w / 2 + obj_min
            for row in dims_obj_base:
                row.append(center)
        else:
            for row, trial_id in zip(dims_obj_base, draw_order):
                value = dim.values[trial_id]
                row.append((value - p_min) / p_w * obj_w + obj_min)

    # Draw multiple line plots and axes.
    # Ref: https://stackoverflow.com/a/50029441
    n_params = len(info.dims_params)
    ax.set_xlim(0, n_params)
    ax.set_ylim(obj_min, obj_max)
    if objective_is_constant:
        ax.set_yticks([info.dim_objective.range[0]])
    xs = [range(n_params + 1) for _ in range(len(dims_obj_base))]
    segments = [np.column_stack([x, y]) for x, y in zip(xs, dims_obj_base)]
    lc = LineCollection(segments, cmap=cmap)
    if info.feasibility is not None:
        lc.set_linestyle(
            ["solid" if info.feasibility[trial_id] else "dotted" for trial_id in draw_order]
        )
        feasibility_legend = [
            (True, "Feasible", "solid"),
            (False, "Infeasible", "dotted"),
        ]
        legend_handles = [
            Line2D([0], [0], color="#2F3B52", linestyle=linestyle, label=label)
            for _, label, linestyle in feasibility_legend
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),
            ncol=len(legend_handles),
            frameon=False,
        )
    color_values = (
        info.color_values if info.color_values is not None else info.dim_objective.values
    )
    color_values = tuple(np.asarray(color_values)[list(draw_order)])
    lc.set_array(np.asarray(color_values))
    if info.is_rank_color:
        max_rank = int(max(color_values))
        lc.set_norm(BoundaryNorm(np.arange(-0.5, max_rank + 1.5), cmap.N))
    axcb = fig.colorbar(lc, pad=0.1, ax=ax)
    if info.is_rank_color:
        axcb.set_ticks(list(range(int(max(color_values)) + 1)))
    axcb.set_label(info.colorbar_title)
    var_names = [info.dim_objective.label] + [dim.label for dim in info.dims_params]
    plt.xticks(range(n_params + 1), var_names, rotation=330)

    for i, dim in enumerate(info.dims_params):
        ax2 = ax.twinx()
        if dim.is_log and not dim.has_missing:
            ax2.set_ylim(np.power(10, dim.range[0]), np.power(10, dim.range[1]))
            ax2.set_yscale("log")
        else:
            ax2.set_ylim(dim.range[0], dim.range[1])
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.spines["right"].set_position(("axes", (i + 1) / n_params))
        if dim.is_cat or dim.has_missing:
            ax2.set_yticks(dim.tickvals)
            ax2.set_yticklabels(dim.ticktext)

    ax.add_collection(lc)

    return ax
