from collections import defaultdict
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy

from optuna._experimental import experimental
from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _get_param_values
from optuna.visualization.matplotlib._matplotlib_imports import _imports
from optuna.visualization.matplotlib._utils import _is_log_scale
from optuna.visualization.matplotlib._utils import _is_numerical


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import ContourSet
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


AXES_PADDING_RATIO = 5e-2


@experimental("2.2.0")
def plot_contour(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the parameter relationship as contour plot in a study with Matplotlib.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_contour` for an example.

    Warnings:
        Output figures of this Matplotlib-based
        :func:`~optuna.visualization.matplotlib.plot_contour` function would be different from
        those of the Plotly-based :func:`~optuna.visualization.plot_contour`.

    Example:

        The following code snippet shows how to plot the parameter relationship as contour plot.

        .. plot::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=30)

            optuna.visualization.matplotlib.plot_contour(study, params=["x", "y"])

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
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    _logger.warning(
        "Output figures of this Matplotlib-based `plot_contour` function would be different from "
        "those of the Plotly-based `plot_contour`."
    )
    return _get_contour_plot(study, params, target, target_name)


def _get_contour_plot(
    study: Study,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    # Calculate basic numbers for plotting.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        _, ax = plt.subplots()
        return ax

    all_params = {p_name for t in trials for p_name in t.params.keys()}

    if params is None:
        sorted_params = sorted(all_params)
    elif len(params) <= 1:
        _logger.warning("The length of params must be greater than 1.")
        _, ax = plt.subplots()
        return ax
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(set(params))
    n_params = len(sorted_params)

    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    if n_params == 2:
        # Set up the graph style.
        fig, axs = plt.subplots()
        axs.set_title("Contour Plot")
        cmap = _set_cmap(study, target)
        contour_point_num = 100

        # Prepare data and draw contour plots.
        if params:
            x_param = params[0]
            y_param = params[1]
        else:
            x_param = sorted_params[0]
            y_param = sorted_params[1]
        cs = _generate_contour_subplot(
            trials, x_param, y_param, axs, cmap, contour_point_num, target
        )
        if isinstance(cs, ContourSet):
            axcb = fig.colorbar(cs)
            axcb.set_label(target_name)
    else:
        # Set up the graph style.
        fig, axs = plt.subplots(n_params, n_params)
        fig.suptitle("Contour Plot")
        cmap = _set_cmap(study, target)
        contour_point_num = 100

        # Prepare data and draw contour plots.
        cs_list = []
        for x_i, x_param in enumerate(sorted_params):
            for y_i, y_param in enumerate(sorted_params):
                ax = axs[y_i, x_i]
                cs = _generate_contour_subplot(
                    trials, x_param, y_param, ax, cmap, contour_point_num, target
                )
                if isinstance(cs, ContourSet):
                    cs_list.append(cs)
        if cs_list:
            axcb = fig.colorbar(cs_list[0], ax=axs)
            axcb.set_label(target_name)

    return axs


def _set_cmap(study: Study, target: Optional[Callable[[FrozenTrial], float]]) -> "Colormap":
    cmap = "Blues_r" if target is None and study.direction == StudyDirection.MAXIMIZE else "Blues"
    return plt.get_cmap(cmap)


def _convert_categorical2int(values: List[str]) -> Tuple[List[int], List[str], List[int]]:
    vocab = defaultdict(lambda: len(vocab))  # type: DefaultDict[str, int]
    [vocab[v] for v in sorted(values)]
    values_converted = [vocab[v] for v in values]
    vocab_item_sorted = sorted(vocab.items(), key=lambda x: x[1])
    cat_param_labels = [v[0] for v in vocab_item_sorted]
    cat_param_pos = [v[1] for v in vocab_item_sorted]

    return values_converted, cat_param_labels, cat_param_pos


def _calculate_griddata(
    trials: List[FrozenTrial],
    x_param: str,
    x_indices: List[Union[str, int, float]],
    y_param: str,
    y_indices: List[Union[str, int, float]],
    contour_point_num: int,
    target: Optional[Callable[[FrozenTrial], float]],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Union[int, float]],
    List[Union[int, float]],
    List[Union[int, float]],
    List[Union[int, float]],
    List[int],
    List[str],
    List[int],
    List[str],
    int,
    int,
]:

    # Extract values for x, y, z axes from each trail.
    x_values = []
    y_values = []
    z_values = []
    for trial in trials:
        if x_param not in trial.params or y_param not in trial.params:
            continue
        x_values.append(trial.params[x_param])
        y_values.append(trial.params[y_param])

        if target is None:
            value = trial.value
        else:
            value = target(trial)

        if isinstance(value, int):
            value = float(value)
        elif not isinstance(value, float):
            raise ValueError(
                "Trial{} has COMPLETE state, but its target value is non-numeric.".format(
                    trial.number
                )
            )
        z_values.append(value)

    # Return empty values when x or y has no value.
    if len(x_values) == 0 or len(y_values) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            x_values,
            y_values,
            [],
            [],
            [],
            [],
            [],
            [],
            0,
            0,
        )

    # Add dummy values for grid data calculation when a parameter has one unique value.
    x_values_dummy = []
    y_values_dummy = []
    if len(set(x_values)) == 1:
        x_values_dummy = [x for x in x_indices if x not in x_values]
        x_values = x_values + x_values_dummy * len(x_values)
        y_values = y_values + (y_values * len(x_values_dummy))
        z_values = z_values + (z_values * len(x_values_dummy))
    if len(set(y_values)) == 1:
        y_values_dummy = [y for y in y_indices if y not in y_values]
        y_values = y_values + y_values_dummy * len(y_values)
        x_values = x_values + (x_values * len(y_values_dummy))
        z_values = z_values + (z_values * len(y_values_dummy))

    # Convert categorical values to int.
    cat_param_labels_x = []  # type: List[str]
    cat_param_pos_x = []  # type: List[int]
    cat_param_labels_y = []  # type: List[str]
    cat_param_pos_y = []  # type: List[int]
    if not _is_numerical(trials, x_param):
        x_values = [str(x) for x in x_values]
        (
            x_values,
            cat_param_labels_x,
            cat_param_pos_x,
        ) = _convert_categorical2int(x_values)
    if not _is_numerical(trials, y_param):
        y_values = [str(y) for y in y_values]
        (
            y_values,
            cat_param_labels_y,
            cat_param_pos_y,
        ) = _convert_categorical2int(y_values)

    # Calculate min and max of x and y.
    x_values_min = min(x_values)
    x_values_max = max(x_values)
    y_values_min = min(y_values)
    y_values_max = max(y_values)

    # Calculate grid data points.
    # For x and y, create 1-D array of evenly spaced coordinates on linear or log scale.
    xi = np.array([])
    yi = np.array([])
    zi = np.array([])

    if _is_log_scale(trials, x_param):
        padding_x = (np.log10(x_values_max) - np.log10(x_values_min)) * AXES_PADDING_RATIO
        x_values_min = np.power(10, np.log10(x_values_min) - padding_x)
        x_values_max = np.power(10, np.log10(x_values_max) + padding_x)
        xi = np.logspace(np.log10(x_values_min), np.log10(x_values_max), contour_point_num)
    else:
        padding_x = (x_values_max - x_values_min) * AXES_PADDING_RATIO
        x_values_min -= padding_x
        x_values_max += padding_x
        xi = np.linspace(x_values_min, x_values_max, contour_point_num)

    if _is_log_scale(trials, y_param):
        padding_y = (np.log10(y_values_max) - np.log10(y_values_min)) * AXES_PADDING_RATIO
        y_values_min = np.power(10, np.log10(y_values_min) - padding_y)
        y_values_max = np.power(10, np.log10(y_values_max) + padding_y)
        yi = np.logspace(np.log10(y_values_min), np.log10(y_values_max), contour_point_num)
    else:
        padding_y = (y_values_max - y_values_min) * AXES_PADDING_RATIO
        y_values_min -= padding_y
        y_values_max += padding_y
        yi = np.linspace(y_values_min, y_values_max, contour_point_num)

    # create irregularly spaced map of trial values
    # and interpolate it with Plotly's interpolation formulation
    if x_param != y_param:
        zmap = _create_zmap(x_values, y_values, z_values, xi, yi)
        zi = _interpolate_zmap(zmap, contour_point_num)

    return (
        xi,
        yi,
        zi,
        x_values,
        y_values,
        [x_values_min, x_values_max],
        [y_values_min, y_values_max],
        cat_param_pos_x,
        cat_param_labels_x,
        cat_param_pos_y,
        cat_param_labels_y,
        len(x_values_dummy),
        len(y_values_dummy),
    )


def _generate_contour_subplot(
    trials: List[FrozenTrial],
    x_param: str,
    y_param: str,
    ax: "Axes",
    cmap: "Colormap",
    contour_point_num: int,
    target: Optional[Callable[[FrozenTrial], float]],
) -> "ContourSet":

    x_indices = sorted(set(_get_param_values(trials, x_param)))
    y_indices = sorted(set(_get_param_values(trials, y_param)))
    if len(x_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(x_param))
        return ax
    if len(y_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(y_param))
        return ax

    (
        xi,
        yi,
        zi,
        x_values,
        y_values,
        x_values_range,
        y_values_range,
        x_cat_param_pos,
        x_cat_param_label,
        y_cat_param_pos,
        y_cat_param_label,
        x_values_dummy_count,
        y_values_dummy_count,
    ) = _calculate_griddata(
        trials, x_param, x_indices, y_param, y_indices, contour_point_num, target
    )
    cs = None
    ax.set(xlabel=x_param, ylabel=y_param)
    ax.set_xlim(x_values_range[0], x_values_range[1])
    ax.set_ylim(y_values_range[0], y_values_range[1])
    if len(zi) > 0:
        if _is_log_scale(trials, x_param):
            ax.set_xscale("log")
        if _is_log_scale(trials, y_param):
            ax.set_yscale("log")
        if x_param != y_param:
            # Contour the gridded data.
            ax.contour(xi, yi, zi, 15, linewidths=0.5, colors="k")
            cs = ax.contourf(xi, yi, zi, 15, cmap=cmap.reversed())
            # Plot data points.
            if x_values_dummy_count > 0:
                x_org_len = int(len(x_values) / (x_values_dummy_count + 1))
                y_org_len = int(len(y_values) / (x_values_dummy_count + 1))
            elif y_values_dummy_count > 0:
                x_org_len = int(len(x_values) / (y_values_dummy_count + 1))
                y_org_len = int(len(y_values) / (y_values_dummy_count + 1))
            else:
                x_org_len = len(x_values)
                y_org_len = len(x_values)
            ax.scatter(
                x_values[:x_org_len],
                y_values[:y_org_len],
                marker="o",
                c="black",
                s=20,
                edgecolors="grey",
                linewidth=2.0,
            )
    if x_cat_param_pos:
        ax.set_xticks(x_cat_param_pos)
        ax.set_xticklabels(x_cat_param_label)
    if y_cat_param_pos:
        ax.set_yticks(y_cat_param_pos)
        ax.set_yticklabels(y_cat_param_label)
    ax.label_outer()
    return cs


def _create_zmap(
    x_values: List[Union[int, float]],
    y_values: List[Union[int, float]],
    z_values: List[float],
    xi: np.ndarray,
    yi: np.ndarray,
) -> Dict[Tuple[int, int], float]:

    # creates z-map from trial values and params.
    # z-map is represented by hashmap of coordinate and trial value pairs
    #
    # coordinates are represented by tuple of integers, where the first item
    # indicates x-axis index and the second item indicates y-axis index
    # and refer to a position of trial value on irregular param grid
    #
    # since params were resampled either with linspace or logspace
    # original params might not be on the x and y axes anymore
    # so we are going with close approximations of trial value positions
    zmap = dict()
    for x, y, z in zip(x_values, y_values, z_values):
        xindex = int(np.argmin(np.abs(xi - x)))
        yindex = int(np.argmin(np.abs(yi - y)))
        zmap[(xindex, yindex)] = z

    return zmap


def _interpolate_zmap(zmap: Dict[Tuple[int, int], float], contour_plot_num: int) -> np.ndarray:

    # implements interpolation formulation used in Plotly
    # to interpolate heatmaps and contour plots
    # https://github.com/plotly/plotly.js/blob/master/src/traces/heatmap/interp2d.js#L30
    # citing their doc:
    #
    # > Fill in missing data from a 2D array using an iterative
    # > poisson equation solver with zero-derivative BC at edges.
    # > Amazingly, this just amounts to repeatedly averaging all the existing
    # > nearest neighbors
    #
    # Plotly's algorithm is equivalent to solve the following linear simultaneous equation.
    # It is discretization form of the Poisson equation.
    #
    #     z[x, y] = zmap[(x, y)]                                  (if zmap[(x, y)] is given)
    # 4 * z[x, y] = z[x-1, y] + z[x+1, y] + z[x, y-1] + z[x, y+1] (if zmap[(x, y)] is not given)

    a_data = []
    a_row = []
    a_col = []
    b = np.zeros(contour_plot_num**2)
    for x in range(contour_plot_num):
        for y in range(contour_plot_num):
            grid_index = y * contour_plot_num + x
            if (x, y) in zmap:
                a_data.append(1)
                a_row.append(grid_index)
                a_col.append(grid_index)
                b[grid_index] = zmap[(x, y)]
            else:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if 0 <= x + dx < contour_plot_num and 0 <= y + dy < contour_plot_num:
                        a_data.append(1)
                        a_row.append(grid_index)
                        a_col.append(grid_index)
                        a_data.append(-1)
                        a_row.append(grid_index)
                        a_col.append(grid_index + dy * contour_plot_num + dx)

    z = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix((a_data, (a_row, a_col))), b)

    return z.reshape((contour_plot_num, contour_plot_num))
