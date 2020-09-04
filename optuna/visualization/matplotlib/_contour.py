from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from scipy.interpolate import griddata

from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports
from optuna.visualization.matplotlib._utils import _is_log_scale

if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import ContourSet
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


def plot_contour(study: Study, params: Optional[List[str]] = None) -> Axes:
    """Plot the parameter relationship as contour plot in a study with Matplotlib.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as contour plot
        with Matplotlib.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            optuna.visualization.matplotlib.plot_contour(study, params=['x', 'y'])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    return _get_contour_plot(study, params)


def _get_contour_plot(study: Study, params: Optional[List[str]] = None) -> Axes:
    # Calculate basic numbers for plotting.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        fig, ax = plt.subplots()
        return ax

    all_params = {p_name for t in trials for p_name in t.params.keys()}

    if params is None:
        sorted_params = sorted(list(all_params))
    elif len(params) <= 1:
        _logger.warning("The length of params must be greater than 1.")
        fig, ax = plt.subplots()
        return ax
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(list(set(params)))
    n_params = len(sorted_params)

    if n_params == 2:
        # Set up the graph style.
        fig, axs = plt.subplots()
        axs.set_title("Contour Plot")
        cmap = _set_cmap(study)
        contour_point_num = 1000

        # Prepare data and draw contour plots.
        if params:
            x_param = params[0]
            y_param = params[1]
        else:
            x_param = sorted_params[0]
            y_param = sorted_params[1]
        cs = _generate_contour_suplot(trials, x_param, y_param, axs, cmap, contour_point_num)
        if cs:
            axcb = fig.colorbar(cs)
            axcb.set_label("Objective Value")
    else:
        # Set up the graph style.
        fig, axs = plt.subplots(n_params, n_params)
        fig.suptitle("Contour Plot")
        cmap = _set_cmap(study)
        contour_point_num = 100

        # Prepare data and draw contour plots.
        cs_list = []
        for x_i, x_param in enumerate(sorted_params):
            for y_i, y_param in enumerate(sorted_params):
                ax = axs[y_i, x_i]
                cs = _generate_contour_suplot(
                    trials, x_param, y_param, ax, cmap, contour_point_num
                )
                if cs:
                    cs_list.append(cs)
        if cs_list:
            axcb = fig.colorbar(cs_list[0], ax=axs)
            axcb.set_label("Objective Value")

    return axs


def _set_cmap(study: Study) -> Colormap:
    cmap = plt.get_cmap("Blues_r")
    if study.direction == StudyDirection.MINIMIZE:
        cmap = plt.get_cmap("Blues")

    return cmap


def _calculate_griddata(
    trials: List[FrozenTrial], x_param: str, y_param: str, contour_point_num: int
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Union[int, float]],
    List[Union[int, float]],
    List[Union[int, float]],
    List[Union[int, float]],
]:

    # Extract values for x, y, z axes from each trail.
    x_values = []
    y_values = []
    z_values = []
    for trial in trials:
        x_values.append(trial.params[x_param])
        y_values.append(trial.params[y_param])
        if isinstance(trial.value, int):
            value = float(trial.value)
        elif isinstance(trial.value, float):
            value = trial.value
        else:
            raise ValueError(
                "Trial{} has COMPLETE state, but its value is non-numeric.".format(trial.number)
            )
        z_values.append(value)

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
    if x_param != y_param:
        if _is_log_scale(trials, x_param):
            xi = np.logspace(np.log10(x_values_min), np.log10(x_values_max), contour_point_num)
        else:
            xi = np.linspace(x_values_min, x_values_max, contour_point_num)
        if _is_log_scale(trials, y_param):
            yi = np.logspace(np.log10(y_values_min), np.log10(y_values_max), contour_point_num)
        else:
            yi = np.linspace(y_values_min, y_values_max, contour_point_num)

        # Interpolate z-axis data on a grid with cubic interpolator.
        # TODO(ytknzw): Implement Plotly-like interpolation algorithm.
        zi = griddata((x_values, y_values), z_values, (xi[None, :], yi[:, None]), method="cubic")

    return (
        xi,
        yi,
        zi,
        x_values,
        y_values,
        [x_values_min, x_values_max],
        [y_values_min, y_values_max],
    )


def _generate_contour_suplot(
    trials: List[FrozenTrial],
    x_param: str,
    y_param: str,
    ax: Axes,
    cmap: Colormap,
    contour_point_num: int,
) -> ContourSet:

    x_indices = sorted(list({t.params[x_param] for t in trials if x_param in t.params}))
    y_indices = sorted(list({t.params[y_param] for t in trials if y_param in t.params}))
    if len(x_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(x_param))
        return ax
    if len(y_indices) < 2:
        _logger.warning("Param {} unique value length is less than 2.".format(y_param))
        return ax

    xi, yi, zi, x_values, y_values, x_values_range, y_values_range = _calculate_griddata(
        trials, x_param, y_param, contour_point_num
    )
    ax.set_xlim(x_values_range[0], x_values_range[1])
    ax.set_ylim(y_values_range[0], y_values_range[1])
    ax.set(xlabel=x_param, ylabel=y_param)
    if _is_log_scale(trials, x_param):
        ax.set_xscale("log")
    if _is_log_scale(trials, y_param):
        ax.set_yscale("log")
    cs = None
    if x_param != y_param:
        # Contour the gridded data.
        ax.contour(xi, yi, zi, 15, linewidths=0.5, colors="k")
        cs = ax.contourf(xi, yi, zi, 15, cmap=cmap)
        # Plot data points.
        ax.scatter(x_values, y_values, marker="o", c="black", s=20, edgecolors="grey")
    ax.label_outer()
    return cs
