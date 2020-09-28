import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports
from optuna.visualization.matplotlib._utils import _is_log_scale


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import matplotlib
    from optuna.visualization.matplotlib._matplotlib_imports import PathCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


def plot_slice(study: Study, params: Optional[List[str]] = None) -> "Axes":
    """Plot the parameter relationship as slice plot in a study with Matplotlib.

    .. seealso::  optuna.visualization.plot_slice

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
    return _get_slice_plot(study, params)


def _get_slice_plot(study: Study, params: Optional[List[str]] = None) -> "Axes":

    # Calculate basic numbers for plotting.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        fig, ax = plt.subplots()
        return ax

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(list(set(params)))

    n_params = len(sorted_params)

    # Set up the graph style.
    cmap = plt.get_cmap("Blues")
    padding_ratio = 0.05
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.

    # Prepare data.
    obj_values = [t.value for t in trials]

    if n_params == 1:
        # Set up the graph style.
        fig, axs = plt.subplots()
        axs.set_title("Slice Plot")

        # Draw a scatter plot.
        sc = _generate_slice_subplot(
            trials, sorted_params[0], axs, cmap, padding_ratio, obj_values  # type: ignore
        )
    else:
        # Set up the graph style.
        min_figwidth = matplotlib.rcParams["figure.figsize"][0] / 2
        fighight = matplotlib.rcParams["figure.figsize"][1]
        # Ensure that each subplot has a minimum width without relying on auto-sizing.
        fig, axs = plt.subplots(
            1, n_params, sharey=True, figsize=(min_figwidth * n_params, fighight)
        )
        fig.suptitle("Slice Plot")

        # Draw scatter plots.
        for i, param in enumerate(sorted_params):
            ax = axs[i]
            sc = _generate_slice_subplot(
                trials, param, ax, cmap, padding_ratio, obj_values  # type: ignore
            )

    axcb = fig.colorbar(sc, ax=axs)
    axcb.set_label("#Trials")

    return axs


def _generate_slice_subplot(
    trials: List[FrozenTrial],
    param: str,
    ax: Axes,
    cmap: Colormap,
    padding_ratio: float,
    obj_values: List[Union[int, float]],
) -> "PathCollection":
    x_values = []
    y_values = []
    trial_numbers = []
    for t in trials:
        if param in t.params:
            x_values.append(t.params[param])
            y_values.append(obj_values[t.number])
            trial_numbers.append(t.number)
    ax.set(xlabel=param, ylabel="Objective Value")
    if _is_log_scale(trials, param):
        ax.set_xscale("log")
        xlim = _calc_lim_with_padding(x_values, padding_ratio, True)
        ax.set_xlim(xlim[0], xlim[1])
    else:
        xlim = _calc_lim_with_padding(x_values, padding_ratio)
        ax.set_xlim(xlim[0], xlim[1])
    sc = ax.scatter(x_values, y_values, c=trial_numbers, cmap=cmap, edgecolors="grey")
    ax.label_outer()

    return sc


def _calc_lim_with_padding(
    values: List[Union[int, float]], padding_ratio: float, is_log_scale: bool = False
) -> Tuple[Union[int, float], Union[int, float]]:
    value_max = max(values)
    value_min = min(values)
    if is_log_scale:
        padding = (math.log10(value_max) - math.log10(value_min)) * padding_ratio
        return (
            math.pow(10, math.log10(value_min) - padding),
            math.pow(10, math.log10(value_max) + padding),
        )
    else:
        padding = (value_max - value_min) * padding_ratio
        return value_min - padding, value_max + padding
