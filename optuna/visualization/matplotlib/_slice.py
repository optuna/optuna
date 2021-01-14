import math
from typing import Callable
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from optuna._experimental import experimental
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _check_plot_args
from optuna.visualization.matplotlib._matplotlib_imports import _imports
from optuna.visualization.matplotlib._utils import _is_categorical
from optuna.visualization.matplotlib._utils import _is_log_scale


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Colormap
    from optuna.visualization.matplotlib._matplotlib_imports import matplotlib
    from optuna.visualization.matplotlib._matplotlib_imports import PathCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental("2.2.0")
def plot_slice(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the parameter relationship as slice plot in a study with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_slice` for an example.

    Example:

        The following code snippet shows how to plot the parameter relationship as slice plot.

        .. plot::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.matplotlib.plot_slice(study, params=["x", "y"])

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
            Target's name to display on the axis label.


    Returns:
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_slice_plot(study, params, target, target_name)


def _get_slice_plot(
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
    if target is None:
        obj_values = [cast(float, t.value) for t in trials]
    else:
        obj_values = [target(t) for t in trials]

    if n_params == 1:
        # Set up the graph style.
        fig, axs = plt.subplots()
        axs.set_title("Slice Plot")

        # Draw a scatter plot.
        sc = _generate_slice_subplot(
            trials, sorted_params[0], axs, cmap, padding_ratio, obj_values, target_name
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
                trials, param, ax, cmap, padding_ratio, obj_values, target_name
            )

    axcb = fig.colorbar(sc, ax=axs)
    axcb.set_label("#Trials")

    return axs


def _generate_slice_subplot(
    trials: List[FrozenTrial],
    param: str,
    ax: "Axes",
    cmap: "Colormap",
    padding_ratio: float,
    obj_values: List[Union[int, float]],
    target_name: str,
) -> "PathCollection":
    x_values = []
    y_values = []
    trial_numbers = []
    scale = None
    for t, obj_v in zip(trials, obj_values):
        if param in t.params:
            x_values.append(t.params[param])
            y_values.append(obj_v)
            trial_numbers.append(t.number)
    ax.set(xlabel=param, ylabel=target_name)
    if _is_log_scale(trials, param):
        ax.set_xscale("log")
        scale = "log"
    elif _is_categorical(trials, param):
        x_values = [str(x) for x in x_values]
        scale = "categorical"
    xlim = _calc_lim_with_padding(x_values, padding_ratio, scale)
    ax.set_xlim(xlim[0], xlim[1])
    sc = ax.scatter(x_values, y_values, c=trial_numbers, cmap=cmap, edgecolors="grey")
    ax.label_outer()

    return sc


def _calc_lim_with_padding(
    values: List[Union[int, float]], padding_ratio: float, scale: Optional[str] = None
) -> Tuple[Union[int, float], Union[int, float]]:
    value_max = max(values)
    value_min = min(values)
    if scale == "log":
        padding = (math.log10(value_max) - math.log10(value_min)) * padding_ratio
        return (
            math.pow(10, math.log10(value_min) - padding),
            math.pow(10, math.log10(value_max) + padding),
        )
    elif scale == "categorical":
        width = len(set(values)) - 1
        padding = width * padding_ratio
        return -padding, width + padding
    else:
        padding = (value_max - value_min) * padding_ratio
        return value_min - padding, value_max + padding
