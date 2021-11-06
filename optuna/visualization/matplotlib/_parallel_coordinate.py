from collections import defaultdict
import math
from typing import Callable
from typing import cast
from typing import DefaultDict
from typing import List
from typing import Optional

import numpy as np

from optuna._experimental import experimental
from optuna.logging import get_logger
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _check_plot_args
from optuna.visualization.matplotlib._matplotlib_imports import _imports
from optuna.visualization.matplotlib._utils import _is_categorical
from optuna.visualization.matplotlib._utils import _is_log_scale
from optuna.visualization.matplotlib._utils import _is_numerical


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import LineCollection
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental("2.2.0")
def plot_parallel_coordinate(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot the high-dimensional parameter relationships in a study with Matplotlib.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_parallel_coordinate` for an example.

    Example:

        The following code snippet shows how to plot the high-dimensional parameter relationships.

        .. plot::

            import optuna

            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=["x", "y"])

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
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_parallel_coordinate_plot(study, params, target, target_name)


def _get_parallel_coordinate_plot(
    study: Study,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":

    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target
        reversescale = study.direction == StudyDirection.MINIMIZE
    else:
        reversescale = True

    # Set up the graph style.
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("Blues_r" if reversescale else "Blues")
    ax.set_title("Parallel Coordinate Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Prepare data for plotting.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return ax

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        all_params = set(params)
    sorted_params = sorted(all_params)

    obj_org = [target(t) for t in trials]
    obj_min = min(obj_org)
    obj_max = max(obj_org)
    obj_w = obj_max - obj_min
    dims_obj_base = [[o] for o in obj_org]

    cat_param_names = []
    cat_param_values = []
    cat_param_ticks = []
    log_param_names = []
    param_values = []
    var_names = [target_name]
    numeric_cat_params_indices: List[int] = []

    for param_index, p_name in enumerate(sorted_params):
        values = [t.params[p_name] if p_name in t.params else np.nan for t in trials]

        if _is_log_scale(trials, p_name):
            values = [math.log10(v) for v in values]
            log_param_names.append(p_name)
        elif _is_categorical(trials, p_name):
            vocab = defaultdict(lambda: len(vocab))  # type: DefaultDict[str, int]

            if _is_numerical(trials, p_name):
                _ = [vocab[v] for v in sorted(values)]
                numeric_cat_params_indices.append(param_index)

            values = [vocab[v] for v in values]

            cat_param_names.append(p_name)
            vocab_item_sorted = sorted(vocab.items(), key=lambda x: x[1])
            cat_param_values.append([v[0] for v in vocab_item_sorted])
            cat_param_ticks.append([v[1] for v in vocab_item_sorted])

        p_min = min(values)
        p_max = max(values)
        p_w = p_max - p_min

        if p_w == 0.0:
            center = obj_w / 2 + obj_min
            for i in range(len(values)):
                dims_obj_base[i].append(center)
        else:
            for i, v in enumerate(values):
                dims_obj_base[i].append((v - p_min) / p_w * obj_w + obj_min)

        var_names.append(p_name if len(p_name) < 20 else "{}...".format(p_name[:17]))
        param_values.append(values)

    if numeric_cat_params_indices:
        # np.lexsort consumes the sort keys the order from back to front.
        # So the values of parameters have to be reversed the order.
        sorted_idx = np.lexsort(
            [param_values[index] for index in numeric_cat_params_indices][::-1]
        )
        # Since the values are mapped to other categories by the index,
        # the index will be swapped according to the sorted index of numeric params.
        param_values = [list(np.array(v)[sorted_idx]) for v in param_values]

    # Draw multiple line plots and axes.
    # Ref: https://stackoverflow.com/a/50029441
    ax.set_xlim(0, len(sorted_params))
    ax.set_ylim(obj_min, obj_max)
    xs = [range(len(sorted_params) + 1) for _ in range(len(dims_obj_base))]
    segments = [np.column_stack([x, y]) for x, y in zip(xs, dims_obj_base)]
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(np.asarray([target(t) for t in trials] + [0]))
    axcb = fig.colorbar(lc, pad=0.1)
    axcb.set_label(target_name)
    plt.xticks(range(len(sorted_params) + 1), var_names, rotation=330)

    for i, p_name in enumerate(sorted_params):
        ax2 = ax.twinx()
        ax2.set_ylim(min(param_values[i]), max(param_values[i]))
        if _is_log_scale(trials, p_name):
            ax2.set_yscale("log")
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.plot([1] * len(param_values[i]), param_values[i], visible=False)
        ax2.spines["right"].set_position(("axes", (i + 1) / len(sorted_params)))
        if p_name in cat_param_names:
            idx = cat_param_names.index(p_name)
            tick_pos = cat_param_ticks[idx]
            tick_labels = cat_param_values[idx]
            ax2.set_yticks(tick_pos)
            ax2.set_yticklabels(tick_labels)

    ax.add_collection(lc)

    return ax
