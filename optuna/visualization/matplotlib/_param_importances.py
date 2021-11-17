from collections import OrderedDict
from typing import Callable
from typing import List
from typing import Optional

import numpy as np

import optuna
from optuna._experimental import experimental
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _check_plot_args
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import cm
    from optuna.visualization.matplotlib._matplotlib_imports import plt


_logger = get_logger(__name__)


AXES_PADDING_RATIO = 1.05


@experimental("2.2.0")
def plot_param_importances(
    study: Study,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot hyperparameter importances with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_param_importances` for an example.

    Example:

        The following code snippet shows how to plot hyperparameter importances.

        .. plot::

            import optuna


            def objective(trial):
                x = trial.suggest_int("x", 0, 2)
                y = trial.suggest_float("y", -1.0, 1.0)
                z = trial.suggest_float("z", 0.0, 1.5)
                return x ** 2 + y ** 3 - z ** 4


            sampler = optuna.samplers.RandomSampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=100)

            optuna.visualization.matplotlib.plot_param_importances(study)

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to
            :class:`~optuna.importance.FanovaImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective
                optimization. For example, to get the hyperparameter importance of the first
                objective, use ``target=lambda t: t.values[0]`` for the target parameter.
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
    return _get_param_importance_plot(study, evaluator, params, target, target_name)


def _get_param_importance_plot(
    study: Study,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Hyperparameter Importances")
    ax.set_xlabel(f"Importance for {target_name}")
    ax.set_ylabel("Hyperparameter")

    # Prepare data for plotting.
    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        _logger.warning("Study instance does not contain completed trials.")
        return ax

    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params, target=target
    )

    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())
    pos = np.arange(len(param_names))

    # Draw horizontal bars.
    ax.barh(
        pos,
        importance_values,
        align="center",
        color=cm.get_cmap("tab20c")(0),
        tick_label=param_names,
    )

    renderer = fig.canvas.get_renderer()
    for idx, val in enumerate(importance_values):
        label = f" {val:.2f}" if val >= 0.01 else " <0.01"
        text = ax.text(val, idx, label, va="center")

        # Sometimes horizontal axis needs to be re-scaled
        # to avoid text going over plot area.
        bbox = text.get_window_extent(renderer)
        bbox = bbox.transformed(ax.transData.inverted())
        _, plot_xmax = ax.get_xlim()
        bbox_xmax = bbox.xmax

        if bbox_xmax > plot_xmax:
            ax.set_xlim(xmax=AXES_PADDING_RATIO * bbox_xmax)

    return ax
