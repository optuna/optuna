from typing import Callable
from typing import List
from typing import Optional

import numpy as np

from optuna._experimental import experimental_func
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._param_importances import _get_importances_info
from optuna.visualization._param_importances import _ImportancesInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


_logger = get_logger(__name__)


AXES_PADDING_RATIO = 1.05


@experimental_func("2.2.0")
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
    """

    _imports.check()

    importances_info = _get_importances_info(study, evaluator, params, target, target_name)
    return _get_importances_plot(importances_info)


def _get_importances_plot(info: _ImportancesInfo) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Hyperparameter Importances")
    ax.set_xlabel(f"Importance for {info.target_name}")
    ax.set_ylabel("Hyperparameter")

    param_names = info.param_names
    pos = np.arange(len(param_names))
    importance_values = info.importance_values

    if len(importance_values) == 0:
        return ax

    # Draw horizontal bars.
    ax.barh(
        pos,
        importance_values,
        align="center",
        color=plt.get_cmap("tab20c")(0),
        tick_label=param_names,
    )

    renderer = fig.canvas.get_renderer()
    for idx, (val, label) in enumerate(zip(importance_values, info.importance_labels)):
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
