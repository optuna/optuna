from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.visualization._param_importances import _get_importances_infos
from optuna.visualization._param_importances import _ImportancesInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.importance._base import BaseImportanceEvaluator
    from optuna.study import Study
    from optuna.trial import FrozenTrial


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import Figure
    from optuna.visualization.matplotlib._matplotlib_imports import plt


_logger = get_logger(__name__)


AXES_PADDING_RATIO = 1.05


@experimental_func("2.2.0")
def plot_param_importances(
    study: Study,
    evaluator: BaseImportanceEvaluator | None = None,
    params: list[str] | None = None,
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot hyperparameter importances with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_param_importances` for an example.

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to :class:`~optuna.importance.PedAnovaImportanceEvaluator`.
            For details on this evaluator, please refer to the following papers:

            - `PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
              <https://arxiv.org/abs/2304.10255>`__ (IJCAI 2023)
            - `Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search
              Spaces <https://arxiv.org/abs/2601.20800>`__ (KDD 2026)

            When using this evaluator in your project, please cite both papers.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, :class:`~optuna.importance.PedAnovaImportanceEvaluator` assesses all
            parameters that appear in completed trials, including conditional parameters, while
            other evaluators assess parameters present in all completed trials.
            If specified, only the specified parameters are assessed.
            When using :class:`~optuna.importance.PedAnovaImportanceEvaluator`, each specified
            parameter must appear in at least one completed trial.
            When using other evaluators, at least one completed trial must contain all specified
            parameters.
        target:
            A function that returns the value used to evaluate and display importances.
            If :obj:`None`, objective values are used for single-objective optimization.
            For multi-objective optimization, all objectives will be plotted if ``target`` is
            :obj:`None`. Specify ``target``, for example ``target=lambda t: t.values[0]``, to
            plot importances for a specific objective.
        target_name:
            Target's name to display on the axis label. Names set via
            :meth:`~optuna.study.Study.set_metric_names` will be used if ``target`` is :obj:`None`,
            overriding this argument.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    importances_infos = _get_importances_infos(study, evaluator, params, target, target_name)
    return _get_importances_plot(importances_infos)


def _get_importances_plot(infos: tuple[_ImportancesInfo, ...]) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Hyperparameter Importances", loc="left")
    ax.set_xlabel("Hyperparameter Importance")
    ax.set_ylabel("Hyperparameter")
    height = 0.8 / len(infos)  # Default height split between objectives.

    for objective_id, info in enumerate(infos):
        param_names = info.param_names
        pos = np.arange(len(param_names))
        offset = height * objective_id
        importance_values = info.importance_values

        if not importance_values:
            continue

        # Draw horizontal bars.
        ax.barh(
            pos + offset,
            importance_values,
            height=height,
            align="center",
            label=info.target_name,
            color=plt.get_cmap("tab20c")(objective_id),
        )

        _set_bar_labels(info, fig, ax, offset)
        ax.set_yticks(pos + offset / 2, param_names)

    ax.legend(loc="best")
    return ax


def _set_bar_labels(info: _ImportancesInfo, fig: "Figure", ax: "Axes", offset: float) -> None:
    # Figure canvas does not necessarily have a get_renderer.
    assert hasattr(fig.canvas, "get_renderer")
    renderer = fig.canvas.get_renderer()
    for idx, (val, label) in enumerate(zip(info.importance_values, info.importance_labels)):
        text = ax.text(val, idx + offset, label, va="center")

        # Sometimes horizontal axis needs to be re-scaled
        # to avoid text going over plot area.
        bbox = text.get_window_extent(renderer)
        bbox = bbox.transformed(ax.transData.inverted())
        _, plot_xmax = ax.get_xlim()
        bbox_xmax = bbox.xmax

        if bbox_xmax > plot_xmax:
            ax.set_xlim(xmax=AXES_PADDING_RATIO * bbox_xmax)
