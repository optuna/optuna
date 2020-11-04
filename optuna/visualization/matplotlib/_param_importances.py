from collections import OrderedDict
from typing import List
from typing import Optional

import numpy as np

import optuna
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import cm
    from optuna.visualization.matplotlib._matplotlib_imports import plt
    from optuna.visualization.matplotlib._matplotlib_imports import Rectangle

    # Color definitions.
    cmap = cm.get_cmap("tab20c")
    _distribution_colors = {
        UniformDistribution: cmap(0),
        LogUniformDistribution: cmap(0),
        DiscreteUniformDistribution: cmap(0),
        IntUniformDistribution: cmap(1),
        IntLogUniformDistribution: cmap(1),
        CategoricalDistribution: cmap(3),
    }
    _legend_elements = [
        Rectangle([0, 0], 0, 0, facecolor=cmap(0), label="Uniform Distribution"),
        Rectangle([0, 0], 0, 0, facecolor=cmap(0), label="Log Uniform Distribution"),
        Rectangle([0, 0], 0, 0, facecolor=cmap(0), label="Discrete Uniform Distribution"),
        Rectangle([0, 0], 0, 0, facecolor=cmap(1), label="Int Uniform Distribution"),
        Rectangle([0, 0], 0, 0, facecolor=cmap(1), label="Int Log Uniform Distribution"),
        Rectangle([0, 0], 0, 0, facecolor=cmap(3), label="Categorical Distribution"),
    ]

_logger = get_logger(__name__)


@experimental("2.2.0")
def plot_param_importances(
    study: Study,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
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
            Defaults to
            :class:`~optuna.importance.FanovaImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    return _get_param_importance_plot(study, evaluator, params)


def _get_param_importance_plot(
    study: Study,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
) -> "Axes":

    # Set up the graph style.
    _, ax = plt.subplots()
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    ax.set_title("Hyperparameter Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Hyperparameter")

    # Prepare data for plotting.
    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        _logger.warning("Study instance does not contain completed trials.")
        return ax

    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params
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
        color=[_get_color(param_name, study) for param_name in param_names],
        tick_label=param_names,
    )
    ax.legend(handles=_legend_elements, title="Distributions", loc="lower right")
    return ax


def _get_distribution(param_name: str, study: Study) -> "BaseDistribution":
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False


def _get_color(param_name: str, study: Study) -> str:
    return _distribution_colors[type(_get_distribution(param_name, study))]
