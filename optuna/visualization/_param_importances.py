from collections import OrderedDict
from typing import Callable
from typing import List
from typing import Optional

import optuna
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
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args


if _imports.is_successful():
    import plotly

    from optuna.visualization._plotly_imports import go

    Blues = plotly.colors.sequential.Blues

    _distribution_colors = {
        UniformDistribution: Blues[-1],
        LogUniformDistribution: Blues[-1],
        DiscreteUniformDistribution: Blues[-1],
        IntUniformDistribution: Blues[-2],
        IntLogUniformDistribution: Blues[-2],
        CategoricalDistribution: Blues[-4],
    }

logger = get_logger(__name__)


def plot_param_importances(
    study: Study,
    evaluator: Optional[BaseImportanceEvaluator] = None,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot hyperparameter importances.

    Example:

        The following code snippet shows how to plot hyperparameter importances.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_int("x", 0, 2)
                y = trial.suggest_float("y", -1.0, 1.0)
                z = trial.suggest_float("z", 0.0, 1.5)
                return x ** 2 + y ** 3 - z ** 4


            sampler = optuna.samplers.RandomSampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_param_importances(study)

    .. seealso::

        This function visualizes the results of :func:`optuna.importance.get_param_importances`.

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
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)

    layout = go.Layout(
        title="Hyperparameter Importances",
        xaxis={"title": f"Importance for {target_name}"},
        yaxis={"title": "Hyperparameter"},
        showlegend=False,
    )

    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return go.Figure(data=[], layout=layout)

    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params, target=target
    )

    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())

    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_values,
                y=param_names,
                text=importance_values,
                texttemplate="%{text:.2f}",
                textposition="outside",
                cliponaxis=False,  # Ensure text is not clipped.
                hovertemplate=[
                    _make_hovertext(param_name, importance, study)
                    for param_name, importance in importances.items()
                ],
                marker_color=[_get_color(param_name, study) for param_name in param_names],
                orientation="h",
            )
        ],
        layout=layout,
    )

    return fig


def _get_distribution(param_name: str, study: Study) -> BaseDistribution:
    for trial in study.trials:
        if param_name in trial.distributions:
            return trial.distributions[param_name]
    assert False


def _get_color(param_name: str, study: Study) -> str:
    return _distribution_colors[type(_get_distribution(param_name, study))]


def _make_hovertext(param_name: str, importance: float, study: Study) -> str:
    return "{} ({}): {}<extra></extra>".format(
        param_name, _get_distribution(param_name, study).__class__.__name__, importance
    )
