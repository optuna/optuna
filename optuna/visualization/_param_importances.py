from collections import OrderedDict
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional

import optuna
from optuna.distributions import BaseDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite


if _imports.is_successful():
    import plotly

    from optuna.visualization._plotly_imports import go


logger = get_logger(__name__)


class _ImportancesInfo(NamedTuple):
    importance_values: List[float]
    param_names: List[str]
    importance_labels: List[str]
    target_name: str


def _get_importances_info(
    study: Study,
    evaluator: Optional[BaseImportanceEvaluator],
    params: Optional[List[str]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
) -> _ImportancesInfo:
    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return _ImportancesInfo(
            importance_values=[],
            param_names=[],
            importance_labels=[],
            target_name=target_name,
        )

    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params, target=target
    )

    importances = OrderedDict(reversed(list(importances.items())))
    importance_values = list(importances.values())
    param_names = list(importances.keys())
    importance_labels = [f"{val:.2f}" if val >= 0.01 else "<0.01" for val in importance_values]

    return _ImportancesInfo(
        importance_values=importance_values,
        param_names=param_names,
        importance_labels=importance_labels,
        target_name=target_name,
    )


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

            fig = optuna.visualization.plot_param_importances(study)
            fig.show()

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
                Specify this argument if ``study`` is being used for multi-objective
                optimization. For example, to get the hyperparameter importance of the first
                objective, use ``target=lambda t: t.values[0]`` for the target parameter.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()

    importances_info = _get_importances_info(study, evaluator, params, target, target_name)
    hover_template = _get_hover_template(importances_info, study)
    return _get_importances_plot(importances_info, hover_template)


def _get_importances_plot(info: _ImportancesInfo, hover_template: List[str]) -> "go.Figure":

    layout = go.Layout(
        title="Hyperparameter Importances",
        xaxis={"title": f"Importance for {info.target_name}"},
        yaxis={"title": "Hyperparameter"},
        showlegend=False,
    )

    param_names = info.param_names
    importance_values = info.importance_values

    if len(importance_values) == 0:
        return go.Figure(data=[], layout=layout)

    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_values,
                y=param_names,
                text=info.importance_labels,
                textposition="outside",
                cliponaxis=False,  # Ensure text is not clipped.
                hovertemplate=hover_template,
                marker_color=plotly.colors.sequential.Blues[-4],
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


def _make_hovertext(param_name: str, importance: float, study: Study) -> str:
    return "{} ({}): {}<extra></extra>".format(
        param_name, _get_distribution(param_name, study).__class__.__name__, importance
    )


def _get_hover_template(importances_info: _ImportancesInfo, study: Study) -> List[str]:
    return [
        _make_hovertext(param_name, importance, study)
        for param_name, importance in zip(
            importances_info.param_names, importances_info.importance_values
        )
    ]
