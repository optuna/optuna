from typing import List
from typing import Optional

import optuna
from optuna._experimental import experimental
from optuna.importance._base import BaseImportanceEvaluator
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

if is_available():
    from optuna.visualization.plotly_imports import go

logger = get_logger(__name__)


@experimental("1.5.0")
def plot_param_importances(
    study: Study, evaluator: BaseImportanceEvaluator = None, params: Optional[List[str]] = None
) -> "go.Figure":
    """Plot hyperparameter importances.

    Example:

        The following code snippet shows how to plot hyperparameter importances.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_int("x", 0, 2)
                y = trial.suggest_float("y", -1.0, 1.0)
                z = trial.suggest_float("z", 0.0, 1.5)
                return x ** 2 + y ** 3 - z ** 4

            study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_param_importances(study)

        .. raw:: html

            <iframe src="../_static/plot_param_importances.html"
             width="100%" height="500px" frameborder="0">
            </iframe>

    .. seealso::

        This function visualizes the results of :func:`optuna.importance.get_param_importances`.

    Args:
        study:
            An optimized study.
        evaluator:
            An importance evaluator object that specifies which algorithm to base the importance
            assessment on.
            Defaults to
            :class:`~optuna.importance._mean_decrease_impurity.MeanDecreaseImpurityImportanceEvaluator`.
        params:
            A list of names of parameters to assess.
            If :obj:`None`, all parameters that are present in all of the completed trials are
            assessed.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _check_plotly_availability()

    layout = go.Layout(
        title="Hyperparameter Importances",
        xaxis={"title": "Feature"},
        yaxis={"title": "Importance"},
        showlegend=False,
    )

    # Importances cannot be evaluated without completed trials.
    # Return an empty figure for consistency with other visualization functions.
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if len(trials) == 0:
        logger.warning("Study instance does not contain completed trials.")
        return go.Figure(data=[], layout=layout)

    importances = optuna.importance.get_param_importances(
        study, evaluator=evaluator, params=params
    )

    fig = go.Figure(
        data=[go.Bar(x=list(importances.keys()), y=list(importances.values()))], layout=layout
    )

    return fig
