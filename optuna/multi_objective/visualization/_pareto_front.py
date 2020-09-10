import json
from typing import List
from typing import Optional

import optuna
from optuna._experimental import experimental
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.visualization._plotly_imports import _imports

if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = optuna.logging.get_logger(__name__)


@experimental("2.0.0")
def plot_pareto_front(
    study: MultiObjectiveStudy, names: Optional[List[str]] = None
) -> "go.Figure":
    """Plot the pareto front of a study.

    Example:

        The following code snippet shows how to plot the pareto front of a study.

        .. testcode::

            import optuna

            def objective(trial):
               x = trial.suggest_float("x", 0, 5)
               y = trial.suggest_float("y", 0, 3)

               v0 = 4 * x ** 2 + 4 * y ** 2
               v1 = (x - 5) ** 2 + (y - 5) ** 2
               return v0, v1

            study = optuna.multi_objective.create_study(["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            optuna.multi_objective.visualization.plot_pareto_front(study)

        .. raw:: html

            <iframe src="../../_static/plot_pareto_front.html" width="100%" height="500px"
            frameborder="0"></iframe>

    Args:
        study:
            A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object whose trials
            are plotted for their objective values.
        names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If the number of objectives of ``study`` isn't 2 or 3.
    """

    _imports.check()

    if study.n_objectives == 2:
        return _get_pareto_front_2d(study, names)
    elif study.n_objectives == 3:
        return _get_pareto_front_3d(study, names)
    else:
        raise ValueError("`plot_pareto_front` function only supports 2 or 3 objective studies.")


def _get_pareto_front_2d(study: MultiObjectiveStudy, names: Optional[List[str]]) -> "go.Figure":
    if names is None:
        names = ["Objective 0", "Objective 1"]
    elif len(names) != 2:
        raise ValueError("The length of `names` is supposed to be 2.")

    trials = study.get_pareto_front_trials()
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    data = go.Scatter(
        x=[t.values[0] for t in trials],
        y=[t.values[1] for t in trials],
        text=[_make_hovertext(t) for t in trials],
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
    )
    layout = go.Layout(title="Pareto-front Plot", xaxis_title=names[0], yaxis_title=names[1])
    return go.Figure(data=data, layout=layout)


def _get_pareto_front_3d(study: MultiObjectiveStudy, names: Optional[List[str]]) -> "go.Figure":
    if names is None:
        names = ["Objective 0", "Objective 1", "Objective 2"]
    elif len(names) != 3:
        raise ValueError("The length of `names` is supposed to be 3.")

    trials = study.get_pareto_front_trials()
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    data = go.Scatter3d(
        x=[t.values[0] for t in trials],
        y=[t.values[1] for t in trials],
        z=[t.values[2] for t in trials],
        text=[_make_hovertext(t) for t in trials],
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
    )
    layout = go.Layout(
        title="Pareto-front Plot",
        scene={"xaxis_title": names[0], "yaxis_title": names[1], "zaxis_title": names[2]},
    )
    return go.Figure(data=data, layout=layout)


def _make_hovertext(trial: FrozenMultiObjectiveTrial) -> str:
    text = json.dumps(
        {"number": trial.number, "values": trial.values, "params": trial.params}, indent=2
    )
    return text.replace("\n", "<br>")
