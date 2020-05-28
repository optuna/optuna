import json
from typing import List
from typing import Optional

from optuna.logging import get_logger
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

if is_available():
    from optuna.visualization.plotly_imports import go

logger = get_logger(__name__)


def plot_pareto_front(
    study: MultiObjectiveStudy, names: Optional[List[str]] = None
) -> "go.Figure":
    _check_plotly_availability()

    if study.n_objectives == 2:
        return _get_pareto_front_2d(study, names)
    elif study.n_objectives == 3:
        return _get_pareto_front_3d(study, names)
    else:
        raise RuntimeError("`plot_pareto_front` function only supports 2 or 3 objective studies.")


def _get_pareto_front_2d(study: MultiObjectiveStudy, names: Optional[List[str]]) -> "go.Figure":
    names = _fill_objective_names(2, names)
    trials = study.get_pareto_front_trials()

    if len(trials) == 0:
        raise ValueError("There must be one or more completed trials to plot a study.")

    data = go.Scatter(
        x=[t.values[0] for t in trials],
        y=[t.values[1] for t in trials],
        text=[_hovertext(t) for t in trials],
        mode="markers",
        showlegend=False,
    )
    layout = go.Layout(
        title="Pareto-front Plot", scene={"xaxis_title": names[0], "yaxis_title": names[1]},
    )
    return go.Figure(data=data, layout=layout)


def _get_pareto_front_3d(study: MultiObjectiveStudy, names: Optional[List[str]]) -> "go.Figure":
    names = _fill_objective_names(3, names)
    trials = study.get_pareto_front_trials()

    if len(trials) == 0:
        raise ValueError("There must be one or more completed trials to plot a study.")

    data = go.Scatter3d(
        x=[t.values[0] for t in trials],
        y=[t.values[1] for t in trials],
        z=[t.values[2] for t in trials],
        text=[_hovertext(t) for t in trials],
        mode="markers",
        showlegend=False,
    )
    layout = go.Layout(
        title="Pareto-front Plot",
        scene={"xaxis_title": names[0], "yaxis_title": names[1], "zaxis_title": names[2]},
    )
    return go.Figure(data=data, layout=layout)


def _fill_objective_names(n_objectives: int, names: Optional[List[str]]) -> List[str]:
    if names is None:
        names = []
    for i in range(n_objectives):
        if len(names) == i:
            names.append("Objective {}".format(i))
    return names


def _hovertext(trial: FrozenMultiObjectiveTrial) -> str:
    return "NUMBER: {}<br>PARAMS:<br>{}".format(
        trial.number, json.dumps(trial.params, indent=2).replace("\n", "<br>")
    )
