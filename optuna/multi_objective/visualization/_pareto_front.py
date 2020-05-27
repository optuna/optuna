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


def plot_pareto_front(study: MultiObjectiveStudy, names: Optional[List[str]] = None) -> go.Figure:
    _check_plotly_availability()

    if study.n_objectives == 2:
        return _get_pareto_front_2d(study, names)
    elif study.n_objectives == 3:
        return _get_pareto_front_3d(study, names)
    else:
        raise RuntimeError("`plot_pareto_front` function only supports 2 or 3 objective studies.")


def _get_pareto_front_2d(study: MultiObjectiveStudy, names: Optional[List[str]]) -> go.Figure:
    if names is None:
        names = ["Objective 0", "Objective 1"]
    if len(names) == 0:
        names.append("Objective 0")
    if len(names) == 1:
        names.append("Objective 1")

    trials = study.get_pareto_front_trials()

    if len(trials) == 0:
        logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    data = go.Scatter(
        x=[t.values[0] for t in trials],
        y=[t.values[1] for t in trials],
        text=["NUMBER: {}, PARAMS:{}".format(t.number, t.params) for t in trials],
        mode="markers",
        showlegend=False,
    )
    layout = go.Layout(
        title="Pareto-front Plot", xaxis={"title": names[0]}, yaxis={"title": names[1]}
    )
    return go.Figure(data=data, layout=layout)


def _get_pareto_front_3d(study: MultiObjectiveStudy, names: Optional[List[str]]) -> go.Figure:
    if names is None:
        names = ["Objective 0", "Objective 1", "Objective 2"]
    if len(names) == 0:
        names.append("Objective 0")
    if len(names) == 1:
        names.append("Objective 1")
    if len(names) == 2:
        names.append("Objective 2")

    trials = study.get_pareto_front_trials()

    if len(trials) == 0:
        logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    data = go.Scatter3d(
        x=[t.values[0] for t in trials],
        y=[t.values[1] for t in trials],
        z=[t.values[2] for t in trials],
        text=[
            "<br>NUMBER: {}<br>PARAMS:<br>{}".format(
                t.number, json.dumps(t.params, indent=2).replace("\n", "<br>")
            )
            for t in trials
        ],
        mode="markers",
        showlegend=False,
    )
    layout = go.Layout(
        title="Pareto-front Plot",
        scene={"xaxis_title": names[0], "yaxis_title": names[1], "zaxis_title": names[2]},
    )
    return go.Figure(data=data, layout=layout)
