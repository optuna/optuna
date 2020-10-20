import json
from typing import List
from typing import Optional
from typing import Union

import optuna
from optuna import multi_objective
from optuna._deprecated import deprecated
from optuna._experimental import experimental
from optuna.multi_objective._selection import _get_pareto_front_trials
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = optuna.logging.get_logger(__name__)


@deprecated("2.3.0", "3.0.0")
@experimental("2.0.0")
def plot_pareto_front(
    study: Union["optuna.Study", MultiObjectiveStudy],
    names: Optional[List[str]] = None,
    include_dominated_trials: bool = False,
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

            <iframe src="../../../_static/plot_pareto_front.html" width="100%" height="500px"
            frameborder="0"></iframe>

    Args:
        study:
            A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object whose trials
            are plotted for their objective values.
        names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.


    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If the number of objectives of ``study`` isn't 2 or 3.
    """

    _imports.check()

    if study.n_objectives == 2:
        return _get_pareto_front_2d(study, names, include_dominated_trials)
    elif study.n_objectives == 3:
        return _get_pareto_front_3d(study, names, include_dominated_trials)
    else:
        raise ValueError("`plot_pareto_front` function only supports 2 or 3 objective studies.")


def _get_non_pareto_front_trials(
    study: Union["optuna.Study", MultiObjectiveStudy],
    pareto_trials: List[
        Union["optuna.trial.FrozenTrial", "multi_objective.trial.FrozenMultiObjectiveTrial"]
    ],
) -> List[Union["optuna.trial.FrozenTrial", "multi_objective.trial.FrozenMultiObjectiveTrial"]]:

    non_pareto_trials = []
    for trial in study.get_trials():
        if trial.state == TrialState.COMPLETE and trial not in pareto_trials:
            non_pareto_trials.append(trial)
    return non_pareto_trials


def _get_pareto_front_2d(
    study: Union["optuna.Study", MultiObjectiveStudy],
    names: Optional[List[str]],
    include_dominated_trials: bool = False,
) -> "go.Figure":
    if names is None:
        names = ["Objective 0", "Objective 1"]
    elif len(names) != 2:
        raise ValueError("The length of `names` is supposed to be 2.")

    trials = _get_pareto_front_trials(study)
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    point_colors = ["blue"] * len(trials)
    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        point_colors += ["red"] * len(non_pareto_trials)
        trials += non_pareto_trials

    value0 = []
    value1 = []
    for t in trials:
        if isinstance(t, FrozenMultiObjectiveTrial):  # Backwards compatibility.
            v0 = t.values[0]
            v1 = t.values[1]
        else:
            v0 = t.value[0]
            v1 = t.value[1]
        value0.append(v0)
        value1.append(v1)

    data = go.Scatter(
        x=value0,
        y=value1,
        text=[_make_hovertext(t) for t in trials],
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
        marker={"color": point_colors},
    )
    layout = go.Layout(title="Pareto-front Plot", xaxis_title=names[0], yaxis_title=names[1])
    return go.Figure(data=data, layout=layout)


def _get_pareto_front_3d(
    study: MultiObjectiveStudy, names: Optional[List[str]], include_dominated_trials: bool = False
) -> "go.Figure":
    if names is None:
        names = ["Objective 0", "Objective 1", "Objective 2"]
    elif len(names) != 3:
        raise ValueError("The length of `names` is supposed to be 3.")

    trials = _get_pareto_front_trials(study)
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    point_colors = ["blue"] * len(trials)
    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        point_colors += ["red"] * len(non_pareto_trials)
        trials += non_pareto_trials

    value0 = []
    value1 = []
    value2 = []
    for t in trials:
        if isinstance(t, FrozenMultiObjectiveTrial):  # Backwards compatibility.
            v0 = t.values[0]
            v1 = t.values[1]
            v2 = t.values[2]
        else:
            v0 = t.value[0]
            v1 = t.value[1]
            v2 = t.value[2]
        value0.append(v0)
        value1.append(v1)
        value2.append(v2)

    data = go.Scatter3d(
        x=value0,
        y=value1,
        z=value2,
        text=[_make_hovertext(t) for t in trials],
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
        marker={"color": point_colors},
    )
    layout = go.Layout(
        title="Pareto-front Plot",
        scene={"xaxis_title": names[0], "yaxis_title": names[1], "zaxis_title": names[2]},
    )
    return go.Figure(data=data, layout=layout)


def _make_hovertext(trial: Union["optuna.trial.FrozenTrial", FrozenMultiObjectiveTrial]) -> str:
    if isinstance(trial, FrozenMultiObjectiveTrial):  # Backwards compatibility.
        value = trial.values
    else:
        value = trial.value

    text = json.dumps({"number": trial.number, "values": value, "params": trial.params}, indent=2)
    return text.replace("\n", "<br>")
