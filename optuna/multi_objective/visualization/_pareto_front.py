import json
from typing import List
from typing import Optional

import optuna
from optuna import multi_objective
from optuna._deprecated import deprecated_func
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = optuna.logging.get_logger(__name__)


@deprecated_func("2.4.0", "4.0.0")
def plot_pareto_front(
    study: MultiObjectiveStudy,
    names: Optional[List[str]] = None,
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "go.Figure":
    """Plot the pareto front of a study.

    Example:

        The following code snippet shows how to plot the pareto front of a study.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x ** 2 + 4 * y ** 2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.multi_objective.create_study(["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            fig = optuna.multi_objective.visualization.plot_pareto_front(study)
            fig.show()

    Args:
        study:
            A :class:`~optuna.multi_objective.study.MultiObjectiveStudy` object whose trials
            are plotted for their objective values. ``study.n_objectives`` must be eigher 2 or 3.
        names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.
        axis_order:
            A list of indices indicating the axis order. If :obj:`None` is specified,
            default order is used.


    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    """

    _imports.check()

    if study.n_objectives == 2:
        return _get_pareto_front_2d(study, names, include_dominated_trials, axis_order)
    elif study.n_objectives == 3:
        return _get_pareto_front_3d(study, names, include_dominated_trials, axis_order)
    else:
        raise ValueError("`plot_pareto_front` function only supports 2 or 3 objective studies.")


def _get_non_pareto_front_trials(
    study: MultiObjectiveStudy,
    pareto_trials: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
) -> List["multi_objective.trial.FrozenMultiObjectiveTrial"]:
    non_pareto_trials = []
    for trial in study.get_trials():
        if trial.state == TrialState.COMPLETE and trial not in pareto_trials:
            non_pareto_trials.append(trial)
    return non_pareto_trials


def _get_pareto_front_2d(
    study: MultiObjectiveStudy,
    names: Optional[List[str]],
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "go.Figure":
    if names is None:
        names = ["Objective 0", "Objective 1"]
    elif len(names) != 2:
        raise ValueError("The length of `names` is supposed to be 2.")

    trials = study.get_pareto_front_trials()
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    point_colors = ["blue"] * len(trials)
    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        point_colors += ["red"] * len(non_pareto_trials)
        trials += non_pareto_trials

    if axis_order is None:
        axis_order = list(range(2))
    else:
        if len(axis_order) != 2:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: 2, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != 2:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!")
        if max(axis_order) > 1:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                "higher than 1."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    data = go.Scatter(
        x=[t.values[axis_order[0]] for t in trials],
        y=[t.values[axis_order[1]] for t in trials],
        text=[_make_hovertext(t) for t in trials],
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
        marker={"color": point_colors},
    )
    layout = go.Layout(
        title="Pareto-front Plot",
        xaxis_title=names[axis_order[0]],
        yaxis_title=names[axis_order[1]],
    )
    return go.Figure(data=data, layout=layout)


def _get_pareto_front_3d(
    study: MultiObjectiveStudy,
    names: Optional[List[str]],
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "go.Figure":
    if names is None:
        names = ["Objective 0", "Objective 1", "Objective 2"]
    elif len(names) != 3:
        raise ValueError("The length of `names` is supposed to be 3.")

    trials = study.get_pareto_front_trials()
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    point_colors = ["blue"] * len(trials)
    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        point_colors += ["red"] * len(non_pareto_trials)
        trials += non_pareto_trials

    if axis_order is None:
        axis_order = list(range(3))
    else:
        if len(axis_order) != 3:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: 3, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != 3:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!.")
        if max(axis_order) > 2:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                "higher than 2."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    data = go.Scatter3d(
        x=[t.values[axis_order[0]] for t in trials],
        y=[t.values[axis_order[1]] for t in trials],
        z=[t.values[axis_order[2]] for t in trials],
        text=[_make_hovertext(t) for t in trials],
        mode="markers",
        hovertemplate="%{text}<extra></extra>",
        marker={"color": point_colors},
    )
    layout = go.Layout(
        title="Pareto-front Plot",
        scene={
            "xaxis_title": names[axis_order[0]],
            "yaxis_title": names[axis_order[1]],
            "zaxis_title": names[axis_order[2]],
        },
    )
    return go.Figure(data=data, layout=layout)


def _make_hovertext(trial: FrozenMultiObjectiveTrial) -> str:
    text = json.dumps(
        {"number": trial.number, "values": trial.values, "params": trial.params}, indent=2
    )
    return text.replace("\n", "<br>")
