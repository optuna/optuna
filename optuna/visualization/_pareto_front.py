import json
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import optuna
from optuna._experimental import experimental
from optuna.study import Study
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = optuna.logging.get_logger(__name__)


@experimental("2.4.0")
def plot_pareto_front(
    study: Study,
    *,
    target_names: Optional[List[str]] = None,
    include_dominated_trials: bool = True,
    axis_order: Optional[List[int]] = None,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
) -> "go.Figure":
    """Plot the Pareto front of a study.

    Example:

        The following code snippet shows how to plot the Pareto front of a study.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x ** 2 + 4 * y ** 2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            fig = optuna.visualization.plot_pareto_front(study)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        target_names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.
        axis_order:
            A list of indices indicating the axis order. If :obj:`None` is specified,
            default order is used.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.
            This specification is the same as in, for example,
            :class:`~optuna.integration.NSGAIISampler`.

            If given, trials are classified into three categories: feasible and best, feasible but
            non-best, and infeasible. Categories are shown in different colors. Here, whether a
            trial is best (on Pareto front) or not is determined ignoring all infeasible trials.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If the number of objectives of ``study`` isn't 2 or 3.
    """

    _imports.check()

    n_dim = len(study.directions)
    if n_dim not in (2, 3):
        raise ValueError("`plot_pareto_front` function only supports 2 or 3 objective studies.")

    if target_names is None:
        target_names = [f"Objective {i}" for i in range(n_dim)]
    elif len(target_names) != n_dim:
        raise ValueError(f"The length of `target_names` is supposed to be {n_dim}.")

    if constraints_func is not None:
        feasible_trials = []
        infeasible_trials = []
        for trial in study.get_trials(states=(TrialState.COMPLETE,)):
            if all(map(lambda x: x <= 0.0, constraints_func(trial))):
                feasible_trials.append(trial)
            else:
                infeasible_trials.append(trial)
        best_trials = _get_pareto_front_trials_by_trials(feasible_trials, study.directions)
        if include_dominated_trials:
            non_best_trials = _get_non_pareto_front_trials(feasible_trials, best_trials)
        else:
            non_best_trials = []

        if len(best_trials) == 0:
            _logger.warning("Your study does not have any completed and feasible trials.")
    else:
        best_trials = study.best_trials
        if len(best_trials) == 0:
            _logger.warning("Your study does not have any completed trials.")

        if include_dominated_trials:
            non_best_trials = _get_non_pareto_front_trials(
                study.get_trials(deepcopy=False), best_trials
            )
        else:
            non_best_trials = []
        infeasible_trials = []

    if axis_order is None:
        axis_order = list(range(n_dim))
    else:
        if len(axis_order) != n_dim:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: {n_dim}, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != n_dim:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!.")
        if max(axis_order) > n_dim - 1:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                f"higher than {n_dim - 1}."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    def _make_scatter_object(
        trials: Sequence[FrozenTrial],
        hovertemplate: str,
        infeasible: bool = False,
        dominated_trials: bool = False,
    ) -> Union["go.Scatter", "go.Scatter3d"]:
        return _make_scatter_object_base(
            n_dim,
            trials,
            axis_order,  # type: ignore
            include_dominated_trials,
            hovertemplate=hovertemplate,
            infeasible=infeasible,
            dominated_trials=dominated_trials,
        )

    if constraints_func is None:
        data = [
            _make_scatter_object(
                non_best_trials,
                hovertemplate="%{text}<extra>Trial</extra>",
                dominated_trials=True,
            ),
            _make_scatter_object(
                best_trials,
                hovertemplate="%{text}<extra>Best Trial</extra>",
                dominated_trials=False,
            ),
        ]
    else:
        data = [
            _make_scatter_object(
                infeasible_trials,
                hovertemplate="%{text}<extra>Infeasible Trial</extra>",
                infeasible=True,
            ),
            _make_scatter_object(
                non_best_trials,
                hovertemplate="%{text}<extra>Feasible Trial</extra>",
                dominated_trials=True,
            ),
            _make_scatter_object(
                best_trials,
                hovertemplate="%{text}<extra>Best Trial</extra>",
                dominated_trials=False,
            ),
        ]

    if n_dim == 2:
        layout = go.Layout(
            title="Pareto-front Plot",
            xaxis_title=target_names[axis_order[0]],
            yaxis_title=target_names[axis_order[1]],
        )
    else:
        layout = go.Layout(
            title="Pareto-front Plot",
            scene={
                "xaxis_title": target_names[axis_order[0]],
                "yaxis_title": target_names[axis_order[1]],
                "zaxis_title": target_names[axis_order[2]],
            },
        )
    return go.Figure(data=data, layout=layout)


def _get_non_pareto_front_trials(
    trials: List[FrozenTrial], pareto_trials: List[FrozenTrial]
) -> List[FrozenTrial]:

    non_pareto_trials = []
    for trial in trials:
        if trial.state == TrialState.COMPLETE and trial not in pareto_trials:
            non_pareto_trials.append(trial)
    return non_pareto_trials


def _make_json_compatible(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        # the value can't be converted to JSON directly, so return a string representation
        return str(value)


def _make_scatter_object_base(
    n_dim: int,
    trials: Sequence[FrozenTrial],
    axis_order: List[int],
    include_dominated_trials: bool,
    hovertemplate: str,
    infeasible: bool = False,
    dominated_trials: bool = False,
) -> Union["go.Scatter", "go.Scatter3d"]:
    assert n_dim in (2, 3)
    marker = _make_marker(
        trials,
        include_dominated_trials,
        dominated_trials=dominated_trials,
        infeasible=infeasible,
    )
    if n_dim == 2:
        return go.Scatter(
            x=[t.values[axis_order[0]] for t in trials],
            y=[t.values[axis_order[1]] for t in trials],
            text=[_make_hovertext(t) for t in trials],
            mode="markers",
            hovertemplate=hovertemplate,
            marker=marker,
            showlegend=False,
        )
    else:
        assert n_dim == 3
        return go.Scatter3d(
            x=[t.values[axis_order[0]] for t in trials],
            y=[t.values[axis_order[1]] for t in trials],
            z=[t.values[axis_order[2]] for t in trials],
            text=[_make_hovertext(t) for t in trials],
            mode="markers",
            hovertemplate=hovertemplate,
            marker=marker,
            showlegend=False,
        )


def _make_hovertext(trial: FrozenTrial) -> str:
    user_attrs = {key: _make_json_compatible(value) for key, value in trial.user_attrs.items()}
    user_attrs_dict = {"user_attrs": user_attrs} if user_attrs else {}
    text = json.dumps(
        {
            "number": trial.number,
            "values": trial.values,
            "params": trial.params,
            **user_attrs_dict,
        },
        indent=2,
    )
    return text.replace("\n", "<br>")


def _make_marker(
    trials: Sequence[FrozenTrial],
    include_dominated_trials: bool,
    dominated_trials: bool = False,
    infeasible: bool = False,
) -> Dict[str, Any]:
    if dominated_trials and not include_dominated_trials:
        assert len(trials) == 0

    if infeasible:
        return {
            "color": "#cccccc",
        }
    elif dominated_trials:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Blues",
            "colorbar": {
                "title": "#Trials",
            },
        }
    else:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Reds",
            "colorbar": {
                "title": "#Best trials",
                "x": 1.1 if include_dominated_trials else 1,
                "xpad": 40,
            },
        }
