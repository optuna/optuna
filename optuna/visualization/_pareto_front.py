import collections
import json
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import warnings

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.study import Study
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = optuna.logging.get_logger(__name__)


class _ParetoFrontInfo(NamedTuple):
    n_targets: int
    target_names: List[str]
    best_trials_with_values: List[Tuple[FrozenTrial, List[float]]]
    non_best_trials_with_values: List[Tuple[FrozenTrial, List[float]]]
    infeasible_trials_with_values: List[Tuple[FrozenTrial, List[float]]]
    axis_order: List[int]
    include_dominated_trials: bool
    has_constraints_func: bool


def plot_pareto_front(
    study: Study,
    *,
    target_names: Optional[List[str]] = None,
    include_dominated_trials: bool = True,
    axis_order: Optional[List[int]] = None,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
) -> "go.Figure":
    """Plot the Pareto front of a study.

    .. seealso::
        Please refer to :ref:`multi_objective` for the tutorial of the Pareto front visualization.

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
            values. ``study.n_objectives`` must be either 2 or 3 when ``targets`` is :obj:`None`.
        target_names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead. If ``targets`` is specified
            for a study that does not contain any completed trial,
            ``target_name`` must be specified.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.
        axis_order:
            A list of indices indicating the axis order. If :obj:`None` is specified,
            default order is used. ``axis_order`` and ``targets`` cannot be used at the same time.

            .. warning::
                Deprecated in v3.0.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v5.0.0, but this schedule is subject to
                change. See https://github.com/optuna/optuna/releases/tag/v3.0.0.
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

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.
        targets:
            A function that returns targets values to display.
            The argument to this function is :class:`~optuna.trial.FrozenTrial`.
            ``axis_order`` and ``targets`` cannot be used at the same time.
            If ``study.n_objectives`` is neither 2 nor 3, ``targets`` must be specified.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()

    info = _get_pareto_front_info(
        study, target_names, include_dominated_trials, axis_order, constraints_func, targets
    )
    return _get_pareto_front_plot(info)


def _get_pareto_front_plot(info: _ParetoFrontInfo) -> "go.Figure":
    include_dominated_trials = info.include_dominated_trials
    has_constraints_func = info.has_constraints_func
    if not has_constraints_func:
        data = [
            _make_scatter_object(
                info.n_targets,
                info.axis_order,
                include_dominated_trials,
                info.non_best_trials_with_values,
                hovertemplate="%{text}<extra>Trial</extra>",
                dominated_trials=True,
            ),
            _make_scatter_object(
                info.n_targets,
                info.axis_order,
                include_dominated_trials,
                info.best_trials_with_values,
                hovertemplate="%{text}<extra>Best Trial</extra>",
                dominated_trials=False,
            ),
        ]
    else:
        data = [
            _make_scatter_object(
                info.n_targets,
                info.axis_order,
                include_dominated_trials,
                info.infeasible_trials_with_values,
                hovertemplate="%{text}<extra>Infeasible Trial</extra>",
                infeasible=True,
            ),
            _make_scatter_object(
                info.n_targets,
                info.axis_order,
                include_dominated_trials,
                info.non_best_trials_with_values,
                hovertemplate="%{text}<extra>Feasible Trial</extra>",
                dominated_trials=True,
            ),
            _make_scatter_object(
                info.n_targets,
                info.axis_order,
                include_dominated_trials,
                info.best_trials_with_values,
                hovertemplate="%{text}<extra>Best Trial</extra>",
                dominated_trials=False,
            ),
        ]

    if info.n_targets == 2:
        layout = go.Layout(
            title="Pareto-front Plot",
            xaxis_title=info.target_names[info.axis_order[0]],
            yaxis_title=info.target_names[info.axis_order[1]],
        )
    else:
        layout = go.Layout(
            title="Pareto-front Plot",
            scene={
                "xaxis_title": info.target_names[info.axis_order[0]],
                "yaxis_title": info.target_names[info.axis_order[1]],
                "zaxis_title": info.target_names[info.axis_order[2]],
            },
        )
    return go.Figure(data=data, layout=layout)


def _get_pareto_front_info(
    study: Study,
    target_names: Optional[List[str]] = None,
    include_dominated_trials: bool = True,
    axis_order: Optional[List[int]] = None,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
) -> _ParetoFrontInfo:
    if axis_order is not None:
        warnings.warn(
            "`axis_order` has been deprecated in v3.0.0. "
            "This feature will be removed in v5.0.0. "
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0.",
            FutureWarning,
        )

    if targets is not None and axis_order is not None:
        raise ValueError(
            "Using both `targets` and `axis_order` is not supported. "
            "Use either `targets` or `axis_order`."
        )

    if constraints_func is not None:
        warnings.warn(
            "``constraints_func`` argument is an experimental feature."
            " The interface can change in the future.",
            ExperimentalWarning,
        )
        feasible_trials = []
        infeasible_trials = []
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
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
                study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), best_trials
            )
        else:
            non_best_trials = []
        infeasible_trials = []

    _targets = targets
    if _targets is None:
        if len(study.directions) in (2, 3):
            _targets = _targets_default
        else:
            raise ValueError(
                "`plot_pareto_front` function only supports 2 or 3 objective"
                " studies when using `targets` is `None`. Please use `targets`"
                " if your objective studies have more than 3 objectives."
            )

    def _make_trials_with_values(
        trials: List[FrozenTrial],
        targets: Callable[[FrozenTrial], Sequence[float]],
    ) -> List[Tuple[FrozenTrial, List[float]]]:
        target_values = [targets(trial) for trial in trials]
        for v in target_values:
            if not isinstance(v, collections.abc.Sequence):
                raise ValueError(
                    "`targets` should return a sequence of target values."
                    " your `targets` returns {}".format(type(v))
                )
        return [(trial, list(v)) for trial, v in zip(trials, target_values)]

    best_trials_with_values = _make_trials_with_values(best_trials, _targets)
    non_best_trials_with_values = _make_trials_with_values(non_best_trials, _targets)
    infeasible_trials_with_values = _make_trials_with_values(infeasible_trials, _targets)

    def _infer_n_targets(
        trials_with_values: Sequence[Tuple[FrozenTrial, Sequence[float]]]
    ) -> Optional[int]:
        if len(trials_with_values) > 0:
            return len(trials_with_values[0][1])
        return None

    # Check for `non_best_trials_with_values` can be skipped, because if `best_trials_with_values`
    # is empty, then `non_best_trials_with_values` will also be empty.
    n_targets = _infer_n_targets(best_trials_with_values) or _infer_n_targets(
        infeasible_trials_with_values
    )
    if n_targets is None:
        if target_names is not None:
            n_targets = len(target_names)
        elif targets is None:
            n_targets = len(study.directions)
        else:
            raise ValueError(
                "If `targets` is specified for empty studies, `target_names` must be specified."
            )

    if n_targets not in (2, 3):
        raise ValueError(
            "`plot_pareto_front` function only supports 2 or 3 targets."
            " you used {} targets now.".format(n_targets)
        )

    if target_names is None:
        target_names = [f"Objective {i}" for i in range(n_targets)]
    elif len(target_names) != n_targets:
        raise ValueError(f"The length of `target_names` is supposed to be {n_targets}.")

    if axis_order is None:
        axis_order = list(range(n_targets))
    else:
        if len(axis_order) != n_targets:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: {n_targets}, "
                f"Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != n_targets:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!.")
        if max(axis_order) > n_targets - 1:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                f"higher than {n_targets - 1}."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    return _ParetoFrontInfo(
        n_targets=n_targets,
        target_names=target_names,
        best_trials_with_values=best_trials_with_values,
        non_best_trials_with_values=non_best_trials_with_values,
        infeasible_trials_with_values=infeasible_trials_with_values,
        axis_order=axis_order,
        include_dominated_trials=include_dominated_trials,
        has_constraints_func=constraints_func is not None,
    )


def _targets_default(trial: FrozenTrial) -> Sequence[float]:
    return trial.values


def _get_non_pareto_front_trials(
    trials: List[FrozenTrial], pareto_trials: List[FrozenTrial]
) -> List[FrozenTrial]:

    non_pareto_trials = []
    for trial in trials:
        if trial not in pareto_trials:
            non_pareto_trials.append(trial)
    return non_pareto_trials


def _make_json_compatible(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        # The value can't be converted to JSON directly, so return a string representation.
        return str(value)


def _make_scatter_object(
    n_targets: int,
    axis_order: Sequence[int],
    include_dominated_trials: bool,
    trials_with_values: Sequence[Tuple[FrozenTrial, Sequence[float]]],
    hovertemplate: str,
    infeasible: bool = False,
    dominated_trials: bool = False,
) -> Union["go.Scatter", "go.Scatter3d"]:
    trials_with_values = trials_with_values or []

    marker = _make_marker(
        [trial for trial, _ in trials_with_values],
        include_dominated_trials,
        dominated_trials=dominated_trials,
        infeasible=infeasible,
    )
    if n_targets == 2:
        return go.Scatter(
            x=[values[axis_order[0]] for _, values in trials_with_values],
            y=[values[axis_order[1]] for _, values in trials_with_values],
            text=[_make_hovertext(trial) for trial, _ in trials_with_values],
            mode="markers",
            hovertemplate=hovertemplate,
            marker=marker,
            showlegend=False,
        )
    elif n_targets == 3:
        return go.Scatter3d(
            x=[values[axis_order[0]] for _, values in trials_with_values],
            y=[values[axis_order[1]] for _, values in trials_with_values],
            z=[values[axis_order[2]] for _, values in trials_with_values],
            text=[_make_hovertext(trial) for trial, _ in trials_with_values],
            mode="markers",
            hovertemplate=hovertemplate,
            marker=marker,
            showlegend=False,
        )
    else:
        assert False, "Must not reach here"


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
                "title": "Trial",
            },
        }
    else:
        return {
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials],
            "colorscale": "Reds",
            "colorbar": {
                "title": "Best Trial",
                "x": 1.1 if include_dominated_trials else 1,
                "xpad": 40,
            },
        }
