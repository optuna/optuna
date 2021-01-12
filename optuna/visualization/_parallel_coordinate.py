from collections import defaultdict
from typing import Any
from typing import Callable
from typing import cast
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional

from optuna._study_direction import StudyDirection
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_parallel_coordinate(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the high-dimentional parameter relationships in a study.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the high-dimentional parameter relationships.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_parallel_coordinate(study, params=["x", "y"])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_parallel_coordinate_plot(study, params, target, target_name)


def _get_parallel_coordinate_plot(
    study: Study,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":

    layout = go.Layout(title="Parallel Coordinate Plot")

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        all_params = set(params)
    sorted_params = sorted(list(all_params))

    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target
        reversescale = study.direction == StudyDirection.MINIMIZE
    else:
        reversescale = True

    dims: List[Dict[str, Any]] = [
        {
            "label": target_name,
            "values": tuple([target(t) for t in trials]),
            "range": (min([target(t) for t in trials]), max([target(t) for t in trials])),
        }
    ]
    for p_name in sorted_params:
        values = []
        for t in trials:
            if p_name in t.params:
                values.append(t.params[p_name])
        is_categorical = False
        try:
            tuple(map(float, values))
        except (TypeError, ValueError):
            vocab: DefaultDict[str, int] = defaultdict(lambda: len(vocab))
            values = [vocab[v] for v in values]
            is_categorical = True
        dim = {
            "label": p_name if len(p_name) < 20 else "{}...".format(p_name[:17]),
            "values": tuple(values),
            "range": (min(values), max(values)),
        }
        if is_categorical:
            dim["tickvals"] = list(range(len(vocab)))
            dim["ticktext"] = list(sorted(vocab.items(), key=lambda x: x[1]))
        dims.append(dim)

    traces = [
        go.Parcoords(
            dimensions=dims,
            labelangle=30,
            labelside="bottom",
            line={
                "color": dims[0]["values"],
                "colorscale": "blues",
                "colorbar": {"title": target_name},
                "showscale": True,
                "reversescale": reversescale,
            },
        )
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure
