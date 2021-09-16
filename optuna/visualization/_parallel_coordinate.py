from collections import defaultdict
import math
from typing import Any
from typing import Callable
from typing import cast
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna.logging import get_logger
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _is_categorical
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical


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
    """Plot the high-dimensional parameter relationships in a study.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the high-dimensional parameter relationships.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            fig = optuna.visualization.plot_parallel_coordinate(study, params=["x", "y"])
            fig.show()

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
    sorted_params = sorted(all_params)

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

    numeric_cat_params_indices: List[int] = []
    for dim_index, p_name in enumerate(sorted_params, start=1):
        values = []
        for t in trials:
            if p_name in t.params:
                values.append(t.params[p_name])

        if _is_log_scale(trials, p_name):
            values = [math.log10(v) for v in values]
            min_value = min(values)
            max_value = max(values)
            tickvals = list(range(math.ceil(min_value), math.ceil(max_value)))
            if min_value not in tickvals:
                tickvals = [min_value] + tickvals
            if max_value not in tickvals:
                tickvals = tickvals + [max_value]
            dim = {
                "label": p_name if len(p_name) < 20 else "{}...".format(p_name[:17]),
                "values": tuple(values),
                "range": (min_value, max_value),
                "tickvals": tickvals,
                "ticktext": ["{:.3g}".format(math.pow(10, x)) for x in tickvals],
            }
        elif _is_categorical(trials, p_name):
            vocab: DefaultDict[str, int] = defaultdict(lambda: len(vocab))

            if _is_numerical(trials, p_name):
                _ = [vocab[v] for v in sorted(values)]
                values = [vocab[v] for v in values]
                ticktext = list(sorted(vocab.keys()))
                numeric_cat_params_indices.append(dim_index)
            else:
                values = [vocab[v] for v in values]
                ticktext = list(sorted(vocab.keys(), key=lambda x: vocab[x]))

            dim = {
                "label": p_name if len(p_name) < 20 else "{}...".format(p_name[:17]),
                "values": tuple(values),
                "range": (min(values), max(values)),
                "tickvals": list(range(len(vocab))),
                "ticktext": ticktext,
            }
        else:
            dim = {
                "label": p_name if len(p_name) < 20 else "{}...".format(p_name[:17]),
                "values": tuple(values),
                "range": (min(values), max(values)),
            }

        dims.append(dim)

    if numeric_cat_params_indices:
        # np.lexsort consumes the sort keys the order from back to front.
        # So the values of parameters have to be reversed the order.
        idx = np.lexsort([dims[index]["values"] for index in numeric_cat_params_indices][::-1])
        for dim in dims:
            # Since the values are mapped to other categories by the index,
            # the index will be swapped according to the sorted index of numeric params.
            dim.update({"values": tuple(np.array(dim["values"])[idx])})

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
