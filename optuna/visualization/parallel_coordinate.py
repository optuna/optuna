from collections import defaultdict
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional

from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

if is_available():
    from optuna.visualization.plotly_imports import go

logger = get_logger(__name__)


def plot_parallel_coordinate(study: Study, params: Optional[List[str]] = None) -> "go.Figure":
    """Plot the high-dimentional parameter relationships in a study.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the high-dimentional parameter relationships.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_parallel_coordinate(study, params=['x', 'y'])

        .. raw:: html

            <iframe src="../_static/plot_parallel_coordinate.html"
             width="100%" height="500px" frameborder="0">
            </iframe>

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _check_plotly_availability()
    return _get_parallel_coordinate_plot(study, params)


def _get_parallel_coordinate_plot(study: Study, params: Optional[List[str]] = None) -> "go.Figure":

    layout = go.Layout(title="Parallel Coordinate Plot",)

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        all_params = set(params)
    sorted_params = sorted(list(all_params))

    dims = [
        {
            "label": "Objective Value",
            "values": tuple([t.value for t in trials]),
            "range": (min([t.value for t in trials]), max([t.value for t in trials])),
        }
    ]  # type: List[Dict[str, Any]]
    for p_name in sorted_params:
        values = []
        for t in trials:
            if p_name in t.params:
                values.append(t.params[p_name])
        is_categorical = False
        try:
            tuple(map(float, values))
        except (TypeError, ValueError):
            vocab = defaultdict(lambda: len(vocab))  # type: DefaultDict[str, int]
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
                "colorbar": {"title": "Objective Value"},
                "showscale": True,
                "reversescale": study.direction == StudyDirection.MINIMIZE,
            },
        )
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure
