import itertools
from typing import List
from typing import Sequence
from typing import Union

import numpy as np

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports

if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_edf(study: Union[Study, Sequence[Study]]) -> "go.Figure":
    """Plot the objective value EDF (empirical distribution function) of a study.

    Note that only the complete trials are considered when plotting the EDF.

    .. note::

        EDF is useful to analyze and improve search spaces.
        For instance, you can see a practical use case of EDF in the paper
        `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    .. note::

        The plotted EDF assumes that the value of the objective function is in
        accordance with the uniform distribution over the objective space.

    Example:

        The following code snippet shows how to plot EDF.

        .. testcode::

            import math

            import optuna


            def ackley(x, y):
                a = 20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2)))
                b = math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
                return -a - b + math.e + 20


            def objective(trial, low, high):
                x = trial.suggest_float("x", low, high)
                y = trial.suggest_float("y", low, high)
                return ackley(x, y)


            sampler = optuna.samplers.RandomSampler()

            # Widest search space.
            study0 = optuna.create_study(study_name="x=[0,5), y=[0,5)", sampler=sampler)
            study0.optimize(lambda t: objective(t, 0, 5), n_trials=500)

            # Narrower search space.
            study1 = optuna.create_study(study_name="x=[0,4), y=[0,4)", sampler=sampler)
            study1.optimize(lambda t: objective(t, 0, 4), n_trials=500)

            # Narrowest search space but it doesn't include the global optimum point.
            study2 = optuna.create_study(study_name="x=[1,3), y=[1,3)", sampler=sampler)
            study2.optimize(lambda t: objective(t, 1, 3), n_trials=500)

            optuna.visualization.plot_edf([study0, study1, study2])

        .. raw:: html

            <iframe src="../../_static/plot_edf.html"
             width="100%" height="500px" frameborder="0">
            </iframe>

    Args:
        study:
            A target :class:`~optuna.study.Study` object.
            You can pass multiple studies if you want to compare those EDFs.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()

    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    return _get_edf_plot(studies)


def _get_edf_plot(studies: List[Study]) -> "go.Figure":
    layout = go.Layout(
        title="Empirical Distribution Function Plot",
        xaxis={"title": "Objective Value"},
        yaxis={"title": "Cumulative Probability"},
    )

    if len(studies) == 0:
        _logger.warning("There are no studies.")
        return go.Figure(data=[], layout=layout)

    all_trials = list(
        itertools.chain.from_iterable(
            (
                trial
                for trial in study.get_trials(deepcopy=False)
                if trial.state == TrialState.COMPLETE
            )
            for study in studies
        )
    )

    if len(all_trials) == 0:
        _logger.warning("There are no complete trials.")
        return go.Figure(data=[], layout=layout)

    min_x_value = min(trial.value for trial in all_trials)
    max_x_value = max(trial.value for trial in all_trials)
    x_values = np.linspace(min_x_value, max_x_value, 100)

    traces = []
    for study in studies:
        values = np.asarray(
            [
                trial.value
                for trial in study.get_trials(deepcopy=False)
                if trial.state == TrialState.COMPLETE
            ]
        )

        y_values = np.sum(values[:, np.newaxis] <= x_values, axis=0) / values.size

        traces.append(go.Scatter(x=x_values, y=y_values, name=study.study_name, mode="lines"))

    figure = go.Figure(data=traces, layout=layout)
    figure.update_yaxes(range=[0, 1])

    return figure
