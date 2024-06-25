from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import cast
from typing import NamedTuple

import numpy as np

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


NUM_SAMPLES_X_AXIS = 100


class _EDFLineInfo(NamedTuple):
    study_name: str
    y_values: np.ndarray


class _EDFInfo(NamedTuple):
    lines: list[_EDFLineInfo]
    x_values: np.ndarray


def plot_edf(
    study: Study | Sequence[Study],
    *,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the objective value EDF (empirical distribution function) of a study.

    Note that only the complete trials are considered when plotting the EDF.

    .. note::

        EDF is useful to analyze and improve search spaces.
        For instance, you can see a practical use case of EDF in the paper
        `Designing Network Design Spaces
        <https://doi.ieeecomputersociety.org/10.1109/CVPR42600.2020.01044>`__.

    .. note::

        The plotted EDF assumes that the value of the objective function is in
        accordance with the uniform distribution over the objective space.

    Example:

        The following code snippet shows how to plot EDF.

        .. plotly::

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


            sampler = optuna.samplers.RandomSampler(seed=10)

            # Widest search space.
            study0 = optuna.create_study(study_name="x=[0,5), y=[0,5)", sampler=sampler)
            study0.optimize(lambda t: objective(t, 0, 5), n_trials=500)

            # Narrower search space.
            study1 = optuna.create_study(study_name="x=[0,4), y=[0,4)", sampler=sampler)
            study1.optimize(lambda t: objective(t, 0, 4), n_trials=500)

            # Narrowest search space but it doesn't include the global optimum point.
            study2 = optuna.create_study(study_name="x=[1,3), y=[1,3)", sampler=sampler)
            study2.optimize(lambda t: objective(t, 1, 3), n_trials=500)

            fig = optuna.visualization.plot_edf([study0, study1, study2])
            fig.show()

    Args:
        study:
            A target :class:`~optuna.study.Study` object.
            You can pass multiple studies if you want to compare those EDFs.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()

    layout = go.Layout(
        title="Empirical Distribution Function Plot",
        xaxis={"title": target_name},
        yaxis={"title": "Cumulative Probability"},
    )

    info = _get_edf_info(study, target, target_name)
    edf_lines = info.lines

    if len(edf_lines) == 0:
        return go.Figure(data=[], layout=layout)

    traces = []
    for study_name, y_values in edf_lines:
        traces.append(go.Scatter(x=info.x_values, y=y_values, name=study_name, mode="lines"))

    figure = go.Figure(data=traces, layout=layout)
    figure.update_yaxes(range=[0, 1])

    return figure


def _get_edf_info(
    study: Study | Sequence[Study],
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
) -> _EDFInfo:
    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    _check_plot_args(studies, target, target_name)

    if len(studies) == 0:
        _logger.warning("There are no studies.")
        return _EDFInfo(lines=[], x_values=np.array([]))

    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    study_names = []
    all_values: list[np.ndarray] = []
    for study in studies:
        trials = _filter_nonfinite(
            study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
        )

        values = np.array([target(trial) for trial in trials])
        all_values.append(values)
        study_names.append(study.study_name)

    if all(len(values) == 0 for values in all_values):
        _logger.warning("There are no complete trials.")
        return _EDFInfo(lines=[], x_values=np.array([]))

    min_x_value = np.min(np.concatenate(all_values))
    max_x_value = np.max(np.concatenate(all_values))
    x_values = np.linspace(min_x_value, max_x_value, NUM_SAMPLES_X_AXIS)

    edf_line_info_list = []
    for study_name, values in zip(study_names, all_values):
        y_values = np.sum(values[:, np.newaxis] <= x_values, axis=0) / values.size
        edf_line_info_list.append(_EDFLineInfo(study_name=study_name, y_values=y_values))

    return _EDFInfo(lines=edf_line_info_list, x_values=x_values)
