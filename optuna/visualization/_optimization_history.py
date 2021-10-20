import itertools
from typing import Callable
from typing import cast
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from optuna.logging import get_logger
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_optimization_history(
    study: Union[Study, Sequence[Study]],
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
    error_bar: bool = False,
) -> "go.Figure":
    """Plot optimization history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            fig = optuna.visualization.plot_optimization_history(study)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
            You can pass multiple studies if you want to compare those optimization histories.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.
        error_bar:
            A flag to show the error bar.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()

    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    _check_plot_args(studies, target, target_name)
    return _get_optimization_history_plot(studies, target, target_name, error_bar)


def _get_optimization_history_plot(
    studies: List[Study],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    error_bar: bool,
) -> "go.Figure":

    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "#Trials"},
        yaxis={"title": target_name},
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

    if error_bar:
        return _get_optimization_histories_with_error_bar(studies, target, target_name, layout)
    else:
        return _get_optimization_histories(studies, target, target_name, layout)


def _get_optimization_histories_with_error_bar(
    studies: List[Study],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    layout: "go.Layout",
) -> "go.Figure":
    max_trial_number = np.max(
        [
            trial.number
            for study in studies
            for trial in study.get_trials(states=(TrialState.COMPLETE,))
        ]
    )

    _target: Callable[[FrozenTrial], float]
    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

    else:
        _target = target

    target_values: List[List[float]] = [[] for _ in range(max_trial_number + 2)]
    for study in studies:
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        for t in trials:
            target_values[t.number].append(_target(t))

    mean_of_target_values = [np.mean(v) if len(v) > 0 else None for v in target_values]
    std_of_target_values = [np.std(v) if len(v) > 0 else None for v in target_values]
    trial_numbers = np.arange(max_trial_number + 2)[[v is not None for v in mean_of_target_values]]
    means = np.asarray(mean_of_target_values)[trial_numbers]
    stds = np.asarray(std_of_target_values)[trial_numbers]
    traces = [
        go.Scatter(
            x=trial_numbers,
            y=means,
            error_y={
                "type": "data",
                "array": stds,
                "visible": True,
            },
            mode="markers",
            name=target_name,
        )
    ]

    if target is None:
        best_values: List[List[float]] = [[] for _ in range(max_trial_number + 2)]
        for study in studies:
            trials = study.get_trials(states=(TrialState.COMPLETE,))

            if study.direction == StudyDirection.MINIMIZE:
                best_vs = np.minimum.accumulate([cast(float, t.value) for t in trials])
            else:
                best_vs = np.maximum.accumulate([cast(float, t.value) for t in trials])

            for i, t in enumerate(trials):
                best_values[t.number].append(best_vs[i])

        mean_of_best_values = [np.mean(v) if len(v) > 0 else None for v in best_values]
        std_of_best_values = [np.std(v) if len(v) > 0 else None for v in best_values]
        means = np.asarray(mean_of_best_values)[trial_numbers]
        stds = np.asarray(std_of_best_values)[trial_numbers]
        traces.append(go.Scatter(x=trial_numbers, y=means, name="Best Value"))
        traces.append(
            go.Scatter(
                x=trial_numbers,
                y=means + stds,
                mode="lines",
                line=dict(width=0.01),
                showlegend=False,
            )
        )
        traces.append(
            go.Scatter(
                x=trial_numbers,
                y=means - stds,
                mode="none",
                showlegend=False,
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.2)",
            )
        )

    figure = go.Figure(data=traces, layout=layout)

    return figure


def _get_optimization_histories(
    studies: List[Study],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    layout: "go.Layout",
) -> "go.Figure":

    traces = []
    for study in studies:
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        if target is None:
            if study.direction == StudyDirection.MINIMIZE:
                best_values = np.minimum.accumulate([cast(float, t.value) for t in trials])
            else:
                best_values = np.maximum.accumulate([cast(float, t.value) for t in trials])
            traces.append(
                go.Scatter(
                    x=[t.number for t in trials],
                    y=[t.value for t in trials],
                    mode="markers",
                    name=target_name
                    if len(studies) == 1
                    else f"{target_name} of {study.study_name}",
                )
            )
            traces.append(
                go.Scatter(
                    x=[t.number for t in trials],
                    y=best_values,
                    name="Best Value"
                    if len(studies) == 1
                    else f"Best Value of {study.study_name}",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[t.number for t in trials],
                    y=[target(t) for t in trials],
                    mode="markers",
                    name=target_name
                    if len(studies) == 1
                    else f"{target_name} of {study.study_name}",
                )
            )

    figure = go.Figure(data=traces, layout=layout)
    figure.update_layout(width=1000, height=400)

    return figure
