from typing import Callable
from typing import cast
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
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


class _ValuesInfo(NamedTuple):
    values: List[float]
    stds: Optional[List[float]]
    label_name: str


class _OptimizationHistoryInfo(NamedTuple):
    trial_numbers: List[int]
    values_info: _ValuesInfo
    best_values_info: Optional[_ValuesInfo]


def _get_optimization_history_info_list(
    study: Union[Study, Sequence[Study]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    error_bar: bool,
) -> List[_OptimizationHistoryInfo]:

    _check_plot_args(study, target, target_name)
    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    info_list: List[_OptimizationHistoryInfo] = []
    for study in studies:
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        label_name = target_name if len(studies) == 1 else f"{target_name} of {study.study_name}"
        if target is not None:
            values = [target(t) for t in trials]
            # We don't calculate best for user-defined target function since we cannot tell
            # which direction is better.
            best_values_info: Optional[_ValuesInfo] = None
        else:
            values = [cast(float, t.value) for t in trials]
            if study.direction == StudyDirection.MINIMIZE:
                best_values = list(np.minimum.accumulate(values))
            else:
                best_values = list(np.maximum.accumulate(values))
            best_label_name = (
                "Best Value" if len(studies) == 1 else f"Best Value of {study.study_name}"
            )
            best_values_info = _ValuesInfo(best_values, None, best_label_name)
        info_list.append(
            _OptimizationHistoryInfo(
                trial_numbers=[t.number for t in trials],
                values_info=_ValuesInfo(values, None, label_name),
                best_values_info=best_values_info,
            )
        )

    if len(info_list) == 0:
        _logger.warning("There are no studies.")

    if sum(len(info.trial_numbers) for info in info_list) == 0:
        _logger.warning("There are no complete trials.")
        info_list.clear()

    if not error_bar:
        return info_list

    # When error_bar=True, a list of 0 or 1 element is returned.
    if len(info_list) == 0:
        return []
    all_trial_numbers = [number for info in info_list for number in info.trial_numbers]
    max_num_trial = max(all_trial_numbers) + 1

    def _aggregate(label_name: str, use_best_value: bool) -> Tuple[List[int], _ValuesInfo]:
        # Calculate mean and std of values for each trial number.
        values: List[List[float]] = [[] for _ in range(max_num_trial)]
        assert info_list is not None
        for trial_numbers, values_info, best_values_info in info_list:
            if use_best_value:
                assert best_values_info is not None
                values_info = best_values_info
            for trial_number, value in zip(trial_numbers, values_info.values):
                values[trial_number].append(value)
        trial_numbers_union = [i for i in range(max_num_trial) if len(values[i]) > 0]
        value_means = [np.mean(v).item() for v in values if len(v) > 0]
        value_stds = [np.std(v).item() for v in values if len(v) > 0]
        return trial_numbers_union, _ValuesInfo(value_means, value_stds, label_name)

    eb_trial_numbers, eb_values_info = _aggregate(target_name, False)
    eb_best_values_info: Optional[_ValuesInfo] = None
    if target is None:
        _, eb_best_values_info = _aggregate("Best Value", True)
    return [_OptimizationHistoryInfo(eb_trial_numbers, eb_values_info, eb_best_values_info)]


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
    """

    _imports.check()

    info_list = _get_optimization_history_info_list(study, target, target_name, error_bar)
    return _get_optimization_history_plot(info_list, target_name)


def _get_optimization_history_plot(
    info_list: List[_OptimizationHistoryInfo],
    target_name: str,
) -> "go.Figure":

    layout = go.Layout(
        title="Optimization History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": target_name},
    )

    traces = []
    for trial_numbers, values_info, best_values_info in info_list:
        if values_info.stds is None:
            error_y = None
        else:
            error_y = {"type": "data", "array": values_info.stds, "visible": True}
        traces.append(
            go.Scatter(
                x=trial_numbers,
                y=values_info.values,
                error_y=error_y,
                mode="markers",
                name=values_info.label_name,
            )
        )

        if best_values_info is not None:
            traces.append(
                go.Scatter(
                    x=trial_numbers,
                    y=best_values_info.values,
                    name=best_values_info.label_name,
                )
            )
            if best_values_info.stds is not None:
                upper = np.array(best_values_info.values) + np.array(best_values_info.stds)
                traces.append(
                    go.Scatter(
                        x=trial_numbers,
                        y=upper,
                        mode="lines",
                        line=dict(width=0.01),
                        showlegend=False,
                    )
                )
                lower = np.array(best_values_info.values) - np.array(best_values_info.stds)
                traces.append(
                    go.Scatter(
                        x=trial_numbers,
                        y=lower,
                        mode="none",
                        showlegend=False,
                        fill="tonexty",
                        fillcolor="rgba(255,0,0,0.2)",
                    )
                )

    return go.Figure(data=traces, layout=layout)
