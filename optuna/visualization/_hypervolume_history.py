from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from optuna._experimental import experimental_func
from optuna._hypervolume import WFG
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import Study
from optuna.study._multi_objective import _dominates
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _HypervolumeHistoryInfo(NamedTuple):
    trial_numbers: list[int]
    values: list[float]


@experimental_func("3.3.0")
def plot_hypervolume_history(
    study: Study,
    reference_point: Sequence[float],
) -> "go.Figure":
    """Plot hypervolume history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

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

            reference_point=[100., 50.]
            fig = optuna.visualization.plot_hypervolume_history(study, reference_point)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their hypervolumes.
            The number of objectives must be 2 or more.

        reference_point:
            A reference point to use for hypervolume computation.
            The dimension of the reference point must be the same as the number of objectives.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()

    if not study._is_multi_objective():
        raise ValueError(
            "Study must be multi-objective. For single-objective optimization, "
            "please use plot_optimization_history instead."
        )

    if len(reference_point) != len(study.directions):
        raise ValueError(
            "The dimension of the reference point must be the same as the number of objectives."
        )

    info = _get_hypervolume_history_info(study, np.asarray(reference_point, dtype=np.float64))
    return _get_hypervolume_history_plot(info)


def _get_hypervolume_history_plot(
    info: _HypervolumeHistoryInfo,
) -> "go.Figure":
    layout = go.Layout(
        title="Hypervolume History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": "Hypervolume"},
    )

    data = go.Scatter(
        x=info.trial_numbers,
        y=info.values,
        mode="lines+markers",
    )
    return go.Figure(data=data, layout=layout)


def _get_hypervolume_history_info(
    study: Study,
    reference_point: np.ndarray,
) -> _HypervolumeHistoryInfo:
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    if len(completed_trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    # Our hypervolume computation module assumes that all objectives are minimized.
    # Here we transform the objective values and the reference point.
    signs = np.asarray([1 if d == StudyDirection.MINIMIZE else -1 for d in study.directions])
    minimization_reference_point = signs * reference_point

    # Only feasible trials are considered in hypervolume computation.
    trial_numbers = []
    values = []
    best_trials: list[FrozenTrial] = []
    hypervolume = 0.0
    for trial in completed_trials:
        trial_numbers.append(trial.number)

        has_constraints = _CONSTRAINTS_KEY in trial.system_attrs
        if has_constraints:
            constraints_values = trial.system_attrs[_CONSTRAINTS_KEY]
            if any(map(lambda x: x > 0.0, constraints_values)):
                # The trial is infeasible.
                values.append(hypervolume)
                continue

        if any(map(lambda t: _dominates(t, trial, study.directions), best_trials)):
            # The trial is not on the Pareto front.
            values.append(hypervolume)
            continue

        best_trials = list(
            filter(lambda t: not _dominates(trial, t, study.directions), best_trials)
        ) + [trial]

        solution_set = np.asarray(
            list(
                filter(
                    lambda v: (v <= minimization_reference_point).all(),
                    [signs * trial.values for trial in best_trials],
                )
            )
        )
        if solution_set.size > 0:
            hypervolume = WFG().compute(solution_set, minimization_reference_point)
        values.append(hypervolume)

    if len(best_trials) == 0:
        _logger.warning("Your study does not have any feasible trials.")

    return _HypervolumeHistoryInfo(trial_numbers, values)
