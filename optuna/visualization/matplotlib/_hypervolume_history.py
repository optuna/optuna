from __future__ import annotations

from typing import Callable
from typing import NamedTuple
from typing import Sequence
import warnings

import numpy as np

from optuna._experimental import experimental_func
from optuna._hypervolume import WFG
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.study import Study
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


class _HypervolumeHistoryInfo(NamedTuple):
    trial_numbers: list[int]
    values: list[float]


@experimental_func("3.2.1")
def plot_hypervolume_history(
    study: Study,
    reference_point: Sequence[float],
    *,
    constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
) -> "Axes":
    """Plot hypervolume history of all trials in a study with Matplotlib.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plot::

            import optuna
            import matplotlib.pyplot as plt


            def objective(trial):
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x ** 2 + 4 * y ** 2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            reference_point=[100, 50]
            optuna.visualization.matplotlib.plot_hypervolume_history(study, reference_point)
            plt.tight_layout()

        .. note::
            You need to adjust the size of the plot by yourself using ``plt.tight_layout()`` or
            ``plt.savefig(IMAGE_NAME, bbox_inches='tight')``.
    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their hypervolumes.
           ``study.n_objectives`` must be 2 or more.

        reference_point:
            A reference point to use for hypervolume computation.

        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.
            This specification is the same as in, for example,
            :class:`~optuna.samplers.NSGAIISampler`.

            If given, infeasible trials are not used to compute hypervolume.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()

    assert study._is_multi_objective(), (
        "Study must be multi-objective. For single-objective optimization, "
        "please use plot_optimization_history instead."
    )

    info = _get_hypervolume_history_info(
        study, np.asarray(reference_point, dtype=np.float64), constraints_func
    )
    return _get_hypervolume_history_plot(info)


def _get_hypervolume_history_plot(
    info: _HypervolumeHistoryInfo,
) -> "Axes":
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Hypervolume History Plot")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Hypervolume")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.plot(
        info.trial_numbers,
        info.values,
        marker="o",
        color=cmap(0),
        alpha=0.5,
    )
    return ax


def _get_hypervolume_history_info(
    study: Study,
    reference_point: np.ndarray,
    constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
) -> _HypervolumeHistoryInfo:
    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if len(completed_trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    if constraints_func is not None:
        warnings.warn(
            "``constraints_func`` argument is an experimental feature."
            " The interface can change in the future.",
            ExperimentalWarning,
        )
        feasible_trials = []
        best_trials_history = []
        trial_numbers = []
        for trial in completed_trials:
            if all(map(lambda x: x <= 0.0, constraints_func(trial))):
                feasible_trials.append(trial)
            trial_numbers.append(trial.number)
            best_trials_history.append(
                _get_pareto_front_trials_by_trials(feasible_trials, study.directions)
            )
    else:
        best_trials_history = []
        trial_numbers = []
        for i, trial in enumerate(completed_trials, start=1):
            trial_numbers.append(trial.number)
            best_trials_history.append(
                _get_pareto_front_trials_by_trials(completed_trials[:i], study.directions)
            )

    # Our hypervolume computation module assumes that all objectives are minimized.
    # Here we transform the objective values and the reference point.
    signs = np.asarray([1 if d == StudyDirection.MINIMIZE else -1 for d in study.directions])
    minimization_reference_point = signs * reference_point
    values = []
    for best_trials in best_trials_history:
        solution_set = np.asarray(
            list(
                filter(
                    lambda v: (v <= minimization_reference_point).all(),
                    [signs * trial.values for trial in best_trials],
                )
            )
        )
        hypervolume = 0.0
        if solution_set.size > 0:
            hypervolume = WFG().compute(solution_set, minimization_reference_point)
        values.append(hypervolume)
    return _HypervolumeHistoryInfo(trial_numbers, values)
