import itertools
from typing import Callable
from typing import cast
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from optuna._experimental import experimental
from optuna.logging import get_logger
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._utils import _check_plot_args
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental("2.2.0")
def plot_optimization_history(
    study: Union[Study, Sequence[Study]],
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
    error_bar: bool = False,
) -> "Axes":
    """Plot optimization history of all trials in a study with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_optimization_history` for an example.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plot::

            import optuna
            import matplotlib.pyplot as plt


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y

            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()

        .. note::
            You need to adjust the size of the plot by yourself using ``plt.tight_layout()`` or
            ``plt.savefig(IMAGE_NAME, bbox_inches='tight')``.
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
        A :class:`matplotlib.axes.Axes` object.

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
    _check_plot_args(study, target, target_name)
    return _get_optimization_history_plot(studies, target, target_name, error_bar)


def _get_optimization_history_plot(
    studies: List[Study],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    error_bar: bool = False,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.

    _, ax = plt.subplots()
    ax.set_title("Optimization History Plot")
    ax.set_xlabel("#Trials")
    ax.set_ylabel(target_name)

    if len(studies) == 0:
        _logger.warning("There are no studies.")
        return ax
    # Prepare data for plotting.
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
        _logger.warning("Study instance does not contain trials.")
        return ax

    if error_bar:
        ax = _get_optimization_histories_with_error_bar(studies, target, target_name, ax)
    else:
        ax = _get_optimization_histories(studies, target, target_name, ax)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    return ax


def _get_optimization_histories_with_error_bar(
    studies: List[Study],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    ax: "Axes",
) -> "Axes":

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

    mean_of_target_values = [np.mean(v) if len(v) > 0 else np.nan for v in target_values]
    std_of_target_values = [np.std(v) if len(v) > 0 else np.nan for v in target_values]
    trial_numbers = np.arange(max_trial_number + 2)[
        [v is not np.nan for v in mean_of_target_values]
    ]
    means = np.asarray(mean_of_target_values)[trial_numbers]
    stds = np.asarray(std_of_target_values)[trial_numbers]

    plt.errorbar(
        x=trial_numbers,
        y=means,
        yerr=stds,
        capsize=5,
        fmt="o",
        color="tab:blue",
    )
    ax.scatter(trial_numbers, means, color="tab:blue", label=target_name)

    if target is None:
        best_values: List[List[float]] = [[] for _ in range(max_trial_number + 2)]
        for study in studies:
            trials = study.get_trials(states=(TrialState.COMPLETE,))
            if study.direction == StudyDirection.MINIMIZE:
                best_vs = np.minimum.accumulate([cast(float, t.values) for t in trials])
            else:
                best_vs = np.maximum.accumulate([cast(float, t.values) for t in trials])

            for i, t in enumerate(trials):
                best_values[t.number].append(best_vs[i])

        mean_of_best_values = [np.mean(v) if len(v) > 0 else np.nan for v in best_values]
        std_of_best_values = [np.std(v) if len(v) > 0 else np.nan for v in best_values]
        means = np.asarray(mean_of_best_values)[trial_numbers]
        stds = np.asarray(std_of_best_values)[trial_numbers]

        ax.plot(trial_numbers, means, color="tab:red", label="Best Value")
        ax.fill_between(
            x=trial_numbers,
            y1=means - stds,
            y2=means + stds,
            color="tab:red",
            alpha=0.4,
        )

        ax.legend()

    return ax


def _get_optimization_histories(
    studies: List[Study],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
    ax: "Axes",
) -> "Axes":
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    # Draw a scatter plot and a line plot.
    for i, study in enumerate(studies):
        trials = study.get_trials(states=(TrialState.COMPLETE,))
        if target is None:
            if study.direction == StudyDirection.MINIMIZE:
                best_values = np.minimum.accumulate([cast(float, t.value) for t in trials])
            else:
                best_values = np.maximum.accumulate([cast(float, t.value) for t in trials])
            ax.scatter(
                x=[t.number for t in trials],
                y=[t.value for t in trials],
                color=cmap(0) if len(studies) == 1 else cmap(2 * i),
                alpha=1,
                label=target_name if len(studies) == 1 else f"{target_name} of {study.study_name}",
            )
            ax.plot(
                [t.number for t in trials],
                best_values,
                marker="o",
                color=cmap(3) if len(studies) == 1 else cmap(2 * i + 1),
                alpha=0.5,
                label="Best Value" if len(studies) == 1 else f"Best Values of {study.study_name}",
            )

            ax.legend()
        else:
            ax.scatter(
                x=[t.number for t in trials],
                y=[target(t) for t in trials],
                color=cmap(0) if len(studies) == 0 else cmap(2 * i),
                alpha=1,
                label=target_name if len(studies) == 1 else f"{target_name} of {study.study_name}",
            )

    return ax
