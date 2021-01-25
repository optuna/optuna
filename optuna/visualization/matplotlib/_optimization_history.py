from typing import Callable
from typing import Optional

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
    study: Study,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "Axes":
    """Plot optimization history of all trials in a study with Matplotlib.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_optimization_history` for an example.

    Example:

        The following code snippet shows how to plot optimization history.

        .. plot::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y

            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.matplotlib.plot_optimization_history(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label and the legend.

    Returns:
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_optimization_history_plot(study, target, target_name)


def _get_optimization_history_plot(
    study: Study,
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Optimization History Plot")
    ax.set_xlabel("#Trials")
    ax.set_ylabel(target_name)
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    # Prepare data for plotting.
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return ax

    best_values = [float("inf")] if study.direction == StudyDirection.MINIMIZE else [-float("inf")]
    comp = min if study.direction == StudyDirection.MINIMIZE else max
    for trial in trials:
        trial_value = trial.value
        assert trial_value is not None  # For mypy
        best_values.append(comp(best_values[-1], trial_value))
    best_values.pop(0)

    # Draw a scatter plot and a line plot.
    if target is None:
        ax.scatter(
            x=[t.number for t in trials],
            y=[t.value for t in trials],
            color=cmap(0),
            alpha=1,
            label=target_name,
        )
        ax.plot(
            [t.number for t in trials],
            best_values,
            marker="o",
            color=cmap(3),
            alpha=0.5,
            label="Best Value",
        )
    else:
        ax.scatter(
            x=[t.number for t in trials],
            y=[target(t) for t in trials],
            color=cmap(0),
            alpha=1,
            label=target_name,
        )
    ax.legend()

    return ax
