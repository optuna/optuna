from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._optimization_history import _get_optimization_history_info_list
from optuna.visualization._optimization_history import _OptimizationHistoryInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("2.2.0")
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
    """

    _imports.check()

    info_list = _get_optimization_history_info_list(study, target, target_name, error_bar)
    return _get_optimization_history_plot(info_list, target_name)


def _get_optimization_history_plot(
    info_list: List[_OptimizationHistoryInfo],
    target_name: str,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Optimization History Plot")
    ax.set_xlabel("Trial")
    ax.set_ylabel(target_name)
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    for i, (trial_numbers, values_info, best_values_info) in enumerate(info_list):
        if values_info.stds is not None:
            plt.errorbar(
                x=trial_numbers,
                y=values_info.values,
                yerr=values_info.stds,
                capsize=5,
                fmt="o",
                color="tab:blue",
            )
        ax.scatter(
            x=trial_numbers,
            y=values_info.values,
            color=cmap(0) if len(info_list) == 1 else cmap(2 * i),
            alpha=1,
            label=values_info.label_name,
        )

        if best_values_info is not None:
            ax.plot(
                trial_numbers,
                best_values_info.values,
                marker="o",
                color=cmap(3) if len(info_list) == 1 else cmap(2 * i + 1),
                alpha=0.5,
                label=best_values_info.label_name,
            )
            if best_values_info.stds is not None:
                lower = np.array(best_values_info.values) - np.array(best_values_info.stds)
                upper = np.array(best_values_info.values) + np.array(best_values_info.stds)
                ax.fill_between(
                    x=trial_numbers,
                    y1=lower,
                    y2=upper,
                    color="tab:red",
                    alpha=0.4,
                )
            ax.legend()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    return ax
