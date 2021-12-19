from typing import List
from typing import Optional

from matplotlib.colors import Colormap

import optuna
from optuna._experimental import experimental
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._pareto_front import _get_pareto_front_info
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = optuna.logging.get_logger(__name__)


@experimental("2.8.0")
def plot_pareto_front(
    study: Study,
    *,
    target_names: Optional[List[str]] = None,
    include_dominated_trials: bool = True,
    axis_order: Optional[List[int]] = None,
) -> "Axes":
    """Plot the Pareto front of a study.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_pareto_front` for an example.

    Example:

        The following code snippet shows how to plot the Pareto front of a study.

        .. plot::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 5)
                y = trial.suggest_float("y", 0, 3)

                v0 = 4 * x ** 2 + 4 * y ** 2
                v1 = (x - 5) ** 2 + (y - 5) ** 2
                return v0, v1


            study = optuna.create_study(directions=["minimize", "minimize"])
            study.optimize(objective, n_trials=50)

            optuna.visualization.matplotlib.plot_pareto_front(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        target_names:
            Objective name list used as the axis titles. If :obj:`None` is specified,
            "Objective {objective_index}" is used instead.
        include_dominated_trials:
            A flag to include all dominated trial's objective values.
        axis_order:
            A list of indices indicating the axis order. If :obj:`None` is specified,
            default order is used.

    Returns:
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If the number of objectives of ``study`` isn't 2 or 3.
    """

    _imports.check()

    info = _get_pareto_front_info(
        study, target_names, include_dominated_trials, axis_order, constraints_func=None
    )
    assert info.n_dim in (2, 3)

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    if info.n_dim == 2:
        _, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    ax.set_xlabel(info.target_names[info.axis_order[0]])
    ax.set_ylabel(info.target_names[info.axis_order[1]])
    if info.n_dim == 3:
        ax.set_zlabel(info.target_names[info.axis_order[2]])

    def scatter_sub(trials: List[FrozenTrial], color: Colormap, label: str) -> None:
        if info.n_dim == 2:
            ax.scatter(
                x=[t.values[info.axis_order[0]] for t in trials],
                y=[t.values[info.axis_order[1]] for t in trials],
                color=color,
                label=label,
            )
        else:
            ax.scatter(
                xs=[t.values[info.axis_order[0]] for t in trials],
                ys=[t.values[info.axis_order[1]] for t in trials],
                zs=[t.values[info.axis_order[2]] for t in trials],
                color=color,
                label=label,
            )

    assert info.infeasible_trials is None
    if info.non_best_trials is not None and len(info.non_best_trials) > 0:
        scatter_sub(info.non_best_trials, cmap(0), "Trial")
    if len(info.best_trials) > 0:
        scatter_sub(info.best_trials, cmap(3), "Best Trial")

    if include_dominated_trials and ax.has_data():
        ax.legend()

    return ax
