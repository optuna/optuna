import collections
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
import warnings

import optuna
from optuna._experimental import experimental
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
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
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
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

            .. warning::
                Deprecated in v3.0.0. This feature will be removed in the future. The removal of
                this feature is currently scheduled for v5.0.0, but this schedule is subject to
                change. See https://github.com/optuna/optuna/releases/tag/v3.0.0.
        targets:
            A function that returns a tuple of target values to display.
            The argument to this function is :class:`~optuna.trial.FrozenTrial`.

            .. note::
                Added in v3.0.0 as an experimental feature. The interface may change in newer
                versions without prior notice.
                See https://github.com/optuna/optuna/releases/tag/v3.0.0.

    Returns:
        A :class:`matplotlib.axes.Axes` object.

    Raises:
        :exc:`ValueError`:
            If ``targets`` is :obj:`None` when your objective studies have more than 3 objectives.
        :exc:`ValueError`:
            If ``targets`` returns something other than sequence.
        :exc:`ValueError`:
            If the number of target values to display isn't 2 or 3.
        :exc:`ValueError`:
            If ``targets`` is specified for empty studies and ``target_names`` is :obj:`None`.
        :exc:`ValueError`:
            If using both ``targets`` and ``axis_order``.
    """

    _imports.check()
    if axis_order is not None:
        warnings.warn(
            "`axis_order` has been deprecated in v3.0.0. "
            "This feature will be removed in v5.0.0. "
            "See https://github.com/optuna/optuna/releases/tag/v3.0.0.",
            DeprecationWarning,
        )
    trials = _get_trials(study, include_dominated_trials)
    if targets is not None and axis_order is not None:
        raise ValueError(
            "Using both `targets` and `axis_order` is not supported. "
            "Use either `targets` or `axis_order`."
        )
    _targets = targets
    if _targets is None:
        if len(study.directions) == 2:
            _targets = _targets_default_2d
        elif len(study.directions) == 3:
            _targets = _targets_default_3d
        else:
            raise ValueError(
                "`plot_pareto_front` function only supports 2 or 3 objective"
                " studies when using `targets` is `None`. Please use `targets`"
                " if your objective studies have more than 3 objectives."
            )
    target_values = [_targets(t) for t in trials]
    if len(target_values) > 0 and not isinstance(target_values[0], collections.abc.Sequence):
        raise ValueError(
            "`targets` should return a sequence of target values."
            " your `targets` returns {}".format(type(target_values[0]))
        )

    if len(target_values) > 0:
        n_targets = len(target_values[0])
    elif target_names is not None:
        n_targets = len(target_names)
    elif targets is None:
        n_targets = len(study.directions)
    else:
        raise ValueError(
            "If `targets` is specified for empty studies, `target_names` must be specified."
        )

    if n_targets == 2:
        return _get_pareto_front_2d(
            study, target_values, target_names, include_dominated_trials, axis_order
        )
    elif n_targets == 3:
        return _get_pareto_front_3d(
            study, target_values, target_names, include_dominated_trials, axis_order
        )
    else:
        raise ValueError(
            "`plot_pareto_front` function only supports 2 or 3 targets."
            " you used {} targets now.".format(n_targets)
        )


def _targets_default_2d(trial: FrozenTrial) -> Sequence[float]:
    return trial.values[0], trial.values[1]


def _targets_default_3d(trial: FrozenTrial) -> Sequence[float]:
    return trial.values[0], trial.values[1], trial.values[2]


def _get_non_pareto_front_trials(
    study: Study, pareto_trials: List[FrozenTrial]
) -> List[FrozenTrial]:

    non_pareto_trials = []
    for trial in study.get_trials():
        if trial.state == TrialState.COMPLETE and trial not in pareto_trials:
            non_pareto_trials.append(trial)
    return non_pareto_trials


def _get_trials(study: Study, include_dominated_trials: bool) -> List[FrozenTrial]:
    trials = study.best_trials
    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")

    if include_dominated_trials:
        non_pareto_trials = _get_non_pareto_front_trials(study, trials)
        trials += non_pareto_trials
    return trials


def _get_pareto_front_2d(
    study: Study,
    target_values: Sequence[Sequence[float]],
    target_names: Optional[List[str]],
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    _, ax = plt.subplots()
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    if target_names is None:
        target_names = ["Objective 0", "Objective 1"]
    elif len(target_names) != 2:
        raise ValueError("The length of `target_names` is supposed to be 2.")

    # Prepare data for plotting.
    if axis_order is None:
        axis_order = list(range(2))
    else:
        if len(axis_order) != 2:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: 2, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != 2:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!")
        if max(axis_order) > 1:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                "higher than 1."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    ax.set_xlabel(target_names[axis_order[0]])
    ax.set_ylabel(target_names[axis_order[1]])

    if len(target_values) - len(study.best_trials) != 0:
        ax.scatter(
            x=[v[axis_order[0]] for v in target_values[len(study.best_trials) :]],
            y=[v[axis_order[1]] for v in target_values[len(study.best_trials) :]],
            color=cmap(0),
            label="Trial",
        )
    if len(study.best_trials):
        ax.scatter(
            x=[v[axis_order[0]] for v in target_values[: len(study.best_trials)]],
            y=[v[axis_order[1]] for v in target_values[: len(study.best_trials)]],
            color=cmap(3),
            label="Best Trial",
        )

    if include_dominated_trials and ax.has_data():
        ax.legend()

    return ax


def _get_pareto_front_3d(
    study: Study,
    target_values: Sequence[Sequence[float]],
    target_names: Optional[List[str]],
    include_dominated_trials: bool = False,
    axis_order: Optional[List[int]] = None,
) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Pareto-front Plot")
    cmap = plt.get_cmap("tab10")  # Use tab10 colormap for similar outputs to plotly.

    if target_names is None:
        target_names = ["Objective 0", "Objective 1", "Objective 2"]
    elif len(target_names) != 3:
        raise ValueError("The length of `target_names` is supposed to be 3.")

    if axis_order is None:
        axis_order = list(range(3))
    else:
        if len(axis_order) != 3:
            raise ValueError(
                f"Size of `axis_order` {axis_order}. Expect: 3, Actual: {len(axis_order)}."
            )
        if len(set(axis_order)) != 3:
            raise ValueError(f"Elements of given `axis_order` {axis_order} are not unique!.")
        if max(axis_order) > 2:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {max(axis_order)} "
                "higher than 2."
            )
        if min(axis_order) < 0:
            raise ValueError(
                f"Given `axis_order` {axis_order} contains invalid index {min(axis_order)} "
                "lower than 0."
            )

    ax.set_xlabel(target_names[axis_order[0]])
    ax.set_ylabel(target_names[axis_order[1]])
    ax.set_zlabel(target_names[axis_order[2]])

    if len(target_values) - len(study.best_trials) != 0:
        ax.scatter(
            xs=[v[axis_order[0]] for v in target_values[len(study.best_trials) :]],
            ys=[v[axis_order[1]] for v in target_values[len(study.best_trials) :]],
            zs=[v[axis_order[2]] for v in target_values[len(study.best_trials) :]],
            color=cmap(0),
            label="Trial",
        )

    if len(study.best_trials):
        ax.scatter(
            xs=[v[axis_order[0]] for v in target_values[: len(study.best_trials)]],
            ys=[v[axis_order[1]] for v in target_values[: len(study.best_trials)]],
            zs=[v[axis_order[2]] for v in target_values[: len(study.best_trials)]],
            color=cmap(3),
            label="Best Trial",
        )

    if include_dominated_trials and ax.has_data():
        ax.legend()

    return ax
