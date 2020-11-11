from optuna._experimental import experimental
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental("2.2.0")
def plot_intermediate_values(study: Study) -> "Axes":
    """Plot intermediate values of all trials in a study with Matplotlib.

    Example:

        The following code snippet shows how to plot intermediate values.

        .. plot::

            import optuna


            def f(x):
                return (x - 2) ** 2


            def df(x):
                return 2 * x - 4


            def objective(trial):
                lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

                x = 3
                for step in range(128):
                    y = f(x)

                    trial.report(y, step=step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    gy = df(x)
                    x -= gy * lr

                return y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=16)

            optuna.visualization.matplotlib.plot_intermediate_values(study)

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_intermediate_values` for an example.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    _imports.check()
    return _get_intermediate_plot(study)


def _get_intermediate_plot(study: Study) -> "Axes":

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Intermediate Values Plot")
    ax.set_xlabel("Step")
    ax.set_ylabel("Intermediate Value")
    cmap = plt.get_cmap("tab20")  # Use tab20 colormap for multiple line plots.

    # Prepare data for plotting.
    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in study.trials if trial.state in target_state]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return ax

    # Draw multiple line plots.
    traces = []
    for i, trial in enumerate(trials):
        if trial.intermediate_values:
            sorted_intermediate_values = sorted(trial.intermediate_values.items())
            trace = ax.plot(
                tuple((x for x, _ in sorted_intermediate_values)),
                tuple((y for _, y in sorted_intermediate_values)),
                color=cmap(i),
                alpha=0.7,
                label="Trial{}".format(trial.number),
            )
            traces.append(trace)

    if not traces:
        _logger.warning(
            "You need to set up the pruning feature to utilize `plot_intermediate_values()`"
        )
        return ax

    return ax
