from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.visualization._intermediate_values import _get_intermediate_plot_info
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import plt

_logger = get_logger(__name__)


@experimental_func("2.2.0")
def plot_intermediate_values(study: Study) -> "Axes":
    """Plot intermediate values of all trials in a study with Matplotlib.

    .. note::
        Please refer to `matplotlib.pyplot.legend
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
        to adjust the style of the generated legend.

    Example:

        The following code snippet shows how to plot intermediate values.

        .. plot::

            import optuna


            def f(x):
                return (x - 2) ** 2


            def df(x):
                return 2 * x - 4


            def objective(trial):
                lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

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
    _, ax = plt.subplots(tight_layout=True)
    ax.set_title("Intermediate Values Plot")
    ax.set_xlabel("Step")
    ax.set_ylabel("Intermediate Value")
    cmap = plt.get_cmap("tab20")  # Use tab20 colormap for multiple line plots.

    info = _get_intermediate_plot_info(study)
    trial_infos = info.trial_infos

    for i, tinfo in enumerate(trial_infos):
        ax.plot(
            tuple((x for x, _ in tinfo.sorted_intermediate_values)),
            tuple((y for _, y in tinfo.sorted_intermediate_values)),
            color=cmap(i),
            alpha=0.7,
            label="Trial{}".format(tinfo.trial_number),
        )

    if len(trial_infos) >= 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    return ax
