from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._timeline import _get_timeline_info
from optuna.visualization._timeline import _TimelineInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import matplotlib
    from optuna.visualization.matplotlib._matplotlib_imports import plt


@experimental_func("3.2.0")
def plot_timeline(study: Study) -> "Axes":
    """Plot the timeline of a study.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_timeline` for an example.

    Example:

        The following code snippet shows how to plot the timeline of a study.

        .. plot::

            import time

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", 0, 1)
                time.sleep(x * 0.1)
                if x > 0.8:
                    raise ValueError()
                if x > 0.4:
                    raise optuna.TrialPruned()
                return x ** 2


            study = optuna.create_study(direction="minimize")
            study.optimize(
                objective, n_trials=50, n_jobs=2, catch=(ValueError,)
            )

            optuna.visualization.matplotlib.plot_timeline(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted with
            their lifetime.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """
    _imports.check()
    info = _get_timeline_info(study)
    return _get_timeline_plot(info)


def _get_timeline_plot(info: _TimelineInfo) -> "Axes":
    _cm = {
        TrialState.COMPLETE: "tab:blue",
        TrialState.FAIL: "tab:red",
        TrialState.PRUNED: "tab:orange",
        TrialState.RUNNING: "tab:green",
        TrialState.WAITING: "tab:gray",
    }

    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Timeline Plot")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Trial")

    if len(info.bars) == 0:
        return ax

    ax.barh(
        y=[b.number for b in info.bars],
        width=[b.complete - b.start for b in info.bars],
        left=[b.start for b in info.bars],
        color=[_cm[b.state] for b in info.bars],
    )

    # There are 5 types of TrialState in total.
    # However, the legend depicts only types present in the arguments.
    legend_handles = []
    for state, color in _cm.items():
        if len([b for b in info.bars if b.state == state]) > 0:
            legend_handles.append(matplotlib.patches.Patch(color=color, label=state.name))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()

    assert len(info.bars) > 0
    start_time = min([b.start for b in info.bars])
    complete_time = max([b.complete for b in info.bars])
    margin = (complete_time - start_time) * 0.05

    ax.set_xlim(right=complete_time + margin, left=start_time - margin)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()
    return ax
