from __future__ import annotations

from optuna._experimental import experimental_func
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._timeline import _get_timeline_info
from optuna.visualization._timeline import _TimelineBarInfo
from optuna.visualization._timeline import _TimelineInfo
from optuna.visualization.matplotlib._matplotlib_imports import _imports


if _imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import Axes
    from optuna.visualization.matplotlib._matplotlib_imports import DateFormatter
    from optuna.visualization.matplotlib._matplotlib_imports import matplotlib
    from optuna.visualization.matplotlib._matplotlib_imports import plt


_INFEASIBLE_KEY = "INFEASIBLE"


@experimental_func("3.2.0")
def plot_timeline(study: Study, n_recent_trials: int | None = None) -> "Axes":
    """Plot the timeline of a study.

    .. seealso::
        Please refer to :func:`optuna.visualization.plot_timeline` for an example.

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted with
            their lifetime.
        n_recent_trials:
            The number of recent trials to plot. If :obj:`None`, all trials are plotted.
            If specified, only the most recent ``n_recent_trials`` will be displayed.
            Must be a positive integer.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.

    Raises:
        ValueError: if ``n_recent_trials`` is 0 or negative.
    """

    if n_recent_trials is not None and n_recent_trials <= 0:
        raise ValueError("n_recent_trials must be a positive integer or None.")

    _imports.check()
    info = _get_timeline_info(study, n_recent_trials)
    return _get_timeline_plot(info)


def _get_state_name(bar_info: _TimelineBarInfo) -> str:
    if bar_info.state == TrialState.COMPLETE and bar_info.infeasible:
        return _INFEASIBLE_KEY
    else:
        return bar_info.state.name


def _get_timeline_plot(info: _TimelineInfo) -> "Axes":
    _cm = {
        TrialState.COMPLETE.name: "tab:blue",
        TrialState.FAIL.name: "tab:red",
        TrialState.PRUNED.name: "tab:orange",
        _INFEASIBLE_KEY: "#CCCCCC",
        TrialState.RUNNING.name: "tab:green",
        TrialState.WAITING.name: "tab:gray",
    }
    # Set up the graph style.
    plt.style.use("ggplot")  # Use ggplot style sheet for similar outputs to plotly.
    fig, ax = plt.subplots()
    ax.set_title("Timeline Plot")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Trial")

    if len(info.bars) == 0:
        return ax

    # According to the `ax.barh` docstring, using list[timedelta] as width and list[datetime] as
    # left is supported, but mypy does not recognize it. Please refer to the following link for
    # more details:
    # https://github.com/matplotlib/matplotlib/blob/v3.10.1/lib/matplotlib/axes/_axes.py#L2701-L2836
    ax.barh(
        y=[b.number for b in info.bars],
        width=[b.complete - b.start for b in info.bars],  # type: ignore[arg-type]
        left=[b.start for b in info.bars],  # type: ignore[arg-type]
        color=[_cm[_get_state_name(b)] for b in info.bars],
    )

    # There are 5 types of TrialState in total.
    # However, the legend depicts only types present in the arguments.
    legend_handles = []
    for state_name, color in _cm.items():
        if any(_get_state_name(b) == state_name for b in info.bars):
            legend_handles.append(matplotlib.patches.Patch(color=color, label=state_name))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()

    assert len(info.bars) > 0
    first_start_time = min([b.start for b in info.bars])
    last_complete_time = max([b.complete for b in info.bars])
    margin = (last_complete_time - first_start_time) * 0.05

    # Officially, ax.set_xlim expects arguments right and left to be float,
    # but ax.barh() accepts datetime, so we leave the type as datetime.
    ax.set_xlim(
        right=last_complete_time + margin,  # type: ignore[arg-type]
        left=first_start_time - margin,  # type: ignore[arg-type]
    )
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))  # type: ignore[no-untyped-call]
    plt.gcf().autofmt_xdate()
    return ax
