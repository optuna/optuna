import datetime
from typing import List
from typing import NamedTuple

from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _make_hovertext


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _TimelineBarInfo(NamedTuple):
    number: int
    start: datetime.datetime
    end: datetime.datetime
    state: TrialState
    hovertext: str


class _TimelineInfo(NamedTuple):
    bars: List[_TimelineBarInfo]


@experimental_func("3.2.0")
def plot_timeline(study: Study) -> "go.Figure":
    """Plot the timeline of a study.

    Example:

        The following code snippet shows how to plot the timeline of a study.

        .. plotly::

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

            fig = optuna.visualization.plot_timeline(study)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted with
            their lifetime.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """
    _imports.check()
    info = _get_timeline_info(study)
    return _get_timeline_plot(info)


def _get_timeline_info(study: Study) -> _TimelineInfo:
    bars = []
    for t in study.get_trials():
        date_end = t.datetime_complete or datetime.datetime.now()
        date_start = t.datetime_start or date_end
        bars.append(
            _TimelineBarInfo(
                number=t.number,
                start=date_start,
                end=date_end,
                state=t.state,
                hovertext=_make_hovertext(t),
            )
        )

    if len(bars) == 0:
        _logger.warning("Your study does not have any trials.")

    return _TimelineInfo(bars)


def _get_timeline_plot(info: _TimelineInfo) -> "go.Figure":
    _cm = {
        "COMPLETE": "blue",
        "FAIL": "red",
        "PRUNED": "orange",
        "RUNNING": "green",
        "WAITING": "gray",
    }

    fig = go.Figure()
    for s in sorted(TrialState, key=lambda x: x.name):
        bars = [b for b in info.bars if b.state == s]
        if len(bars) == 0:
            continue
        fig.add_trace(
            go.Bar(
                name=s.name,
                x=[(b.end - b.start).total_seconds() * 1000 for b in bars],
                y=[b.number for b in bars],
                base=[b.start.isoformat() for b in bars],
                text=[b.hovertext for b in bars],
                hovertemplate="%{text}<extra>" + s.name + "</extra>",
                orientation="h",
                marker=dict(color=_cm[s.name]),
                textposition="none",  # avoid drawing hovertext in a bar
            )
        )
    fig.update_xaxes(type="date")
    fig.update_layout(
        go.Layout(
            title="Timeline Plot",
            xaxis={"title": "Datetime"},
            yaxis={"title": "Trial"},
        )
    )
    return fig
