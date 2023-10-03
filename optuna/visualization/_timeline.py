from __future__ import annotations

import datetime
from typing import NamedTuple

from optuna._experimental import experimental_func
from optuna.logging import get_logger
from optuna.samplers._base import _CONSTRAINTS_KEY
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
    complete: datetime.datetime
    state: TrialState
    hovertext: str
    infeasible: bool


class _TimelineInfo(NamedTuple):
    bars: list[_TimelineBarInfo]


@experimental_func("3.2.0")
def plot_timeline(study: Study) -> "go.Figure":
    """Plot the timeline of a study.

    Example:

        The following code snippet shows how to plot the timeline of a study.
        Timeline plot can visualize trials with overlapping execution time
        (e.g., in distributed environments).

        .. plotly::

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
    for t in study.get_trials(deepcopy=False):
        date_complete = t.datetime_complete or datetime.datetime.now()
        date_start = t.datetime_start or date_complete
        infeasible = (
            False
            if _CONSTRAINTS_KEY not in t.system_attrs
            else any([x > 0 for x in t.system_attrs[_CONSTRAINTS_KEY]])
        )
        if date_complete < date_start:
            _logger.warning(
                (
                    f"The start and end times for Trial {t.number} seem to be reversed. "
                    f"The start time is {date_start} and the end time is {date_complete}."
                )
            )
        bars.append(
            _TimelineBarInfo(
                number=t.number,
                start=date_start,
                complete=date_complete,
                state=t.state,
                hovertext=_make_hovertext(t),
                infeasible=infeasible,
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
        if s.name == "COMPLETE":
            infeasible_bars = [b for b in info.bars if b.state == s and b.infeasible]
            feasible_bars = [b for b in info.bars if b.state == s and not b.infeasible]
            _plot_bars(infeasible_bars, "#cccccc", "INFEASIBLE", fig)
            _plot_bars(feasible_bars, _cm[s.name], s.name, fig)
        else:
            bars = [b for b in info.bars if b.state == s]
            _plot_bars(bars, _cm[s.name], s.name, fig)
    fig.update_xaxes(type="date")
    fig.update_layout(
        go.Layout(
            title="Timeline Plot",
            xaxis={"title": "Datetime"},
            yaxis={"title": "Trial"},
        )
    )
    fig.update_layout(showlegend=True)  # Draw a legend even if all TrialStates are the same.
    return fig


def _plot_bars(bars: list[_TimelineBarInfo], color: str, name: str, fig: go.Figure) -> None:
    if len(bars) == 0:
        return

    fig.add_trace(
        go.Bar(
            name=name,
            x=[(b.complete - b.start).total_seconds() * 1000 for b in bars],
            y=[b.number for b in bars],
            base=[b.start.isoformat() for b in bars],
            text=[b.hovertext for b in bars],
            hovertemplate="%{text}<extra>" + name + "</extra>",
            orientation="h",
            marker=dict(color=color),
            textposition="none",  # Avoid drawing hovertext in a bar.
        )
    )
