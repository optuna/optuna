from typing import List
from typing import NamedTuple
from typing import Tuple

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


class _TrialInfo(NamedTuple):
    trial_number: int
    sorted_intermediate_values: List[Tuple[int, float]]


class _IntermediatePlotInfo(NamedTuple):
    trial_infos: List[_TrialInfo]


def _get_intermediate_plot_info(study: Study) -> _IntermediatePlotInfo:
    trials = study.get_trials(
        deepcopy=False, states=(TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING)
    )
    trial_infos = [
        _TrialInfo(trial.number, sorted(trial.intermediate_values.items()))
        for trial in trials
        if len(trial.intermediate_values) > 0
    ]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
    elif len(trial_infos) == 0:
        _logger.warning(
            "You need to set up the pruning feature to utilize `plot_intermediate_values()`"
        )

    return _IntermediatePlotInfo(trial_infos)


def plot_intermediate_values(study: Study) -> "go.Figure":
    """Plot intermediate values of all trials in a study.

    Example:

        The following code snippet shows how to plot intermediate values.

        .. plotly::

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

            fig = optuna.visualization.plot_intermediate_values(study)
            fig.show()

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()
    return _get_intermediate_plot(_get_intermediate_plot_info(study))


def _get_intermediate_plot(info: _IntermediatePlotInfo) -> "go.Figure":

    layout = go.Layout(
        title="Intermediate Values Plot",
        xaxis={"title": "Step"},
        yaxis={"title": "Intermediate Value"},
        showlegend=False,
    )

    trial_infos = info.trial_infos

    if len(trial_infos) == 0:
        return go.Figure(data=[], layout=layout)

    traces = [
        go.Scatter(
            x=tuple((x for x, _ in tinfo.sorted_intermediate_values)),
            y=tuple((y for _, y in tinfo.sorted_intermediate_values)),
            mode="lines+markers",
            marker={"maxdisplayed": 10},
            name="Trial{}".format(tinfo.trial_number),
        )
        for tinfo in trial_infos
    ]

    return go.Figure(data=traces, layout=layout)
