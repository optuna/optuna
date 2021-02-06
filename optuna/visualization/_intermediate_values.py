from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


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

            optuna.visualization.plot_intermediate_values(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _imports.check()
    return _get_intermediate_plot(study)


def _get_intermediate_plot(study: Study) -> "go.Figure":

    layout = go.Layout(
        title="Intermediate Values Plot",
        xaxis={"title": "Step"},
        yaxis={"title": "Intermediate Value"},
        showlegend=False,
    )

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in study.trials if trial.state in target_state]

    if len(trials) == 0:
        _logger.warning("Study instance does not contain trials.")
        return go.Figure(data=[], layout=layout)

    traces = []
    for trial in trials:
        if trial.intermediate_values:
            sorted_intermediate_values = sorted(trial.intermediate_values.items())
            trace = go.Scatter(
                x=tuple((x for x, _ in sorted_intermediate_values)),
                y=tuple((y for _, y in sorted_intermediate_values)),
                mode="lines+markers",
                marker={"maxdisplayed": 10},
                name="Trial{}".format(trial.number),
            )
            traces.append(trace)

    if not traces:
        _logger.warning(
            "You need to set up the pruning feature to utilize `plot_intermediate_values()`"
        )
        return go.Figure(data=[], layout=layout)

    figure = go.Figure(data=traces, layout=layout)

    return figure
