from optuna.logging import get_logger
from optuna.trial import TrialState
from optuna import type_checking
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

if type_checking.TYPE_CHECKING:
    from optuna.study import Study  # NOQA

if is_available():
    from optuna.visualization.plotly_imports import go

logger = get_logger(__name__)


def plot_intermediate_values(study):
    # type: (Study) -> go.Figure
    """Plot intermediate values of all trials in a study.

    Example:

        The following code snippet shows how to plot intermediate values.

        .. testcode::

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

            study = optuna.create_study()
            study.optimize(objective, n_trials=16)

            optuna.visualization.plot_intermediate_values(study)

        .. raw:: html

            <iframe src="../_static/plot_intermediate_values.html"
             width="100%" height="500px" frameborder="0">
            </iframe>

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _check_plotly_availability()
    return _get_intermediate_plot(study)


def _get_intermediate_plot(study):
    # type: (Study) -> go.Figure

    layout = go.Layout(
        title="Intermediate Values Plot",
        xaxis={"title": "Step"},
        yaxis={"title": "Intermediate Value"},
        showlegend=False,
    )

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in study.trials if trial.state in target_state]

    if len(trials) == 0:
        logger.warning("Study instance does not contain trials.")
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
        logger.warning(
            "You need to set up the pruning feature to utilize `plot_intermediate_values()`"
        )
        return go.Figure(data=[], layout=layout)

    figure = go.Figure(data=traces, layout=layout)

    return figure
