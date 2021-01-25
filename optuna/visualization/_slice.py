from typing import Callable
from typing import cast
from typing import List
from typing import Optional

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _is_log_scale


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import Scatter

_logger = get_logger(__name__)


def plot_slice(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the parameter relationship as slice plot in a study.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as slice plot.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_slice(study, params=["x", "y"])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their target values.
        params:
            Parameter list to visualize. The default is all parameters.
        target:
            A function to specify the value to display. If it is :obj:`None` and ``study`` is being
            used for single-objective optimization, the objective values are plotted.

            .. note::
                Specify this argument if ``study`` is being used for multi-objective optimization.
        target_name:
            Target's name to display on the axis label.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.

    Raises:
        :exc:`ValueError`:
            If ``target`` is :obj:`None` and ``study`` is being used for multi-objective
            optimization.
    """

    _imports.check()
    _check_plot_args(study, target, target_name)
    return _get_slice_plot(study, params, target, target_name)


def _get_slice_plot(
    study: Study,
    params: Optional[List[str]] = None,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":

    layout = go.Layout(title="Slice Plot")

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError("Parameter {} does not exist in your study.".format(input_p_name))
        sorted_params = sorted(list(set(params)))

    n_params = len(sorted_params)

    if n_params == 1:
        figure = go.Figure(
            data=[_generate_slice_subplot(study, trials, sorted_params[0], target)], layout=layout
        )
        figure.update_xaxes(title_text=sorted_params[0])
        figure.update_yaxes(title_text=target_name)
        if _is_log_scale(trials, sorted_params[0]):
            figure.update_xaxes(type="log")
    else:
        figure = make_subplots(rows=1, cols=len(sorted_params), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for i, param in enumerate(sorted_params):
            trace = _generate_slice_subplot(study, trials, param, target)
            trace.update(marker={"showscale": showscale})  # showscale's default is True.
            if showscale:
                showscale = False
            figure.add_trace(trace, row=1, col=i + 1)
            figure.update_xaxes(title_text=param, row=1, col=i + 1)
            if i == 0:
                figure.update_yaxes(title_text=target_name, row=1, col=1)
            if _is_log_scale(trials, param):
                figure.update_xaxes(type="log", row=1, col=i + 1)
        if n_params > 3:
            # Ensure that each subplot has a minimum width without relying on autusizing.
            figure.update_layout(width=300 * n_params)

    return figure


def _generate_slice_subplot(
    study: Study,
    trials: List[FrozenTrial],
    param: str,
    target: Optional[Callable[[FrozenTrial], float]],
) -> "Scatter":

    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    return go.Scatter(
        x=[t.params[param] for t in trials if param in t.params],
        y=[target(t) for t in trials if param in t.params],
        mode="markers",
        marker={
            "line": {"width": 0.5, "color": "Grey"},
            "color": [t.number for t in trials if param in t.params],
            "colorscale": "Blues",
            "colorbar": {
                "title": "#Trials",
                "x": 1.0,  # Offset the colorbar position with a fixed width `xpad`.
                "xpad": 40,
            },
        },
        showlegend=False,
    )
