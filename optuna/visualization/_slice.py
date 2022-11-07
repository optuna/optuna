from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import NamedTuple
from typing import Optional

from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import _imports
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _is_numerical


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go
    from optuna.visualization._plotly_imports import make_subplots
    from optuna.visualization._plotly_imports import Scatter
    from optuna.visualization._utils import COLOR_SCALE

_logger = get_logger(__name__)


class _SliceSubplotInfo(NamedTuple):
    param_name: str
    x: List[Any]
    y: List[float]
    trial_numbers: List[int]
    is_log: bool
    is_numerical: bool


class _SlicePlotInfo(NamedTuple):
    target_name: str
    subplots: List[_SliceSubplotInfo]


def _get_slice_subplot_info(
    trials: List[FrozenTrial],
    param: str,
    target: Optional[Callable[[FrozenTrial], float]],
    log_scale: bool,
    numerical: bool,
) -> _SliceSubplotInfo:

    if target is None:

        def _target(t: FrozenTrial) -> float:
            return cast(float, t.value)

        target = _target

    return _SliceSubplotInfo(
        param_name=param,
        x=[t.params[param] for t in trials if param in t.params],
        y=[target(t) for t in trials if param in t.params],
        trial_numbers=[t.number for t in trials if param in t.params],
        is_log=log_scale,
        is_numerical=numerical,
    )


def _get_slice_plot_info(
    study: Study,
    params: Optional[List[str]],
    target: Optional[Callable[[FrozenTrial], float]],
    target_name: str,
) -> _SlicePlotInfo:

    _check_plot_args(study, target, target_name)

    trials = _filter_nonfinite(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), target=target
    )

    if len(trials) == 0:
        _logger.warning("Your study does not have any completed trials.")
        return _SlicePlotInfo(target_name, [])

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(all_params)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError(f"Parameter {input_p_name} does not exist in your study.")
        sorted_params = sorted(set(params))

    return _SlicePlotInfo(
        target_name=target_name,
        subplots=[
            _get_slice_subplot_info(
                trials=trials,
                param=param,
                target=target,
                log_scale=_is_log_scale(trials, param),
                numerical=_is_numerical(trials, param),
            )
            for param in sorted_params
        ],
    )


def plot_slice(
    study: Study,
    params: Optional[List[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
) -> "go.Figure":
    """Plot the parameter relationship as slice plot in a study.

    Note that, if a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as slice plot.

        .. plotly::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                y = trial.suggest_categorical("y", [-1, 0, 1])
                return x ** 2 + y


            sampler = optuna.samplers.TPESampler(seed=10)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=10)

            fig = optuna.visualization.plot_slice(study, params=["x", "y"])
            fig.show()

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
    """

    _imports.check()
    return _get_slice_plot(_get_slice_plot_info(study, params, target, target_name))


def _get_slice_plot(info: _SlicePlotInfo) -> "go.Figure":

    layout = go.Layout(title="Slice Plot")

    if len(info.subplots) == 0:
        return go.Figure(data=[], layout=layout)
    elif len(info.subplots) == 1:
        figure = go.Figure(data=[_generate_slice_subplot(info.subplots[0])], layout=layout)
        figure.update_xaxes(title_text=info.subplots[0].param_name)
        figure.update_yaxes(title_text=info.target_name)
        if info.subplots[0].is_log:
            figure.update_xaxes(type="log")
    else:
        figure = make_subplots(rows=1, cols=len(info.subplots), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True  # showscale option only needs to be specified once.
        for column_index, subplot_info in enumerate(info.subplots, start=1):
            trace = _generate_slice_subplot(subplot_info)
            trace.update(marker={"showscale": showscale})  # showscale's default is True.
            if showscale:
                showscale = False
            figure.add_trace(trace, row=1, col=column_index)
            figure.update_xaxes(title_text=subplot_info.param_name, row=1, col=column_index)
            if column_index == 1:
                figure.update_yaxes(title_text=info.target_name, row=1, col=column_index)
            if subplot_info.is_log:
                figure.update_xaxes(type="log", row=1, col=column_index)
        if len(info.subplots) > 3:
            # Ensure that each subplot has a minimum width without relying on autusizing.
            figure.update_layout(width=300 * len(info.subplots))

    return figure


def _generate_slice_subplot(subplot_info: _SliceSubplotInfo) -> "Scatter":
    x = [x if x is not None else "None" for x in subplot_info.x]
    y = [y if y is not None else "None" for y in subplot_info.y]

    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker={
            "line": {"width": 0.5, "color": "Grey"},
            "color": subplot_info.trial_numbers,
            "colorscale": COLOR_SCALE,
            "colorbar": {
                "title": "Trial",
                "x": 1.0,  # Offset the colorbar position with a fixed width `xpad`.
                "xpad": 40,
            },
        },
        showlegend=False,
    )
