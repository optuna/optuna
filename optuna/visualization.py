from collections import defaultdict
import math

from optuna.distributions import LogUniformDistribution
from optuna.logging import get_logger
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna import type_checking

logger = get_logger(__name__)

if type_checking.TYPE_CHECKING:
    from plotly.graph_objs import Contour  # NOQA
    from plotly.graph_objs import Scatter  # NOQA
    from typing import DefaultDict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from optuna.structs import FrozenTrial  # NOQA

try:
    import plotly
    import plotly.graph_objs as go
    from plotly.graph_objs._figure import Figure  # NOQA
    from plotly.subplots import make_subplots
    _available = True
except ImportError as e:
    _import_error = e
    # Visualization features are disabled because plotly is not available.
    _available = False


def is_available():
    # type: () -> bool
    """Returns whether visualization is available or not.

    .. note::

        :mod:`~optuna.visualization` module depends on plotly version 4.0.0 or higher. If a
        supported version of plotly isn't installed in your environment, this function will return
        :obj:`False`. In such case, please execute ``$ pip install -U plotly>=4.0.0`` to install
        plotly.

    Returns:
        :obj:`True` if visualization is available, :obj:`False` otherwise.
    """

    return _available


def plot_intermediate_values(study):
    # type: (Study) -> None
    """Plot intermediate values of all trials in a study.

    Example:

        The following code snippet shows how to plot intermediate values.

        .. code::

            import optuna

            def objective(trial):
                # Intermediate values are supposed to be reported inside the objective function.
                ...

            study = optuna.create_study()
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_intermediate_values(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.
    """

    _check_plotly_availability()
    figure = _get_intermediate_plot(study)
    figure.show()


def _get_intermediate_plot(study):
    # type: (Study) -> Figure

    layout = go.Layout(
        title='Intermediate Values Plot',
        xaxis={'title': 'Step'},
        yaxis={'title': 'Intermediate Value'},
        showlegend=False
    )

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in study.trials if trial.state in target_state]

    if len(trials) == 0:
        logger.warning('Study instance does not contain trials.')
        return go.Figure(data=[], layout=layout)
    if hasattr(trials[0], 'intermediate_values') is False:
        logger.warning(
            'You need to set up the pruning feature to utilize plot_intermediate_values()')
        return go.Figure(data=[], layout=layout)

    traces = []
    for trial in trials:
        trace = go.Scatter(
            x=tuple(trial.intermediate_values.keys()),
            y=tuple(trial.intermediate_values.values()),
            mode='lines+markers',
            marker={
                'maxdisplayed': 10
            },
            name='Trial{}'.format(trial.number)
        )
        traces.append(trace)

    figure = go.Figure(data=traces, layout=layout)

    return figure


def plot_optimization_history(study):
    # type: (Study) -> None
    """Plot optimization history of all trials in a study.

    Example:

        The following code snippet shows how to plot optimization history.

        .. code::

            import optuna

            def objective(trial):
                ...

            study = optuna.create_study()
            study.optimize(objective ,n_trials=100)

            optuna.visualization.plot_optimization_history(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
    """

    _check_plotly_availability()
    figure = _get_optimization_history_plot(study)
    figure.show()


def _get_optimization_history_plot(study):
    # type: (Study) -> Figure

    layout = go.Layout(
        title='Optimization History Plot',
        xaxis={'title': '#Trials'},
        yaxis={'title': 'Objective Value'},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning('Study instance does not contain trials.')
        return go.Figure(data=[], layout=layout)

    best_values = [float('inf')] if study.direction == StudyDirection.MINIMIZE else [-float('inf')]
    for trial in trials:
        if isinstance(trial.value, float):
            trial_value = trial.value
        else:
            raise ValueError(
                'Trial{} has COMPLETE state, but its value is non float.'.format(trial.number))
        if study.direction == StudyDirection.MINIMIZE:
            best_values.append(min(best_values[-1], trial_value))
        else:
            best_values.append(max(best_values[-1], trial_value))
    best_values.pop(0)
    traces = [
        go.Scatter(x=[t.number for t in trials], y=[t.value for t in trials],
                   mode='markers', name='Objective Value'),
        go.Scatter(x=[t.number for t in trials], y=best_values, name='Best Value')
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure


def plot_contour(study, params=None):
    # type: (Study, Optional[List[str]]) -> None
    """Plot the parameter relationship as contour plot in a study.

        Note that, If a parameter contains missing values, a trial with missing values is not
        plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as contour plot.

        .. code::

            import optuna

            def objective(trial):
                ...

            study = optuna.create_study()
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_contour(study, params=['param_a', 'param_b'])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.
    """

    _check_plotly_availability()
    figure = _get_contour_plot(study, params)
    figure.show()


def _get_contour_plot(study, params=None):
    # type: (Study, Optional[List[str]]) -> Figure

    layout = go.Layout(
        title='Contour Plot',
    )

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning('Your study does not have any completed trials.')
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    elif len(params) <= 1:
        logger.warning('The length of params must be greater than 1.')
        return go.Figure(data=[], layout=layout)
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError('Parameter {} does not exist in your study.'.format(input_p_name))
        sorted_params = sorted(list(set(params)))

    param_values_range = dict()
    for p_name in sorted_params:
        values = [t.params[p_name] for t in trials if p_name in t.params]
        param_values_range[p_name] = (min(values), max(values))

    if len(sorted_params) == 2:
        x_param = sorted_params[0]
        y_param = sorted_params[1]
        sub_plots = _generate_contour_subplot(
            trials, x_param, y_param, study.direction)
        figure = go.Figure(data=sub_plots)
        figure.update_xaxes(title_text=x_param, range=param_values_range[x_param])
        figure.update_yaxes(title_text=y_param, range=param_values_range[y_param])
        if _is_log_scale(trials, x_param):
            log_range = [math.log10(p) for p in param_values_range[x_param]]
            figure.update_xaxes(range=log_range, type='log')
        if _is_log_scale(trials, y_param):
            log_range = [math.log10(p) for p in param_values_range[y_param]]
            figure.update_yaxes(range=log_range, type='log')
    else:
        figure = make_subplots(rows=len(sorted_params),
                               cols=len(sorted_params), shared_xaxes=True, shared_yaxes=True)
        showscale = True   # showscale option only needs to be specified once
        for x_i, x_param in enumerate(sorted_params):
            for y_i, y_param in enumerate(sorted_params):
                if x_param == y_param:
                    figure.add_trace(go.Scatter(), row=y_i + 1, col=x_i + 1)
                else:
                    sub_plots = _generate_contour_subplot(
                        trials, x_param, y_param, study.direction)
                    contour = sub_plots[0]
                    scatter = sub_plots[1]
                    contour.update(showscale=showscale)  # showscale's default is True
                    if showscale:
                        showscale = False
                    figure.add_trace(contour, row=y_i + 1, col=x_i + 1)
                    figure.add_trace(scatter, row=y_i + 1, col=x_i + 1)
                figure.update_xaxes(range=param_values_range[x_param],
                                    row=y_i + 1, col=x_i + 1)
                figure.update_yaxes(range=param_values_range[y_param],
                                    row=y_i + 1, col=x_i + 1)
                if _is_log_scale(trials, x_param):
                    log_range = [math.log10(p) for p in param_values_range[x_param]]
                    figure.update_xaxes(range=log_range, type='log', row=y_i + 1, col=x_i + 1)
                if _is_log_scale(trials, y_param):
                    log_range = [math.log10(p) for p in param_values_range[y_param]]
                    figure.update_yaxes(range=log_range, type='log', row=y_i + 1, col=x_i + 1)
                if x_i == 0:
                    figure.update_yaxes(title_text=y_param, row=y_i + 1, col=x_i + 1)
                if y_i == len(sorted_params) - 1:
                    figure.update_xaxes(title_text=x_param, row=y_i + 1, col=x_i + 1)

    return figure


def _generate_contour_subplot(trials, x_param, y_param, direction):
    # type: (List[FrozenTrial], str, str, StudyDirection) -> Tuple[Contour, Scatter]

    x_indices = sorted(list({t.params[x_param] for t in trials if x_param in t.params}))
    y_indices = sorted(list({t.params[y_param] for t in trials if y_param in t.params}))
    if len(x_indices) < 2:
        logger.warning('Param {} unique value length is less than 2.'.format(x_param))
        return go.Contour(), go.Scatter()
    if len(y_indices) < 2:
        logger.warning('Param {} unique value length is less than 2.'.format(y_param))
        return go.Contour(), go.Scatter()
    z = [[float('nan') for _ in range(len(x_indices))] for _ in range(len(y_indices))]

    x_values = []
    y_values = []
    for trial in trials:
        if x_param not in trial.params or y_param not in trial.params:
            continue
        x_values.append(trial.params[x_param])
        y_values.append(trial.params[y_param])
        x_i = x_indices.index(trial.params[x_param])
        y_i = y_indices.index(trial.params[y_param])
        if isinstance(trial.value, int):
            value = float(trial.value)
        elif isinstance(trial.value, float):
            value = trial.value
        else:
            raise ValueError(
                'Trial{} has COMPLETE state, but its value is non-numeric.'.format(trial.number))
        z[y_i][x_i] = value

    # TODO(Yanase): Use reversescale argument to reverse colorscale if Plotly's bug is fixed.
    # If contours_coloring='heatmap' is specified, reversesecale argument of go.Contour does not
    # work correctly. See https://github.com/pfnet/optuna/issues/606.
    colorscale = plotly.colors.PLOTLY_SCALES['Blues']
    if direction == StudyDirection.MINIMIZE:
        colorscale = [[1 - t[0], t[1]] for t in colorscale]
        colorscale.reverse()

    contour = go.Contour(
        x=x_indices, y=y_indices, z=z,
        colorbar={'title': 'Objective Value'},
        colorscale=colorscale,
        connectgaps=True,
        contours_coloring='heatmap',
        hoverinfo='none',
        line_smoothing=1.3,
    )

    scatter = go.Scatter(
        x=x_values,
        y=y_values,
        marker={'color': 'black'},
        mode='markers',
        showlegend=False
    )

    return (contour, scatter)


def plot_parallel_coordinate(study, params=None):
    # type: (Study, Optional[List[str]]) -> None
    """Plot the high-dimentional parameter relationships in a study.

        Note that, If a parameter contains missing values, a trial with missing values is not
        plotted.

    Example:

        The following code snippet shows how to plot the high-dimentional parameter relationships.

        .. code::

            import optuna

            def objective(trial):
                ...

            study = optuna.create_study()
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_parallel_coordinate(study, params=['param_a', 'param_b'])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.
    """

    _check_plotly_availability()
    figure = _get_parallel_coordinate_plot(study, params)
    figure.show()


def _get_parallel_coordinate_plot(study, params=None):
    # type: (Study, Optional[List[str]]) -> Figure

    layout = go.Layout(
        title='Parallel Coordinate Plot',
    )

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning('Your study does not have any completed trials.')
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is not None:
        for input_p_name in params:
            if input_p_name not in all_params:
                logger.warning('Parameter {} does not exist in your study.'.format(input_p_name))
                return go.Figure(data=[], layout=layout)
        all_params = set(params)
    sorted_params = sorted(list(all_params))

    dims = [{
        'label': 'Objective Value',
        'values': tuple([t.value for t in trials]),
        'range': (min([t.value for t in trials]), max([t.value for t in trials]))
    }]
    for p_name in sorted_params:
        values = []
        for t in trials:
            if p_name in t.params:
                values.append(t.params[p_name])
        is_categorical = False
        try:
            tuple(map(float, values))
        except (TypeError, ValueError):
            vocab = defaultdict(lambda: len(vocab))  # type: DefaultDict[str, int]
            values = [vocab[v] for v in values]
            is_categorical = True
        dim = {
            'label': p_name,
            'values': tuple(values),
            'range': (min(values), max(values))
        }
        if is_categorical:
            dim['tickvals'] = list(range(len(vocab)))
            dim['ticktext'] = list(sorted(vocab.items(), key=lambda x: x[1]))
        dims.append(dim)

    traces = [
        go.Parcoords(
            dimensions=dims,
            line=dict(
                color=[t.value for t in trials],
                colorscale='blues',
                colorbar=dict(
                    title='Objective Value'
                ),
                showscale=True,
                reversescale=study.direction == StudyDirection.MINIMIZE
            )
        )
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure


def plot_slice(study, params=None):
    # type: (Study, Optional[List[str]]) -> None
    """Plot the parameter relationship as slice plot in a study.

        Note that, If a parameter contains missing values, a trial with missing values is not
        plotted.

    Example:

        The following code snippet shows how to plot the parameter relationship as slice plot.

        .. code::

            import optuna

            def objective(trial):
                ...

            study = optuna.create_study()
            study.optimize(objective, n_trials=100)

            optuna.visualization.plot_slice(study, params=['param_a', 'param_b'])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.
    """

    _check_plotly_availability()
    figure = _get_slice_plot(study, params)
    figure.show()


def _get_slice_plot(study, params=None):
    # type: (Study, Optional[List[str]]) -> Figure

    layout = go.Layout(
        title='Slice Plot',
    )

    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]

    if len(trials) == 0:
        logger.warning('Your study does not have any completed trials.')
        return go.Figure(data=[], layout=layout)

    all_params = {p_name for t in trials for p_name in t.params.keys()}
    if params is None:
        sorted_params = sorted(list(all_params))
    else:
        for input_p_name in params:
            if input_p_name not in all_params:
                raise ValueError('Parameter {} does not exist in your study.'.format(input_p_name))
        sorted_params = sorted(list(set(params)))

    if len(sorted_params) == 1:
        figure = go.Figure(
            data=[_generate_slice_subplot(study, trials, sorted_params[0])],
            layout=layout
        )
        figure.update_xaxes(title_text=sorted_params[0])
        figure.update_yaxes(title_text='Objective Value')
        if _is_log_scale(trials, sorted_params[0]):
            figure.update_xaxes(type='log')
    else:
        figure = make_subplots(rows=1, cols=len(sorted_params), shared_yaxes=True)
        figure.update_layout(layout)
        showscale = True   # showscale option only needs to be specified once.
        for i, param in enumerate(sorted_params):
            trace = _generate_slice_subplot(study, trials, param)
            trace.update(marker=dict(showscale=showscale))  # showscale's default is True.
            if showscale:
                showscale = False
            figure.add_trace(trace, row=1, col=i + 1)
            figure.update_xaxes(title_text=param, row=1, col=i + 1)
            if i == 0:
                figure.update_yaxes(title_text='Objective Value', row=1, col=1)
            if _is_log_scale(trials, param):
                figure.update_xaxes(type='log', row=1, col=i + 1)

    return figure


def _is_log_scale(trials, param):
    # type: (List[FrozenTrial], str) -> bool

    return any(isinstance(t.distributions[param], LogUniformDistribution)
               for t in trials if param in t.params)


def _generate_slice_subplot(study, trials, param):
    # type: (Study, List[FrozenTrial], str) -> Scatter

    return go.Scatter(
        x=[t.params[param] for t in trials if param in t.params],
        y=[t.value for t in trials if param in t.params],
        mode='markers',
        marker={
            'line': {
                'width': 0.5,
                'color': 'Grey',
            },
            'color': [t.number for t in trials if param in t.params],
            'colorscale': 'Blues',
            'colorbar': {'title': '#Trials'}
        },
        showlegend=False,
    )


def _check_plotly_availability():
    # type: () -> None

    if not is_available():
        raise ImportError(
            'Plotly is not available. Please install plotly to use this feature. '
            'Plotly can be installed by executing `$ pip install plotly`. '
            'For further information, please refer to the installation guide of plotly. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')

    from distutils.version import StrictVersion
    from plotly import __version__ as plotly_version
    if StrictVersion(plotly_version) < StrictVersion('4.0.0'):
        raise ImportError(
            'Your version of Plotly is ' + plotly_version + ' . '
            'Please install plotly version 4.0.0 or higher. '
            'Plotly can be installed by executing `$ pip install -U plotly>=4.0.0`. '
            'For further information, please refer to the installation guide of plotly. ')
