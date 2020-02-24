from collections import defaultdict

from optuna.logging import get_logger
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna import type_checking
from optuna.visualization.utils import _check_plotly_availability
from optuna.visualization.utils import is_available

if type_checking.TYPE_CHECKING:
    from typing import DefaultDict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna.study import Study  # NOQA

if is_available():
    from optuna.visualization.plotly_imports import go

logger = get_logger(__name__)


def plot_parallel_coordinate(study, params=None):
    # type: (Study, Optional[List[str]]) -> go.Figure
    """Plot the high-dimentional parameter relationships in a study.

    Note that, If a parameter contains missing values, a trial with missing values is not plotted.

    Example:

        The following code snippet shows how to plot the high-dimentional parameter relationships.

        .. testcode::

            import optuna

            def objective(trial):
                x = trial.suggest_uniform('x', -100, 100)
                y = trial.suggest_categorical('y', [-1, 0, 1])
                return x ** 2 + y

            study = optuna.create_study()
            study.optimize(objective, n_trials=10)

            optuna.visualization.plot_parallel_coordinate(study, params=['x', 'y'])

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their objective
            values.
        params:
            Parameter list to visualize. The default is all parameters.

    Returns:
        A :class:`plotly.graph_objs.Figure` object.
    """

    _check_plotly_availability()
    return _get_parallel_coordinate_plot(study, params)


def _get_parallel_coordinate_plot(study, params=None):
    # type: (Study, Optional[List[str]]) -> go.Figure

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
                ValueError('Parameter {} does not exist in your study.'.format(input_p_name))
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
            line={
                'color': dims[0]['values'],
                'colorscale': 'blues',
                'colorbar': {'title': 'Objective Value'},
                'showscale': True,
                'reversescale': study.direction == StudyDirection.MINIMIZE,
            }
        )
    ]

    figure = go.Figure(data=traces, layout=layout)

    return figure
