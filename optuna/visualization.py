import time

from optuna.logging import get_logger
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.study import Study  # NOQA
from optuna import type_checking

logger = get_logger(__name__)

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

try:
    from IPython.display import display, HTML
    import plotly.graph_objs as go
    from plotly.graph_objs._figure import Figure  # NOQA
    from plotly.offline import init_notebook_mode
    _available = True
except ImportError as e:
    _import_error = e
    # Visualization features are disabled because plotly is not available.
    _available = False


def plot_intermediate_values(study):
    # type: (Study) -> None
    """Plot intermediate values of all trials in a study.

    Example:

        The following code snippet shows how to plot intermediate values inside Jupyter Notebook.

        .. code::

            import optuna

            def objective(trial):
                # Intermediate values are supposed to be reported inside the objective function.
                ...

            study = optuna.create_study()
            study.optimize(objective ,n_trials=100)

            optuna.visualization.plot_intermediate_values(study)

    Args:
        study:
            A :class:`~optuna.study.Study` object whose trials are plotted for their intermediate
            values.
    """

    _check_plotly_availability()
    init_notebook_mode(connected=True)
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

    trials = study.trials

    if len(trials) == 0:
        logger.warning('Study instance does not contain trials.')
        return go.Figure(data=[], layout=layout)
    if hasattr(trials[0], 'intermediate_values') is False:
        logger.warning(
            'You need to set up the pruning feature to utilize plot_intermediate_values()')
        return go.Figure(data=[], layout=layout)

    target_state = [TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING]
    trials = [trial for trial in trials if trial.state in target_state]
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

        The following code snippet shows how to plot optimization history inside Jupyter Notebook.

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
    init_notebook_mode(connected=True)
    figure = _get_optimization_history_plot(study)
    figure.show()


def _get_optimization_history_plot(study):
    # type: (Study) -> Figure

    layout = go.Layout(
        title='Optimization History Plot',
        xaxis={'title': 'Number of Trial'},
        yaxis={'title': 'Objective Value'},
    )

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    best_values = [float('inf')] if study.direction == StudyDirection.MINIMIZE else [-float('inf')]
    for trial in trials:
        if isinstance(trial.value, int):
            trial_value = float(trial.value)
        elif isinstance(trial.value, float):
            trial_value = trial.value
        else:
            raise ValueError(
                'Trial{} has COMPLETE state, but its value is non-numeric.'.format(trial.number))
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


def _check_plotly_availability():
    # type: () -> None

    if not _available:
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


class ProgressBar:
    def __init__(self, gen):
        self.gen = gen
        if self.gen <= 0 or isinstance(self.gen, int) == False:
            print('gen must be an int and greater than 1.')
        self.now_iter = 0
        self.progress = self._html_progress_bar(self.now_iter, '')
        self.out = display(HTML(self.progress), display_id=True)
        self.start_t = time.time()
        self.comment = ''
        self.text = ''

    # TODO: get Lock
    def update(self):
        self.now_iter += 1
        if self.now_iter > self.gen:
            print('Now iter is greater than gen.')
            return False
        cur_t = time.time()
        avg_t = (cur_t - self.start_t) / self.now_iter
        self.pred_t = avg_t * self.gen
        self.last_t = cur_t
        self._update_bar()
        if self.now_iter == self.gen:
            self._on_iter_end()

    def _update_bar(self):
        elapsed_t = self.last_t - self.start_t
        remaining_t = self._format_time(self.pred_t - elapsed_t)
        elapsed_t = self._format_time(elapsed_t)
        end = '' if len(self.comment) == 0 else f' {self.comment}'
        self._on_update(
            f'{100 * self.now_iter/self.gen:.2f}% [{self.now_iter}/{self.gen} {elapsed_t}<{remaining_t}{end}]')

    def _on_update(self, text):
        self.progress = self._html_progress_bar(self.now_iter, text)
        self.out.update(HTML(self.progress))

    def _on_iter_end(self):
        if self.text.endswith('<p>'):
            self.text = self.text[:-3]
        self.out.update(HTML(self.text))

    def _html_progress_bar(self, value, label):
        return f"""
        <div>
            <style>
                /* Turns off some styling */
                progress {{
                    /* gets rid of default border in Firefox and Opera. */
                    border: none;
                    /* Needs to be in here for Safari polyfill so background images work as expected. */
                    background-size: auto;
                }}
            </style>
        <progress value='{value}' max='{self.gen}', style='width:300px; height:20px; vertical-align: middle;'></progress>
        {label}
        </div>
        """

    def _format_time(self, time):
        time = int(time)
        hour, minute, second = time//3600, (time//60) % 60, time % 60
        if hour != 0:
            return f'{hour}:{minute:02d}:{second:02d}'
        else:
            return f'{minute:02d}:{second:02d}'


class ProgressPlot:
    def __init__(self, study, plot_type=['history']):
        self.plot_type = plot_type
        self.figures = self._initialize_figures(study)
        init_notebook_mode(connected=True)
        for figure in self.figures:
            display(figure, display_id=True)

    def update(self, study):
        df = study.trials_dataframe()
        df = df.rename(columns={'number': 'trial_id'})
        for i, plot in enumerate(self.plot_type):
            if plot == 'history':
                updated_figure = _get_optimization_history_plot(study)
                if len(self.figures[i].data) == 0:
                    self.figures[i].add_scatter(updated_figure.data[0])
                    self.figures[i].add_scatter(updated_figure.data[1])
                else:
                    self.figures[i].data[0].x = updated_figure.data[0].x
                    self.figures[i].data[0].y = updated_figure.data[0].y
                    self.figures[i].data[1].x = updated_figure.data[1].x
                    self.figures[i].data[1].y = updated_figure.data[1].y
            elif plot == 'intermediate_value':
                updated_figure = _get_intermediate_plot(study)
                self.figures[i].data = updated_figure.data
            elif plot == 'parallel_coordinate':
                updated_figure = _get_parallel_coordinate_plot(study)
                self.figures[i].data = updated_figure.data
            elif plot == 'contour':
                df = study.trials_dataframe()
                params = df['params'].columns
                updated_figure = _get_contour_plot(study, params)
                self.figures[i].data = updated_figure.data
            elif plot == 'slice':
                df = study.trials_dataframe()
                params = df['params'].columns
                updated_figure = _get_slice_plot(study, params)
                self.figures[i].data = updated_figure.data

    def _initialize_figures(self, study):
        figures = []
        for plot in self.plot_type:
            if plot == 'history':
                figures.append(go.FigureWidget(_get_optimization_history_plot(study)))
            elif plot == 'intermediate_value':
                figures.append(go.FigureWidget(_get_intermediate_plot(study)))
            elif plot == 'parallel_coordinate':
                figures.append(go.FigureWidget(_get_parallel_coordinate_plot(study)))
            elif plot == 'contour':
                df = study.trials_dataframe()
                params = df['params'].columns
                figures.append(go.FigureWidget(_get_contour_plot(study, params)))
            elif plot == 'slice':
                df = study.trials_dataframe()
                params = df['params'].columns
                figures.append(go.FigureWidget(_get_slice_plot(study, params)))

        return figures
