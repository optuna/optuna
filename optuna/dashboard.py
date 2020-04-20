try:
    import bokeh.command.bootstrap
    import bokeh.document  # NOQA
    import bokeh.layouts
    import bokeh.models
    import bokeh.models.widgets
    import bokeh.plotting
    import bokeh.themes
    import tornado.gen

    _available = True
except ImportError as e:
    _available = False
    _import_error = e

import collections
import threading
import time

import numpy as np

import optuna.logging
import optuna.study
from optuna.study import StudyDirection
import optuna.trial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

_mode = None  # type: Optional[str]
_study = None  # type: Optional[optuna.study.Study]

_HEADER_FORMAT = """
<style>
body {{
    margin: 20px;
}}
h1, p {{
    margin: 10px 0px;
}}
</style>

<h1>Optuna Dashboard (Beta)</h1>
<p>
<b>Study name:</b> {study_name}<br>
</p>
"""

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

if _available:

    class _CompleteTrialsWidget(object):
        def __init__(self, trials, direction):
            # type: (List[optuna.trial.FrozenTrial], StudyDirection) -> None

            complete_trials = [
                trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE
            ]
            self.trial_ids = set([trial._trial_id for trial in complete_trials])

            self.direction = direction
            values = [trial.value for trial in complete_trials]
            if direction == StudyDirection.MINIMIZE:
                best_values = np.minimum.accumulate(values, axis=0)
            else:
                best_values = np.maximum.accumulate(values, axis=0)

            self.cds = bokeh.models.ColumnDataSource(
                {
                    "#": list(range(len(complete_trials))),
                    "value": values,
                    "best_value": best_values,
                }
            )

            self.best_value = best_values[-1] if complete_trials else np.inf

        def create_figure(self):
            # type: () -> bokeh.plotting.Figure

            figure = bokeh.plotting.figure(height=150)
            figure.circle(x="#", y="value", source=self.cds, alpha=0.3, color="navy")
            figure.line(x="#", y="best_value", source=self.cds, color="firebrick")
            figure.xaxis[0].axis_label = "Number of Trials"
            figure.yaxis[0].axis_label = "Objective Value"
            return figure

        def update(self, new_trials):
            # type: (List[optuna.trial.FrozenTrial]) -> None

            stream_dict = collections.defaultdict(list)  # type: Dict[str, List[Any]]

            for trial in new_trials:
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    continue
                if trial._trial_id in self.trial_ids:
                    continue
                stream_dict["#"].append(len(self.trial_ids))
                stream_dict["value"].append(trial.value)
                if self.direction == StudyDirection.MINIMIZE:
                    self.best_value = min(self.best_value, trial.value)
                else:
                    self.best_value = max(self.best_value, trial.value)
                stream_dict["best_value"].append(self.best_value)
                self.trial_ids.add(trial._trial_id)

            if stream_dict:
                self.cds.stream(stream_dict)

    class _AllTrialsWidget(object):
        def __init__(self, trials):
            # type: (List[optuna.trial.FrozenTrial]) -> None

            self.cds = bokeh.models.ColumnDataSource(self.trials_to_dict(trials))

        def create_table(self):
            # type: () -> bokeh.models.widgets.DataTable

            return bokeh.models.widgets.DataTable(
                source=self.cds,
                columns=[
                    bokeh.models.widgets.TableColumn(field=field, title=field)
                    for field in [
                        "number",
                        "state",
                        "value",
                        "params",
                        "datetime_start",
                        "datetime_complete",
                    ]
                ],
            )

        def update(
            self,
            old_trials,  # type: List[optuna.trial.FrozenTrial]
            new_trials,  # type: List[optuna.trial.FrozenTrial]
        ):
            # type: (...) -> None

            modified_indices = []
            modified_trials = []
            for i, old_trial in enumerate(old_trials):
                new_trial = new_trials[i]
                if old_trial != new_trial:
                    modified_indices.append(i)
                    modified_trials.append(new_trial)

            patch_dict = self.trials_to_dict(modified_trials)
            patch_dict = {k: list(zip(modified_indices, v)) for k, v in patch_dict.items()}
            self.cds.patch(patch_dict)

            self.cds.stream(self.trials_to_dict(new_trials[len(old_trials) :]))

        @staticmethod
        def trials_to_dict(trials):
            # type: (List[optuna.trial.FrozenTrial]) -> Dict[str, List[Any]]

            return {
                "number": [trial.number for trial in trials],
                "state": [trial.state.name for trial in trials],
                "value": [trial.value for trial in trials],
                "params": [str(trial.params) for trial in trials],
                "datetime_start": [
                    trial.datetime_start.strftime(_DATETIME_FORMAT)
                    if trial.datetime_start is not None
                    else None
                    for trial in trials
                ],
                "datetime_complete": [
                    trial.datetime_complete.strftime(_DATETIME_FORMAT)
                    if trial.datetime_complete is not None
                    else None
                    for trial in trials
                ],
            }

    class _DashboardApp(object):
        def __init__(self, study, launch_update_thread):
            # type: (optuna.study.Study, bool) -> None

            self.study = study
            self.launch_update_thread = launch_update_thread
            self.lock = threading.Lock()

        def __call__(self, doc):
            # type: (bokeh.document.Document) -> None

            self.doc = doc
            self.current_trials = (
                self.study.trials
            )  # type: Optional[List[optuna.trial.FrozenTrial]]
            self.new_trials = None  # type: Optional[List[optuna.trial.FrozenTrial]]
            self.complete_trials_widget = _CompleteTrialsWidget(
                self.current_trials, self.study.direction
            )
            self.all_trials_widget = _AllTrialsWidget(self.current_trials)

            self.doc.title = "Optuna Dashboard (Beta)"
            header = _HEADER_FORMAT.format(study_name=self.study.study_name)
            self.doc.add_root(
                bokeh.layouts.layout(
                    [
                        [bokeh.models.widgets.Div(text=header)],
                        [self.complete_trials_widget.create_figure()],
                        [self.all_trials_widget.create_table()],
                    ],
                    sizing_mode="scale_width",
                )
            )

            if self.launch_update_thread:
                thread = threading.Thread(target=self.thread_loop)
                thread.daemon = True
                thread.start()

        def thread_loop(self):
            # type: () -> None

            while True:
                time.sleep(1)
                new_trials = self.study.trials
                with self.lock:
                    need_to_add_callback = self.new_trials is None
                    self.new_trials = new_trials
                    if need_to_add_callback:
                        self.doc.add_next_tick_callback(self.update_callback)

        @tornado.gen.coroutine
        def update_callback(self):
            # type: () -> None

            with self.lock:
                current_trials = self.current_trials
                new_trials = self.new_trials
                self.current_trials = self.new_trials
                self.new_trials = None

            assert current_trials is not None
            assert new_trials is not None
            self.complete_trials_widget.update(new_trials)
            self.all_trials_widget.update(current_trials, new_trials)


def _check_bokeh_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "Bokeh is not available. Please install Bokeh to use the dashboard. "
            "Bokeh can be installed by executing `$ pip install bokeh`. "
            "For further information, please refer to the installation guide of Bokeh. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )


def _show_experimental_warning():
    # type: () -> None

    logger = optuna.logging.get_logger(__name__)
    logger.warning("Optuna dashboard is still highly experimental. Please use with caution!")


def _get_this_source_path():
    # type: () -> str

    path = __file__

    # Sometimes __file__ points to a *.pyc file, but Bokeh doesn't accept it.
    if path.endswith(".pyc"):
        path = path[:-1]
    return path


def _serve(study, bokeh_allow_websocket_origins):
    # type: (optuna.study.Study, List[str]) -> None

    global _mode, _study

    _check_bokeh_availability()
    _show_experimental_warning()

    # We want to pass the mode (launching a server? or, just writing an HTML?) and a target study
    # to our Bokeh app. Unfortunately, as we are using `bokeh.command.bootstrap.main` to launch
    # our Bokeh app, we cannot directly pass Python objects to it. Therefore, we have no choice but
    # to use global variables to pass them.
    _mode = "serve"
    _study = study

    # TODO(akiba): Stop using Bokeh's CLI entry point, and start the HTTP server by ourselves.

    # This is not a very clean way to launch Bokeh server.
    # Another seemingly better way is to
    # instantiate and launch `bokeh.server.server.Server` by ourselves. However, in this way,
    # for some reason, we found that the CDS update is not reflected to browsers, at least on Bokeh
    # version 0.12.15. In addition, we will need to do many configuration to servers, which can be
    # done automatically with the following one line. So, for now, we decided to use this way.
    command = ["bokeh", "serve", "--show", _get_this_source_path()]
    for bokeh_allow_websocket_origin in bokeh_allow_websocket_origins:
        command.extend(["--allow-websocket-origin", bokeh_allow_websocket_origin])
    bokeh.command.bootstrap.main(command)


def _write(study, out_path):
    # type: (optuna.study.Study, str) -> None

    global _mode, _study

    _check_bokeh_availability()
    _show_experimental_warning()

    _mode = "html"
    _study = study
    bokeh.command.bootstrap.main(["bokeh", "html", _get_this_source_path(), "-o", out_path])


def _run():
    # type: () -> None

    # Please note that `_study` and `optuna.dashboard._study` are different here. Here, this module
    # is loaded inside Bokeh, and thus it is not `optuna.dashboard`, but `bk_script_????`.
    study = optuna.dashboard._study
    mode = optuna.dashboard._mode

    assert study is not None
    app = _DashboardApp(study, launch_update_thread=(mode == "serve"))
    doc = bokeh.plotting.curdoc()
    app(doc)


if __name__.startswith("bk_script_"):
    # Here, this module is loaded inside Bokeh. Therefore, we should launch the Bokeh app.
    _run()
