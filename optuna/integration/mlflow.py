import mlflow

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA


class MlflowCallback(object):
    """Callback to track optuna trials with Mlflow.

    This callback adds relevant information that is
    tracked by Optuna to Mlflow.

    Example:

        Add Mlflow callback to optuna optimization.

        .. testcode::

            def objective(trial):
                x = trial.suggest_uniform('x', -10, 10)
                return (x - 2) ** 2

            mlflc = MlflowCallback(
                tracking_uri='https://my-tracking-server:5000',
                metric_name='my metric score',
                )

            study = optuna.create_study(study_name='my_study')
            study.optimize(objective, n_trials=10, callbacks=[mlflc])

    Args:
        tracking_uri:
            Set the tracking server URI.

            - An empty string, or a local file path, prefixed with ``file:/``. Data is stored
              locally at the provided file (or ``./mlruns`` if empty).
            - An HTTP URI like ``https://my-tracking-server:5000``.
            - A Databricks workspace, provided as the string ``databricks`` or, to use a
              Databricks CLI.
              `profile <https://github.com/databricks/databricks-cli#installation>`_,
              ``databricks://<profileName>``.
        experiment:
            Name of Mlflow experiment to be activated. If not set ``study.study_name``
            will be taken. If ``study.study_name`` is not set the Mlflow default will be used.
        metric_name:
            Name of the metric. If not provided this will be called ``trial_value``.
    """

    def __init__(self, tracking_uri=None, experiment=None, metric_name=None):
        # type: (str, str, str) -> None

        self._tracking_uri = tracking_uri
        self._experiment = experiment
        self._metric_name = metric_name

    def __call__(self, study, trial):
        # type: (optuna.study.Study, optuna.structs.FrozenTrial) -> None

        # This sets the tracking_uri for Mlflow.
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)

        # This sets the experiment of Mlflow.
        if self._experiment is not None:
            mlflow.set_experiment(self._experiment)
        elif study.study_name is not None:
            mlflow.set_experiment(study.study_name)

        with mlflow.start_run(run_name=trial.number):

            # This sets the metric for Mlflow.
            trial_value = trial.value if trial.value is not None else float("nan")
            metric_name = self._metric_name if self._metric_name is not None else "trial_value"
            mlflow.log_metric(metric_name, trial_value)

            # This sets the params for Mlflow.
            mlflow.log_params(trial.params)

            # This sets the tags for Mlflow.
            tags = {}  # type: Dict[str, str]
            tags["trial_number"] = str(trial.number)
            tags["trial_datetime_start"] = str(trial.datetime_start)
            tags["trial_datetime_complete"] = str(trial.datetime_complete)
            tags["trial_state"] = str(trial.state)
            tags["study_direction"] = str(study.direction)
            tags.update(trial.user_attrs)
            distributions = {k: str(v) for (k, v) in trial.distributions.items()}
            tags.update(distributions)
            mlflow.set_tags(tags)
