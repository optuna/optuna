import optuna
from optuna._experimental import experimental
from optuna import structs
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

try:
    import mlflow

    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    mlflow = object


def _check_mlflow_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "MLflow is not available. Please install MLflow to use this "
            "feature. It can be installed by executing `$ pip install "
            "mlflow`. For further information, please refer to the installation guide "
            "of MLflow. (The actual import error is as follows: " + str(_import_error) + ")"
        )


@experimental("1.4.0")
class MLflowCallback(object):
    """Callback to track optuna trials with MLflow.

    This callback adds relevant information that is
    tracked by Optuna to MLflow.

    Example:

        Add MLflow callback to optuna optimization.

        .. testsetup::

            import pathlib
            import tempfile
            tempdir = tempfile.mkdtemp()
            YOUR_TRACKING_URI = pathlib.Path(tempdir).as_uri()

        .. testcode::

            import optuna
            from optuna.integration.mlflow import MLflowCallback

            def objective(trial):
                x = trial.suggest_uniform('x', -10, 10)
                return (x - 2) ** 2

            mlflc = MLflowCallback(
                tracking_uri=YOUR_TRACKING_URI,
                metric_name='my metric score',
            )

            study = optuna.create_study(study_name='my_study')
            study.optimize(objective, n_trials=10, callbacks=[mlflc])

        .. testcleanup::

            import shutil
            shutil.rmtree(tempdir)

        .. testoutput::
            :hide:
            :options: +NORMALIZE_WHITESPACE

            INFO: 'my_study' does not exist. Creating a new experiment

    Args:
        tracking_uri:
            The URI of the MLflow tracking server.

            Please refer to `mlflow.set_tracking_uri
            <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri>`_
            for more details.
        experiment_name:
            Name of MLflow experiment to be activated. If not set ``study.study_name``
            will be taken. Either ``experiment`` or ``study.study_name`` must be set.
        metric_name:
            Name of the metric. If not provided this will be called ``trial_value``.
    """

    def __init__(self, tracking_uri=None, experiment_name=None, metric_name=None):
        # type: (Optional[str], Optional[str], Optional[str]) -> None

        _check_mlflow_availability()

        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._metric_name = metric_name

    def __call__(self, study, trial):
        # type: (optuna.study.Study, optuna.structs.FrozenTrial) -> None

        # This sets the tracking_uri for MLflow.
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)

        # This sets the experiment of MLflow.
        if self._experiment_name is not None:
            mlflow.set_experiment(self._experiment_name)
        elif (
            study.study_name is not None
            and study.study_name != "no-name-00000000-0000-0000-0000-000000000000"
        ):
            mlflow.set_experiment(study.study_name)
        else:
            raise ValueError("Either 'experiment' or 'study.study_name' must be set!")

        with mlflow.start_run(run_name=str(trial.number)):

            # This sets the metric for MLflow.
            trial_value = trial.value if trial.value is not None else float("nan")
            metric_name = self._metric_name if self._metric_name is not None else "value"
            mlflow.log_metric(metric_name, trial_value)

            # This sets the params for MLflow.
            mlflow.log_params(trial.params)

            # This sets the tags for MLflow.
            tags = {}  # type: Dict[str, str]
            tags["number"] = str(trial.number)
            tags["datetime_start"] = str(trial.datetime_start)
            tags["datetime_complete"] = str(trial.datetime_complete)

            # Set TrialState and convert it to str and remove the common prefix.
            trial_state = trial.state
            if isinstance(trial_state, structs.TrialState):
                tags["state"] = str(trial_state).split(".")[-1]

            tags["direction"] = str(study.direction)
            tags.update(trial.user_attrs)
            distributions = {
                (k + "_distribution"): str(v) for (k, v) in trial.distributions.items()
            }
            tags.update(distributions)
            mlflow.set_tags(tags)
