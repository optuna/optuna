import mlflow


class MlflowCallback():

    def __init__(self, tracking_uri=None, experiment=None, metric_name=None):
        self._tracking_uri = tracking_uri
        self._experiment = experiment
        self._metric_name = metric_name

    def __call__(self, study, trial):
        # set experiment
        if self._experiment is None:
            if study.study_name is None or study.study_name == 'no-name-00000000-0000-0000-0000-000000000000':
                raise ValueError('Either \'experiment\' or \'study.study_name\' must be set!')
            else:
                mlflow.set_experiment(study.study_name)
        else:
            mlflow.set_experiment(self._experiment)

        # set tracking_uri
        if self._tracking_uri is not None:
            mlflow.set_tracking_uri(self._tracking_uri)

        with mlflow.start_run(run_name=trial.number):
            # set metric
            trial_value = trial.value if trial.value is not None else float('nan')
            metric_name = self._metric_name if self._metric_name is not None else 'trial_value'
            mlflow.log_metric(metric_name, trial_value)

            # set params
            mlflow.log_params(trial.params)

            # set tags
            tags = {}
            tags['trial_number'] = trial.number
            tags['trial_datetime_start'] = trial.datetime_start
            tags['trial_datetime_complete'] = trial.datetime_complete
            tags['trial_state'] = trial.state
            tags['study_direction'] = study.direction
            tags.update(trial.user_attrs)
            tags.update(trial.distributions)
            mlflow.set_tags(tags)
