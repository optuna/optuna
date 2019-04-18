import optuna


class DeterministicPruner(optuna.pruners.BasePruner):
    def __init__(self, is_pruning):
        # type: (bool) -> None

        self.is_pruning = is_pruning

    def prune(self, storage, study_id, trial_id, step):
        # type: (optuna.storages.BaseStorage, int, int, int) -> bool

        return self.is_pruning


def create_running_trial(study, value):
    # type: (optuna.study.Study, float) -> optuna.trial.Trial

    trial_id = study.storage.create_new_trial_id(study.study_id)
    study.storage.set_trial_value(trial_id, value)
    return optuna.trial.Trial(study, trial_id)
