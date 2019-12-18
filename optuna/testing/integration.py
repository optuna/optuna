import optuna


class DeterministicPruner(optuna.pruners.BasePruner):
    def __init__(self, is_pruning):
        # type: (bool) -> None

        self.is_pruning = is_pruning

    def prune(self, study, trial):
        # type: (optuna.study.Study, optuna.structs.FrozenTrial) -> bool

        return self.is_pruning

    def get_trial_pruner_auxiliary_data(self, study_name, trial_number):
        # type: (str, int) -> str

        return ''

    def should_filter_trials(self):
        # type: () -> bool

        return False


def create_running_trial(study, value):
    # type: (optuna.study.Study, float) -> optuna.trial.Trial

    trial_id = study._storage.create_new_trial(study._study_id)
    study._storage.set_trial_value(trial_id, value)
    return optuna.trial.Trial(study, trial_id)
