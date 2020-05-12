import optuna


class DeterministicPruner(optuna.pruners.BasePruner):
    def __init__(self, is_pruning, interval=1):
        # type: (bool, int) -> None

        self.is_pruning = is_pruning
        self._interval = interval

    def prune(self, study, trial):
        # type: (optuna.study.Study, optuna.trial.FrozenTrial) -> bool

        return self.is_pruning

    def is_target_step(self, step):
        # type: (int) -> bool

        return step % self._interval == 0


def create_running_trial(study, value):
    # type: (optuna.study.Study, float) -> optuna.trial.Trial

    trial_id = study._storage.create_new_trial(study._study_id)
    study._storage.set_trial_value(trial_id, value)
    return optuna.trial.Trial(study, trial_id)
