import optuna


class DeterministicPruner(optuna.pruners.BasePruner):
    def __init__(self, is_pruning: bool, interval: int = 1) -> None:

        self.is_pruning = is_pruning
        self.interval = interval

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:

        step = trial.last_step
        if step is None:
            return False

        if not self.is_taret_step(step, trial):
            return False

        return self.is_pruning

    def is_taret_step(self, step: int, trial: "optuna.trial.FrozenTrial") -> bool:
        return (step + 1) % self.interval == 0


def create_running_trial(study: "optuna.study.Study", value: float) -> optuna.trial.Trial:

    trial_id = study._storage.create_new_trial(study._study_id)
    study._storage.set_trial_values(trial_id, [value])
    return optuna.trial.Trial(study, trial_id)
