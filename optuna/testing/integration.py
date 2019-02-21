import optuna


class DeterministicPruner(optuna.pruners.BasePruner):
    def __init__(self, is_pruning):
        # type: (bool) -> None

        self.is_pruning = is_pruning

    def prune(self, storage, study_id, trial_id, step):
        # type: (optuna.storages.BaseStorage, int, int, int) -> bool

        return self.is_pruning
