from typing import Any

import optuna


with optuna._imports.try_import() as _imports:
    import catboost

class CatBoostPruningCallback(object):

    def __init__(self, trial: optuna.trial.Trial, observation_key: str, metric: str) -> None:
        self._trial = trial
        self._observation_key = observation_key
        self._metric = metric
        self._pruned = False
        self._message = ""

    def after_iteration(self, info):
        metrics = info.metrics[self._observation_key]
        scores = metrics[self._metric]
        current_score = scores[-1]
        epoch = info.iteration - 1
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at iteration {}.".format(epoch)
            self._message = message
            self._pruned = True
            return False
#            print(message)
#            return False
            # The training should not stop.
        return True

    def check_pruned(self):
        if self._pruned is True:
            raise optuna.TrialPruned(self._message)



