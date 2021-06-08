from typing import Any

from packaging import version

import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import catboost as cb

    if version.parse(cb.__version__) < version.parse("0.26"):
        raise NotImplementedError(
            f"This function is available since version 0.26!  catboost version: {cb.__version__}"
        )


class CatBoostPruningCallback(object):
    def __init__(self, trial: optuna.trial.Trial, observation_key: str, metric: str) -> None:
        self._trial = trial
        self._observation_key = observation_key
        self._metric = metric
        self._pruned = False
        self._message = ""

    def after_iteration(self, info: Any) -> bool:
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
        return True

    def check_pruned(self) -> None:
        if self._pruned is True:
            raise optuna.TrialPruned(self._message)
