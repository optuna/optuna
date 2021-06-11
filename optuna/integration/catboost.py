from typing import Any

from packaging import version

import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import catboost as cb

    if version.parse(cb.__version__) < version.parse("0.26"):
        raise ImportError(
            f"This function is available since version 0.26!  catboost version: {cb.__version__}"
        )

_doc = """Callback for catoost to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    catboost/catboost_integration.py>`__
    if you want to add a pruning handler which observes validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        metric:
            An evaluation metric for pruning, e.g., ``Logloss`` and ``AUC``.
            Please refer to
            `CatBoost reference
            <https://catboost.ai/docs/references/eval-metric__supported-metrics.html>`_
            for further details.
        valid_name:
            Validation names.
            If you set only one eval_set, ``validation`` is used.
            If you set multiple eval_sets, the index number of ``eval_set`` must be
            included in the valid_name, e.g., ``validation_0`` and ``validation_0``
    """


class CatBoostPruningCallback(object):
    def __init__(
        self, trial: optuna.trial.Trial, metric: str, valid_name: str = "validation"
    ) -> None:
        self._trial = trial
        self._metric = metric
        self._valid_name = valid_name
        self._pruned = False
        self._message = ""

    def after_iteration(self, info: Any) -> bool:
        epoch = info.iteration - 1
        if self._valid_name not in info.metrics:
            raise ValueError(
                'The entry associated with the validation name "{}" '
                "is not found in the evaluation result list {}.".format(self._valid_name, info)
            )
        metrics = info.metrics[self._valid_name]
        if self._metric not in metrics:
            raise ValueError(
                'The entry associated with the metric name "{}" '
                "is not found in the evaluation result list {}.".format(self._metric, info)
            )
        scores = metrics[self._metric]
        current_score = scores[-1]
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
