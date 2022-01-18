from typing import Any

from packaging import version

import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import catboost as cb

    if version.parse(cb.__version__) < version.parse("0.26"):
        raise ImportError(f"You don't have CatBoost installed! CatBoost version: {cb.__version__}")


class CatBoostPruningCallback(object):
    """Callback for catboost to prune unpromising trials.

    If TrialPruned is raised at after_iteration in CatBoostPruningCallback, then catboost exits
    with error. So we add `checked_pruned` function in which TrialPruned is raised if pruning is
    nessasary.
    You must call `check_pruned` after training manually unlike other pruning callbacks.

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
            The name of the target validation.
            If you set only one ``eval_set``, ``validation`` is used.
            If you set multiple datasets as ``eval_set``, the index number of ``eval_set`` must be
            included in the valid_name, e.g., ``validation_0`` or ``validation_1``
            when ``eval_set`` contains two datasets.
    """

    def __init__(
        self, trial: optuna.trial.Trial, metric: str, valid_name: str = "validation"
    ) -> None:
        self._trial = trial
        self._metric = metric
        self._valid_name = valid_name
        self._pruned = False
        self._message = ""

    def after_iteration(self, info: Any) -> bool:
        """Run after each iteration.

        Args:
            info:
                A ``simpleNamespace`` containing iteraion, ``validation_name``, ``metric_name``
                and history of losses.
                For example ``namespace(iteration=2, metrics= {
                'learn': {'Logloss': [0.6, 0.5]},
                'validation': {'Logloss': [0.7, 0.6], 'AUC': [0.8, 0.9]}
                })``.

        Returns:
            A boolean value. If :obj:`True`, the trial should be pruned.
            Otherwise, the trial should continue.
        """
        step = info.iteration - 1
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
        current_score = metrics[self._metric][-1]
        self._trial.report(current_score, step=step)
        if self._trial.should_prune():
            self._message = "Trial was pruned at iteration {}.".format(step)
            self._pruned = True
            return False
        return True

    def check_pruned(self) -> None:
        """Check whether pruend."""
        if self._pruned:
            raise optuna.TrialPruned(self._message)
