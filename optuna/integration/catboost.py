from typing import Any

from packaging import version

import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import catboost as cb

    if version.parse(cb.__version__) < version.parse("0.26"):
        raise ImportError(f"You don't have CatBoost>=0.26! CatBoost version: {cb.__version__}")


class CatBoostPruningCallback(object):
    """Callback for catboost to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    catboost/catboost_pruning.py>`__
    if you want to add a pruning callback which observes validation accuracy of
    a CatBoost model.

    If :class:`optuna.TrialPruned` is raised in ``after_iteration`` via
    ``CatBoostPruningCallback``, then catboost exits.
    You must call ``check_pruned`` after training manually unlike other pruning callbacks
    to raise :class:`optuna.TrialPruned`.

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
        eval_set_index:
            The index of the target validation dataset.
            If you set only one ``eval_set``, ``eval_set_index`` is None.
            If you set multiple datasets as ``eval_set``, the index of ``eval_set`` must be
            ``eval_set_index``, e.g., ``0`` or ``1`` when ``eval_set`` contains two datasets.
    """

    def __init__(self, trial: optuna.trial.Trial, metric: str, eval_set_index: int = None) -> None:
        default_valid_name = "validation"
        self._trial = trial
        self._metric = metric
        if eval_set_index is None:
            self._valid_name = default_valid_name
        else:
            self._valid_name = default_valid_name + "_" + str(eval_set_index)
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
