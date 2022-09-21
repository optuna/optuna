from typing import Any
from typing import Optional

from packaging import version

import optuna
from optuna._experimental import experimental_class
from optuna._imports import try_import


with try_import() as _imports:
    import catboost as cb

    if version.parse(cb.__version__) < version.parse("0.26"):
        raise ImportError(f"You don't have CatBoost>=0.26! CatBoost version: {cb.__version__}")


@experimental_class("3.0.0")
class CatBoostPruningCallback:
    """Callback for catboost to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    catboost/catboost_pruning.py>`__
    if you want to add a pruning callback which observes validation accuracy of
    a CatBoost model.

    .. note::
        :class:`optuna.TrialPruned` cannot be raised
        in :meth:`~optuna.integration.CatBoostPruningCallback.after_iteration`
        that is called in CatBoost via ``CatBoostPruningCallback``.
        You must call :meth:`~optuna.integration.CatBoostPruningCallback.check_pruned`
        after training manually unlike other pruning callbacks
        to raise :class:`optuna.TrialPruned`.

    .. note::
        This callback cannot be used with CatBoost on GPUs because CatBoost doesn't support
        a user-defined callback for GPU.
        Please refer to `CatBoost issue <https://github.com/catboost/catboost/issues/1792>`_.

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

    def __init__(
        self, trial: optuna.trial.Trial, metric: str, eval_set_index: Optional[int] = None
    ) -> None:
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
        """Report an evaluation metric value for Optuna pruning after each CatBoost's iteration.

        This method is called by CatBoost.

        Args:
            info:
                A ``SimpleNamespace`` containing iteraion, ``validation_name``, ``metric_name``
                and history of losses.
                For example ``SimpleNamespace(iteration=2, metrics={
                'learn': {'Logloss': [0.6, 0.5]},
                'validation': {'Logloss': [0.7, 0.6], 'AUC': [0.8, 0.9]}
                })``.

        Returns:
            A boolean value. If :obj:`False`, CatBoost internally stops the optimization
            with Optuna's pruning logic without raising :class:`optuna.TrialPruned`.
            Otherwise, the optimization continues.
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
        """Raise :class:`optuna.TrialPruned` manually if the CatBoost optimization is pruned."""
        if self._pruned:
            raise optuna.TrialPruned(self._message)
