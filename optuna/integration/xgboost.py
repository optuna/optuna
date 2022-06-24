from typing import Any

import optuna


use_callback_cls = True

with optuna._imports.try_import() as _imports:
    import xgboost as xgb

    xgboost_version = xgb.__version__.split(".")
    xgboost_major_version = int(xgboost_version[0])
    xgboost_minor_version = int(xgboost_version[1])
    use_callback_cls = xgboost_major_version >= 1 and xgboost_minor_version >= 3

_doc = """Callback for XGBoost to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    xgboost/xgboost_integration.py>`__
    if you want to add a pruning callback which observes validation accuracy of
    a XGBoost model.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        observation_key:
            An evaluation metric for pruning, e.g., ``validation-error`` and
            ``validation-merror``. When using the Scikit-Learn API, the index number of
            ``eval_set`` must be included in the ``observation_key``, e.g.,
            ``validation_0-error`` and ``validation_0-merror``. Please refer to ``eval_metric``
            in `XGBoost reference <https://xgboost.readthedocs.io/en/latest/parameter.html>`_
            for further details.
    """

if _imports.is_successful() and use_callback_cls:

    class XGBoostPruningCallback(xgb.callback.TrainingCallback):
        __doc__ = _doc

        def __init__(self, trial: optuna.trial.Trial, observation_key: str) -> None:
            self._trial = trial
            self._observation_key = observation_key
            self._is_cv = False

        def before_training(self, model: Any) -> Any:
            # The use of Any type is due to _PackedBooster is not yet being exposed
            # to public interface as of xgboost 1.3.
            if isinstance(model, xgb.Booster):
                self._is_cv = False
            else:
                self._is_cv = True
            return model

        def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:
            evaluation_results = {}
            # Flatten the evaluation history to `{dataset-metric: score}` layout.
            for dataset, metrics in evals_log.items():
                for metric, scores in metrics.items():
                    assert isinstance(scores, list), scores
                    key = dataset + "-" + metric
                    if self._is_cv:
                        # Remove stddev of the metric across the cross-valdation
                        # folds.
                        evaluation_results[key] = scores[-1][0]
                    else:
                        evaluation_results[key] = scores[-1]

            current_score = evaluation_results[self._observation_key]
            self._trial.report(current_score, step=epoch)
            if self._trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(epoch)
                raise optuna.TrialPruned(message)
            # The training should not stop.
            return False

elif _imports.is_successful():

    def _get_callback_context(env: "xgb.core.CallbackEnv") -> str:  # type: ignore
        """Return whether the current callback context is cv or train.

        .. note::
            `Reference
            <https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/callback.py>`_.
        """

        if env.model is None and env.cvfolds is not None:
            context = "cv"
        else:
            context = "train"
        return context

    class XGBoostPruningCallback:  # type: ignore
        __doc__ = _doc

        def __init__(self, trial: optuna.trial.Trial, observation_key: str) -> None:
            self._trial = trial
            self._observation_key = observation_key

        def __call__(self, env: "xgb.core.CallbackEnv") -> None:  # type: ignore

            context = _get_callback_context(env)
            evaluation_result_list = env.evaluation_result_list
            if context == "cv":
                # Remove a third element: the stddev of the metric across the
                # cross-valdation folds.
                evaluation_result_list = [
                    (key, metric) for key, metric, _ in evaluation_result_list
                ]
            current_score = dict(evaluation_result_list)[self._observation_key]
            self._trial.report(current_score, step=env.iteration)
            if self._trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(env.iteration)
                raise optuna.TrialPruned(message)

else:

    class XGBoostPruningCallback:  # type: ignore
        __doc__ = _doc

        def __init__(self, trial: optuna.trial.Trial, observation_key: str) -> None:
            _imports.check()
