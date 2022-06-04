import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import mxnet as mx


class MXNetPruningCallback:
    """MXNet callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    mxnet/mxnet_integration.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        eval_metric:
            An evaluation metric name for pruning, e.g., ``cross-entropy`` and
            ``accuracy``. If using default metrics like mxnet.metrics.Accuracy, use it's
            default metric name. For custom metrics, use the metric_name provided to
            constructor. Please refer to `mxnet.metrics reference
            <https://mxnet.apache.org/api/python/metric/metric.html>`_ for further details.
    """

    def __init__(self, trial: optuna.trial.Trial, eval_metric: str) -> None:

        _imports.check()

        self._trial = trial
        self._eval_metric = eval_metric

    def __call__(self, param: "mx.model.BatchEndParam") -> None:

        if param.eval_metric is not None:
            metric_names, metric_values = param.eval_metric.get()
            if type(metric_names) == list and self._eval_metric in metric_names:
                current_score = metric_values[metric_names.index(self._eval_metric)]
            elif metric_names == self._eval_metric:
                current_score = metric_values
            else:
                raise ValueError(
                    'The entry associated with the metric name "{}" '
                    "is not found in the evaluation result list {}.".format(
                        self._eval_metric, str(metric_names)
                    )
                )
            self._trial.report(current_score, step=param.epoch)
            if self._trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(param.epoch)
                raise optuna.TrialPruned(message)
