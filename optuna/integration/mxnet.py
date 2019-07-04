from __future__ import absolute_import

import optuna

try:
    import mxnet as mx  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class MXNetPruningCallback(object):
    """MXNet callback to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation accuracy.

        .. code::

            model.fit(train_data=X, eval_data=Y,
                      eval_end_callback=MXNetPruningCallback(trial, eval_metric='accuracy'))

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

    def __init__(self, trial, eval_metric):
        # type: (optuna.trial.Trial, str) -> None

        _check_mxnet_availability()

        self.trial = trial
        self.eval_metric = eval_metric

    def __call__(self, param):
        # type: (mx.model.BatchEndParams,) -> None

        if param.eval_metric is not None:
            metric_names, metric_values = param.eval_metric.get()
            if type(metric_names) == list and self.eval_metric in metric_names:
                current_score = metric_values[metric_names.index(self.eval_metric)]
            elif metric_names == self.eval_metric:
                current_score = metric_values
            else:
                raise ValueError('The entry associated with the metric name "{}" '
                                 'is not found in the evaluation result list {}.'.format(
                                     self.eval_metric, str(metric_names)))
            self.trial.report(current_score, step=param.epoch)
            if self.trial.should_prune():
                message = "Trial was pruned at epoch {}.".format(param.epoch)
                raise optuna.structs.TrialPruned(message)


def _check_mxnet_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'MXNet is not available. Please install MXNet to use this feature. '
            'MXNet can be installed by executing `$ pip install mxnet`. '
            'For further information, please refer to the installation guide of MXNet. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
