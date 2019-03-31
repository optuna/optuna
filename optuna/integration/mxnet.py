from __future__ import absolute_import

import optuna

try:
    import mxnet as mx  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class MxnetPruningCallback(object):
    """Mxnet callback to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation losses.

        .. code::

            model.fit(X, y, batch_end_callback=MxnetPruningCallback(trial, eval_metric='accuracy'))

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

    def __init__(self, trial, eval_metric='accuracy'):
        # type: (optuna.trial.Trial, str) -> None

        _check_mxnet_availability()

        self.trial = trial
        self.eval_metric = eval_metric

    def __call__(self, param):
        # type: (mx.model.BatchEndParams,) -> None

        if param.nbatch == 1 and param.eval_metric is not None:
            metric_name, metric_value = param.eval_metric.get()
            if type(metric_name) == list and self.eval_metric in metric_name:
                current_score = metric_value[metric_name.index(self.eval_metric)]
            elif metric_name == self.eval_metric:
                current_score = metric_value
            else:
                return
            self.trial.report(current_score, step=param.epoch)
            if self.trial.should_prune(param.epoch):
                message = "Trial was pruned at epoch {}.".format(param.epoch)
                raise optuna.structs.TrialPruned(message)


def _check_mxnet_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Mxnet is not available. Please install Mxnet to use this feature. '
            'Mxnet for cudaX.Y support can be installed by executing `$ pip install mxnet-cuXY`. '
            'For further information, please refer to the installation guide of Mxnet. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
