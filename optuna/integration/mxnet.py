from __future__ import absolute_import

import mxnet as mx  # NOQA
import optuna


class MxnetPruningCallback(object):
    """Mxnet callback to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation losses.

        .. code::

            model.fit(X, y, batch_end_callback=MxnetPruningCallback(trial, eval_metric='acc'))

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``cross-entropy`` and
            ``accuracy``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
    """

    def __init__(self, trial, eval_metric='accuracy'):
        # type: (optuna.trial.Trial, str) -> None

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
            self.trial.report(current_score)
            if self.trial.should_prune(param.epoch):
                message = "Trial was pruned at epoch {}.".format(param.epoch)
                raise optuna.structs.TrialPruned(message)
