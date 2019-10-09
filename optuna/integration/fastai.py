from __future__ import print_function

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Collection  # NOQA
    from typing import Dict  # NOQA

try:
    import torch
    from fastai.callback import Callback
    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    Callback = object


class FastaiPruningCallback(Callback):
    """FastAI callback to prune unpromising trials.

    Example:

    Args:
        learn:
            A entity of fastai.basic
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the objective function.
        monitor:
            Ane evaluation metric for pruning, e.g. ``valid_loss`` and ``Accuracy``.
            Please refer to `fastai.Callback reference
            <https://docs.fast.ai/callback.html#Callback>`_ for further
            details.
    """

    def __init__(self, learn, trial, monitor):
        # type: (fastai.basic_train.Learner, optuna.trial.Trial, str) -> None

        super(FastaiPruningCallback, self).__init__(learn)

        _check_fastai_availability()

        self.trial = trial
        self.monitor = monitor

    def on_train_begin(self, **kwargs):
        # type: (Dict[Any, Any]) -> None

        """Initialize recording status at beginning of training."""

        if self.monitor not in self.learn.recorder.names:
            raise RuntimeError(
                'Invalid `monitor` argument. '
                'Available monitors are {}'.format(
                    ', '.join(self.learn.recorder.names[1:])))
        self._index_to_monitor = self.learn.recorder.names.index(self.monitor)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        # type: (...) -> None

        if last_metrics is None:
            raise RuntimeError('Empty `last_metrics`')

        epoch_stats = [epoch, smooth_loss] + last_metrics
        value_to_monitor = epoch_stats[self._index_to_monitor]
        if isinstance(value_to_monitor, torch.Tensor):
            value_to_monitor = value_to_monitor.item()

        assert isinstance(value_to_monitor, float)
        self.trial.report(value_to_monitor, step=epoch)
        if self.trial.should_prune():
            message = 'Trial was pruned at epoch {}.'.format(epoch)
            raise optuna.structs.TrialPruned(message)


def _check_fastai_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'fastai is not available. Please install fastai to use this feature. '
            'fastai can be installed by executing `$ pip install fastai`. '
            'For further information, please refer to the installation guide of fastai. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
