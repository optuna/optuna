from __future__ import print_function

import numpy

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Collection  # NOQA
    from typing import Dict  # NOQA

try:
    import torch
    from fastai.basic_train import Learner
    from fastai.basic_train import LearnerCallback
    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    Callback = object


class FastaiPruningCallback(LearnerCallback):
    """FastAI callback to prune unpromising trials for fastai<=2.0.

    Example:

        Add a pruning callback which observes validation losses.

        .. code::

            learn.fit(n_epochs, callbacks=[FastaiPruningCallback(learn, trial, 'valid_loss')])
            # If you want to use `fit_one_cycle`
            learn.fit_one_cycle(
                n_epochs, cyc_len, max_lr,
                callbacks=[FastaiPruningCallback(learn, trial, 'valid_loss')])
            # For `fit`
            learn.fit(2, 1e-3, callbacks=[FastaiPruningCallback(learn, trial, 'valid_loss')])

    Args:
        learn:
            An entity of fastai.basic
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current
            evaluation of the objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``valid_loss`` and ``Accuracy``.
            Please refer to `fastai.Callback reference
            <https://docs.fast.ai/callback.html#Callback>`_ for further
            details.
    """

    def __init__(self, learn, trial=None, monitor=''):
        # type: (Learner, optuna.trial.Trial, str) -> None

        super(FastaiPruningCallback, self).__init__(learn)

        _check_fastai_availability()

        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        # type: (...) -> None

        if last_metrics is None:
            raise RuntimeError('Empty `last_metrics`')

        # NOTE(crcrpar): In `LearnerCallback` implementation,
        #         setattr(self.learn, self.cb_name, self)
        # the above snippet exists.
        # This makes it impossible to set the index of ``self.monitor`` to
        # this callback.
        if self.monitor not in self.learn.recorder.names:
            raise RuntimeError('Invalid `monitor` argument ({}). '
                               'Available monitors are {}'.format(
                                   self.monitor, ', '.join(self.learn.recorder.names[1:])))

        epoch_stats = [epoch, smooth_loss] + last_metrics
        value_to_monitor = epoch_stats[self.learn.recorder.names.index(self.monitor)]
        if isinstance(value_to_monitor, (torch.Tensor, numpy.ndarray)):
            value_to_monitor = value_to_monitor.item()

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
