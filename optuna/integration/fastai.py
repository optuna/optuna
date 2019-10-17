from __future__ import print_function

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA

try:
    from fastai.basic_train import Learner  # NOQA
    from fastai.callbacks import TrackerCallback
    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    Callback = object


class FastaiPruningCallback(TrackerCallback):
    """FastAI callback to prune unpromising trials for fastai<=2.0.

    Example:

        Add a pruning callback which observes validation losses.

        .. code::

            # If registering this callback in construction
            learn = cnn_learner(
                data, models.resnet18, metrics=[accuracy],
                callback_fns=[partial(FastaiPruningCallback, trial=trial, monitor='valid_loss')])
            # If use `fit`
            # learn.fit(n_epochs, callbacks=[FastaiPruningCallback(learn, trial, 'valid_loss')])
            # If you want to use `fit_one_cycle`
            # learn.fit_one_cycle(
            #     n_epochs, cyc_len, max_lr,
            #     callbacks=[FastaiPruningCallback(learn, trial, 'valid_loss')])

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

    def __init__(self, learn, trial, monitor):
        # type: (Learner, optuna.trial.Trial, str) -> None

        super(FastaiPruningCallback, self).__init__(learn, monitor)

        _check_fastai_availability()

        self.trial = trial

    def on_epoch_end(self, epoch, **kwargs):
        # type: (epoch, Any) -> None

        value = self.get_monitor_value()
        if value is None:
            return

        self.trial.report(value, step=epoch)
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
