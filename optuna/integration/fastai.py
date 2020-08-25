from typing import Any

import optuna
from optuna._imports import try_import

with try_import() as _imports:
    from fastai.basic_train import Learner
    from fastai.callbacks import TrackerCallback

if not _imports.is_successful():
    TrackerCallback = object  # NOQA


class FastAIPruningCallback(TrackerCallback):
    """FastAI callback to prune unpromising trials for fastai.

    .. note::
        This callback is for fastai<2.0, not the coming version developed in fastai/fastai_dev.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/fastai_simple.py>`__
    if you want to add a pruning callback which monitors validation loss of a ``Learner``.

    Example:

        Register a pruning callback to ``learn.fit`` and ``learn.fit_one_cycle``.

        .. code::

            learn.fit(n_epochs, callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(
                n_epochs, cyc_len, max_lr,
                callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])


    Args:
        learn:
            `fastai.basic_train.Learner <https://docs.fast.ai/basic_train.html#Learner>`_.
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current
            evaluation of the objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``valid_loss`` and ``Accuracy``.
            Please refer to `fastai.Callback reference
            <https://docs.fast.ai/callback.html#Callback>`_ for further
            details.
    """

    def __init__(self, learn: "Learner", trial: optuna.trial.Trial, monitor: str) -> None:

        super(FastAIPruningCallback, self).__init__(learn, monitor)

        _imports.check()

        self._trial = trial

    def on_epoch_end(self, epoch: int, **kwargs: Any) -> None:

        value = self.get_monitor_value()
        if value is None:
            return

        # This conversion is necessary to avoid problems reported in issues.
        # - https://github.com/optuna/optuna/issue/642
        # - https://github.com/optuna/optuna/issue/655.
        self._trial.report(float(value), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
