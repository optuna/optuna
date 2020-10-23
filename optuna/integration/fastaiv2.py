from typing import Any

from packaging import version

import optuna
from optuna._imports import try_import


with try_import() as _imports:
    import fastai
    if version.parse(fastai.__version__) < version.parse("2.0.0"):
        raise ImportError(f"You don't have fastai V2 installed! Fastai version: {fastai.__version__}")

    from fastai.callback.core import CancelFitException
    from fastai.callback.tracker import TrackerCallback

if not _imports.is_successful():
    TrackerCallback = object  # NOQA


class FastAIV2PruningCallback(TrackerCallback):
    """FastAI callback to prune unpromising trials for fastai.

    .. note::
        This callback is for fastai>2.0.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/fastai_simple.py>`__
    if you want to add a pruning callback which monitors validation loss of a ``Learner``.

    Example:

        Register a pruning callback to ``learn.fit`` and ``learn.fit_one_cycle``.

        .. code::
            learn = cnn_learner(dls, resnet18, metrics=[error_rate])
            learn.fit(n_epochs, cbs=[FastAIPruningCallback(trial)]) # Monitor "valid_loss"
            learn.fit_one_cycle(
                n_epochs,
                lr_max,
                cbs=[FastAIPruningCallback(trial, monitor="error_rate")], # Monitor "error_rate"
            )


    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current
            evaluation of the objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``valid_loss`` or ``accuracy``.
            Please refer to `fastai.callback.TrackerCallback reference
            <https://docs.fast.ai/callback.tracker#TrackerCallback>`_ for further
            details.
    """
    
    
    # Implementation notes: it's a subclass of TrackerCallback to benefit from it. For example, 
    # when to run (after the Recorder callback), when not to (like with lr_find), etc.

    def __init__(self, optuna_trial: optuna.Trial, monitor: str = 'valid_loss'):
        super(FastAIV2PruningCallback, self).__init__(monitor=monitor)
        _imports.check()
        self.optuna_trial = optuna_trial
        
    def after_epoch(self):
        super().after_epoch()
        # self.idx is set by TrackTrackerCallback
        self.optuna_trial.report(self.recorder.final_record[self.idx], step=self.epoch)
        
        if self.optuna_trial.should_prune():
            raise CancelFitException()
            
    def after_fit(self):
        super().after_fit()
        if self.optuna_trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at epoch {self.epoch}.")