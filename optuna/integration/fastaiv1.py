from typing import Any

from packaging import version

import optuna
from optuna._deprecated import deprecated_class
from optuna._imports import try_import


with try_import() as _imports:
    import fastai

    if version.parse(fastai.__version__) >= version.parse("2.0.0"):
        raise ImportError(
            f"You don't have fastai V1 installed! Fastai version: {fastai.__version__}"
        )

    from fastai.basic_train import Learner
    from fastai.callbacks import TrackerCallback

if not _imports.is_successful():
    TrackerCallback = object  # NOQA


@deprecated_class("2.4.0", "4.0.0")
class FastAIV1PruningCallback(TrackerCallback):
    """FastAI callback to prune unpromising trials for fastai.

    .. note::
        This callback is for fastai<2.0.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    fastai/fastaiv1_simple.py>`__
    if you want to add a pruning callback which monitors validation loss of a ``Learner``.

    Example:

        Register a pruning callback to ``learn.fit`` and ``learn.fit_one_cycle``.

        .. code::

            learn.fit(n_epochs, callbacks=[FastAIPruningCallback(learn, trial, "valid_loss")])
            learn.fit_one_cycle(
                n_epochs,
                cyc_len,
                max_lr,
                callbacks=[FastAIPruningCallback(learn, trial, "valid_loss")],
            )


    Args:
        learn:
            `fastai.basic_train.Learner <https://docs.fast.ai/basic_train.html#Learner>`_.
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current
            evaluation of the objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``valid_loss`` and ``Accuracy``.
            Please refer to `fastai.callbacks.TrackerCallback reference
            <https://fastai1.fast.ai/callbacks.tracker.html#TrackerCallback>`_ for further
            details.
    """

    def __init__(self, learn: "Learner", trial: optuna.trial.Trial, monitor: str) -> None:

        super().__init__(learn, monitor)

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
