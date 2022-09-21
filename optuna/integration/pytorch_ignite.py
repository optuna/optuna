import optuna
from optuna.trial import Trial


with optuna._imports.try_import() as _imports:
    from ignite.engine import Engine


class PyTorchIgnitePruningHandler:
    """PyTorch Ignite handler to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    pytorch/pytorch_ignite_simple.py>`__
    if you want to add a pruning handler which observes validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        metric:
            A name of metric for pruning, e.g., ``accuracy`` and ``loss``.
        trainer:
            A trainer engine of PyTorch Ignite. Please refer to `ignite.engine.Engine reference
            <https://pytorch.org/ignite/engine.html#ignite.engine.Engine>`_ for further details.
    """

    def __init__(self, trial: Trial, metric: str, trainer: "Engine") -> None:

        _imports.check()

        self._trial = trial
        self._metric = metric
        self._trainer = trainer

    def __call__(self, engine: "Engine") -> None:

        score = engine.state.metrics[self._metric]
        self._trial.report(score, self._trainer.state.epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at {} epoch.".format(self._trainer.state.epoch)
            raise optuna.TrialPruned(message)
