import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA

try:
    from ignite.engine import Engine  # NOQA

    _available = True
except ImportError as e:
    _import_error = e
    # PyTorchIgnitePruningHandler is disabled because pytorch-ignite is not available.
    _available = False


class PyTorchIgnitePruningHandler(object):
    """PyTorch Ignite handler to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pytorch_ignite_simple.py>`__
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
        interval:
            Check if trial should be pruned every n-th epoch. By default `interval=1` and
            pruning is performed after every epoch. Increase `interval` to run several
            epochs faster before applying pruning.
    """

    def __init__(self, trial, metric, trainer, interval=1):
        # type: (Trial, str, Engine, int) -> None

        self._trial = trial
        self._metric = metric
        self._trainer = trainer
        self._interval = interval

    def __call__(self, engine):
        # type: (Engine) -> None

        if self._trainer.state.epoch % self._interval != 0:
            return

        score = engine.state.metrics[self._metric]
        self._trial.report(score, self._trainer.state.epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at {} epoch.".format(self._trainer.state.epoch)
            raise optuna.exceptions.TrialPruned(message)


def _check_pytorch_ignite_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "PyTorch Ignite is not available. Please install PyTorch Ignite to use this feature. "
            "PyTorch Ignite can be installed by executing `$ pip install pytorch-ignite`. "
            "For further information, please refer to the installation guide of PyTorch Ignite. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
