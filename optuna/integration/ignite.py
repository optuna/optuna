import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA

try:
    from ignite.engine import Engine  # NOQA
    _available = True
except ImportError as e:
    _import_error = e
    # IgnitePruningHandler is disabled because pytorch-ignite is not available.
    _available = False


class IgnitePruningHandler(object):

    def __init__(self, trial, metric, trainer):
        # type: (Trial, str, Engine) -> None

        self.trial = trial
        self.metric = metric
        self.trainer = trainer

    def __call__(self, engine):
        # type: (Engine) -> None

        score = engine.state.metrics[self.metric]
        self.trial.report(score, engine.state.epoch)
        if self.trial.should_prune():
            self.trainer.terminate()
            message = "Trial was pruned at {} epoch.".format(engine.state.epoch)
            raise optuna.structs.TrialPruned(message)


def _check_pytorch_ignite_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'PyTorch Ignite is not available. Please install PyTorch Ignite to use this feature. '
            'PyTorch Ignite can be installed by executing `$ pip install pytorch-ignite`. '
            'For further information, please refer to the installation guide of PyTorch Ignite. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
