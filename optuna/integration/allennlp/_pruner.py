from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from packaging import version

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import


with try_import() as _imports:
    import allennlp
    import allennlp.commands
    import allennlp.common.cached_transformers
    import allennlp.common.util

if _imports.is_successful():
    from allennlp.training import GradientDescentTrainer
    from allennlp.training import TrainerCallback
else:
    # I disable mypy here since `allennlp.training.TrainerCallback` is a subclass of `Registrable`
    # (https://docs.allennlp.org/main/api/training/trainer/#trainercallback) but `TrainerCallback`
    # defined here is not `Registrable`, which causes a mypy checking failure.
    class TrainerCallback:  # type: ignore
        """Stub for TrainerCallback."""

        @classmethod
        def register(cls: Any, *args: Any, **kwargs: Any) -> Callable:
            """Stub method for `TrainerCallback.register`.

            This method has the same signature as
            `Registrable.register <https://docs.allennlp.org/master/
            api/common/registrable/#registrable>`_ in AllenNLP.

            """

            def wrapper(subclass: Any, *args: Any, **kwargs: Any) -> Any:
                return subclass

            return wrapper


@experimental("2.0.0")
@TrainerCallback.register("optuna_pruner")
class AllenNLPPruningCallback(TrainerCallback):
    """AllenNLP callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/tree/main/
    allennlp/allennlp_simple.py>`__
    if you want to add a pruning callback which observes a metric.

    You can also see the tutorial of our AllenNLP integration on
    `AllenNLP Guide <https://guide.allennlp.org/hyperparameter-optimization>`_.

    .. note::
        When :class:`~optuna.integration.AllenNLPPruningCallback` is instantiated in Python script,
        trial and monitor are mandatory.

        On the other hand, when :class:`~optuna.integration.AllenNLPPruningCallback` is used with
        :class:`~optuna.integration.AllenNLPExecutor`, ``trial`` and ``monitor``
        would be ``None``. :class:`~optuna.integration.AllenNLPExecutor` sets
        environment variables for a study name, trial id, monitor, and storage.
        Then :class:`~optuna.integration.AllenNLPPruningCallback`
        loads them to restore ``trial`` and ``monitor``.

    .. note::
        Currently, build-in pruners are supported except for
        :class:`~optuna.pruners.PatientPruner`.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``validation_loss`` or
            ``validation_accuracy``.

    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        monitor: Optional[str] = None,
    ):
        _imports.check()

        if version.parse(allennlp.__version__) < version.parse("2.0.0"):
            raise ImportError(
                "`AllenNLPPruningCallback` requires AllenNLP>=v2.0.0."
                "If you want to use a callback with an old version of AllenNLP, "
                "please install Optuna v2.5.0 by executing `pip install 'optuna==2.5.0'`."
            )

        if monitor is None:
            self._monitor = trial.system_attrs["allennlp:monitor"]
        else:
            self._monitor = monitor

        self._trial = trial

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs: Any,
    ) -> None:
        """Check if a training reaches saturation.

        Args:
            trainer:
                AllenNLP's trainer
            metrics:
                Dictionary of metrics.
            epoch:
                Number of current epoch.
            is_primary:
                A flag for AllenNLP internal.

        """
        if not is_primary:
            return None

        value = metrics.get(self._monitor)
        if value is None:
            return

        self._trial.report(float(value), epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()
