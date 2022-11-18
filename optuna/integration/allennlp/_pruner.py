import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from packaging import version

from optuna import load_study
from optuna import pruners
from optuna import Trial
from optuna import TrialPruned
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.integration.allennlp._variables import _VariableManager
from optuna.integration.allennlp._variables import OPTUNA_ALLENNLP_DISTRIBUTED_FLAG


with try_import() as _imports:
    import allennlp
    import allennlp.commands
    import allennlp.common.cached_transformers
    import allennlp.common.util

if _imports.is_successful():
    from allennlp.training import GradientDescentTrainer
    from allennlp.training import TrainerCallback
    import psutil

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


def _create_pruner(
    pruner_class: str,
    pruner_kwargs: Dict[str, Any],
) -> Optional[pruners.BasePruner]:

    """Restore a pruner which is defined in `create_study`.

    `AllenNLPPruningCallback` is launched as a sub-process of
    a main script that defines search spaces.
    An instance cannot be passed directly from the parent process
    to its sub-process. For this reason, we set information about
    pruner as environment variables and load them and
    re-create the same pruner in `AllenNLPPruningCallback`.

    """
    pruner = getattr(pruners, pruner_class, None)
    if pruner is None:
        return None

    return pruner(**pruner_kwargs)


@experimental_class("2.0.0")
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
        would be :obj:`None`. :class:`~optuna.integration.AllenNLPExecutor` sets
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
        trial: Optional[Trial] = None,
        monitor: Optional[str] = None,
    ):
        _imports.check()

        if version.parse(allennlp.__version__) < version.parse("2.0.0"):  # type: ignore
            raise ImportError(
                "`AllenNLPPruningCallback` requires AllenNLP>=v2.0.0."
                "If you want to use a callback with an old version of AllenNLP, "
                "please install Optuna v2.5.0 by executing `pip install 'optuna==2.5.0'`."
            )

        # When `AllenNLPPruningCallback` is instantiated in Python script,
        # trial and monitor should not be `None`.
        if trial is not None and monitor is not None:
            self._trial = trial
            self._monitor = monitor

        # When `AllenNLPPruningCallback` is used with `AllenNLPExecutor`,
        # `trial` and `monitor` would be None. `AllenNLPExecutor` sets information
        # for a study name, trial id, monitor, and storage in environment variables.
        else:
            current_process = psutil.Process()

            if os.getenv(OPTUNA_ALLENNLP_DISTRIBUTED_FLAG) == "1":
                os.environ.pop(OPTUNA_ALLENNLP_DISTRIBUTED_FLAG)
                parent_process = current_process.parent()
                target_pid = parent_process.ppid()

            else:
                target_pid = current_process.ppid()

            variable_manager = _VariableManager(target_pid)

            study_name = variable_manager.get_value("study_name")
            trial_id = variable_manager.get_value("trial_id")
            monitor = variable_manager.get_value("monitor")
            storage = variable_manager.get_value("storage_name")

            if study_name is None or trial_id is None or monitor is None or storage is None:
                message = (
                    "Fail to load study. Perhaps you attempt to use `AllenNLPPruningCallback`"
                    " without `AllenNLPExecutor`. If you want to use a callback"
                    " without an executor, you have to instantiate a callback with"
                    "`trial` and `monitor. Please see the Optuna example: https://github.com/"
                    "optuna/optuna-examples/tree/main/allennlp/allennlp_simple.py."
                )
                raise RuntimeError(message)

            else:
                # If `stoage` is empty despite `study_name`, `trial_id`,
                # and `monitor` are not `None`, users attempt to use `AllenNLPPruningCallback`
                # with `AllenNLPExecutor` and in-memory storage.
                # `AllenNLPruningCallback` needs RDB or Redis storages to work.
                if storage == "":
                    message = (
                        "If you want to use AllenNLPExecutor and AllenNLPPruningCallback,"
                        " you have to use RDB or Redis storage."
                    )
                    raise RuntimeError(message)

                pruner = _create_pruner(
                    variable_manager.get_value("pruner_class"),
                    variable_manager.get_value("pruner_kwargs"),
                )

                study = load_study(
                    study_name=study_name,
                    storage=storage,
                    pruner=pruner,
                )
                self._trial = Trial(study, trial_id)
                self._monitor = monitor

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **_: Any,
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
            raise TrialPruned()
