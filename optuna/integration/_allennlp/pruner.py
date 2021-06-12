import json
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

from packaging import version

import optuna
from optuna import load_study
from optuna import Trial
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.integration._allennlp.environment import _environment_variables


with try_import() as _imports:
    import allennlp
    import allennlp.commands
    import allennlp.common.cached_transformers
    import allennlp.common.util

if _imports.is_successful():
    import _jsonnet
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


_PPID = os.getppid()

"""
User might want to launch multiple studies that uses `AllenNLPExecutor`.
Because `AllenNLPExecutor` uses environment variables for communicating
between a parent process and a child process. A parent process creates a study,
defines a search space, and a child process trains a AllenNLP model by
`allennlp.commands.train.train_model`. If multiple processes use `AllenNLPExecutor`,
the one's configuration could be loaded in the another's configuration.
To avoid this hazard, we add ID of a parent process to each key of
environment variables.
"""
_PREFIX = "{}_OPTUNA_ALLENNLP".format(_PPID)
_MONITOR = "{}_MONITOR".format(_PREFIX)
_PRUNER_CLASS = "{}_PRUNER_CLASS".format(_PREFIX)
_PRUNER_KEYS = "{}_PRUNER_KEYS".format(_PREFIX)
_STORAGE_NAME = "{}_STORAGE_NAME".format(_PREFIX)
_STUDY_NAME = "{}_STUDY_NAME".format(_PREFIX)
_TRIAL_ID = "{}_TRIAL_ID".format(_PREFIX)


def _create_pruner() -> Optional[optuna.pruners.BasePruner]:
    """Restore a pruner which is defined in `create_study`.

    `AllenNLPPruningCallback` is launched as a sub-process of
    a main script that defines search spaces.
    An instance cannot be passed directly from the parent process
    to its sub-process. For this reason, we set information about
    pruner as environment variables and load them and
    re-create the same pruner in `AllenNLPPruningCallback`.

    """
    pruner_class = os.getenv(_PRUNER_CLASS)
    if pruner_class is None:
        return None

    pruner_params = _get_environment_variables_for_pruner()
    pruner = getattr(optuna.pruners, pruner_class, None)

    if pruner is None:
        return None

    return pruner(**pruner_params)


def _get_environment_variables_for_trial() -> Dict[str, Optional[str]]:
    return {
        "study_name": os.getenv(_STUDY_NAME),
        "trial_id": os.getenv(_TRIAL_ID),
        "storage": os.getenv(_STORAGE_NAME),
        "monitor": os.getenv(_MONITOR),
    }


def _get_environment_variables_for_pruner() -> Dict[str, Optional[Union[str, int, float, bool]]]:
    keys = os.getenv(_PRUNER_KEYS)

    # keys would be empty when `_PRUNER_CLASS` is `NopPruner`
    if keys is None or keys == "":
        return {}

    kwargs = {}
    for key in keys.split(","):
        key_without_prefix = key.replace("{}_".format(_PREFIX), "")
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"{key} is not found in environment variables.")
        kwargs[key_without_prefix] = eval(value)

    return kwargs


def dump_best_config(input_config_file: str, output_config_file: str, study: optuna.Study) -> None:
    """Save JSON config file with environment variables and best performing hyperparameters.

    Args:
        input_config_file:
            Input Jsonnet config file used with
            :class:`~optuna.integration.AllenNLPExecutor`.
        output_config_file:
            Output JSON config file.
        study:
            Instance of :class:`~optuna.study.Study`.
            Note that :func:`~optuna.study.Study.optimize` must have been called.

    """
    _imports.check()

    # Get environment variables.
    ext_vars = _environment_variables()

    # Get the best hyperparameters.
    best_params = study.best_params
    for key, value in best_params.items():
        best_params[key] = str(value)

    # If keys both appear in environment variables and best_params,
    # values in environment variables are overwritten, which means best_params is prioritized.
    ext_vars.update(best_params)

    best_config = json.loads(_jsonnet.evaluate_file(input_config_file, ext_vars=ext_vars))

    # `optuna_pruner` only works with Optuna.
    # It removes when dumping configuration since
    # the result of `dump_best_config` can be passed to
    # `allennlp train`.
    if "callbacks" in best_config["trainer"]:
        new_callbacks = []
        callbacks = best_config["trainer"]["callbacks"]
        for callback in callbacks:
            if callback["type"] == "optuna_pruner":
                continue
            new_callbacks.append(callback)

        if len(new_callbacks) == 0:
            best_config["trainer"].pop("callbacks")
        else:
            best_config["trainer"]["callbacks"] = new_callbacks

    with open(output_config_file, "w") as f:
        json.dump(best_config, f, indent=4)


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
        trial: Optional[optuna.trial.Trial] = None,
        monitor: Optional[str] = None,
    ):
        _imports.check()

        if version.parse(allennlp.__version__) < version.parse("2.0.0"):
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
            environment_variables = _get_environment_variables_for_trial()
            study_name = environment_variables["study_name"]
            trial_id = environment_variables["trial_id"]
            monitor = environment_variables["monitor"]
            storage = environment_variables["storage"]

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

                study = load_study(study_name, storage, pruner=_create_pruner())
                self._trial = Trial(study, int(trial_id))
                self._monitor = monitor

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
