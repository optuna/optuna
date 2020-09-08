import json
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna import load_study
from optuna import Trial


with try_import() as _imports:
    import allennlp
    import allennlp.commands
    import allennlp.common.util

# EpochCallback is conditionally imported for CI because allennlp is not available on
# the eivironment that builds documentation.
if _imports.is_successful():
    import _jsonnet
    from allennlp.training import EpochCallback
else:
    # `allennlp.training.EpochCallback` is a subclass of `Registerable`
    # https://docs.allennlp.org/master/api/training/trainer/#epochcallback
    # On the other hand, `EpochCallback` defined here is not `Registerable`,
    # which causes a mypy checking feailure.
    class EpochCallback:  # type: ignore
        """Stub for EpochCallback.

        This class is introduced for documentation CI.

        """

        @classmethod
        def register(cls: Any, *args: Any, **kwargs: Any) -> Callable:
            def wrapper(subclass: Any, *args: Any, **kwargs: Any) -> None:
                return subclass

            return wrapper


PPID = os.getppid()


def _create_pruner() -> Optional[optuna.pruners.BasePruner]:
    """Restore a pruner which is defined in `create_study`.

    AllenNLPPruningCallback is launched as a sub-process of
    a main script that defines search spaces.
    An instance cannot be passed directly from the parent process
    to its sub-process. For this reason, we set information about
    pruner as environment variables and load them and
    re-create the same pruner in `AllenNLPPruningCallback`.

    """
    pruner_class = os.getenv("{}_OPTUNA_ALLENNLP_PRUNER_CLASS".format(PPID))
    if pruner_class is None:
        return None

    pruner_params = _get_environment_variables_for_pruner()

    pruner = getattr(optuna.pruners, pruner_class, None)

    if pruner is None:
        return None

    return pruner(**pruner_params)


def _infer_and_cast(value: Optional[str]) -> Any:
    """Infer and cast a string to desired types.

    We are only able to set strings as environment variables.
    However, parameters of a pruner could be integer, float,
    boolean, or else. We infer and cast environment variables
    to desired types.

    """
    if value is None:
        return None

    result = value  # type: Any

    try:
        result = int(value)
    except ValueError:
        try:
            result = float(value)
        except ValueError:
            if result == "True":
                result = True
            elif value == "False":
                result = False

    return result


def _get_environment_variables_for_trial() -> Dict[str, Optional[str]]:
    return {
        "study_name": os.getenv("{}_OPTUNA_ALLENNLP_STUDY_NAME".format(PPID)),
        "trial_id": os.getenv("{}_OPTUNA_ALLENNLP_TRIAL_ID".format(PPID)),
        "storage": os.getenv("{}_OPTUNA_ALLENNLP_STORAGE_NAME".format(PPID)),
        "monitor": os.getenv("{}_OPTUNA_ALLENNLP_MONITOR".format(PPID)),
    }


def _get_environment_variables_for_pruner() -> Dict[str, Optional[str]]:
    keys = os.getenv("{}_OPTUNA_ALLENNLP_PRUNER_KEYS".format(PPID))
    if keys is None:
        return {}

    kwargs = {}
    for key in keys.split(","):
        value = os.getenv("{}_OPTUNA_ALLENNLP_{}".format(PPID, key))
        kwargs[key] = _infer_and_cast(value)

    return kwargs


def _fetch_pruner_config(trial: optuna.Trial) -> Dict[str, Any]:
    pruner = trial.study.pruner
    kwargs = {}  # type: Dict[str, Any]

    if isinstance(pruner, optuna.pruners.HyperbandPruner):
        kwargs["min_resource"] = pruner._min_resource
        kwargs["max_resource"] = pruner._max_resource
        kwargs["reduction_factor"] = pruner._reduction_factor
    elif isinstance(pruner, optuna.pruners.PercentilePruner):
        kwargs["percentile"] = pruner._percentile
        kwargs["n_startup_tirals"] = pruner._n_startup_trials
        kwargs["n_warmup_steps"] = pruner._n_warmup_steps
        kwargs["interval_steps"] = pruner._interval_steps
    elif isinstance(pruner, optuna.pruners.ThresholdPruner):
        kwargs["lower"] = pruner._lower
        kwargs["upper"] = pruner._upper
        kwargs["n_warmup_steps"] = pruner._n_warmup_steps
        kwargs["interval_steps"] = pruner._interval_steps
    elif isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner):
        kwargs["min_resource"] = pruner._min_resource
        kwargs["reduction_factor"] = pruner._reduction_factor
        kwargs["min_early_stopping_rate"] = pruner._min_early_stopping_rate

    return kwargs


def dump_best_config(input_config_file: str, output_config_file: str, study: optuna.Study) -> None:
    """Save JSON config file after updating with parameters from the best trial in the study.

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

    best_params = study.best_params
    for key, value in best_params.items():
        best_params[key] = str(value)
    best_config = json.loads(_jsonnet.evaluate_file(input_config_file, ext_vars=best_params))

    # `optuner_pruner` only works with Optuna.
    # It removes when dumping configuration since
    # the result of `dump_best_config` can be passed to
    # `allennlp train`.
    if "epoch_callbacks" in best_config["trainer"]:
        new_epoch_callbacks = []
        epoch_callbacks = best_config["trainer"]["epoch_callbacks"]
        for callback in epoch_callbacks:
            if callback["type"] == "optuna_pruner":
                continue
            new_epoch_callbacks.append(callback)

        if len(new_epoch_callbacks) == 0:
            best_config["trainer"].pop("epoch_callbacks")
        else:
            best_config["trainer"]["epoch_callbacks"] = new_epoch_callbacks

    with open(output_config_file, "w") as f:
        json.dump(best_config, f, indent=4)


@experimental("1.4.0")
class AllenNLPExecutor(object):
    """AllenNLP extension to use optuna with Jsonnet config file.

    This feature is experimental since AllenNLP major release will come soon.
    The interface may change without prior notice to correspond to the update.

    See the examples of `objective function <https://github.com/optuna/optuna/blob/
    master/examples/allennlp/allennlp_jsonnet.py>`_.

    From Optuna v2.1.0, users have to cast their parameters by using methods in Jsonnet.
    Call ``std.parseInt`` for integer, or ``std.parseJson`` for floating point.
    Please see the `example configuration <https://github.com/optuna/optuna/blob/master/
    examples/allennlp/classifier.jsonnet>`_.

    .. note::
        In :class:`~optuna.integration.AllenNLPExecutor`,
        you can pass parameters to AllenNLP by either defining a search space using
        Optuna suggest methods or setting environment variables just like AllenNLP CLI.
        If a value is set in both a search space in Optuna and the environment variables,
        the executor will use the value specified in the search space in Optuna.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation
            of the objective function.
        config_file:
            Config file for AllenNLP.
            Hyperparameters should be masked with ``std.extVar``.
            Please refer to `the config example <https://github.com/allenai/allentune/blob/
            master/examples/classifier.jsonnet>`_.
        serialization_dir:
            A path which model weights and logs are saved.
        metrics:
            An evaluation metric for the result of ``objective``.
        include_package:
            Additional packages to include.
            For more information, please see
            `AllenNLP documentation <https://docs.allennlp.org/master/api/commands/train/>`_.

    """

    def __init__(
        self,
        trial: optuna.Trial,
        config_file: str,
        serialization_dir: str,
        metrics: str = "best_validation_accuracy",
        *,
        include_package: Optional[Union[str, List[str]]] = None
    ):
        _imports.check()

        self._params = trial.params
        self._config_file = config_file
        self._serialization_dir = serialization_dir
        self._metrics = metrics
        if include_package is None:
            include_package = []
        if isinstance(include_package, str):
            self._include_package = [include_package]
        else:
            self._include_package = include_package

        system_attrs = {
            "OPTUNA_ALLENNLP_STUDY_NAME": trial.study.study_name,
            "OPTUNA_ALLENNLP_TRIAL_ID": str(trial._trial_id),
            "OPTUNA_ALLENNLP_STORAGE_NAME": trial.study._storage.get_url() or "",
            "OPTUNA_ALLENNLP_MONITOR": metrics,
        }

        pruner_params = _fetch_pruner_config(trial)
        system_attrs["OPTUNA_ALLENNLP_PRUNER_KEYS"] = ",".join(pruner_params.keys())

        pruner_params = {
            "OPTUNA_ALLENNLP_{}".format(key): str(value) for key, value in pruner_params.items()
        }
        system_attrs.update(pruner_params)

        if trial.study.pruner is not None:
            system_attrs["OPTUNA_ALLENNLP_PRUNER_CLASS"] = type(trial.study.pruner).__name__

        self._system_attrs = system_attrs

    def _build_params(self) -> Dict[str, Any]:
        """Create a dict of params for AllenNLP.

        _build_params is based on allentune's ``train_func``.
        For more detail, please refer to
        https://github.com/allenai/allentune/blob/master/allentune/modules/allennlp_runner.py#L34-L65

        """
        params = self._environment_variables()
        params.update({key: str(value) for key, value in self._params.items()})
        params.update(self._system_attrs)
        return json.loads(_jsonnet.evaluate_file(self._config_file, ext_vars=params))

    def _set_environment_variables(self) -> None:
        for key, value in self._system_attrs.items():
            key_with_ppid = "{}_{}".format(PPID, key)
            os.environ[key_with_ppid] = value

    @staticmethod
    def _is_encodable(value: str) -> bool:
        # https://github.com/allenai/allennlp/blob/master/allennlp/common/params.py#L77-L85
        return (value == "") or (value.encode("utf-8", "ignore") != b"")

    def _environment_variables(self) -> Dict[str, str]:
        return {key: value for key, value in os.environ.items() if self._is_encodable(value)}

    def run(self) -> float:
        """Train a model using AllenNLP."""
        try:
            import_func = allennlp.common.util.import_submodules
        except AttributeError:
            import_func = allennlp.common.util.import_module_and_submodules

        for package_name in self._include_package:
            import_func(package_name)

        self._set_environment_variables()
        params = allennlp.common.params.Params(self._build_params())
        allennlp.commands.train.train_model(params, self._serialization_dir)

        metrics = json.load(open(os.path.join(self._serialization_dir, "metrics.json")))
        return metrics[self._metrics]


@experimental("2.0.0")
@EpochCallback.register("optuna_pruner")
class AllenNLPPruningCallback(EpochCallback):
    """AllenNLP callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/allennlp/allennlp_simple.py>`__
    if you want to add a proning callback which observes a metric.

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

        if allennlp.__version__ < "1.0.0":
            raise Exception("AllenNLPPruningCallback requires `allennlp`>=1.0.0.")

        if trial is not None and monitor is not None:
            self._trial = trial
            self._monitor = monitor
        else:
            _environment_variables = _get_environment_variables_for_trial()
            study_name = _environment_variables["study_name"]
            trial_id = _environment_variables["trial_id"]
            monitor = _environment_variables["monitor"]
            storage = _environment_variables["storage"] or None

            if (
                study_name is not None
                and trial_id is not None
                and monitor is not None
                and storage is not None
            ):
                study = load_study(study_name, storage, pruner=_create_pruner())
                self._trial = Trial(study, int(trial_id))
                self._monitor = monitor
            else:
                message = "Fail to load study.\n"
                message += "AllenNLPPruningCallback works only with Optuna with RDB or Redis."
                raise Exception(message)

    def __call__(
        self,
        trainer: "allennlp.training.GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        value = metrics.get(self._monitor)
        if value is None:
            return

        self._trial.report(float(value), epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned()
