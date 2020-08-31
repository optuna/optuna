import json
import os
from typing import Any
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
    from allennlp.training import EpochCallback

if _imports.is_successful():
    import _jsonnet
else:
    EpochCallback = object  # NOQA


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

        self._system_attrs = {
            "OPTUNA_ALLENNLP_STUDY_NAME": trial.study.study_name,
            "OPTUNA_ALLENNLP_TRIAL_ID": str(trial._trial_id),
            "OPTUNA_ALLENNLP_STORAGE_NAME": trial.study._storage.get_url() or "",
            "OPTUNA_ALLENNLP_MONITOR": metrics,
        }

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
            key_with_ppid = "{}_{}".format(os.getppid(), key)
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
        monitor:
            An evaluation metric for pruning, e.g. ``validation_loss`` or
            ``validation_accuracy``.
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
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
            _environment_variables = self._get_environment_variables()
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
                _study = load_study(study_name, storage)
                self._trial = Trial(_study, int(trial_id))
                self._monitor = monitor
            else:
                message = "Fail to load study.\n"
                message += "AllenNLPPruningCallback is only available with Optuna."
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

    def _get_environment_variables(self) -> Dict[str, Optional[str]]:
        return {
            "study_name": os.getenv("{}_OPTUNA_ALLENNLP_STUDY_NAME".format(os.getppid())),
            "trial_id": os.getenv("{}_OPTUNA_ALLENNLP_TRIAL_ID".format(os.getppid())),
            "storage": os.getenv("{}_OPTUNA_ALLENNLP_STORAGE_NAME".format(os.getppid())),
            "monitor": os.getenv("{}_OPTUNA_ALLENNLP_MONITOR".format(os.getppid())),
        }
