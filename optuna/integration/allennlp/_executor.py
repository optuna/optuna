import json
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import warnings

import optuna
from optuna import TrialPruned
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.integration.allennlp._environment import _environment_variables
from optuna.integration.allennlp._variables import _VariableManager
from optuna.integration.allennlp._variables import OPTUNA_ALLENNLP_DISTRIBUTED_FLAG


with try_import() as _imports:
    import allennlp
    import allennlp.commands
    import allennlp.common.cached_transformers
    import allennlp.common.util

# TrainerCallback is conditionally imported because allennlp may be unavailable in
# the environment that builds the documentation.
if _imports.is_successful():
    import _jsonnet
    import psutil
    from torch.multiprocessing.spawn import ProcessRaisedException


def _fetch_pruner_config(trial: optuna.Trial) -> Dict[str, Any]:
    pruner = trial.study.pruner
    kwargs: Dict[str, Any] = {}

    if isinstance(pruner, optuna.pruners.HyperbandPruner):
        kwargs["min_resource"] = pruner._min_resource
        kwargs["max_resource"] = pruner._max_resource
        kwargs["reduction_factor"] = pruner._reduction_factor

    elif isinstance(pruner, optuna.pruners.MedianPruner):
        kwargs["n_startup_trials"] = pruner._n_startup_trials
        kwargs["n_warmup_steps"] = pruner._n_warmup_steps
        kwargs["interval_steps"] = pruner._interval_steps

    elif isinstance(pruner, optuna.pruners.PercentilePruner):
        kwargs["percentile"] = pruner._percentile
        kwargs["n_startup_trials"] = pruner._n_startup_trials
        kwargs["n_warmup_steps"] = pruner._n_warmup_steps
        kwargs["interval_steps"] = pruner._interval_steps

    elif isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner):
        kwargs["min_resource"] = pruner._min_resource
        kwargs["reduction_factor"] = pruner._reduction_factor
        kwargs["min_early_stopping_rate"] = pruner._min_early_stopping_rate

    elif isinstance(pruner, optuna.pruners.ThresholdPruner):
        kwargs["lower"] = pruner._lower
        kwargs["upper"] = pruner._upper
        kwargs["n_warmup_steps"] = pruner._n_warmup_steps
        kwargs["interval_steps"] = pruner._interval_steps
    elif isinstance(pruner, optuna.pruners.NopPruner):
        pass
    else:
        raise ValueError("Unsupported pruner is specified: {}".format(type(pruner)))

    return kwargs


@experimental_class("1.4.0")
class AllenNLPExecutor:
    """AllenNLP extension to use optuna with Jsonnet config file.

    See the examples of `objective function <https://github.com/optuna/optuna-examples/tree/
    main/allennlp/allennlp_jsonnet.py>`_.

    You can also see the tutorial of our AllenNLP integration on
    `AllenNLP Guide <https://guide.allennlp.org/hyperparameter-optimization>`_.

    .. note::
        From Optuna v2.1.0, users have to cast their parameters by using methods in Jsonnet.
        Call ``std.parseInt`` for integer, or ``std.parseJson`` for floating point.
        Please see the `example configuration <https://github.com/optuna/optuna-examples/tree/main/
        allennlp/classifier.jsonnet>`_.

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
            An evaluation metric. `GradientDescrentTrainer.train() <https://docs.allennlp.org/
            main/api/training/gradient_descent_trainer/#train>`_ of AllenNLP
            returns a dictionary containing metrics after training.
            :class:`~optuna.integration.AllenNLPExecutor` accesses the dictionary
            by the key ``metrics`` you specify and use it as a objective value.
        force:
            If :obj:`True`, an executor overwrites the output directory if it exists.
        file_friendly_logging:
            If :obj:`True`, tqdm status is printed on separate lines and slows tqdm refresh rate.
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
        include_package: Optional[Union[str, List[str]]] = None,
        force: bool = False,
        file_friendly_logging: bool = False,
    ):
        _imports.check()

        self._params = trial.params
        self._config_file = config_file
        self._serialization_dir = serialization_dir
        self._metrics = metrics
        self._force = force
        self._file_friendly_logging = file_friendly_logging

        if include_package is None:
            include_package = []
        if isinstance(include_package, str):
            include_package = [include_package]

        self._include_package = include_package + ["optuna.integration.allennlp"]

        storage = trial.study._storage

        if isinstance(storage, optuna.storages.RDBStorage):
            url = storage.url

        elif isinstance(storage, optuna.storages._CachedStorage):
            assert isinstance(storage._backend, optuna.storages.RDBStorage)
            url = storage._backend.url

        else:
            url = ""

        target_pid = psutil.Process().ppid()
        variable_manager = _VariableManager(target_pid)

        pruner_kwargs = _fetch_pruner_config(trial)
        variable_manager.set_value("study_name", trial.study.study_name)
        variable_manager.set_value("trial_id", trial._trial_id)
        variable_manager.set_value("storage_name", url)
        variable_manager.set_value("monitor", metrics)

        if trial.study.pruner is not None:
            variable_manager.set_value("pruner_class", type(trial.study.pruner).__name__)
            variable_manager.set_value("pruner_kwargs", pruner_kwargs)

    def _build_params(self) -> Dict[str, Any]:
        """Create a dict of params for AllenNLP.

        _build_params is based on allentune's ``train_func``.
        For more detail, please refer to
        https://github.com/allenai/allentune/blob/master/allentune/modules/allennlp_runner.py#L34-L65

        """
        params = _environment_variables()
        params.update({key: str(value) for key, value in self._params.items()})
        return json.loads(_jsonnet.evaluate_file(self._config_file, ext_vars=params))

    def _set_environment_variables(self) -> None:
        for key, value in _environment_variables().items():
            if key is None:
                continue
            os.environ[key] = value

    def run(self) -> float:
        """Train a model using AllenNLP."""
        for package_name in self._include_package:
            allennlp.common.util.import_module_and_submodules(package_name)

        # Without the following lines, the transformer model construction only takes place in the
        # first trial (which would consume some random numbers), and the cached model will be used
        # in trials afterwards (which would not consume random numbers), leading to inconsistent
        # results between single trial and multiple trials. To make results reproducible in
        # multiple trials, we clear the cache before each trial.
        # TODO(MagiaSN) When AllenNLP has introduced a better API to do this, one should remove
        # these lines and use the new API instead. For example, use the `_clear_caches()` method
        # which will be in the next AllenNLP release after 2.4.0.
        allennlp.common.cached_transformers._model_cache.clear()
        allennlp.common.cached_transformers._tokenizer_cache.clear()

        self._set_environment_variables()
        params = allennlp.common.params.Params(self._build_params())

        if "distributed" in params:

            if OPTUNA_ALLENNLP_DISTRIBUTED_FLAG in os.environ:
                warnings.warn(
                    "Other process may already exists."
                    " If you have trouble, please unset the environment"
                    " variable `OPTUNA_ALLENNLP_USE_DISTRIBUTED`"
                    " and try it again."
                )

            os.environ[OPTUNA_ALLENNLP_DISTRIBUTED_FLAG] = "1"

        try:
            allennlp.commands.train.train_model(
                params=params,
                serialization_dir=self._serialization_dir,
                file_friendly_logging=self._file_friendly_logging,
                force=self._force,
                include_package=self._include_package,
            )
        except ProcessRaisedException as e:
            if "raise TrialPruned()" in str(e):
                raise TrialPruned()

        metrics = json.load(open(os.path.join(self._serialization_dir, "metrics.json")))
        return metrics[self._metrics]
