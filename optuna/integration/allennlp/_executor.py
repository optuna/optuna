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
from optuna.integration.allennlp._environment import _environment_variables


with try_import() as _imports:
    import allennlp
    import allennlp.commands
    import allennlp.common.cached_transformers
    import allennlp.common.util

    import optuna.integration.allennlp._train

# TrainerCallback is conditionally imported because allennlp may be unavailable in
# the environment that builds the documentation.
if _imports.is_successful():
    import _jsonnet


@experimental("1.4.0")
class AllenNLPExecutor(object):
    """AllenNLP extension to use optuna with Jsonnet config file.

    This feature is experimental since AllenNLP major release will come soon.
    The interface may change without prior notice to correspond to the update.

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
            An evaluation metric for the result of ``objective``.
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

        self._trial = trial
        self._params = trial.params
        self._config_file = config_file
        self._serialization_dir = serialization_dir
        self._metrics = metrics
        self._force = force
        self._file_friendly_logging = file_friendly_logging

        if include_package is None:
            include_package = []

        if isinstance(include_package, str):
            self._include_package = [include_package]
        else:
            self._include_package = include_package

        self._trial.set_system_attr("monitor", metrics)

    def _build_params(self) -> Dict[str, Any]:
        """Create a dict of params for AllenNLP.

        _build_params is based on allentune's ``train_func``.
        For more detail, please refer to
        https://github.com/allenai/allentune/blob/master/allentune/modules/allennlp_runner.py#L34-L65

        """
        params = _environment_variables()
        params.update({key: str(value) for key, value in self._params.items()})
        return json.loads(_jsonnet.evaluate_file(self._config_file, ext_vars=params))

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

        params = allennlp.common.params.Params(self._build_params())
        optuna.integration.allennlp._train.train_model_with_optuna(
            params=params,
            serialization_dir=self._serialization_dir,
            file_friendly_logging=self._file_friendly_logging,
            force=self._force,
            include_package=self._include_package,
            trial=self._trial,
        )
        metrics = json.load(open(os.path.join(self._serialization_dir, "metrics.json")))
        return metrics[self._metrics]
