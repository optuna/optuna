import json
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import optuna

try:
    import _jsonnet
    import allennlp.commands
    import allennlp.common.util

    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    TrackerCallback = object


class AllenNLPExecutor(object):
    """Allennlp extension to use optuna with an allennlp config file.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation
            of the objective function.
        config_file:
            An allennlp config file.
            Hyperparameters should be masked with `std.extVar`.
            Please refer to `the config example <https://github.com/allenai/allentune/blob/
            master/examples/classifier.jsonnet>`_.
        serialization_dir:
            A path which model weights and logs are saved.
        metrics:
            An evaluation metric for the result of ``objective``.
        include_package:
            Additional packages to include.
            For more information, please see
            `allennlp documentation <https://docs.allennlp.org/master/api/commands/train/>`_.

    """

    def __init__(
        self,
        trial: optuna.Trial,
        config_file: str,
        serialization_dir: str,
        metrics: str = "best_validation_accuracy",
        *,
        include_package: Union[str, List[str]] = []
    ):

        self._params = trial.params
        self._config_file = config_file
        self._serialization_dir = serialization_dir
        self._metrics = metrics
        if isinstance(include_package, str):
            self._include_package = [include_package]
        else:
            self._include_package = include_package

    def _build_params(self) -> Dict[str, Any]:
        """Create a dict of params for allennlp."""

        # _build_params is based on allentune's train_func.
        # https://github.com/allenai/allentune/blob/master/allentune/modules/allennlp_runner.py#L34-L65
        for key, value in self._params.items():
            self._params[key] = str(value)
        _params = json.loads(_jsonnet.evaluate_file(self._config_file, ext_vars=self._params))

        # _params contains a list of string or string as value values.
        # Some params couldn't be casted correctly.
        # infer_and_cast converts them into desired values.
        return allennlp.common.params.infer_and_cast(_params)

    def run(self) -> float:
        """Train a model using allennlp."""

        for package_name in self._include_package:
            allennlp.common.util.import_submodules(package_name)

        params = allennlp.common.params.Params(self._build_params())
        allennlp.commands.train.train_model(params, self._serialization_dir)

        metrics = json.load(open(os.path.join(self._serialization_dir, "metrics.json")))
        return metrics[self._metrics]


def _check_allennlp_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "allennlp is not available. Please install allennlp to use this feature. "
            "allennlp can be installed by executing `$ pip install allennlp`. "
            "For further information, please refer to the installation guide of allennlp. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
