import json
import os

import optuna

try:
    import allennlp.commands

    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    TrackerCallback = object


class AllenNLPExecutor(object):
    """Allennlp extension to use optuna with an allennlp config file.

    .. warning::
        AllenNLPExecutor uses environment variables on OS.
        This could cause problems when AllenNLPExecutor runs in multi-threading.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation
            of the objective function.
        config_file:
            An allennlp config file.
            Hyperparameters should be masked with `std.extVar`.
            Please refer to `the config example <https://github.com/allenai/allentune/blob/
            f2b7de2cad2026c2a50625b939b2db3c1d9bc580/examples/classifier.jsonnet>`_.
        serialization_dir:
            A path which model weights and logs are saved.
        metrics:
            An evaluation metric for the result of ``objective``.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        config_file: str,
        serialization_dir: str,
        metrics: str = "best_validation_accuracy",
    ):

        self._params = trial.params
        self._config_file = config_file
        self._serialization_dir = serialization_dir
        self._metrics = metrics

    def _set_params(self) -> None:
        """Register hyperparameters as environment variables."""

        for key, value in self._params.items():
            os.environ[key] = str(value)

    def _clean_params(self) -> None:
        """Clear registered hyperparameters."""

        for key, value in self._params.items():
            if key not in os.environ:
                continue
            os.environ.pop(key)

    def run(self) -> float:
        """Train a model using allennlp."""

        self._set_params()
        allennlp.commands.train.train_model_from_file(self._config_file, self._serialization_dir)
        self._clean_params()

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
