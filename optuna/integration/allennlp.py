import json
import os

import allennlp.commands

import optuna


class AllenNLPExecutor(object):
    """Allennlp extension to use optuna with an allennlp config file.

    .. note::
        AllenNLPExecutor uses environment variables on OS.
        This could cause problems when AllenNLPExecutor runs in multi-threading.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation
            of the objective function.
        config_file:
            An allennlp config file.
            Hyperparameters should be masked with `std.extVar`.
            Please refer to `the example <https://github.com/allenai/allentune/blob/
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

    def run(self) -> float:
        for key, value in self.params.items():
            os.environ[key] = str(value)
        allennlp.commands.train.train_model_from_file(self.config_file, self.serialization_dir)
        metrics = json.load(open(os.path.join(self.serialization_dir, "metrics.json")))
        return metrics[self.metrics]
