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
        allennlp_executable_path:
            A path to allennlp cli executable file.
        use_poetry:
            A flag of `poetry <https://python-poetry.org/>`_.
            If ``use_poetry`` is true, allennlp will be called with poetry.
        use_pipenv:
            A flag of `pipenv <https://pipenv-fork.readthedocs.io/en/latest/>`_.
            If ``use_pipenv`` is true, allennlp will be called with pipenv.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        config_file: str,
        serialization_dir: str,
        metrics: str = "best_validation_accuracy",
        *,
        allennlp_executable_path: Optional[str] = None,
        use_poetry: bool = False,
        use_pipenv: bool = False
    ):
        self.params = trial.params
        self.config_file = config_file
        self.serialization_dir = serialization_dir
        self.metrics = metrics
        self.allennlp_executable_path = allennlp_executable_path
        self.use_poetry = use_poetry
        self.use_pipenv = use_pipenv

    def run(self) -> float:
        for key, value in self.params.items():
            os.environ[key] = str(value)
        allennlp.commands.train.train_model_from_file(self.config_file, self.serialization_dir)
        metrics = json.load(open(os.path.join(self.serialization_dir, "metrics.json")))
        return metrics[self.metrics]
