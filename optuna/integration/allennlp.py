import json
import os
import subprocess
from typing import Optional

import optuna


class AllenNLPExecutor(object):
    """Allennlp extension to use optuna with an allennlp config file.

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

        if self.use_poetry:
            allennlp_command = "poetry run allennlp"
        elif self.use_pipenv:
            allennlp_command = "pipenv run allennlp"
        elif self.allennlp_executable_path is not None:
            allennlp_command = self.allennlp_executable_path
        else:
            allennlp_command = "allennlp"

        command = "{} train --serialization-dir={} {}".format(
            allennlp_command,
            self.serialization_dir,
            self.config_file,
        )

        env = {k: str(v) for k, v in self.params.items()}
        current_path = os.getenv("PATH")
        if current_path is None:
            raise ValueError("PATH is empty")
        env["PATH"] = current_path
        subprocess.run(command, env=env, shell=True)
        metrics = json.load(open(os.path.join(self.serialization_dir, "metrics.json")))
        return metrics[self.metrics]
