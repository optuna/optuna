import json
import os
from typing import Any


OPTUNA_ALLENNLP_DISTRIBUTED_FLAG = "OPTUNA_ALLENNLP_USE_DISTRIBUTED"


class _VariableManager:
    """A environment variable manager for AllenNLP integration.

    User might want to launch multiple studies that uses `AllenNLPExecutor`.
    Because `AllenNLPExecutor` uses environment variables for communicating
    between a parent process and a child process. A parent process creates a study,
    defines a search space, and a child process trains a AllenNLP model by
    `allennlp.commands.train.train_model`. If multiple processes use `AllenNLPExecutor`,
    the one's configuration could be loaded in the another's configuration.
    To avoid this hazard, we add ID of a parent process to each key of
    environment variables.

    Note that you must invoke `set_value` only in `AllenNLPExecutor`.
    Methods in `AllenNLPPruingCallback` could be called in multiple
    processes when enabling distributed optimization. If `set_value`
    is invoked in the pruning callback, a consistency would break.
    So, after initializing `AllenNLPExecutor`, `_VariableManager` provides
    an interface to access environment variables in a read-only manner.

    """

    NAME_OF_KEY = {
        "monitor": "{}_MONITOR",
        "pruner_class": "{}_PRUNER_CLASS",
        "pruner_kwargs": "{}_PRUNER_PARAMS",
        "storage_name": "{}_STORAGE_NAME",
        "study_name": "{}_STUDY_NAME",
        "trial_id": "{}_TRIAL_ID",
    }
    NAME_OF_PATH = "optuna.integration.allennlp._variables._VariableManager.NAME_OF_KEY"

    def __init__(self, target_pid: int) -> None:
        self.target_pid = target_pid

    @property
    def prefix(self) -> str:
        return "{}_OPTUNA_ALLENNLP".format(self.target_pid)

    def _get_key(self, name: str) -> str:
        key = self.NAME_OF_KEY.get(name)
        if key is None:
            raise KeyError(f"{name} is not found in `{self.NAME_OF_PATH}`.")
        return key

    def set_value(self, name: str, value: Any) -> None:
        """Set values to environment variables.

        `set_value` is only invoked in `optuna.integration.allennlp.AllenNLPExecutor`.

        """
        key = self._get_key(name).format(self.target_pid)
        os.environ[key] = json.dumps(value)

    def get_value(self, name: str) -> Any:
        """Fetch parameters from environment variables.

        `get_value` is only called in `optuna.integration.allennlp.AllenNLPPruningCallback`.

        """
        key = self._get_key(name).format(self.target_pid)
        value = os.environ.get(key)
        if value is None:
            raise KeyError(f"{key} is not found in environment variables.")
        return json.loads(value)
