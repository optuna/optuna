import os
from typing import Optional


SPECIAL_DELIMITER = "[OPTUNA_ALLENNLP_INTEGRATION_DELIMITER]"


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
    an interface to access environment variables in a readonly manner.

    """

    NAME_OF_KEY = {
        "monitor": "{}_MONITOR",
        "pruner_class": "{}_PRUNER_CLASS",
        "pruner_keys": "{}_PRUNER_KEYS",
        "pruner_values": "{}_PRUNER_VALUES",
        "storage_name": "{}_STORAGE_NAME",
        "study_name": "{}_STUDY_NAME",
        "trial_id": "{}_TRIAL_ID",
    }

    def __init__(self, target_pid: int) -> None:
        self.target_pid = target_pid

    @property
    def prefix(self) -> str:
        return "{}_OPTUNA_ALLENNLP".format(self.target_pid)

    def get_key(self, name: str) -> Optional[str]:
        return self.NAME_OF_KEY.get(name)

    def set_value(self, name: str, value: str) -> None:
        """Set values to environment variables.

        `set_value` is only invoked in `optuna.integration.allennlp.AllenNLPExecutor`.

        """
        key = self.get_key(name)
        if key is None:
            return
        key = key.format(self.target_pid)
        os.environ[key] = value

    def get_value(self, name: str) -> Optional[str]:
        """Fetch parameters from environment variables.

        `get_value` is only called in `optuna.integration.allennlp.AllenNLPPruningCallback`.

        """

        key = self.get_key(name)
        name_of_path = "optuna.integration.allennlp._variables._VariableManager.NAME_OF_KEY"
        assert key is not None, f"{name} is not found in `{name_of_path}`."
        key = key.format(self.target_pid)
        value = os.environ.get(key)
        assert value is not None, f"{key} is not found in environment variables."
        return value
