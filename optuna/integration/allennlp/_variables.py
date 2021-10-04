import os
from typing import Optional

from optuna._imports import try_import


with try_import():
    import psutil


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
        key = self.get_key(name)
        if key is None:
            return
        key = key.format(self.target_pid)
        os.environ[key] = value

    def get_value(self, name: str) -> Optional[str]:
        key = self.get_key(name)
        name_of_path = "optuna.integration.allennlp._variables._VariableManager.NAME_OF_KEY"
        assert key is not None, f"{name} is not found in `{name_of_path}`."
        key = key.format(self.target_pid)
        value = os.environ.get(key)
        assert value is not None, f"{key} is not found in environment variables."
        return value

    def acquire_lock(self, is_distributed: bool) -> None:
        if "OPTUNA_ALLENNLP_USE_DISTRIBUTED" in os.environ:
            raise EnvironmentError(
                "Lock is already acquired. This error may occur when multiple"
                " `AllenNLPExecutor` are initialized simuletaneously."
                " If you see this error even you don't run multiple optimization"
                " simultaneously, please unset `OPTUNA_ALLENNLP_USE_DISTRIBUTED` from"
                " environment variable and re-run the optimization."
            )

        status = "1" if is_distributed else "0"
        os.environ["OPTUNA_ALLENNLP_USE_DISTRIBUTED"] = status

    @classmethod
    def check_and_release_lock(cls) -> "_VariableManager":
        status = os.getenv("OPTUNA_ALLENNLP_USE_DISTRIBUTED")
        if status is None:
            raise KeyError(
                "`OPTUNA_ALLENNLP_USE_DISTRIBUTED` is not found in" "environment variable."
            )

        # release_lock
        del os.environ["OPTUNA_ALLENNLP_USE_DISTRIBUTED"]

        current_process = psutil.Process()
        if status == "1":
            parent_process = current_process.parent()
            return cls(parent_process.ppid())

        elif status == "0":
            return cls(current_process.ppid())

        else:
            raise ValueError(f"{status} is not valid value for `OPTUNA_ALLENNLP_USE_DISTRIBUTED`")
