import os


_PPID = os.getppid()

"""
User might want to launch multiple studies that uses `AllenNLPExecutor`.
Because `AllenNLPExecutor` uses environment variables for communicating
between a parent process and a child process. A parent process creates a study,
defines a search space, and a child process trains a AllenNLP model by
`allennlp.commands.train.train_model`. If multiple processes use `AllenNLPExecutor`,
the one's configuration could be loaded in the another's configuration.
To avoid this hazard, we add ID of a parent process to each key of
environment variables.
"""
_PREFIX = "{}_OPTUNA_ALLENNLP".format(_PPID)
_MONITOR = "{}_MONITOR".format(_PREFIX)
_PRUNER_CLASS = "{}_PRUNER_CLASS".format(_PREFIX)
_PRUNER_KEYS = "{}_PRUNER_KEYS".format(_PREFIX)
_STORAGE_NAME = "{}_STORAGE_NAME".format(_PREFIX)
_STUDY_NAME = "{}_STUDY_NAME".format(_PREFIX)
_TRIAL_ID = "{}_TRIAL_ID".format(_PREFIX)
