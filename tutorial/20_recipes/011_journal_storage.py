"""
.. _journal_storage:

(File-based) Journal Storage
============================

Optuna provides :class:`~optuna.storages.JournalStorage`. With this feature, you can easily run a
distributed optimization over network using NFS as the shared storage, without need for setting up
RDB or Redis.

"""

import logging
import sys

import optuna


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage("./journal.log"),  # NFS path for distributed optimization
)

study = optuna.create_study(study_name=study_name, storage=storage)


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study.optimize(objective, n_trials=3)

###################################################################################################
# Although the optimization in this example is too short to run in parallel, you can extend this
# example to write a optimization script which can be run in parallel.
#

###################################################################################################
# .. note::
#     In a Windows environment, an error message "A required privilege is not held by the client"
#     may appear. In this case, you can solve the problem with creating storage by specifying
#     :class:`~optuna.storages.JournalFileOpenLock`. See the reference of
#     :class:`~optuna.storages.JournalStorage` for any details.
