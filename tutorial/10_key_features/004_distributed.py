"""
.. _distributed:

Easy Parallelization
====================

Optuna supports several ways to run distributed optimization.

1. **Multi-thread optimization**:
    You can run multiple trials in parallel within a single process using the `n_jobs` parameter in `optuna.create_study()`.
2. **Multi-process optimization**:
    You can run multiple processes sharing the same storage backend, such as RDB or Redis.
3. **Multi-node optimization**:
    If you whant the thousands of process nodes, you can use `GRPCProxyStorage` to run distributed optimization across multiple machines.

Following diagram shows which strategy is suitable for which use case.


image!!!


Multi Thread Optimization
-------------------------

.. Note::
    **Recommended backends**:
        - In-Memory Storage
        - JournalStorage
        - RDBStorage


You can run multiple trials in parallel just by setting the ``n_jobs``
parameter in :func:`~optuna.create_study()`.

Multi-thread optimization is not powerful in python due to the Global Interpreter Lock (GIL),
but from Python 3.14, GIL is no longer exists, so Multi-thread optimization is now a good option.

"""

import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend
from optuna.trial import Trial


def objective(trial: Trial):
    print(f"{trial.number} Job Started!")
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./journal_example.log")),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, n_jobs=10)


################################################################################
#
# Multi Process Optimization with JournalStorage
# ----------------------------------------------
#
# .. Note::
#    **Recommended backends**:
#         - JournalStorage
#         - RDBStorage
#
# You can run multiple processes optimization using shared storage,
# Since :class:`~optuna.storages.InMemoryStorage` is not meant to be shared across processes,
# it cannot be used for multi-process optimization.
#
# The following example shows how to use :class:`~optuna.storages.journal.JournalStorage`
# for multi-process optimization.

import optuna
from optuna.storages.journal import JournalStorage, JournalFileBackend


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="journal_storage_multiprocess",
        storage=JournalStorage(JournalFileBackend(file_path="./journal_example.log")),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, n_jobs=2)

################################################################################
# You can run this example with multiple processes:
#
# .. code-block:: console
#
#     $ seq 12 | xargs -n 1 -P 4 python3 multi_process_example.py
#
# Multi Node Optimization
# -----------------------
#
# Since :class:`~optuna.storages.JournalFileBackend` is using host filesystem,
# it is likely to couse a race condition when accessing by multiple machines.
#
# Therefore, it's time to use a RDB backend for multi-node optimization.
# You can use ``mysql`` or other RDB backends.
#
# You need to set up a MySQL server and create a database for Optuna.
#
# .. code-block:: console
#
#    $ mysql -u username -e "CREATE DATABASE IF NOT EXISTS example"
#
# Then, you can use this mysql database as a storage backend just setting the
# MySQL url to the ``storage`` parameter in :func:`~optuna.create_study()`.

import optuna


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="distributed_test",
        storage="mysql://username:password@127.0.0.1:3306/example",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)

################################################################################
# You can run this example accross multiple machines
#
#
# GRPC Proxy Storage
# ------------------
#
# But if you are running thousands of process nodes,
# RDB server may not be able to handle the load.
# In that case, you can use :class:`~optuna.storages.grpc.GrpcStorageProxy`
# for distributeing the server load.
#
# :class:`~optuna.storages.grpc.GrpcStorageProxy` is a proxy storage that
# use another RDB storage as a backend.
# But it can handle multiple requests from multiple machines,
#
# Following example shows how to use :class:`~optuna.storages.grpc.GrpcStorageProxy`
# Since :class:`~optuna.storages.grpc.GrpcStorageProxy` is a proxy storage,
# you need to run a gRPC server with a RDB storage backend first.

from optuna.storages import run_grpc_proxy_server
from optuna.storages import get_storage

storage = get_storage("mysql+pymysql://username:password@127.0.0.1:3306/example")
run_grpc_proxy_server(storage, host="localhost", port=13000)

################################################################################
# Then you can run the following example to use the gRPC proxy storage

import optuna

from optuna.storages import GrpcStorageProxy


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    storage = GrpcStorageProxy(host="localhost", port=13000)
    study = optuna.create_study(
        study_name="grpc_proxy_multinode", load_if_exists=True, storage=storage
    )
    study.optimize(objective, n_trials=50)
