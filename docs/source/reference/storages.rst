.. module:: optuna.storages

optuna.storages
===============

The :mod:`~optuna.storages` module defines a :class:`~optuna.storages.BaseStorage` class which abstracts a backend database and provides library-internal interfaces to the read/write histories of the studies and trials. Library users who wish to use storage solutions other than the default :class:`~optuna.storages.InMemoryStorage` should use one of the child classes of :class:`~optuna.storages.BaseStorage` documented below.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   RDBStorage
   RetryFailedTrialCallback
   fail_stale_trials
   JournalStorage
   InMemoryStorage
   run_grpc_proxy_server
   GrpcStorageProxy

optuna.storages.journal
-----------------------

:class:`~optuna.storages.JournalStorage` requires its backend specification and here is the list of the supported backends:

.. note::
   If users would like to use any backends not supported by Optuna, it is possible to do so by creating a customized class by inheriting :class:`optuna.storages.journal.BaseJournalBackend`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   journal.JournalFileBackend
   journal.JournalRedisBackend

Users can flexibly choose a lock object for :class:`~optuna.storages.journal.JournalFileBackend` and here is the list of supported lock objects:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   journal.JournalFileSymlinkLock
   journal.JournalFileOpenLock

Deprecated Modules
------------------

.. note::
   The following modules are deprecated at v4.0.0 and will be removed in the future.
   Please use the modules defined in :mod:`optuna.storages.journal`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BaseJournalLogStorage
   JournalFileStorage
   JournalRedisStorage
