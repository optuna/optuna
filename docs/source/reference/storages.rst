.. module:: optuna.storages

optuna.storages
===============

The :mod:`~optuna.storages` module defines a :class:`~optuna.storages.BaseStorage` class which abstracts a backend database and provides library-internal interfaces to the read/write histories of the studies and trials. Library users who wish to use storage solutions other than the default in-memory storage should use one of the child classes of :class:`~optuna.storages.BaseStorage` documented below.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.storages.RDBStorage
   optuna.storages.RetryFailedTrialCallback
   optuna.storages.fail_stale_trials
   optuna.storages.JournalStorage
   optuna.storages.InMemoryStorage

optuna.storages.journal
-----------------------

:class:`~optuna.storages.JournalStorage` requires its backend specification and here is the list of the supported backends:

.. note::
   If users would like to use any backends not supported by Optuna, it is possible to do so by creating a customized class by inheriting :class:`optuna.storages.journal.BaseJournalBackend`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.storages.journal.JournalFileBackend
   optuna.storages.journal.JournalRedisBackend

Users can flexibly choose a lock object for :class:`~optuna.storages.journal.JournalFileBackend` and here is the list of supported lock objects:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.storages.journal.JournalFileSymlinkLock
   optuna.storages.journal.JournalFileOpenLock

Deprecated Modules
------------------

.. note::
   The following modules are deprecated at v4.0.0 and will be removed in the future.
   Please use the modules defined in :mod:`optuna.storages.journal`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.storages.BaseJournalLogStorage
   optuna.storages.JournalFileStorage
   optuna.storages.JournalRedisStorage
