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

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.storages.journal.BaseJournalBackend
   optuna.storages.journal.JournalFileBackend
   optuna.storages.journal.JournalRedisBackend
   optuna.storages.journal.JournalFileSymlinkLock
   optuna.storages.journal.JournalFileOpenLock
   optuna.storages.journal.BaseJournalLogStorage
   optuna.storages.journal.JournalFileStorage
   optuna.storages.journal.JournalRedisStorage
