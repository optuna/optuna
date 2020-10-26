.. module:: optuna.storages

optuna.storages
===============

The :mod:`~optuna.storages` module defines a :class:`~optuna.storages.BaseStorage` class which abstracts a backend database and provides library-internal interfaces to read/write histories of studies and trials. Library users who wish to use storage solutions other than the default in-memory storage should use one of the child classes of :class:`~optuna.storages.BaseStorage` documented below.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.storages.RDBStorage
   optuna.storages.RedisStorage
