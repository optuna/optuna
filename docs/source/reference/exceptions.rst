.. module:: optuna.exceptions

optuna.exceptions
=================

The :mod:`~optuna.exceptions` module defines Optuna-specific exceptions deriving from a base :class:`~optuna.exceptions.OptunaError` class. Of special importance for library users is the :class:`~optuna.exceptions.TrialPruned` exception to be raised if :func:`optuna.trial.Trial.should_prune` returns ``True`` for a trial that should be pruned.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.exceptions.OptunaError
   optuna.exceptions.TrialPruned
   optuna.exceptions.CLIUsageError
   optuna.exceptions.StorageInternalError
   optuna.exceptions.DuplicatedStudyError
