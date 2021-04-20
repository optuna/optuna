.. module:: optuna

optuna
======

The :mod:`optuna` module is primarily used as an alias for basic Optuna functionality coded in other modules. Currently, two modules are aliased: (1) from :mod:`optuna.study`, functions regarding the Study lifecycle, and (2) from :mod:`optuna.exceptions`, the TrialPruned Exception raised when a trial is pruned.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.create_study
   optuna.load_study
   optuna.delete_study
   optuna.copy_study
   optuna.get_all_study_summaries
   optuna.TrialPruned
