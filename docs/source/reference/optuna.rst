.. module:: optuna

optuna
======

The :mod:`optuna` module is primarily used as an alias for basic Optuna functionality coded in other modules. Currently, two modules are aliased: (1) from :mod:`optuna.study`, functions regarding the Study lifecycle, and (2) from :mod:`optuna.exceptions`, the TrialPruned Exception raised when a trial is pruned.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   create_study
   load_study
   delete_study
   copy_study
   get_all_study_names
   get_all_study_summaries
   TrialPruned
