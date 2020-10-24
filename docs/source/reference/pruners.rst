.. module:: optuna.pruners

optuna.pruners
==============

The :mod:`~optuna.pruners` module defines a :class:`~optuna.pruners.BasePruner` class characterized by an abstract :meth:`~optuna.pruners.BasePruner.prune` method, which, for a given trial and its associated study, returns a boolean value representing whether the trial should be pruned. This determination is made based on stored intermediate values of the objective function, as previously reported for the trial using :meth:`optuna.trial.Trial.report`. The remaining classes in this module represent child classes, inheriting from :class:`~optuna.pruners.BasePruner`, which implement different pruning strategies.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.pruners.BasePruner
   optuna.pruners.MedianPruner
   optuna.pruners.NopPruner
   optuna.pruners.PercentilePruner
   optuna.pruners.SuccessiveHalvingPruner
   optuna.pruners.HyperbandPruner
   optuna.pruners.ThresholdPruner
