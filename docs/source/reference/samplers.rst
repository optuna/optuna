.. module:: optuna.samplers

optuna.samplers
===============

The :mod:`~optuna.samplers` module defines a base class for parameter sampling as described extensively in :class:`~optuna.samplers.BaseSampler`. The remaining classes in this module represent child classes, deriving from :class:`~optuna.samplers.BaseSampler`, which implement different sampling strategies.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.samplers.BaseSampler
   optuna.samplers.GridSampler
   optuna.samplers.RandomSampler
   optuna.samplers.TPESampler
   optuna.samplers.CmaEsSampler
   optuna.samplers.PartialFixedSampler
   optuna.samplers.NSGAIISampler
   optuna.samplers.MOTPESampler
   optuna.samplers.IntersectionSearchSpace
   optuna.samplers.intersection_search_space
