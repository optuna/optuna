.. module:: optuna.samplers

optuna.samplers
===============

The :mod:`~optuna.samplers` module defines a base class for parameter sampling as described extensively in :class:`~optuna.samplers.BaseSampler`. The remaining classes in this module represent child classes, deriving from :class:`~optuna.samplers.BaseSampler`, which implement different sampling strategies.

.. seealso::
    | :ref:`pruning` tutorial explains the overview of the sampler classes.
    | :ref:`user_defined_sampler` tutorial could be helpful if you want to implement your own sampler classes.

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
   optuna.samplers.QMCSampler
   optuna.samplers.IntersectionSearchSpace
   optuna.samplers.intersection_search_space

.. toctree::
    :maxdepth: 1

    nsgaii
