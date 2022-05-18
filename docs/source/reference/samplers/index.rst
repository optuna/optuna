.. module:: optuna.samplers

optuna.samplers
===============

The :mod:`~optuna.samplers` module defines a base class for parameter sampling as described extensively in :class:`~optuna.samplers.BaseSampler`. The remaining classes in this module represent child classes, deriving from :class:`~optuna.samplers.BaseSampler`, which implement different sampling strategies.

.. seealso::
    :ref:`pruning` tutorial explains the overview of the sampler classes.

.. seealso::
    :ref:`user_defined_sampler` tutorial could be helpful if you want to implement your own sampler classes.

+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
|                                  | RandomSampler | GridSampler | TPESampler  | CmaEsSampler |              NSGAIISampler             | QMCSampler | BoTorchSampler |
+==================================+===============+=============+=============+==============+========================================+============+================+
| Float parameters                 |       ✅      |     ✅      |     ✅      |      ✅      | ▲(✅ if you use non-default crossover) |     ✅     |       ✅       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Integer parameters               |       ✅      |     ✅      |     ✅      |      ✅      | ▲(✅ if you use non-default crossover) |     ✅     |       ✅       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Categorical parameters           |       ✅      |     ✅      |     ✅      |      ❌      |                   ✅                   |     ❌     |       ✅       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Multivariate optimization        |       ❌      |     ❌      |     ✅      |      ✅      |                   ❌                   |     ❌     |       ✅       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Conditional search space         |       ❌      |     ❌      |     ✅      |      ❌      |                   ❌                   |     ❌     |       ❌       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Multi-objective optimization (*) |       ▲       |     ▲       |     ✅      |      ❌      |       ✅(▲ for single-objective)       |     ▲      |       ✅       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Batch optimization               |       ❌      |     ❌      |     ✅      |      ❌      |                   ❌                   |     ❌     |       ❌       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Constrained optimization         |       ❌      |     ❌      |     ✅      |      ❌      |                   ✅                   |     ❌     |       ✅       |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+
| Time complecity (per trial)      |      O(d)     |    O(dn)    | O(dnlog(n)) |   O(dn+d^3)  |                O(dmn^2)                |    O(d)    |     O(dn^3)    |
+----------------------------------+---------------+-------------+-------------+--------------+----------------------------------------+------------+----------------+

.. note::
    ✅: Support this feature. 
    ▲ : Work, but inefficiently.
    ❌: Does not support the feature, but does not cause any errors.

    (*) represents as follows.
    ✅: Support this feature.
    ▲ : Does not support the feature, but does not cause any errors.
    ❌: Cause an error.

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
