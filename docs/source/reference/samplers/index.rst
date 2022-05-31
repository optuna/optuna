.. module:: optuna.samplers

optuna.samplers
===============

The :mod:`~optuna.samplers` module defines a base class for parameter sampling as described extensively in :class:`~optuna.samplers.BaseSampler`. The remaining classes in this module represent child classes, deriving from :class:`~optuna.samplers.BaseSampler`, which implement different sampling strategies.

.. seealso::
    :ref:`pruning` tutorial explains the overview of the sampler classes.

.. seealso::
    :ref:`user_defined_sampler` tutorial could be helpful if you want to implement your own sampler classes.

+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
|                                  | RandomSampler | GridSampler  | TPESampler  | CmaEsSampler |              NSGAIISampler             | QMCSampler | BoTorchSampler |
+==================================+===============+==============+=============+==============+========================================+============+================+
| Float parameters                 |       ✅      |     ✅       |     ✅      |      ✅      |                   ▲                    |     ✅     |       ✅       |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Integer parameters               |       ✅      |     ✅       |     ✅      |      ✅      |                   ▲                    |     ✅     |       ✅       |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Categorical parameters           |       ✅      |     ✅       |     ✅      |      ▲       |                   ✅                   |     ▲      |       ✅       |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Pruning                          |       ✅      |     ✅       |     ✅      |      ▲       |                   ❌                   |     ✅     |       ▲        |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Multivariate optimization        |       ▲       |     ▲        |     ✅      |      ✅      |                   ▲                    |     ▲      |       ✅       |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Conditional search space         |       ✅      |     ▲        |     ✅      |      ▲       |                   ▲                    |     ▲      |       ▲        |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Multi-objective optimization     |       ✅      |     ▲        |     ✅      |      ❌      |       ✅(▲ for single-objective)       |     ▲      |       ✅       |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Batch optimization               |       ✅      |     ✅       |     ✅      |      ✅      |                   ✅                   |     ✅     |       ▲        |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Distributed optimization         |       ✅      |     ✅       |     ✅      |      ✅      |                   ✅                   |     ✅     |       ▲        |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Constrained optimization         |       ❌      |     ❌       |     ✅      |      ❌      |                   ✅                   |     ❌     |       ✅       |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Time complexity (per trial) (*)  |      O(1)     |    O(dn^2)   | O(dnlog(n)) |    O(d^3)    |                 O(mnp)                 |    O(dn)   |     O(n^3)     |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+
| Recommended budgets (#trials)    | as many as    | number of    | 100 ~ 1000  | 1000 ~ 10000 |                100 ~ 10000             | as many as |    10 ~ 100    |
| (**)                             | one likes     | combinations |             |              |                                        | one likes  |                |
+----------------------------------+---------------+--------------+-------------+--------------+----------------------------------------+------------+----------------+

.. note::
    ✅: Supports this feature. 
    ▲ : Works, but inefficiently.
    ❌: Causes an error, or has no interface.

    (*): We assumes that `d` is the dimension of the search space, `n` is the number of finished trials, `m` is the number of objectives, and `p` is the population size (algorithm specific parameter).
    This table shows the time complexity of the sampling algorithms. We may omit other terms that depend on the implementation in Optuna, including O(d) to call the sampling methods and O(n) to collect the completed trials.
    This means that, for example, the actual time complexity of :class:`~optuna.samplers.RandomSampler` is O(d+n+1) = O(d+n).
    From another perspective, with the exception of :class:`~optuna.samplers.NSGAIISampler`, all time complexity is written for single-objective optimization.

    (**): The budget depends on the number of parameters and the number of objectives.

.. note::
   For float, integer, or categorical parameters, see :ref:`configurations` tutorial.

   For pruning, see :ref:`pruning` tutorial.
   
   For multivariate optimization, see :class:`~optuna.samplers.BaseSampler`. The multivariate optimization is implemented as :func:`~optuna.samplers.BaseSampler.sample_relative` in Optuna. Please check the concrete documents of samplers for more details.

   For conditional search space, see :ref:`configurations` tutorial and :class:`~optuna.samplers.TPESampler`. The ``group`` option of :class:`~optuna.samplers.TPESampler` allows :class:`~optuna.samplers.TPESampler` to handle the conditional search space.

   For multi-objective optimization, see :ref:`multi_objective` tutorial.

   For batch optimization, see :ref:`Batch-Optimization` tutorial. Note that the ``constant_liar`` option of :class:`~optuna.samplers.TPESampler` allows :class:`~optuna.samplers.TPESampler` to handle the batch optimization.

   For distributed optimization, see :ref:`distributed` tutorial. Note that the ``constant_liar`` option of :class:`~optuna.samplers.TPESampler` allows :class:`~optuna.samplers.TPESampler` to handle the distributed optimization.

   For constrained optimization, see an `example <https://github.com/optuna/optuna-examples/blob/main/multi_objective/botorch_simple.py>`_.

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
