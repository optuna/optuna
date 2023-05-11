.. module:: optuna.samplers

optuna.samplers
===============

The :mod:`~optuna.samplers` module defines a base class for parameter sampling as described extensively in :class:`~optuna.samplers.BaseSampler`. The remaining classes in this module represent child classes, deriving from :class:`~optuna.samplers.BaseSampler`, which implement different sampling strategies.

.. seealso::
    :ref:`pruning` tutorial explains the overview of the sampler classes.

.. seealso::
    :ref:`user_defined_sampler` tutorial could be helpful if you want to implement your own sampler classes.

+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
|                                  |         RandomSampler         |          GridSampler          |          TPESampler           |         CmaEsSampler          |                                NSGAIISampler                                |          QMCSampler           |        BoTorchSampler         |                               BruteForceSampler                               |
+==================================+===============================+===============================+===============================+===============================+=============================================================================+===============================+===============================+===============================================================================+
| Float parameters                 |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                           :math:`\blacktriangle`                            |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark` (:math:`\color{red}\times` for infinite domain)|
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Integer parameters               |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                           :math:`\blacktriangle`                            |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Categorical parameters           |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |                       :math:`\color{green}\checkmark`                       |    :math:`\blacktriangle`     |:math:`\color{green}\checkmark`|                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Pruning                          |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |                         :math:`\color{red}\times`                           |:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Multivariate optimization        |    :math:`\blacktriangle`     |    :math:`\blacktriangle`     |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                           :math:`\blacktriangle`                            |    :math:`\blacktriangle`     |:math:`\color{green}\checkmark`|                            :math:`\blacktriangle`                             |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Conditional search space         |:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |                           :math:`\blacktriangle`                            |    :math:`\blacktriangle`     |    :math:`\blacktriangle`     |                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Multi-objective optimization     |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|   :math:`\color{red}\times`   |:math:`\color{green}\checkmark` (:math:`\blacktriangle` for single-objective)|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Batch optimization               |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                       :math:`\color{green}\checkmark`                       |:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Distributed optimization         |:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|:math:`\color{green}\checkmark`|                       :math:`\color{green}\checkmark`                       |:math:`\color{green}\checkmark`|    :math:`\blacktriangle`     |                        :math:`\color{green}\checkmark`                        |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Constrained optimization         |   :math:`\color{red}\times`   |   :math:`\color{red}\times`   |:math:`\color{green}\checkmark`|   :math:`\color{red}\times`   |                       :math:`\color{green}\checkmark`                       |   :math:`\color{red}\times`   |:math:`\color{green}\checkmark`|                           :math:`\color{red}\times`                           |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Time complexity (per trial) (*)  |         :math:`O(d)`          |        :math:`O(dn)`          |     :math:`O(dn \log n)`      |        :math:`O(d^3)`         |                           :math:`O(mp^2)` (\*\*\*)                          |         :math:`O(dn)`         |        :math:`O(n^3)`         |                                 :math:`O(d)`                                  |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+
| Recommended budgets (#trials)    | as many as one likes          | number of combinations        |          100 – 1000           |         1000 – 10000          |                                100 – 10000                                  | as many as one likes          |           10 – 100            |                            number of combinations                             |
| (**)                             |                               |                               |                               |                               |                                                                             |                               |                               |                                                                               |
+----------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-----------------------------------------------------------------------------+-------------------------------+-------------------------------+-------------------------------------------------------------------------------+

.. note::
   :math:`\color{green}\checkmark`: Supports this feature.
   :math:`\blacktriangle`: Works, but inefficiently.
   :math:`\color{red}\times`: Causes an error, or has no interface.

    (*): We assumes that :math:`d` is the dimension of the search space, :math:`n` is the number of finished trials, :math:`m` is the number of objectives, and :math:`p` is the population size (algorithm specific parameter).
    This table shows the time complexity of the sampling algorithms. We may omit other terms that depend on the implementation in Optuna, including :math:`O(d)` to call the sampling methods and :math:`O(n)` to collect the completed trials.
    This means that, for example, the actual time complexity of :class:`~optuna.samplers.RandomSampler` is :math:`O(d+n+d) = O(d+n)`.
    From another perspective, with the exception of :class:`~optuna.samplers.NSGAIISampler`, all time complexity is written for single-objective optimization.

    (**): The budget depends on the number of parameters and the number of objectives.

    (\*\*\*): This time complexity assumes that the number of population size :math:`p` and the number of parallelization are regular.
    This means that the number of parallelization should not exceed the number of population size :math:`p`.

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
    optuna.samplers.NSGAIIISampler
    optuna.samplers.MOTPESampler
    optuna.samplers.QMCSampler
    optuna.samplers.BruteForceSampler
    optuna.samplers.IntersectionSearchSpace
    optuna.samplers.intersection_search_space

.. note::
    The following :mod:`optuna.samplers.nsgaii` module defines crossover operations used by :class:`~optuna.samplers.NSGAIISampler`.

.. toctree::
    :maxdepth: 1

    nsgaii
