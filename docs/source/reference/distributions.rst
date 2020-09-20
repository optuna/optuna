optuna.distributions
====================

The :mod:`~optuna.distributions` module defines various Classes representing probability distributions, mainly used to suggest starting hyperparameter values for an optimization trial. A Distribution inherits from a library-internal BaseDistribution, and is initialized with specific parameters, such as the ``low`` and ``high`` endpoints for a :class:`~optuna.distributions.UniformDistribution`.

Library users should not call use Distribution classes directly, but instead use utility functions provided by :class:`optuna.trial.Trial` such as :meth:`optuna.trial.Trial.suggest_int`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.distributions.UniformDistribution
   optuna.distributions.LogUniformDistribution
   optuna.distributions.DiscreteUniformDistribution
   optuna.distributions.IntUniformDistribution
   optuna.distributions.IntLogUniformDistribution
   optuna.distributions.CategoricalDistribution
   optuna.distributions.distribution_to_json
   optuna.distributions.json_to_distribution
   optuna.distributions.check_distribution_compatibility
