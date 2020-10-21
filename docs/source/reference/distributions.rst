.. module:: optuna.distributions

optuna.distributions
====================

The :mod:`~optuna.distributions` module defines various classes representing probability distributions, mainly used to suggest initial hyperparameter values for an optimization trial. Distribution classes inherit from a library-internal :class:`~optuna.distributions.BaseDistribution`, and is initialized with specific parameters, such as the ``low`` and ``high`` endpoints for a :class:`~optuna.distributions.UniformDistribution`.

Optuna users should not use distribution classes directly, but instead use utility functions provided by :class:`~optuna.trial.Trial` such as :meth:`~optuna.trial.Trial.suggest_int`.

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
