.. module:: optuna.distributions

Distributions
=============

.. autoclass:: UniformDistribution
    :members:

.. autoclass:: LogUniformDistribution
    :members:

.. autoclass:: DiscreteUniformDistribution
    :members:

.. autoclass:: IntUniformDistribution
    :members:
    :exclude-members: to_external_repr, to_internal_repr

.. autoclass:: IntLogUniformDistribution
    :members:
    :exclude-members: to_external_repr, to_internal_repr

.. autoclass:: CategoricalDistribution
    :members:
    :exclude-members: to_external_repr, to_internal_repr

.. autofunction:: distribution_to_json

.. autofunction:: json_to_distribution

.. autofunction:: check_distribution_compatibility
