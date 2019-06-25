.. module:: optuna.distributions

Distributions
=============

.. autoclass:: BaseDistribution
    :members:
    :inherited-members:
    :exclude-members: count, index, high, low

.. autoclass:: UniformDistribution
    :members:
    :inherited-members:
    :exclude-members: count, index, high, low

.. autoclass:: LogUniformDistribution
    :members:
    :inherited-members:
    :exclude-members: count, index, high, low

.. autoclass:: DiscreteUniformDistribution
    :members:
    :inherited-members:
    :exclude-members: count, index, high, low, q

.. autoclass:: IntUniformDistribution
    :members:
    :inherited-members:
    :exclude-members: count, index, high, low

.. autoclass:: CategoricalDistribution
    :members:
    :inherited-members:
    :exclude-members: count, index, choices

.. autofunction:: distribution_to_json

.. autofunction:: json_to_distribution

.. autofunction:: check_distribution_compatibility
