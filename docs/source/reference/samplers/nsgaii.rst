.. module:: optuna.samplers.nsgaii

optuna.samplers.nsgaii
======================

The :mod:`~optuna.samplers.nsgaii` module defines crossover and mutation operations used by
:class:`~optuna.samplers.NSGAIISampler` and :class:`~optuna.samplers.NSGAIIISampler`.

Crossover
---------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    BaseCrossover
    UniformCrossover
    BLXAlphaCrossover
    SPXCrossover
    SBXCrossover
    VSBXCrossover
    UNDXCrossover

Mutation
--------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    BaseMutation
    PolynomialMutation
