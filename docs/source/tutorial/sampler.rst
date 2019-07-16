.. _sampler:

Custom Sampler
==============

This feature enables you to define your own samplers.

A sampler inherits :class:`~optuna.samplers.BaseSampler` and has the responsibility to determine the parameter values to be evaluated in a trial.
When a `suggest` API (e.g., :func:`~optuna.trial.Trial.suggest_uniform`) is called inside an objective function, the corresponding distribution object (e.g., :class:`~optuna.distributions.UniformDistribution`) is created internally. A sampler samples a value from the distribution.

Optuna provides built-in samplers (e.g., :class:`~optuna.samplers.TPESampler`, :class:`~optuna.samplers.RandomSampler`) that work well for a wide range of cases.
However, if you are interested in optimizing hyperparameters in a specific domain, it may be possible to improve optimization performance by using a sampling algorithm specialized to the domain.
The custom sampler feature helps you in this case.

In addition, this feature allows you to use algorithms defined by other existing libraries.
For instance, Optuna provides :class:`~optuna.integration.SkoptSampler` that wraps
`skopt <https://scikit-optimize.github.io/>`_ library.
