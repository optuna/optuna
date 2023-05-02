"""
.. _user_defined_sampler:

User-Defined Sampler
====================

Thanks to user-defined samplers, you can:

- experiment your own sampling algorithms,
- implement task-specific algorithms to refine the optimization performance, or
- wrap other optimization libraries to integrate them into Optuna pipelines (e.g., :class:`~optuna.integration.SkoptSampler`).

This section describes the internal behavior of sampler classes and shows an example of implementing a user-defined sampler.


Overview of Sampler
-------------------

A sampler has the responsibility to determine the parameter values to be evaluated in a trial.
When a `suggest` API (e.g., :func:`~optuna.trial.Trial.suggest_float`) is called inside an objective function, the corresponding distribution object (e.g., :class:`~optuna.distributions.FloatDistribution`) is created internally. A sampler samples a parameter value from the distribution. The sampled value is returned to the caller of the `suggest` API and evaluated in the objective function.

To create a new sampler, you need to define a class that inherits :class:`~optuna.samplers.BaseSampler`.
The base class has three abstract methods;
:meth:`~optuna.samplers.BaseSampler.infer_relative_search_space`,
:meth:`~optuna.samplers.BaseSampler.sample_relative`, and
:meth:`~optuna.samplers.BaseSampler.sample_independent`.

As the method names imply, Optuna supports two types of sampling: one is **relative sampling** that can consider the correlation of the parameters in a trial, and the other is **independent sampling** that samples each parameter independently.

At the beginning of a trial, :meth:`~optuna.samplers.BaseSampler.infer_relative_search_space` is called to provide the relative search space for the trial. Then, :meth:`~optuna.samplers.BaseSampler.sample_relative` is invoked to sample relative parameters from the search space. During the execution of the objective function, :meth:`~optuna.samplers.BaseSampler.sample_independent` is used to sample parameters that don't belong to the relative search space.

.. note::
    Please refer to the document of :class:`~optuna.samplers.BaseSampler` for further details.


An Example: Implementing SimulatedAnnealingSampler
--------------------------------------------------

For example, the following code defines a sampler based on
`Simulated Annealing (SA) <https://en.wikipedia.org/wiki/Simulated_annealing>`_:
"""

import numpy as np
import optuna


class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):
    def __init__(self, temperature=100):
        self._rng = np.random.RandomState()
        self._temperature = temperature  # Current temperature.
        self._current_trial = None  # Current state.

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}

        # Simulated Annealing algorithm.
        # 1. Calculate transition probability.
        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            probability = 1.0
        else:
            probability = np.exp(
                (self._current_trial.value - prev_trial.value) / self._temperature
            )
        self._temperature *= 0.9  # Decrease temperature.

        # 2. Transit the current state if the previous result is accepted.
        if self._rng.uniform(0, 1) < probability:
            self._current_trial = prev_trial

        # 3. Sample parameters from the neighborhood of the current point.
        # The sampled parameters will be used during the next execution of
        # the objective function passed to the study.
        params = {}
        for param_name, param_distribution in search_space.items():
            if (
                not isinstance(param_distribution, optuna.distributions.FloatDistribution)
                or (param_distribution.step is not None and param_distribution.step != 1)
                or param_distribution.log
            ):
                msg = (
                    "Only suggest_float() with `step` `None` or 1.0 and"
                    " `log` `False` is supported"
                )
                raise NotImplementedError(msg)

            current_value = self._current_trial.params[param_name]
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params

    # The rest are unrelated to SA algorithm: boilerplate
    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)


###################################################################################################
# .. note::
#    In favor of code simplicity, the above implementation doesn't support some features (e.g., maximization).
#    If you're interested in how to support those features, please see
#    `examples/samplers/simulated_annealing.py
#    <https://github.com/optuna/optuna-examples/blob/main/samplers/simulated_annealing_sampler.py>`_.
#
#
# You can use ``SimulatedAnnealingSampler`` in the same way as built-in samplers as follows:


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y


sampler = SimulatedAnnealingSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

best_trial = study.best_trial
print("Best value: ", best_trial.value)
print("Parameters that achieve the best value: ", best_trial.params)


###################################################################################################
# In this optimization, the values of ``x`` and ``y`` parameters are sampled by using
# ``SimulatedAnnealingSampler.sample_relative`` method.
#
# .. note::
#     Strictly speaking, in the first trial,
#     ``SimulatedAnnealingSampler.sample_independent`` method is used to sample parameter values.
#     Because :func:`~optuna.search_space.intersection_search_space` used in
#     ``SimulatedAnnealingSampler.infer_relative_search_space`` cannot infer the search space
#     if there are no complete trials.
