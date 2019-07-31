.. _sampler:

User-Defined Sampler
=====================

Optuna allows users to create user-defined samplers.

A sampler has the responsibility to determine the parameter values to be evaluated in a trial.
When a `suggest` API (e.g., :func:`~optuna.trial.Trial.suggest_uniform`) is called inside an objective function, the corresponding distribution object (e.g., :class:`~optuna.distributions.UniformDistribution`) is created internally. A sampler samples a value from the distribution. The sampled value is returned to the caller of the `suggest` API and evaluated in the objective function.

Optuna provides built-in samplers (e.g., :class:`~optuna.samplers.TPESampler`, :class:`~optuna.samplers.RandomSampler`) that work well for a wide range of cases.
However, if you are only interested in optimizing hyperparameters in a specific domain, optimization performance may be improved if you use a sampling algorithm specialized to the domain.
Thanks to user-defined sampler feature, you can use such specialized algorithms within Optuna framework.

In addition, this feature allows you to use algorithms implemented by other libraries.
For instance, Optuna provides :class:`~optuna.integration.SkoptSampler` that wraps
`skopt <https://scikit-optimize.github.io/>`_ library.


An Example: SimulatedAnnealingSampler
-------------------------------------

For creating a new sampler, you need to define a class that inherits :class:`~optuna.samplers.BaseSampler`,
and implement the three abstract methods of the base class (
:meth:`~optuna.samplers.BaseSampler.infer_relative_search_space`,
:meth:`~optuna.samplers.BaseSampler.sample_relative` and
:meth:`~optuna.samplers.BaseSampler.sample_independent`).

As an example, the following code defines a sampler named ``SimulatedAnnealingSampler`` that based on
`Simulate Annealing (SA) <https://en.wikipedia.org/wiki/Simulated_annealing>`_ algorithm:

.. code-block:: python

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

            #
            # An implementation of SA algorithm.
            #

            # Calculate transition probability.
            prev_trial = study.trials[-2]
            if self._current_trial is None or prev_trial.value <= self._current_trial.value:
                probability = 1.0
            else:
                probability = np.exp((self._current_trial.value - prev_trial.value) / self._temperature)
            self._temperature *= 0.9  # Decrease temperature.

            # Transit the current state if the previous result is accepted.
            if self._rng.uniform(0, 1) < probability:
                self._current_trial = prev_trial

            # Sample parameters from the neighborhood of the current point.
            #
            # The sampled parameters will be used during the next execution of
            # the objective function passed to the study.
            params = {}
            for param_name, param_distribution in search_space.items():
                if not isinstance(param_distribution, optuna.distributions.UniformDistribution):
                    raise NotImplementedError('Only suggest_uniform() is supported')

                current_value = self._current_trial.params[param_name]
                width = (param_distribution.high - param_distribution.low) * 0.1
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

            return params

        #
        # The rest is boilerplate code and unrelated to SA algorithm.
        #
        def infer_relative_search_space(self, study, trial):
            return optuna.samplers.intersection_search_space(study)

        def sample_independent(self, study, trial, param_name, param_distribution):
            independent_sampler = optuna.samplers.RandomSampler()
            return independent_sampler.sample_independent(study, trial, param_name, param_distribution)


.. note::
   In favor of code simplicity, the above implementation doesn't support some features (e.g., maximization).
   If you are interested, more complete version is found in
   `simulated_annealing.py <https://github.com/pfnet/optuna/tree/master/examples/samplers/simulated_annealing.py>`_
   example.


You can use ``SimulatedAnnealingSampler`` in the same way as built-in samplers as follows:

.. code-block:: python

    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        y = trial.suggest_uniform('y', -5, 5)
        return x**2 + y

    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)


In this optimization, the values of ``x`` and ``y`` parameters are sampled by using
``SimulatedAnnealingSampler.sample_relative`` method.

.. note::
    Strictly speaking, in the first trial,
    ``SimulatedAnnealingSampler.sample_independent`` method is used for sampling parameter values
    because ``SimulatedAnnealingSampler.infer_relative_search_space`` cannot infer the search space
    if there are no complete trials.
