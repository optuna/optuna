.. _sampler:

User-Defined Sampler
=====================

This feature enables you to define your samplers.

A sampler has the responsibility to determine the parameter values to be evaluated in a trial.
When a `suggest` API (e.g., :func:`~optuna.trial.Trial.suggest_uniform`) is called inside an objective function, the corresponding distribution object (e.g., :class:`~optuna.distributions.UniformDistribution`) is created internally. A sampler samples a value from the distribution. The sampled value is returned to the caller of the `suggest` API and evaluated in the objective function.

Optuna provides built-in samplers (e.g., :class:`~optuna.samplers.TPESampler`, :class:`~optuna.samplers.RandomSampler`) that work well for a wide range of cases.
However, if you are only interested in optimizing hyperparameters in a specific domain, optimization performance may be improved if you use a sampling algorithm specialized to the domain.
Thanks to custom sampler feature, you can use such specialized algorithms within the Optuna framework.

In addition, this feature allows you to use algorithms implemented by other libraries.
For instance, Optuna provides :class:`~optuna.integration.SkoptSampler` that wraps
`skopt <https://scikit-optimize.github.io/>`_ library.


An example: Implementing SimulatedAnnealingSampler
--------------------------------------------------

In this section, let's implement a sampler based on
`Simulate Annealing (SA) <https://en.wikipedia.org/wiki/Simulated_annealing>`_ algorithm.

.. note::
   For simplicity, the following implementation doesn't support some features (e.g., maximization).
   If you are interested, more completed version is found in
   `simulated_annealing.py <https://github.com/pfnet/optuna/tree/master/examples/sampler/simulated_annealing.py>`_
   example.

First, you need to define a class that inherits :class:`~optuna.samplers.BaseSampler`.
In the constructor of ``SimulatedAnnealingSampler``, the temperature and state of SA are initialized.

.. code-block:: python

    import numpy as np
    import optuna


    class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):
        def __init__(self, temperature=100, seed=None):
            # type: (int, Optional[int]) -> None

            self._rng = np.random.RandomState(seed)
            self._temperature = temperature  # Current temperature.
            self._current_params = {}  # Current state (parameter names and values).


Then, let's define :meth:`~optuna.samplers.BaseSampler.sample_relative` method.
:meth:`~optuna.samplers.BaseSampler.sample_relative` is called at the beginning of a trial for sampling parameters from the given search space (i.e., distributions).
The following code is the core part of this sampler.

.. code-block:: python

    ...

        def sample_relative(self, study, trial, search_space):
            # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

            if search_space == {}:
                return {}

            # Calculate transition probability.
            prev_trial = study.trials[-2]
            best_trial = study.best_trial
            if prev_trial.value <= best_trial.value:
                probability = 1.0
            else:
                probability = np.exp((best_trial.value - prev_trial.value) / self._temperature)
            self._temperature *= 0.9  # Decrease temperature.

            # Transit the current state if the previous result is accepted.
            if self._rng.uniform(0, 1) < probability:
                self._current_params = prev_trial.params

            # Sample parameters for the trial.
            params = {}
            for param_name, param_distribution in search_space.items():
                if not isinstance(param_distribution, optuna.distributions.UniformDistribution):
                    raise NotImplementedError('Only suggest_uniform() is supported')

                current_value = self._current_params[param_name]
                width = (param_distribution.high - param_distribution.low) * 0.1
                neighbor_low = max(current_value - width, param_distribution.low)
                neighbor_high = min(current_value + width, param_distribution.high)
                params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

            return params


Finally, it's needed to implement other abstract methods of :class:`~optuna.samplers.BaseSampler` as the following code.
About the details of those methods, please read the next section.

.. code-block:: python

    ...

        def infer_relative_search_space(self, study, trial):
            # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

            return optuna.samplers.product_search_space(study)

        def sample_independent(self, study, trial, param_name, param_distribution):
            # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> Any

            independent_sampler = optuna.samplers.RandomSampler()
            return independent_sampler.sample_independent(study, trial, param_name, param_distribution)


``SimulatedAnnealingSampler`` is complete.
The custom sampler can be used in the same way as built-in samplers (see below).

.. code-block:: python

    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        y = trial.suggest_uniform('x', -5, 5)
        return x**2 + y

    sampler = SimulatedAnnealingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
