FAQ
===

How do I use Optuna for Chainer, XGBoost, LightGBM, and others?
---------------------------------------------------------------

Using Optuna with those is easy.
Please refer to `examples <https://github.com/pfnet/optuna/tree/master/examples>`_ for code that works.
Besides, we provide various :doc:`integrations <reference/integration>` for advanced features (e.g., pruning, distributed optimization).


How to define objective functions that have own arguments?
----------------------------------------------------------

There are two methods to realize it.

First, callable classes can be used for that purpose as follows:

.. code-block:: python

    class Objective(object):
        def __init__(self, min, max):
            # Hold this implementation specific arguments as the fields of the class.
            self.min = min
            self.max = max

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            x = trial.sugget_uniform('x', self.min, self.max)
            return (x - 2) ** 2

    study = optuna.create_study()
    study.optimize(Object(-100, 100), n_trials=100)


Second, using ``lambda`` or ``functools.partial`` for creating functions (closures) that hold extra parameters.
Below is an example that uses ``lambda``:

.. code-block:: python

    min = -100
    max = 100

    objective = lamba trial: (trial.sugget_uniform('x', min, max) - 2) ** 2

    study = optuna.create_study()
    study.optimize(Object(-100, 100), n_trials=100)
