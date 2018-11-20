FAQ
===

How do I use Optuna for my favorite ML library?
-----------------------------------------------

Optuna is compatible with most ML libraries, and it's easy to use Optuna for those.
For examples that actually perform optimization using Optuna, please refer to `examples <https://github.com/pfnet/optuna/tree/master/examples>`_.


How to define objective functions that have own arguments?
----------------------------------------------------------

There are two ways to realize it.

First, callable classes can be used for that purpose as follows:

.. code-block:: python

    import optuna

    class Objective(object):
        def __init__(self, min, max):
            # Hold this implementation specific arguments as the fields of the class.
            self.min = min
            self.max = max

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            x = trial.suggest_uniform('x', self.min, self.max)
            return (x - 2) ** 2

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study()
    study.optimize(Objective(-100, 100), n_trials=100)


Second, using ``lambda`` or ``functools.partial`` for creating functions (closures) that hold extra arguments.
Below is an example that uses ``functools.partial``:

.. code-block:: python

    from functools import partial
    import optuna

    # Objective function that takes three arguments.
    def objective(min, max, trial):
        x = trial.suggest_uniform('x', min, max)
        return (x - 2) ** 2

    # Extra arguments.
    min = -100
    max = 100

    # Execute an optimization by using a partial evaluated objective function.
    study = optuna.create_study()
    study.optimize(partial(objective, min, max), n_trials=100)


Can I use Optuna without remote RDB servers?
--------------------------------------------

Yes, it's possible.

In the simplest form, Optuna works with in-memory storage:

.. code-block:: python

    study = optuna.create_study()
    study.optimize(objective)


If you want to save and resume studies,  it's handy to use SQLite as the local storage:

.. code-block:: python

    study = optuna.create_study(study_name='foo_study', storage='sqlite://example.db')
    study.optimize(objective)  # The state of `study` will be persisted to the local SQLite file

Please see :ref:`rdb` for more details.
