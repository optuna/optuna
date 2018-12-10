FAQ
===

Can I use Optuna with X? (where X is your favorite ML library)
--------------------------------------------------------------

Optuna is compatible with most ML libraries, and it's easy to use Optuna with those.
Please refer to `examples <https://github.com/pfnet/optuna/tree/master/examples>`_.


How to define objective functions that have own arguments?
----------------------------------------------------------

There are two ways to realize it.

First, callable classes can be used for that purpose as follows:

.. code-block:: python

    import optuna

    class Objective(object):
        def __init__(self, min_x, max_x):
            # Hold this implementation specific arguments as the fields of the class.
            self.min_x = min_x
            self.max_x = max_x

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            x = trial.suggest_uniform('x', self.min_x, self.max_x)
            return (x - 2) ** 2

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study()
    study.optimize(Objective(-100, 100), n_trials=100)


Second, you can use ``lambda`` or ``functools.partial`` for creating functions (closures) that hold extra arguments.
Below is an example that uses ``lambda``:

.. code-block:: python

    import optuna

    # Objective function that takes three arguments.
    def objective(trial, min_x, max_x):
        x = trial.suggest_uniform('x', min_x, max_x)
        return (x - 2) ** 2

    # Extra arguments.
    min_x = -100
    max_x = 100

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, min_x, max_x), n_trials=100)


Can I use Optuna without remote RDB servers?
--------------------------------------------

Yes, it's possible.

In the simplest form, Optuna works with in-memory storage:

.. code-block:: python

    study = optuna.create_study()
    study.optimize(objective)


If you want to save and resume studies,  it's handy to use SQLite as the local storage:

.. code-block:: python

    study = optuna.create_study(study_name='foo_study', storage='sqlite:///example.db')
    study.optimize(objective)  # The state of `study` will be persisted to the local SQLite file.

Please see :ref:`rdb` for more details.


How to suppress log messages of Optuna?
---------------------------------------

By default, Optuna shows log messages at the ``optuna.logging.INFO`` level.
You can change logging levels by using  :func:`optuna.logging.set_verbosity`.

For instance, you can stop showing each trial result as follows:

.. code-block:: python

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study()
    study.optimize(objective)
    # Logs like '[I 2018-12-05 11:41:42,324] Finished a trial resulted in value:...' are disabled.


Please refer to :class:`optuna.logging` for further details.


How does Optuna handle NaNs and exceptions reported by the objective function?
--------------------------------------------------------------------------

Optuna treats such trials as failures (i.e., :obj:`~optuna.structs.TrialState.FAIL`) and continues the study.
The Optuna's system process will not be crashed by any objective values or exceptions raised in objective functions.

You can find the failed trials in log messages.
Errors raised in objective functions are shown as follows:

.. code-block:: sh

    [W 2018-12-07 16:38:36,889] Setting trial status as TrialState.FAIL because of \
    the following error: ValueError('A sample error in objective.')

And trials which returned :obj:`NaN` are shown as follows:


.. code-block:: sh

    [W 2018-12-07 16:41:59,000] Setting trial status as TrialState.FAIL because the \
    objective function returned nan.

You can also find the failed trials by checking the trial states as follows:

.. code-block:: python

    study.trials_dataframe()

.. csv-table::

    trial_id,state,value,...,params,system_attrs
    0,TrialState.FAIL,,...,0,Setting trial status as TrialState.FAIL because of the following error: ValueError('A test error in objective.')
    1,TrialState.COMPLETE,1269,...,1,
