.. _firstopt:

First Optimization
==================


Quadratic Function Example
--------------------------

Usually, Optuna is used to optimize hyper-parameters, but as an example, let us directly optimize a quadratic function in an IPython shell.

.. code-block:: python

    import optuna

The objective function is what will be optimized.

.. code-block:: python

    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2
    

This function returns the value of :math:`(x - 2)^2`. Our goal is to find the value of ``x`` that minimizes the output of the ``objective`` function. This is the "optimization." During the optimization, Optuna repeatedly calls and evaluates the objective function with different values of ``x``.

A :class:`~optuna.trial.Trial` object corresponds to a single execution of the objective function and is internally instantiated upon each invocation of the function.

The `suggest` APIs (for example, :func:`~optuna.trial.Trial.suggest_uniform`) are called inside the objective function to obtain parameters for a trial. :func:`~optuna.trial.Trial.suggest_uniform` selects parameters uniformly within the range provided. In our example, from -10 to 10.

To start the optimization, we create a study object and pass the objective function to method :func:`~optuna.study.Study.optimize` as follows.

.. code-block:: python

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

Out:

.. code-block:: console

    [I 2018-05-09 10:03:22,469] Finished trial#0 resulted in value: 52.9345515866657. Current best value is 52.9345515866657 with parameters: {'x': -5.275613485244093}.
    [I 2018-05-09 10:03:22,474] Finished trial#1 resulted in value: 32.82718929591965. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,475] Finished trial#2 resulted in value: 46.89428737068025. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,476] Finished trial#3 resulted in value: 100.99613064563654. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,477] Finished trial#4 resulted in value: 110.56391159932272. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,478] Finished trial#5 resulted in value: 42.486606942847395. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,479] Finished trial#6 resulted in value: 1.130813338091735. Current best value is 1.130813338091735 with parameters: {'x': 3.063397074517198}.
    ...
    [I 2018-05-09 10:03:23,431] Finished trial#99 resulted in value: 8.760381111220335. Current best value is 0.0026232243068543526 with parameters: {'x': 1.9487825780924659}.

You can get the best parameter as follows.

.. code-block:: python

    study.best_params

Out:

.. code-block:: console

    {'x': 1.9487825780924659}

We can see that Optuna found the best ``x`` value ``1.9487825780924659``, which is close to the optimal value of ``2``.

.. note::
    When used to search for hyper-parameters in machine learning, usually the objective function would return the loss or accuracy of the model.

Study Object
------------

Let us clarify the terminology in Optuna as follows:

* **Trial**: A single call of the objective function
* **Study**: An optimization session, which is a set of trials
* **Parameter**: A variable whose value is to be optimized, such as ``x`` in the above example

In Optuna, we use the study object to manage optimization. Method :func:`~optuna.study.create_study` returns a study object.
A study object has useful properties for analyzing the optimization outcome.

To get the best parameter:

.. code-block:: python

    study.best_params

Out:

.. code-block:: console

    {'x': 1.9926578647650126}

To get the best value:

.. code-block:: python

    study.best_value

Out:

.. code-block:: console

    5.390694980884334e-05

To get the best trial:

.. code-block:: python

    study.best_trial

Out:

.. code-block:: console

    FrozenTrial(number=26, state=<TrialState.COMPLETE: 1>, params={'x': 1.9926578647650126}, user_attrs={}, system_attrs={'_number': 26}, value=5.390694980884334e-05, intermediate_values={}, datetime_start=datetime.datetime(2018, 5, 9, 10, 23, 0, 87060), datetime_complete=datetime.datetime(2018, 5, 9, 10, 23, 0, 91010), trial_id=26)

To get all trials:

.. code-block:: python

    study.trials

Out:

.. code-block:: console

    [FrozenTrial(number=0, state=<TrialState.COMPLETE: 1>, params={'x': -4.219801301030433}, user_attrs={}, system_attrs={'_number': 0}, value=38.685928224299865, intermediate_values={}, datetime_start=datetime.datetime(2018, 5, 9, 10, 22, 59, 983824), datetime_complete=datetime.datetime(2018, 5, 9, 10, 22, 59, 984053), trial_id=0),
     ...
     user_attrs={}, system_attrs={'_number': 99}, value=8.2881000286123179, intermediate_values={}, datetime_start=datetime.datetime(2018, 5, 9, 10, 23, 0, 886434), datetime_complete=datetime.datetime(2018, 5, 9, 10, 23, 0, 891347), trial_id=99)]

To get the number of trials:

.. code-block:: python

    len(study.trials)

Out:

.. code-block:: console

    100

By executing :func:`~optuna.study.Study.optimize` again, we can continue the optimization.

.. code-block:: python

    study.optimize(objective, n_trials=100)

To get the updated number of trials:

.. code-block:: python

    len(study.trials)

Out:

.. code-block:: console

    200
