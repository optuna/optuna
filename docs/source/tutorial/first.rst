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

    [I 2020-04-08 10:42:09,028] Trial 0 finished with value: 25.77382032395108 with parameters: {'x': 7.076792326257898}. Best is trial 0 with value: 25.77382032395108.
    [I 2020-04-08 10:42:09,064] Trial 1 finished with value: 1.5189812248635903 with parameters: {'x': 0.7675304365366298}. Best is trial 1 with value: 1.5189812248635903.
    [I 2020-04-08 10:42:09,106] Trial 2 finished with value: 34.4074691838153 with parameters: {'x': -3.865788027521562}. Best is trial 1 with value: 1.5189812248635903.
    [I 2020-04-08 10:42:09,145] Trial 3 finished with value: 3.3601305753722657 with parameters: {'x': 3.8330658949891205}. Best is trial 1 with value: 1.5189812248635903.
    [I 2020-04-08 10:42:09,185] Trial 4 finished with value: 61.16797535698886 with parameters: {'x': -5.820995803412048}. Best is trial 1 with value: 1.5189812248635903.
    [I 2020-04-08 10:42:09,228] Trial 5 finished with value: 90.08665552769618 with parameters: {'x': -7.491399028999686}. Best is trial 1 with value: 1.5189812248635903.
    [I 2020-04-08 10:42:09,274] Trial 6 finished with value: 25.254236332163032 with parameters: {'x': 7.025359323686519}. Best is trial 1 with value: 1.5189812248635903.
    ...
    [I 2020-04-08 10:42:14,237] Trial 99 finished with value: 0.5227007740782738 with parameters: {'x': 2.7229804797352926}. Best is trial 67 with value: 2.916284393762304e-06.

You can get the best parameter as follows.

.. code-block:: python

    study.best_params

Out:

.. code-block:: console

    {'x': 2.001707713205946}

We can see that Optuna found the best ``x`` value ``2.001707713205946``, which is close to the optimal value of ``2``.

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

    {'x': 2.001707713205946}

To get the best value:

.. code-block:: python

    study.best_value

Out:

.. code-block:: console

    2.916284393762304e-06

To get the best trial:

.. code-block:: python

    study.best_trial

Out:

.. code-block:: console

    FrozenTrial(number=67, value=2.916284393762304e-06, datetime_start=datetime.datetime(2020, 4, 8, 10, 42, 12, 595884), datetime_complete=datetime.datetime(2020, 4, 8, 10, 42, 12, 639969), params={'x': 2.001707713205946}, distributions={'x': UniformDistribution(high=10, low=-10)}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=67, state=TrialState.COMPLETE)

To get all trials:

.. code-block:: python

    study.trials

Out:

.. code-block:: console

    [FrozenTrial(number=0, value=25.77382032395108, datetime_start=datetime.datetime(2020, 4, 8, 10, 42, 8, 987277), datetime_complete=datetime.datetime(2020, 4, 8, 10, 42, 9, 27959), params={'x': 7.076792326257898}, distributions={'x': UniformDistribution(high=10, low=-10)}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=0, state=TrialState.COMPLETE),
     ...
     user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=99, state=TrialState.COMPLETE)]

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
