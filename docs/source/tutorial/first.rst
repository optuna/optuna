.. _firstopt:

First Optimization
==================


Quadratic Function Example
--------------------------

Let us try very simple optimization in IPython shell.

.. code-block:: python

    In [1]: import optuna

Here, we use a very simple quadratic function as an example of objective function.

.. code-block:: python

    In [2]: def objective(trial):
       ...:     x = trial.suggest_uniform('x', -10, 10)
       ...:     return (x - 2) ** 2
       ...:

Our goal is to find out ``x`` that minimizes the output of ``objective`` function, which we refer to as "optimization." During the optimization, Optuna repeatedly invokes and evaluates the objective function with different values of ``x``.

A :class:`~optuna.trial.Trial` object corresponds to a single execution of the objective function and is internally instantiated upon each invocation of the function.

The `suggest` APIs (e.g., :func:`~optuna.trial.Trial.suggest_uniform`) are called inside the objective function to obtain parameters for a trial.

To start the optimization, we create a study object and pass the objective function to method :func:`~optuna.study.Study.optimize` as follows.

.. code-block:: python

    In [3]: study = optuna.create_study()
    In [4]: study.optimize(objective, n_trials=100)
    [I 2018-05-09 10:03:22,469] Finished trial#0 resulted in value: 52.9345515866657. Current best value is 52.9345515866657 with parameters: {'x': -5.275613485244093}.
    [I 2018-05-09 10:03:22,474] Finished trial#1 resulted in value: 32.82718929591965. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,475] Finished trial#2 resulted in value: 46.89428737068025. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,476] Finished trial#3 resulted in value: 100.99613064563654. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,477] Finished trial#4 resulted in value: 110.56391159932272. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,478] Finished trial#5 resulted in value: 42.486606942847395. Current best value is 32.82718929591965 with parameters: {'x': -3.7295016620924066}.
    [I 2018-05-09 10:03:22,479] Finished trial#6 resulted in value: 1.130813338091735. Current best value is 1.130813338091735 with parameters: {'x': 3.063397074517198}.
    ...
    [I 2018-05-09 10:03:23,431] Finished trial#99 resulted in value: 8.760381111220335. Current best value is 0.0026232243068543526 with parameters: {'x': 1.9487825780924659}.
    In [5]: study.best_params
    Out[5]: {'x': 1.9487825780924659}

We can see that Optuna found the best ``x`` value ``1.9487825780924659``, which is close to the optimal value of ``2``.

.. note::
    In practice, it is expected that training of machine learning algorithms is invoked in objective functions, and metrics such as loss or error are reported.

Study Object
------------

Let us clarify the terminology in Optuna as follows.

* **Trial**: A single call of the objective function.
* **Study**: An optimization session, i.e., a set of trials.
* **Parameter**: A variable whose value is to be optimized, e.g., ``x`` in the above example.

In Optuna, we use study object to manage optimization. Method :func:`~optuna.study.create_study` returns a study object.
A study object has useful properties to analyze the optimization outcome.

.. code-block:: python

    In [5]: study.best_params
    Out[5]: {'x': 1.9926578647650126}

    In [6]: study.best_value
    Out[6]: 5.390694980884334e-05

    In [7]: study.best_trial
    Out[7]: FrozenTrial(number=26, state=<TrialState.COMPLETE: 1>, params={'x': 1.9926578647650126}, user_attrs={}, system_attrs={'_number': 26}, value=5.390694980884334e-05, intermediate_values={}, datetime_start=datetime.datetime(2018, 5, 9, 10, 23, 0, 87060), datetime_complete=datetime.datetime(2018, 5, 9, 10, 23, 0, 91010), params_in_internal_repr={'x': 1.9926578647650126}, trial_id=26)

    In [8]: study.trials  # all trials
    Out[8]:
    [FrozenTrial(number=0, state=<TrialState.COMPLETE: 1>, params={'x': -4.219801301030433}, user_attrs={}, system_attrs={'_number': 0}, value=38.685928224299865, intermediate_values={}, datetime_start=datetime.datetime(2018, 5, 9, 10, 22, 59, 983824), datetime_complete=datetime.datetime(2018, 5, 9, 10, 22, 59, 984053), params_in_internal_repr={'x': -4.219801301030433}, trial_id=0),
     ...
     user_attrs={}, system_attrs={'_number': 99}, value=8.2881000286123179, intermediate_values={}, datetime_start=datetime.datetime(2018, 5, 9, 10, 23, 0, 886434), datetime_complete=datetime.datetime(2018, 5, 9, 10, 23, 0, 891347), params_in_internal_repr={'x': 4.8789060472013182}, trial_id=99)]

    In [9]: len(study.trials)
    Out[9]: 100


By executing :func:`~optuna.study.Study.optimize` again, we can continue the optimization.

.. code-block:: python

    In [10]: study.optimize(objective, n_trials=100)
    ...

    In [11]: len(study.trials)
    Out[11]: 200
