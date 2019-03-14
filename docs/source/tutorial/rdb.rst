.. _rdb:

Saving/Resuming Study with RDB Backend
==========================================

An RDB backend enables persistent experiments (i.e., to save and resume a study) as well as access to history of studies.
In addition, we can run multi-node optimization tasks with this feature, which is described in :ref:`distributed`.

In this section, let's try simple examples running on a local environment with SQLite DB.

.. note::
    You can also utilize other RDB backends, e.g., PostgreSQL or MySQL, by setting the storage argument to the DB's URL.
    Please refer to `SQLAlchemy's document <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ for how to set up the URL.


New Study
---------

We can create a persistent study by calling :func:`~optuna.study.create_study` function as follows.
An SQLite file ``example.db`` is automatically initialized with a new study record.

.. code-block:: python

    import optuna
    study_name = 'example-study'  # Unique identifier of the study.
    study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')

To run a study, call :func:`~optuna.study.Study.optimize` method passing an objective function.

.. code-block:: python

    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    study.optimize(objective, n_trials=3)

Resume Study
------------

To resume a study, instantiate a :class:`~optuna.study.Study` object passing the study name ``example-study`` and the DB URL ``sqlite:///example.db``.

.. code-block:: python

    study = optuna.create_study(study_name='example-study', storage='sqlite:///example.db', load_if_exists=True)
    study.optimize(objective, n_trials=3)

Experimental History
--------------------

We can access histories of studies and trials via the :class:`~optuna.study.Study` class.
For example, we can get all trials of ``example-study`` as:

.. code-block:: python

    import optuna
    study = optuna.create_study(study_name='example-study', storage='sqlite:///example.db', load_if_exists=True)
    df = study.trials_dataframe()

The method :func:`~optuna.study.Study.trials_dataframe` returns a pandas dataframe like:

.. code-block:: bash

    number                state       value             datetime_start          datetime_complete    params system_attrs
                                                                                                          x      _number
         0  TrialState.COMPLETE   25.301959 2019-03-14 10:57:27.716141 2019-03-14 10:57:27.746354 -3.030105            0
         1  TrialState.COMPLETE    1.406223 2019-03-14 10:57:27.774461 2019-03-14 10:57:27.835520  0.814157            1
         2  TrialState.COMPLETE   44.010366 2019-03-14 10:57:27.871365 2019-03-14 10:57:27.926247 -4.634031            2
         3  TrialState.COMPLETE   55.872181 2019-03-14 10:59:00.845565 2019-03-14 10:59:00.899305  9.474770            3
         4  TrialState.COMPLETE  113.039223 2019-03-14 10:59:00.921534 2019-03-14 10:59:00.947233 -8.631991            4
         5  TrialState.COMPLETE   57.319570 2019-03-14 10:59:00.985909 2019-03-14 10:59:01.028819  9.570969            5

A :class:`~optuna.study.Study` object also provides properties such as :attr:`~optuna.study.Study.trials`, :attr:`~optuna.study.Study.best_value`, :attr:`~optuna.study.Study.best_params` (see also :ref:`firstopt`).

.. code-block:: bash

    study.best_params  # Get best parameters for the objective function.
    study.best_value  # Get best objective value.
    study.best_trial  # Get best trial's information.
    study.trials  # Get all trials' information.
