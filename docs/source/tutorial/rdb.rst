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
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))

The method :func:`~optuna.study.Study.trials_dataframe` returns a pandas dataframe like:

.. code-block:: python

    print(df)

Out:

.. code-block:: console

            number       value  params_x     state
         0       0   25.301959 -3.030105  COMPLETE
         1       1    1.406223  0.814157  COMPLETE
         2       2   44.010366 -4.634031  COMPLETE
         3       3   55.872181  9.474770  COMPLETE
         4       4  113.039223 -8.631991  COMPLETE
         5       5   57.319570  9.570969  COMPLETE

A :class:`~optuna.study.Study` object also provides properties such as :attr:`~optuna.study.Study.trials`, :attr:`~optuna.study.Study.best_value`, :attr:`~optuna.study.Study.best_params` (see also :ref:`firstopt`).

.. code-block:: python

    study.best_params  # Get best parameters for the objective function.
    study.best_value  # Get best objective value.
    study.best_trial  # Get best trial's information.
    study.trials  # Get all trials' information.
