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

    study = optuna.Study(study_name='example-study', storage='sqlite:///example.db')
    study.optimize(objective, n_trials=3)

Experimental History
--------------------

We can access histories of studies and trials via the :class:`~optuna.study.Study` class.
For example, we can get all trials of ``example-study`` as:

.. code-block:: python

    import optuna
    study = optuna.Study(study_name='example-study', storage='sqlite:///example.db')
    df = study.trials_dataframe()

The method :func:`~optuna.study.Study.trials_dataframe` returns a pandas dataframe like:

.. code-block:: bash

    trial_id                state       value             datetime_start          datetime_complete    params
                                                                                                            x
           1  TrialState.COMPLETE   46.904095 2018-10-31 16:06:28.264950 2018-10-31 16:06:28.296937  8.848656
           2  TrialState.COMPLETE   25.416075 2018-10-31 16:06:28.310073 2018-10-31 16:06:28.333799 -3.041436
           3  TrialState.COMPLETE   50.302101 2018-10-31 16:06:28.344672 2018-10-31 16:06:28.364514  9.092397
           4  TrialState.COMPLETE   53.415845 2018-10-31 16:06:28.380938 2018-10-31 16:06:28.400815 -5.308614
           5  TrialState.COMPLETE   29.780800 2018-10-31 16:06:28.415496 2018-10-31 16:06:28.449833  7.457179
           6  TrialState.COMPLETE    6.950141 2018-10-31 16:06:28.466843 2018-10-31 16:06:28.484284  4.636312

A :class:`~optuna.study.Study` object also provides properties such as :attr:`~optuna.study.Study.trials`, :attr:`~optuna.study.Study.best_value`, :attr:`~optuna.study.Study.best_params` (see also :ref:`firstopt`).

.. code-block:: bash

    study.best_params  # Get best parameters for the objective function.
    study.best_value  # Get best objective value.
    study.best_trial  # Get best trial's information.
    study.trials  # Get all trials' information.
