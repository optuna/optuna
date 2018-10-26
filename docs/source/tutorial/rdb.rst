.. _rdb:

Using RDB Backend
=================

An RDB backend enables persistent experiments (i.e., to save and resume a study) as well as access to history of studies.
In addition, we can run multi-node optimization tasks with this feature, which is described in :ref:`distributed`.

In this section, let's try simple examples running on a local environment with sqlite DB.


Creating a Study
----------------

The first step is to run ``optuna create-study`` command specifying a DB's URL.

.. code-block:: bash

    $ optuna create-study --storage=sqlite:///example.db
    [I 2018-05-09 11:26:34,565] A new study created with name: no-name-cf809196-5431-4a2c-8dbb-954d656c69bd
    no-name-cf809196-5431-4a2c-8dbb-954d656c69bd

A name is populated to identify this study.
We can run optimization tasks and access the study's information with this name.

Note: see also ``optuna.create_study`` function, which creates a study via Python API.


Persistent Optimization
-----------------------

To run a study, create a Study object with additional arguments of the DB's URL and the study's name,
and then invoke ``study.optimize`` method.

.. code-block:: python

    study_name = ...  # Put a populated name by `optuna create-study`.
    study = Study(study_name, 'sqlite:///example.db')
    study.optimize(objective, n_trials=10)

To resume a study, all we need to do is just invoking the same method in the same way.

``optuna study optimize`` command is also available to start or resume a study, by passing storage and study arguments.
(Replace STUDY_NAME with a name populated by ``optuna create-study`` command.)

.. code-block:: bash

    $ optuna study optimize hoge.py objective --n-trials=100 --study=<STUDY_NAME> --storage='sqlite:///example.db'
    [I 2018-05-09 11:53:42,084] Finished a trial resulted in value: 1.5611912329002966. Current best value is 0.0006759211500324974 with parameters: {'x': 1.9740015163897489}.
    ...


Study History
-------------

``optuna studies`` command lists all study records stored in a DB.

.. code-block:: bash

    $ optuna studies --storage='sqlite:///example.db'
    +----------------------------------------------+-----------+----------+---------------------+
    | NAME                                         | DIRECTION | N_TRIALS | DATETIME_START      |
    +----------------------------------------------+-----------+----------+---------------------+
    | no-name-cf809196-5431-4a2c-8dbb-954d656c69bd | MINIMIZE  |      110 | 2018-05-09 11:53:22 |
    | no-name-3dd632e9-24a1-43a5-9c9d-bf125bb39f3c | NOT_SET   |        0 | None                |
    | no-name-7ccf1253-3c89-4811-8a8e-945dd5244fe6 | MINIMIZE  |       98 | 2018-05-09 11:56:28 |
    +----------------------------------------------+-----------+----------+---------------------+


Note: see also ``optuna.get_all_study_summaries`` function, which gets summary of studies via Python API.

To access details of a single study, instantiate ``Study`` object by passing the study's name and DB URL.
The object provides properties such as ``trials``, ``best_value``, ``best_params``, described in :ref:`firstopt`.

.. code-block:: python

    study_name = ...  # Put a populated name by `optuna create-study`.
    study = Study(study_name, 'sqlite:///example.db')

.. code-block:: bash

    study.best_params  # Get best parameters for the objective function.
    study.best_value  # Get best objective value.
    study.best_trial  # Get best trial's information.
    study.trials  # Get all trials' information.
