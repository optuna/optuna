.. _distributed:

Distributed Optimization
========================

There is no complicated setup but just sharing the same study name among nodes/processes.

First, create a shared study using ``optuna create-study`` command (or using :func:`optuna.create_study` in a Python script).

.. code-block:: bash

    $ optuna create-study --study-name "distributed-example" --storage "sqlite:///example.db"
    [I 2018-10-31 18:21:57,885] A new study created with name: distributed-example

Then, write an optimization script. Let's assume that ``foo.py`` contains the following code.

.. code-block:: python

    import optuna

    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2

    if __name__ == '__main__':
        study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
        study.optimize(objective, n_trials=100)

Finally, run the shared study from multiple processes.
For example, run ``Process 1`` in a terminal, and do ``Process 2`` in another one.
They get parameter suggestions based on shared trials' history.

Process 1:

.. code-block:: bash

    $ python foo.py
    [I 2018-10-31 18:46:44,308] Finished a trial resulted in value: 1.1097007755908204. Current best value is 0.00020881104123229936 with parameters: {'x': 2.014450295541348}.
    [I 2018-10-31 18:46:44,361] Finished a trial resulted in value: 0.5186699439824186. Current best value is 0.00020881104123229936 with parameters: {'x': 2.014450295541348}.
    ...

Process 2 (the same command as process 1):

.. code-block:: bash

    $ python foo.py
    [I 2018-10-31 18:47:02,912] Finished a trial resulted in value: 29.821448668796563. Current best value is 0.00020881104123229936 with parameters: {'x': 2.014450295541348}.
    [I 2018-10-31 18:47:02,968] Finished a trial resulted in value: 0.7962498978463782. Current best value is 0.00020881104123229936 with parameters: {'x': 2.014450295541348}.
    ...

.. note::
    We do not recommend SQLite for large scale distributed optimizations because it may cause serious performance issues. Please consider to use another database engine like PostgreSQL or MySQL.

.. note::
    Please avoid putting the SQLite database on NFS when running distributed optimizations. See also: https://www.sqlite.org/faq.html#q5
