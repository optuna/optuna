"""
.. _distributed:

Distributed Optimization
========================

There is no complicated setup but just sharing the same study name among nodes/processes.

First, create a shared study using ``optuna create-study`` command (or using :func:`optuna.create_study` in a Python script).

.. code-block:: bash

    $ optuna create-study --study-name "distributed-example" --storage "sqlite:///example.db"
    [I 2020-07-21 13:43:39,642] A new study created with name: distributed-example


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
    [I 2020-07-21 13:45:02,973] Trial 0 finished with value: 45.35553104173011 and parameters: {'x': 8.73465151598285}. Best is trial 0 with value: 45.35553104173011.
    [I 2020-07-21 13:45:04,013] Trial 2 finished with value: 4.6002397305938905 and parameters: {'x': 4.144816945707463}. Best is trial 1 with value: 0.028194513284051464.
    ...

Process 2 (the same command as process 1):

.. code-block:: bash

    $ python foo.py
    [I 2020-07-21 13:45:03,748] Trial 1 finished with value: 0.028194513284051464 and parameters: {'x': 1.8320877810162361}. Best is trial 1 with value: 0.028194513284051464.
    [I 2020-07-21 13:45:05,783] Trial 3 finished with value: 24.45966755098074 and parameters: {'x': 6.945671597566982}. Best is trial 1 with value: 0.028194513284051464.
    ...

.. note::
    We do not recommend SQLite for large scale distributed optimizations because it may cause serious performance issues. Please consider to use another database engine like PostgreSQL or MySQL.

.. note::
    Please avoid putting the SQLite database on NFS when running distributed optimizations. See also: https://www.sqlite.org/faq.html#q5
"""
