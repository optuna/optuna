"""
.. _distributed:

Easy Parallelization
====================

It's straightforward to parallelize :func:`optuna.study.Study.optimize`.

If you want to manually execute Optuna optimization:

    1. start an RDB server (this example uses MySQL)
    2. create a study with ``--storage`` argument
    3. share the study among multiple nodes and processes

Of course, you can use Kubernetes as in `the kubernetes examples <https://github.com/optuna/optuna-examples/tree/main/kubernetes>`_.

To just see how parallel optimization works in Optuna, check the below video.

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/J_aymk4YXhg?start=427" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


Create a Study
--------------

You can create a study using ``optuna create-study`` command.
Alternatively, in Python script you can use :func:`optuna.create_study`.


.. code-block:: bash

    $ mysql -u root -e "CREATE DATABASE IF NOT EXISTS example"
    $ optuna create-study --study-name "distributed-example" --storage "mysql://root@localhost/example"
    [I 2020-07-21 13:43:39,642] A new study created with name: distributed-example


Then, write an optimization script. Let's assume that ``foo.py`` contains the following code.

.. code-block:: python

    import optuna


    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2


    if __name__ == "__main__":
        study = optuna.load_study(
            study_name="distributed-example", storage="mysql://root@localhost/example"
        )
        study.optimize(objective, n_trials=100)


Share the Study among Multiple Nodes and Processes
--------------------------------------------------

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
    ``n_trials`` is the number of trials each process will run, not the total number of trials across all processes. For example, the script given above runs 100 trials for each process, 100 trials * 2 processes = 200 trials. :class:`optuna.study.MaxTrialsCallback` can ensure how many times trials will be performed across all processes.

.. note::
    We do not recommend SQLite for distributed optimizations at scale because it may cause deadlocks and serious performance issues. Please consider to use another database engine like PostgreSQL or MySQL.

.. note::
    Please avoid putting the SQLite database on NFS when running distributed optimizations. See also: https://www.sqlite.org/faq.html#q5

"""
