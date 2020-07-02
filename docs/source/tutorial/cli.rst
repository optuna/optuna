.. _cli:

Command-Line Interface
======================

.. csv-table::
   :header: Command, Description
   :widths: 20, 40

    create-study, Create a new study.
    delete-study, Delete a specified study.
    dashboard, Launch web dashboard (beta).
    storage upgrade, Upgrade the schema of a storage.
    studies, Show a list of studies.
    study optimize, Start optimization of a study.
    study set-user-attr, Set a user attribute to a study.

Optuna provides command-line interface as shown in the above table.

Let us assume you are not in IPython shell and writing Python script files instead.
It is totally fine to write scripts like the following:

.. code-block:: python

    import optuna


    def objective(trial):
        x = trial.suggest_uniform('x', -10, 10)
        return (x - 2) ** 2


    if __name__ == '__main__':
        study = optuna.create_study()
        study.optimize(objective, n_trials=100)
        print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

