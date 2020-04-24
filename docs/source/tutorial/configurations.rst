.. _configurations:

Advanced Configurations
=======================

Defining Parameter Spaces
-------------------------

Optuna supports five kinds of parameters.

.. code-block:: python

    def objective(trial):
        # Categorical parameter
        optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])

        # Int parameter
        num_layers = trial.suggest_int('num_layers', 1, 3)

        # Uniform parameter
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

        # Loguniform parameter
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        # Discrete-uniform parameter
        drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)

        ...

Branches and Loops
------------------

You can use branches or loops depending on the parameter values.

.. code-block:: python

    def objective(trial):
        classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForest'])
        if classifier_name == 'SVC':
            svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
            classifier_obj = sklearn.svm.SVC(C=svc_c)
        else:
            rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
            classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)

        ...

.. code-block:: python

    def create_model(trial):
        n_layers = trial.suggest_int('n_layers', 1, 3)

        layers = []
        for i in range(n_layers):
            n_units = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
            layers.append(L.Linear(None, n_units))
            layers.append(F.relu)
        layers.append(L.Linear(None, 10))

        return chainer.Sequential(*layers)

Please also refer to `examples <https://github.com/optuna/optuna/tree/master/examples>`_.


Note on the Number of Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The difficulty of optimization increases roughly exponentially with regard to the number of parameters. That is, the number of necessary trials increases exponentially when you increase the number of parameters, so it is recommended to not add unimportant parameters.


Arguments for `Study.optimize`
--------------------------------

The method :func:`~optuna.study.Study.optimize` (and ``optuna study optimize`` CLI command as well)
has several useful options such as ``timeout``.
For details, please refer to the API reference for :func:`~optuna.study.Study.optimize`.

**FYI**: If you give neither ``n_trials`` nor ``timeout`` options, the optimization continues until it receives a termination signal such as Ctrl+C or SIGTERM.
This is useful for use cases such as when it is hard to estimate the computational costs required to optimize your objective function.
