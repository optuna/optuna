.. _configurations:

Advanced Configurations
=======================

Defining Parameter Spaces
-------------------------

Currently, we support five kinds of parameters.

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

You can use branches or loops depending on parameter values.

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

Please also refer to `examples <https://github.com/pfnet/optuna/tree/master/examples>`_.


Note on the Number of Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The difficulty of optimization increases roughly exponentially with regard to the number of parameters. That is, the number of necessary trials increases exponentially when you increase the number of parameters.
We recommend not to add unimportant parameters.


Arguments for `Study.optimize`
--------------------------------

Method :func:`~optuna.study.Study.optimize` (and ``optuna study optimize`` CLI command as well)
has several useful options such as ``timeout``.
Please refer to its docstring.

**FYI**: If you give neither ``n_trials`` nor ``timeout`` options, the optimization continues until it receives a termination signal such as Ctrl+C or SIGTERM.
This feature is useful for certain use cases, e.g., when it is hard to estimate computational costs required to optimize your objective function.
