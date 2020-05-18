.. _pruning:

Pruning Unpromising Trials
==========================

This feature automatically stops unpromising trials at the early stages of the training (a.k.a., automated early-stopping).
Optuna provides interfaces to concisely implement the pruning mechanism in iterative training algorithms.


Activating Pruners
------------------
To turn on the pruning feature, you need to call :func:`~optuna.trial.Trial.report` and :func:`~optuna.trial.Trial.should_prune` after each step of the iterative training.
:func:`~optuna.trial.Trial.report` periodically monitors the intermediate objective values.
:func:`~optuna.trial.Trial.should_prune` decides termination of the trial that does not meet a predefined condition.

.. code-block:: python

    """filename: prune.py"""

    import sklearn.datasets
    import sklearn.linear_model
    import sklearn.model_selection

    import optuna

    def objective(trial):
        iris = sklearn.datasets.load_iris()
        classes = list(set(iris.target))
        train_x, valid_x, train_y, valid_y = \
            sklearn.model_selection.train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
        clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

        for step in range(100):
            clf.partial_fit(train_x, train_y, classes=classes)

            # Report intermediate objective value.
            intermediate_value = 1.0 - clf.score(valid_x, valid_y)
            trial.report(intermediate_value, step)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return 1.0 - clf.score(valid_x, valid_y)

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)


Executing the script above:

.. code-block:: bash

    $ python prune.py
    [I 2018-11-21 17:27:57,836] Finished trial#0 resulted in value: 0.052631578947368474. Current best value is 0.052631578947368474 with parameters: {'alpha': 0.011428158279113485}.
    [I 2018-11-21 17:27:57,963] Finished trial#1 resulted in value: 0.02631578947368418. Current best value is 0.02631578947368418 with parameters: {'alpha': 0.01862693201743629}.
    [I 2018-11-21 17:27:58,164] Finished trial#2 resulted in value: 0.21052631578947367. Current best value is 0.02631578947368418 with parameters: {'alpha': 0.01862693201743629}.
    [I 2018-11-21 17:27:58,333] Finished trial#3 resulted in value: 0.02631578947368418. Current best value is 0.02631578947368418 with parameters: {'alpha': 0.01862693201743629}.
    [I 2018-11-21 17:27:58,617] Finished trial#4 resulted in value: 0.23684210526315785. Current best value is 0.02631578947368418 with parameters: {'alpha': 0.01862693201743629}.
    [I 2018-11-21 17:27:58,642] Setting status of trial#5 as TrialState.PRUNED.
    [I 2018-11-21 17:27:58,666] Setting status of trial#6 as TrialState.PRUNED.
    [I 2018-11-21 17:27:58,675] Setting status of trial#7 as TrialState.PRUNED.
    [I 2018-11-21 17:27:59,183] Finished trial#8 resulted in value: 0.39473684210526316. Current best value is 0.02631578947368418 with parameters: {'alpha': 0.01862693201743629}.
    [I 2018-11-21 17:27:59,202] Setting status of trial#9 as TrialState.PRUNED.
    ...

We can see ``Setting status of trial#{} as TrialState.PRUNED`` in the log messages.
This means several trials are stopped before they finish all iterations.


Integration Modules for Pruning
-------------------------------
To implement pruning mechanism in much simpler forms, Optuna provides integration modules for the following libraries.

- XGBoost: :class:`optuna.integration.XGBoostPruningCallback`
- LightGBM: :class:`optuna.integration.LightGBMPruningCallback`
- Chainer: :class:`optuna.integration.ChainerPruningExtension`
- Keras: :class:`optuna.integration.KerasPruningCallback`
- TensorFlow :class:`optuna.integration.TensorFlowPruningHook`
- tf.keras :class:`optuna.integration.TFKerasPruningCallback`
- MXNet :class:`optuna.integration.MXNetPruningCallback`
- PyTorch Ignite :class:`optuna.integration.PyTorchIgnitePruningHandler`
- PyTorch Lightning :class:`optuna.integration.PyTorchLightningPruningCallback`
- FastAI :class:`optuna.integration.FastAIPruningCallback`

For example, :class:`~optuna.integration.XGBoostPruningCallback` introduces pruning without directly changing the logic of training iteration.
(See also `example <https://github.com/optuna/optuna/blob/master/examples/pruning/xgboost_integration.py>`_ for the entire script.)

.. code-block:: python

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-error')
        bst = xgb.train(param, dtrain, evals=[(dvalid, 'validation')], callbacks=[pruning_callback])
