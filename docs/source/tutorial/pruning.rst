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
                raise optuna.TrialPruned()

        return 1.0 - clf.score(valid_x, valid_y)

    # Set up the median stopping rule as the pruning condition.
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)


Executing the script above:

.. code-block:: bash

    $ python prune.py
    [I 2020-06-12 16:54:23,876] Trial 0 finished with value: 0.3157894736842105 and parameters: {'alpha': 0.00181467547181131}. Best is trial 0 with value: 0.3157894736842105.
    [I 2020-06-12 16:54:23,981] Trial 1 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.015378744419287613}. Best is trial 1 with value: 0.07894736842105265.
    [I 2020-06-12 16:54:24,083] Trial 2 finished with value: 0.21052631578947367 and parameters: {'alpha': 0.04089428832878595}. Best is trial 1 with value: 0.07894736842105265.
    [I 2020-06-12 16:54:24,185] Trial 3 finished with value: 0.052631578947368474 and parameters: {'alpha': 0.004018735937374473}. Best is trial 3 with value: 0.052631578947368474.
    [I 2020-06-12 16:54:24,303] Trial 4 finished with value: 0.07894736842105265 and parameters: {'alpha': 2.805688697062864e-05}. Best is trial 3 with value: 0.052631578947368474.
    [I 2020-06-12 16:54:24,315] Trial 5 pruned. 
    [I 2020-06-12 16:54:24,355] Trial 6 pruned. 
    [I 2020-06-12 16:54:24,511] Trial 7 finished with value: 0.052631578947368474 and parameters: {'alpha': 2.243775785299103e-05}. Best is trial 3 with value: 0.052631578947368474.
    [I 2020-06-12 16:54:24,625] Trial 8 finished with value: 0.1842105263157895 and parameters: {'alpha': 0.007021209286214553}. Best is trial 3 with value: 0.052631578947368474.
    [I 2020-06-12 16:54:24,629] Trial 9 pruned. 
    ...

``Trial 5 pruned.``, etc. in the log messages means several trials were stopped before they finished all of the iterations.


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
