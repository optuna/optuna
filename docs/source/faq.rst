FAQ
===

Can I use Optuna with X? (where X is your favorite ML library)
--------------------------------------------------------------

Optuna is compatible with most ML libraries, and it's easy to use Optuna with those.
Please refer to `examples <https://github.com/pfnet/optuna/tree/master/examples>`_.


How to define objective functions that have own arguments?
----------------------------------------------------------

There are two ways to realize it.

First, callable classes can be used for that purpose as follows:

.. code-block:: python

    import optuna

    class Objective(object):
        def __init__(self, min_x, max_x):
            # Hold this implementation specific arguments as the fields of the class.
            self.min_x = min_x
            self.max_x = max_x

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            x = trial.suggest_uniform('x', self.min_x, self.max_x)
            return (x - 2) ** 2

    # Execute an optimization by using an `Objective` instance.
    study = optuna.create_study()
    study.optimize(Objective(-100, 100), n_trials=100)


Second, you can use ``lambda`` or ``functools.partial`` for creating functions (closures) that hold extra arguments.
Below is an example that uses ``lambda``:

.. code-block:: python

    import optuna

    # Objective function that takes three arguments.
    def objective(trial, min_x, max_x):
        x = trial.suggest_uniform('x', min_x, max_x)
        return (x - 2) ** 2

    # Extra arguments.
    min_x = -100
    max_x = 100

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, min_x, max_x), n_trials=100)


Can I use Optuna without remote RDB servers?
--------------------------------------------

Yes, it's possible.

In the simplest form, Optuna works with in-memory storage:

.. code-block:: python

    study = optuna.create_study()
    study.optimize(objective)


If you want to save and resume studies,  it's handy to use SQLite as the local storage:

.. code-block:: python

    study = optuna.create_study(study_name='foo_study', storage='sqlite:///example.db')
    study.optimize(objective)  # The state of `study` will be persisted to the local SQLite file.

Please see :ref:`rdb` for more details.


How to suppress log messages of Optuna?
---------------------------------------

By default, Optuna shows log messages at the ``optuna.logging.INFO`` level.
You can change logging levels by using  :func:`optuna.logging.set_verbosity`.

For instance, you can stop showing each trial result as follows:

.. code-block:: python

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study()
    study.optimize(objective)
    # Logs like '[I 2018-12-05 11:41:42,324] Finished a trial resulted in value:...' are disabled.


Please refer to :class:`optuna.logging` for further details.


How to save machine learning models trained in objective functions?
-------------------------------------------------------------------

Optuna saves hyperparameter values with its corresponding objective value to storage,
but it discards intermediate objects such as machine learning models and neural network weights.
To save models or weights, please use features of the machine learning library you used.

We recommend saving :obj:`~optuna.trial.Trial.trail_id` with a model in order to identify its corresponding trial.
For example, you can save SVM models trained in the objective function as follows:

.. code-block:: python

    def objective(trial):
        svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
        clf = sklearn.svm.SVC(C=svc_c)
        clf.fit(X_train, y_train)

        # Save a trained model to a file.
        with open('{}.pickle'.format(trial.trial_id), 'wb') as fout:
            pickle.dump(clf, fout)
        return 1.0 - accuracy_score(y_test, clf.predict(X_test))


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    # Load the best model.
    with open('{}.pickle'.format(study.best_trial.trial_id), 'rb') as fin:
        best_clf = pickle.load(fin)
    print(accuracy_score(y_test, best_clf.predict(X_test)))


How can I obtain reproducible optimization results?
---------------------------------------------------

To make the parameters suggested by Optuna reproducible, you can specify a fixed random seed via ``seed`` argument of :class:`~optuna.samplers.RandomSampler` or :class:`~optuna.samplers.TPESampler` as follows:

.. code-block:: python

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective)

However, there are two caveats.

First, when optimizing a study in distributed or parallel mode, there is inherent non-determinism.
Thus it is very difficult to reproduce the same results in such condition.
We recommend executing optimization of a study sequentially if you would like to reproduce the result.

Second, if your objective function behaves in a non-deterministic way (i.e., it does not return the same value even if the same parameters were suggested), you cannot reproduce an optimization.
To deal with this problem, please set an option (e.g., random seed) to make the behavior deterministic if your optimization target (e.g., an ML library) provides it.


How does Optuna handle NaNs and exceptions reported by the objective function?
------------------------------------------------------------------------------

Optuna treats such trials as failures (i.e., :obj:`~optuna.structs.TrialState.FAIL`) and continues the study.
The Optuna's system process will not be crashed by any objective values or exceptions raised in objective functions.

You can find the failed trials in log messages.
Errors raised in objective functions are shown as follows:

.. code-block:: sh

    [W 2018-12-07 16:38:36,889] Setting trial status as TrialState.FAIL because of \
    the following error: ValueError('A sample error in objective.')

And trials which returned :obj:`NaN` are shown as follows:

.. code-block:: sh

    [W 2018-12-07 16:41:59,000] Setting trial status as TrialState.FAIL because the \
    objective function returned nan.

You can also find the failed trials by checking the trial states as follows:

.. code-block:: python

    study.trials_dataframe()

.. csv-table::

    trial_id,state,value,...,params,system_attrs
    0,TrialState.FAIL,,...,0,Setting trial status as TrialState.FAIL because of the following error: ValueError('A test error in objective.')
    1,TrialState.COMPLETE,1269,...,1,


How can I use two GPUs for evaluating two trials simultaneously?
----------------------------------------------------------------

If your optimization target supports GPU (CUDA) acceleration and you want to specify which GPU is used, the easiest way is to set ``CUDA_VISIBLE_DEVICES`` environment variable:

.. code-block:: bash

    # On a terminal.
    #
    # Specify to use the first GPU, and run an optimization.
    $ export CUDA_VISIBLE_DEVICES=0
    $ optuna study optimize foo.py objective --study foo --storage sqlite:///example.db

    # On another terminal.
    #
    # Specify to use the second GPU, and run another optimization.
    $ export CUDA_VISIBLE_DEVICES=1
    $ optuna study optimize bar.py objective --study bar --storage sqlite:///example.db

Please refer to `CUDA C Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`_ for further details.
