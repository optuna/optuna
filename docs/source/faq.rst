FAQ
===

.. contents::
    :local:

Can I use Optuna with X? (where X is your favorite ML library)
--------------------------------------------------------------

Optuna is compatible with most ML libraries, and it's easy to use Optuna with those.
Please refer to `examples <https://github.com/optuna/optuna-examples/>`__.


.. _objective-func-additional-args:

How to define objective functions that have own arguments?
----------------------------------------------------------

There are two ways to realize it.

First, callable classes can be used for that purpose as follows:

.. code-block:: python

    import optuna


    class Objective:
        def __init__(self, min_x, max_x):
            # Hold this implementation specific arguments as the fields of the class.
            self.min_x = min_x
            self.max_x = max_x

        def __call__(self, trial):
            # Calculate an objective value by using the extra arguments.
            x = trial.suggest_float("x", self.min_x, self.max_x)
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
        x = trial.suggest_float("x", min_x, max_x)
        return (x - 2) ** 2


    # Extra arguments.
    min_x = -100
    max_x = 100

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, min_x, max_x), n_trials=100)

Please also refer to `sklearn_additional_args.py <https://github.com/optuna/optuna-examples/tree/main/sklearn/sklearn_additional_args.py>`__ example,
which reuses the dataset instead of loading it in each trial execution.


Can I use Optuna without remote RDB servers?
--------------------------------------------

Yes, it's possible.

In the simplest form, Optuna works with :class:`~optuna.storages.InMemoryStorage`:

.. code-block:: python

    study = optuna.create_study()
    study.optimize(objective)


If you want to save and resume studies,  it's handy to use SQLite as the local storage:

.. code-block:: python

    study = optuna.create_study(study_name="foo_study", storage="sqlite:///example.db")
    study.optimize(objective)  # The state of `study` will be persisted to the local SQLite file.

Please see :ref:`rdb` for more details.


How can I save and resume studies?
----------------------------------------------------

There are two ways of persisting studies, which depend if you are using
:class:`~optuna.storages.InMemoryStorage` (default) or remote databases (RDB). In-memory studies can be
saved and loaded like usual Python objects using ``pickle`` or ``joblib``. For
example, using ``joblib``:

.. code-block:: python

    study = optuna.create_study()
    joblib.dump(study, "study.pkl")

And to resume the study:

.. code-block:: python

    study = joblib.load("study.pkl")
    print("Best trial until now:")
    print(" Value: ", study.best_trial.value)
    print(" Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

Note that Optuna does not support saving/reloading across different Optuna
versions with ``pickle``. To save/reload a study across different Optuna versions,
please use RDBs and `upgrade storage schema <reference/cli.html#storage-upgrade>`__
if necessary. If you are using RDBs, see :ref:`rdb` for more details.

How to suppress log messages of Optuna?
---------------------------------------

By default, Optuna shows log messages at the ``optuna.logging.INFO`` level.
You can change logging levels by using  :func:`optuna.logging.set_verbosity`.

For instance, you can stop showing each trial result as follows:

.. code-block:: python

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study()
    study.optimize(objective)
    # Logs like '[I 2020-07-21 13:41:45,627] Trial 0 finished with value:...' are disabled.


Please refer to :class:`optuna.logging` for further details.


How to save machine learning models trained in objective functions?
-------------------------------------------------------------------

Optuna saves hyperparameter values with their corresponding objective values to storage,
but it discards intermediate objects such as machine learning models and neural network weights.

To save models or weights, we recommend utilizing Optuna's built-in ``ArtifactStore``.
For example, you can use the :func:`~optuna.artifacts.upload_artifact` as follows:

.. code-block:: python

    base_path = "./artifacts"
    os.makedirs(base_path, exist_ok=True)
    artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=base_path)

    def objective(trial):
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        clf = sklearn.svm.SVC(C=svc_c)
        clf.fit(X_train, y_train)

        # Save the model using ArtifactStore
        with open("model.pickle", "wb") as fout:
            pickle.dump(clf, fout)
        artifact_id = optuna.artifacts.upload_artifact(
            artifact_store=artifact_store,
            file_path="model.pickle",
            study_or_trial=trial.study,
        )
        trial.set_user_attr("artifact_id", artifact_id)
        return 1.0 - accuracy_score(y_valid, clf.predict(X_valid))

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

To retrieve models or weights, you can list and download them using :func:`~optuna.artifacts.get_all_artifact_meta` and :func:`~optuna.artifacts.download_artifact` as shown below:

.. code-block:: python

    # List all models
    for artifact_meta in optuna.artifacts.get_all_artifact_meta(study_or_trial=study):
        print(artifact_meta)
    # Download the best model
    trial = study.best_trial
    best_artifact_id = trial.user_attrs["artifact_id"]
    optuna.artifacts.download_artifact(
        artifact_store=artifact_store,
        file_path='best_model.pickle',
        artifact_id=best_artifact_id,
    )

For a more comprehensive guide, refer to the `ArtifactStore tutorial <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html>`_.

How can I obtain reproducible optimization results?
---------------------------------------------------

To make the parameters suggested by Optuna reproducible, you can specify a fixed random seed via ``seed`` argument of an instance of :mod:`~optuna.samplers` as follows:

.. code-block:: python

    sampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective)

To make the pruning by :class:`~optuna.pruners.HyperbandPruner` reproducible, please specify a fixed ``study_name`` of :class:`~optuna.study.Study` in addition to the ``seed`` argument.


However, there are two caveats.

First, when optimizing a study in distributed or parallel mode, there is inherent non-determinism.
Thus it is very difficult to reproduce the same results in such condition.
We recommend executing optimization of a study sequentially if you would like to reproduce the result.

Second, if your objective function behaves in a non-deterministic way (i.e., it does not return the same value even if the same parameters were suggested), you cannot reproduce an optimization.
To deal with this problem, please set an option (e.g., random seed) to make the behavior deterministic if your optimization target (e.g., an ML library) provides it.


How are exceptions from trials handled?
---------------------------------------

Trials that raise exceptions without catching them will be treated as failures, i.e. with the :obj:`~optuna.trial.TrialState.FAIL` status.

By default, all exceptions except :class:`~optuna.exceptions.TrialPruned` raised in objective functions are propagated to the caller of :func:`~optuna.study.Study.optimize`.
In other words, studies are aborted when such exceptions are raised.
It might be desirable to continue a study with the remaining trials.
To do so, you can specify in :func:`~optuna.study.Study.optimize` which exception types to catch using the ``catch`` argument.
Exceptions of these types are caught inside the study and will not propagate further.

You can find the failed trials in log messages.

.. code-block:: sh

    [W 2018-12-07 16:38:36,889] Setting status of trial#0 as TrialState.FAIL because of \
    the following error: ValueError('A sample error in objective.')

You can also find the failed trials by checking the trial states as follows:

.. code-block:: python

    study.trials_dataframe()

.. csv-table::

    number,state,value,...,params,system_attrs
    0,TrialState.FAIL,,...,0,Setting status of trial#0 as TrialState.FAIL because of the following error: ValueError('A test error in objective.')
    1,TrialState.COMPLETE,1269,...,1,

.. seealso::

    The ``catch`` argument in :func:`~optuna.study.Study.optimize`.


How are NaNs returned by trials handled?
----------------------------------------

Trials that return NaN (``float('nan')``) are treated as failures, but they will not abort studies.

Trials which return NaN are shown as follows:

.. code-block:: sh

    [W 2018-12-07 16:41:59,000] Setting status of trial#2 as TrialState.FAIL because the \
    objective function returned nan.


What happens when I dynamically alter a search space?
-----------------------------------------------------

Since parameters search spaces are specified in each call to the suggestion API, e.g.
:func:`~optuna.trial.Trial.suggest_float` and :func:`~optuna.trial.Trial.suggest_int`,
it is possible to, in a single study, alter the range by sampling parameters from different search
spaces in different trials.
The behavior when altered is defined by each sampler individually.

.. note::

    Discussion about the TPE sampler. https://github.com/optuna/optuna/issues/822


How can I use two GPUs for evaluating two trials simultaneously?
----------------------------------------------------------------

If your optimization target supports GPU (CUDA) acceleration and you want to specify which GPU is used in your script,
``main.py``, the easiest way is to set ``CUDA_VISIBLE_DEVICES`` environment variable:

.. code-block:: bash

    # On a terminal.
    #
    # Specify to use the first GPU, and run an optimization.
    $ export CUDA_VISIBLE_DEVICES=0
    $ python main.py

    # On another terminal.
    #
    # Specify to use the second GPU, and run another optimization.
    $ export CUDA_VISIBLE_DEVICES=1
    $ python main.py

Please refer to `CUDA C Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`__ for further details.


How can I test my objective functions?
--------------------------------------

When you test objective functions, you may prefer fixed parameter values to sampled ones.
In that case, you can use :class:`~optuna.trial.FixedTrial`, which suggests fixed parameter values based on a given dictionary of parameters.
For instance, you can input arbitrary values of :math:`x` and :math:`y` to the objective function :math:`x + y` as follows:

.. code-block:: python

    def objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        y = trial.suggest_int("y", -5, 5)
        return x + y


    objective(FixedTrial({"x": 1.0, "y": -1}))  # 0.0
    objective(FixedTrial({"x": -1.0, "y": -4}))  # -5.0


Using :class:`~optuna.trial.FixedTrial`, you can write unit tests as follows:

.. code-block:: python

    # A test function of pytest
    def test_objective():
        assert 1.0 == objective(FixedTrial({"x": 1.0, "y": 0}))
        assert -1.0 == objective(FixedTrial({"x": 0.0, "y": -1}))
        assert 0.0 == objective(FixedTrial({"x": -1.0, "y": 1}))


.. _out-of-memory-gc-collect:

How do I avoid running out of memory (OOM) when optimizing studies?
-------------------------------------------------------------------

If the memory footprint increases as you run more trials, try to periodically run the garbage collector.
Specify ``gc_after_trial`` to :obj:`True` when calling :func:`~optuna.study.Study.optimize` or call :func:`gc.collect` inside a callback.

.. code-block:: python

    def objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        y = trial.suggest_int("y", -5, 5)
        return x + y


    study = optuna.create_study()
    study.optimize(objective, n_trials=10, gc_after_trial=True)

    # `gc_after_trial=True` is more or less identical to the following.
    study.optimize(objective, n_trials=10, callbacks=[lambda study, trial: gc.collect()])

There is a performance trade-off for running the garbage collector, which could be non-negligible depending on how fast your objective function otherwise is. Therefore, ``gc_after_trial`` is :obj:`False` by default.
Note that the above examples are similar to running the garbage collector inside the objective function, except for the fact that :func:`gc.collect` is called even when errors, including :class:`~optuna.exceptions.TrialPruned` are raised.

.. note::

    :class:`~optuna.integration.ChainerMNStudy` does currently not provide ``gc_after_trial`` nor callbacks for :func:`~optuna.integration.ChainerMNStudy.optimize`.
    When using this class, you will have to call the garbage collector inside the objective function.

How can I output a log only when the best value is updated?
-----------------------------------------------------------

Here's how to replace the logging feature of optuna with your own logging callback function.
The implemented callback can be passed to :func:`~optuna.study.Study.optimize`.
Here's an example:

.. code-block:: python

    import optuna


    # Turn off optuna log notes.
    optuna.logging.set_verbosity(optuna.logging.WARN)


    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        return x ** 2


    def logging_callback(study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
                )
            )


    study = optuna.create_study()
    study.optimize(objective, n_trials=100, callbacks=[logging_callback])

Note that this callback may show incorrect values when you try to optimize an objective function with ``n_jobs!=1``
(or other forms of distributed optimization) due to its reads and writes to storage that are prone to race conditions.

How do I suggest variables which represent the proportion, that is, are in accordance with Dirichlet distribution?
------------------------------------------------------------------------------------------------------------------

When you want to suggest :math:`n` variables which represent the proportion, that is, :math:`p[0], p[1], ..., p[n-1]` which satisfy :math:`0 \le p[k] \le 1` for any :math:`k` and :math:`p[0] + p[1] + ... + p[n-1] = 1`, try the below.
For example, these variables can be used as weights when interpolating the loss functions.
These variables are in accordance with the flat `Dirichlet distribution <https://en.wikipedia.org/wiki/Dirichlet_distribution>`__.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import optuna


    def objective(trial):
        n = 5
        x = []
        for i in range(n):
            x.append(- np.log(trial.suggest_float(f"x_{i}", 0, 1)))

        p = []
        for i in range(n):
            p.append(x[i] / sum(x))

        for i in range(n):
            trial.set_user_attr(f"p_{i}", p[i])

        return 0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=1000)

    n = 5
    p = []
    for i in range(n):
        p.append([trial.user_attrs[f"p_{i}"] for trial in study.trials])
    axes = plt.subplots(n, n, figsize=(20, 20))[1]

    for i in range(n):
        for j in range(n):
            axes[j][i].scatter(p[i], p[j], marker=".")
            axes[j][i].set_xlim(0, 1)
            axes[j][i].set_ylim(0, 1)
            axes[j][i].set_xlabel(f"p_{i}")
            axes[j][i].set_ylabel(f"p_{j}")

    plt.savefig("sampled_ps.png")

This method is justified in the following way:
First, if we apply the transformation :math:`x = - \log (u)` to the variable :math:`u` sampled from the uniform distribution :math:`Uni(0, 1)` in the interval :math:`[0, 1]`, the variable :math:`x` will follow the exponential distribution :math:`Exp(1)` with scale parameter :math:`1`.
Furthermore, for :math:`n` variables :math:`x[0], ..., x[n-1]` that follow the exponential distribution of scale parameter :math:`1` independently, normalizing them with :math:`p[i] = x[i] / \sum_i x[i]`, the vector :math:`p` follows the Dirichlet distribution :math:`Dir(\alpha)` of scale parameter :math:`\alpha = (1, ..., 1)`.
You can verify the transformation by calculating the elements of the Jacobian.

How can I optimize a model with some constraints?
-------------------------------------------------

When you want to optimize a model with constraints, you can use the following classes: :class:`~optuna.samplers.TPESampler`, :class:`~optuna.samplers.NSGAIISampler` or `BoTorchSampler <https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.BoTorchSampler.html>`__.
The following example is a benchmark of Binh and Korn function, a multi-objective optimization, with constraints using :class:`~optuna.samplers.NSGAIISampler`. This one has two constraints :math:`c_0 = (x-5)^2 + y^2 - 25 \le 0` and :math:`c_1 = -(x - 8)^2 - (y + 3)^2 + 7.7 \le 0` and finds the optimal solution satisfying these constraints.


.. code-block:: python

    import optuna


    def objective(trial):
        # Binh and Korn function with constraints.
        x = trial.suggest_float("x", -15, 30)
        y = trial.suggest_float("y", -15, 30)

        # Constraints which are considered feasible if less than or equal to zero.
        # The feasible region is basically the intersection of a circle centered at (x=5, y=0)
        # and the complement to a circle centered at (x=8, y=-3).
        c0 = (x - 5) ** 2 + y ** 2 - 25
        c1 = -((x - 8) ** 2) - (y + 3) ** 2 + 7.7

        # Store the constraints as user attributes so that they can be restored after optimization.
        trial.set_user_attr("constraint", (c0, c1))

        v0 = 4 * x ** 2 + 4 * y ** 2
        v1 = (x - 5) ** 2 + (y - 5) ** 2

        return v0, v1


    def constraints(trial):
        return trial.user_attrs["constraint"]


    sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=32, timeout=600)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("  Trial#{}".format(trial.number))
        print(
            "    Values: Values={}, Constraint={}".format(
                trial.values, trial.user_attrs["constraint"][0]
            )
        )
        print("    Params: {}".format(trial.params))

If you are interested in an example for `BoTorchSampler <https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.BoTorchSampler.html>`__, please refer to `this sample code <https://github.com/optuna/optuna-examples/blob/main/multi_objective/botorch_simple.py>`__.


There are two kinds of constrained optimizations, one with soft constraints and the other with hard constraints.
Soft constraints do not have to be satisfied, but an objective function is penalized if they are unsatisfied. On the other hand, hard constraints must be satisfied.

Optuna is adopting the soft one and **DOES NOT** support the hard one. In other words, Optuna **DOES NOT** have built-in samplers for the hard constraints.

How can I parallelize optimization?
-----------------------------------

The variations of parallelization are in the following three cases.

1. Multi-threading parallelization with single node
2. Multi-processing parallelization with single node
3. Multi-processing parallelization with multiple nodes

1. Multi-threading parallelization with a single node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parallelization can be achieved by setting the argument ``n_jobs`` in :func:`optuna.study.Study.optimize`.
However, the python code will not be faster due to GIL because :func:`optuna.study.Study.optimize` with ``n_jobs!=1`` uses multi-threading.

While optimizing, it will be faster in limited situations, such as waiting for other server requests or C/C++ processing with numpy, etc., but it will not be faster in other cases.

For more information about 1., see APIReference_.

.. _APIReference: https://optuna.readthedocs.io/en/stable/reference/index.html

2. Multi-processing parallelization with single node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This can be achieved by using :class:`~optuna.storages.journal.JournalFileBackend` or client/server RDBs (such as PostgreSQL and MySQL).

For more information about 2., see TutorialEasyParallelization_.

.. _TutorialEasyParallelization: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html

3. Multi-processing parallelization with multiple nodes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This can be achieved by using client/server RDBs (such as PostgreSQL and MySQL).
However, if you are in the environment where you can not install a client/server RDB, you can not run multi-processing parallelization with multiple nodes.

For more information about 3., see TutorialEasyParallelization_.

.. _sqlite_concurrency:

How can I solve the error that occurs when performing parallel optimization with SQLite3?
-----------------------------------------------------------------------------------------

We would never recommend SQLite3 for parallel optimization in the following reasons.

- To concurrently evaluate trials enqueued by :func:`~optuna.study.Study.enqueue_trial`, :class:`~optuna.storages.RDBStorage` uses `SELECT ... FOR UPDATE` syntax, which is unsupported in `SQLite3 <https://github.com/sqlalchemy/sqlalchemy/blob/rel_1_4_41/lib/sqlalchemy/dialects/sqlite/base.py#L1265-L1267>`__.
- As described in `the SQLAlchemy's documentation <https://docs.sqlalchemy.org/en/14/dialects/sqlite.html#sqlite-concurrency>`__,
  SQLite3 (and pysqlite driver) does not support a high level of concurrency.
  You may get a "database is locked" error, which occurs when one thread or process has an exclusive lock on a database connection (in reality a file handle) and another thread times out waiting for the lock to be released.
  You can increase the default `timeout <https://docs.python.org/3/library/sqlite3.html#sqlite3.connect>`__ value like `optuna.storages.RDBStorage("sqlite:///example.db", engine_kwargs={"connect_args": {"timeout": 20.0}})` though.
- For distributed optimization via NFS, SQLite3 does not work as described at `FAQ section of sqlite.org <https://www.sqlite.org/faq.html#q5>`__.

If you want to use a file-based Optuna storage for these scenarios, please consider using :class:`~optuna.storages.journal.JournalFileBackend` instead.

.. code-block:: python

   import optuna
   from optuna.storages import JournalStorage
   from optuna.storages.journal import JournalFileBackend

   storage = JournalStorage(JournalFileBackend("optuna_journal_storage.log"))

   study = optuna.create_study(storage=storage)
   ...

See `the Medium blog post <https://medium.com/optuna/distributed-optimization-via-nfs-using-optunas-new-operation-based-logging-storage-9815f9c3f932>`__ for details.

.. _heartbeat_monitoring:

Can I monitor trials and make them failed automatically when they are killed unexpectedly?
------------------------------------------------------------------------------------------

.. note::

  Heartbeat mechanism is experimental. API would change in the future.

A process running a trial could be killed unexpectedly, typically by a job scheduler in a cluster environment.
If trials are killed unexpectedly, they will be left on the storage with their states `RUNNING` until we remove them or update their state manually.
For such a case, Optuna supports monitoring trials using `heartbeat <https://en.wikipedia.org/wiki/Heartbeat_(computing)>`__ mechanism.
Using heartbeat, if a process running a trial is killed unexpectedly,
Optuna will automatically change the state of the trial that was running on that process to :obj:`~optuna.trial.TrialState.FAIL`
from :obj:`~optuna.trial.TrialState.RUNNING`.

.. code-block:: python

    import optuna

    def objective(trial):
        (Very time-consuming computation)

    # Recording heartbeats every 60 seconds.
    # Other processes' trials where more than 120 seconds have passed
    # since the last heartbeat was recorded will be automatically failed.
    storage = optuna.storages.RDBStorage(url="sqlite:///:memory:", heartbeat_interval=60, grace_period=120)
    study = optuna.create_study(storage=storage)
    study.optimize(objective, n_trials=100)

.. note::

  The heartbeat is supposed to be used with :meth:`~optuna.study.Study.optimize`. If you use :meth:`~optuna.study.Study.ask` and
  :meth:`~optuna.study.Study.tell`, please change the state of the killed trials by calling :meth:`~optuna.study.Study.tell`
  explicitly.

You can also execute a callback function to process the failed trial.
Optuna provides a callback to retry failed trials as :class:`~optuna.storages.RetryFailedTrialCallback`.
Note that a callback is invoked at a beginning of each trial, which means :class:`~optuna.storages.RetryFailedTrialCallback`
will retry failed trials when a new trial starts to evaluate.

.. code-block:: python

    import optuna
    from optuna.storages import RetryFailedTrialCallback

    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )

    study = optuna.create_study(storage=storage)


How can I deal with permutation as a parameter?
-----------------------------------------------

Although it is not straightforward to deal with combinatorial search spaces like permutations with existing API, there exists a convenient technique for handling them.
It involves re-parametrization of permutation search space of :math:`n` items as an independent :math:`n`-dimensional integer search space.
This technique is based on the concept of `Lehmer code <https://en.wikipedia.org/wiki/Lehmer_code>`__.

A Lehmer code of a sequence is the sequence of integers in the same size, whose :math:`i`-th entry denotes how many inversions the :math:`i`-th entry of the permutation has after itself.
In other words, the :math:`i`-th entry of the Lehmer code represents the number of entries that are located after and are smaller than the :math:`i`-th entry of the original sequence.
For instance, the Lehmer code of the permutation :math:`(3, 1, 4, 2, 0)` is :math:`(3, 1, 2, 1, 0)`.

Not only does the Lehmer code provide a unique encoding of permutations into an integer space, but it also has some desirable properties.
For example, the sum of Lehmer code entries is equal to the minimum number of adjacent transpositions necessary to transform the corresponding permutation into the identity permutation.
Additionally, the lexicographical order of the encodings of two permutations is the same as that of the original sequence.
Therefore, Lehmer code preserves "closeness" among permutations in some sense, which is important for the optimization algorithm.
An Optuna implementation example to solve Euclid TSP is as follows:

.. code-block:: python

    import numpy as np

    import optuna


    def decode(lehmer_code: list[int]) -> list[int]:
        """Decode Lehmer code to permutation.

        This function decodes Lehmer code represented as a list of integers to a permutation.
        """
        all_indices = list(range(n))
        output = []
        for k in lehmer_code:
            value = all_indices[k]
            output.append(value)
            all_indices.remove(value)
        return output


    # Euclidean coordinates of cities for TSP.
    city_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]]
    )
    n = len(city_coordinates)


    def objective(trial: optuna.Trial) -> float:
        # Suggest a permutation in the Lehmer code representation.
        lehmer_code = [trial.suggest_int(f"x{i}", 0, n - i - 1) for i in range(n)]
        permutation = decode(lehmer_code)

        # Calculate the total distance of the suggested path.
        total_distance = 0.0
        for i in range(n):
            total_distance += np.linalg.norm(
                city_coordinates[permutation[i]] - city_coordinates[np.roll(permutation, 1)[i]]
            )
        return total_distance


    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    lehmer_code = study.best_params.values()
    print(decode(lehmer_code))

How can I ignore duplicated samples?
------------------------------------

Optuna may sometimes suggest parameters evaluated in the past and if you would like to avoid this problem, you can try out the following workaround:

.. code-block:: python

    import optuna
    from optuna.trial import TrialState


    def objective(trial):
        # Sample parameters.
        x = trial.suggest_int("x", -5, 5)
        y = trial.suggest_int("y", -5, 5)
        # Fetch all the trials to consider.
        # In this example, we use only completed trials, but users can specify other states
        # such as TrialState.PRUNED and TrialState.FAIL.
        states_to_consider = (TrialState.COMPLETE,)
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
        # Check whether we already evaluated the sampled `(x, y)`.
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                # Use the existing value as trial duplicated the parameters.
                return t.value

        # Compute the objective function if the parameters are not duplicated.
        # We use the 2D sphere function in this example.
        return x ** 2 + y ** 2


    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

.. _remove_for_artifact_store:

How can I delete all the artifacts uploaded to a study?
-------------------------------------------------------

Optuna supports :mod:`~optuna.artifacts` for large data storage during an optimization.
After you conduct enormous amount of experiments, you may want to remove the artifacts stored during optimizations.

We strongly recommend to create a new directory or bucket for each study so that all the artifacts linked to a study can be entirely removed by deleting the directory or the bucket.

However, if it is necessary to remove artifacts from a Python script, users can use the following code:

.. warning::

    :func:`~optuna.study.Study.add_trial` and :meth:`~optuna.study.copy_study` do not copy artifact files linked to :class:`~optuna.study.Study` or :class:`~optuna.trial.Trial`.
    Please make sure **NOT** to delete the artifacts from the source study or trial.
    Failing to do so may lead to unexpected behaviors as Optuna does not guarantee expected behaviors when users call :meth:`remove` externally.
    Due to the Optuna software design, it is hard to officially support the delete feature and we are not planning to support this feature in the future either.

.. code-block:: python

    from optuna.artifacts import get_all_artifact_meta


    def remove_artifacts(study, artifact_store):
        # NOTE: ``artifact_store.remove`` is discouraged to use because it is an internal feature.
        storage = study._storage
        for trial in study.trials:
            for artifact_meta in get_all_artifact_meta(trial, storage=storage):
                # For each trial, remove the artifacts uploaded to ``base_path``.
                artifact_store.remove(artifact_meta.artifact_id)

        for artifact_meta in get_all_artifact_meta(study):
            # Remove the artifacts uploaded to ``base_path``.
            artifact_store.remove(artifact_meta.artifact_id)

How can I resolve case sensitivity issues with MySQL?
-----------------------------------------------------

By default, MySQL performs case-insensitive string comparisons.
However, Optuna treats string parameters in a case-sensitive manner, leading to conflicts in MySQL if parameter names differ only by case.

For example,

.. code-block:: python

    def objective(trial):
        a = trial.suggest_int("a", 0, 10)
        A = trial.suggest_int("A", 0, 10)
        return a + A

In this case, Optuna treats `a` and `A` distinctively while MySQL does not due to its default collation settings.
As a result, only one of the parameters will be registered in MySQL.

The following workarounds should be considered:

1. Use a different storage backend.
    Please consider using PostgreSQL or SQLite, which supports case-sensitive handling.
2. Rename the parameters to avoid case conflicts.
    For example, use "a" and "b" instead of "a" and "A".
3. Change MySQL’s collation settings to be case-sensitive.
    You can configure case sensitivity at the database, table, or column level.
    For more details, refer to `the MySQL documentation <https://dev.mysql.com/doc/refman/9.3/en/charset-syntax.html>`__.
