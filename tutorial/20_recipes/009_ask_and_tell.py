"""
.. _ask_and_tell:

Ask-and-Tell Interface
=======================

Optuna has an `Ask-and-Tell` interface, which provides a more flexible interface for hyperparameter optimization.
This tutorial explains three use-cases when the ask-and-tell interface is beneficial:

- :ref:`Apply-optuna-to-an-existing-optimization-problem-with-minimum-modifications`
- :ref:`Define-and-Run`
- :ref:`Batch-Optimization`

.. _Apply-optuna-to-an-existing-optimization-problem-with-minimum-modifications:

----------------------------------------------------------------------------
Apply Optuna to an existing optimization problem with minimum modifications
----------------------------------------------------------------------------

Let's consider the traditional supervised classification problem; you aim to maximize the validation accuracy.
To do so, you train `LogisticRegression` as a simple model.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import optuna


X, y = make_classification(n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, y)

C = 0.01
clf = LogisticRegression(C=C)
clf.fit(X_train, y_train)
val_accuracy = clf.score(X_test, y_test)  # the objective

###################################################################################################
# Then you try to optimize hyperparameters ``C`` and ``solver`` of the classifier by using optuna.
# When you introduce optuna naively, you define an ``objective`` function
# such that it takes ``trial`` and calls ``suggest_*`` methods of ``trial`` to sample the hyperparameters:


def objective(trial):
    X, y = make_classification(n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    C = trial.suggest_loguniform("C", 1e-7, 10.0)
    solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))

    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)

    return val_accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

###################################################################################################
# This interface is not flexible enough.
# For example, if ``objective`` requires additional arguments other than ``trial``,
# you need to define a class as in
# `How to define objective functions that have own arguments? <../../faq.html#how-to-define-objective-functions-that-have-own-arguments>`_.
# The ask-and-tell interface provides a more flexible syntax to optimize hyperparameters.
# The following example is equivalent to the previous code block.

study = optuna.create_study(direction="maximize")

n_trials = 10
for _ in range(n_trials):
    trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.

    C = trial.suggest_loguniform("C", 1e-7, 10.0)
    solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))

    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)

    study.tell(trial, val_accuracy)  # tell the pair of trial and objective value

###################################################################################################
# The main difference is to use two methods: :func:`optuna.study.Study.ask`
# and :func:`optuna.study.Study.tell`.
# :func:`optuna.study.Study.ask` creates a trial that can sample hyperparameters, and
# :func:`optuna.study.Study.tell` finishes the trial by passing ``trial`` and an objective value.
# You can apply Optuna's hyperparameter optimization to your original code
# without an ``objective`` function.
#
# If you want to make your optimization faster with a pruner, you need to explicitly pass the state of trial
# to the argument of :func:`optuna.study.Study.tell` method as follows:
#
# .. code-block:: python
#
#    import numpy as np
#    from sklearn.datasets import load_iris
#    from sklearn.linear_model import SGDClassifier
#    from sklearn.model_selection import train_test_split
#
#    import optuna
#
#
#    X, y = load_iris(return_X_y=True)
#    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
#    classes = np.unique(y)
#    n_train_iter = 100
#
#    # define study with hyperband pruner.
#    study = optuna.create_study(
#        direction="maximize",
#        pruner=optuna.pruners.HyperbandPruner(
#            min_resource=1, max_resource=n_train_iter, reduction_factor=3
#        ),
#    )
#
#    for _ in range(20):
#        trial = study.ask()
#
#        alpha = trial.suggest_uniform("alpha", 0.0, 1.0)
#
#        clf = SGDClassifier(alpha=alpha)
#        pruned_trial = False
#
#        for step in range(n_train_iter):
#            clf.partial_fit(X_train, y_train, classes=classes)
#
#            intermediate_value = clf.score(X_valid, y_valid)
#            trial.report(intermediate_value, step)
#
#            if trial.should_prune():
#                pruned_trial = True
#                break
#
#        if pruned_trial:
#            study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # tell the pruned state
#        else:
#            score = clf.score(X_valid, y_valid)
#            study.tell(trial, score)  # tell objective value

###################################################################################################
# .. note::
#
#     :func:`optuna.study.Study.tell` method can take a trial number rather than the trial object.
#     ``study.tell(trial.number, y)`` is equivalent to ``study.tell(trial, y)``.


###################################################################################################
# .. _Define-and-Run:
#
# ---------------
# Define-and-Run
# ---------------
# The ask-and-tell interface supports both `define-by-run` and `define-and-run` APIs.
# This section shows the example of the `define-and-run` API
# in addition to the define-by-run example above.
#
# Define distributions for the hyperparameters before calling the
# :func:`optuna.study.Study.ask` method for define-and-run API.
# For example,

distributions = {
    "C": optuna.distributions.LogUniformDistribution(1e-7, 10.0),
    "solver": optuna.distributions.CategoricalDistribution(("lbfgs", "saga")),
}

###################################################################################################
# Pass ``distributions`` to :func:`optuna.study.Study.ask` method at each call.
# The retuned ``trial`` contains the suggested hyperparameters.

study = optuna.create_study(direction="maximize")
n_trials = 10
for _ in range(n_trials):
    trial = study.ask(distributions)  # pass the pre-defined distributions.

    # two hyperparameters are already sampled from the pre-defined distributions
    C = trial.params["C"]
    solver = trial.params["solver"]

    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)

    study.tell(trial, val_accuracy)


###################################################################################################
# .. _Batch-Optimization:
#
# -------------------
# Batch Optimization
# -------------------
# The ask-and-tell interface enables us to optimize a batched objective for faster optimization.
# For example, parallelizable evaluation, operation over vectors, etc.

###################################################################################################
# The following objective takes batched hyperparameters ``xs`` and ``ys`` instead of a single
# pair of hyperparameters ``x`` and ``y`` and calculates the objective over the full vectors.


def batched_objective(xs: np.ndarray, ys: np.ndarray):
    return xs ** 2 + ys


###################################################################################################
# In the following example, the number of pairs of hyperparameters in a batch is :math:`10`,
# and ``batched_objective`` is evaluated three times.
# Thus, the number of trials is :math:`30`.
# Note that you need to store either ``trial_ids`` or ``trial`` to call
# :func:`optuna.study.Study.tell` method after the batched evaluations.

batch_size = 10
study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())

for _ in range(3):

    # create batch
    trial_ids = []
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        trial = study.ask()
        trial_ids.append(trial.number)
        x_batch.append(trial.suggest_float("x", -10, 10))
        y_batch.append(trial.suggest_float("y", -10, 10))

    # evaluate batched objective
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    objectives = batched_objective(x_batch, y_batch)

    # finish all trials in the batch
    for trial_id, objective in zip(trial_ids, objectives):
        study.tell(trial_id, objective)
