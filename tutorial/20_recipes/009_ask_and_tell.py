"""
.. _ask_and_tell:

Ask-and-Tell Interface
=======================

Optuna provides `Ask-and-Tell` interface, a more flexible interface for hyper-parameter optimization.
This tutorial explains three use-cases when the ask-and-tell interface is beneficial:

- :ref:`without-objective`
- :ref:`define-and-run`
- :ref:`batch-optimization`


.. _without-objective:

----------------------------------------------------------------------------
Apply optuna to an existing optimization problem with minimum modifications
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
# Then you try to optimize hyper-parameters ``C`` and ``solver`` of the classifier by using optuna.
# When you introduce optuna naively, you define an ``objective`` function
# such that it takes ``trial`` and contains ``suggest_*`` functions to sample the hyper-parameters:

def objective(trial):
    X, y = make_classification(n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    C = trial.suggest_loguniform("C", 1e-7, 10.0)
    solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))

    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)

    return val_accuracy

study = optuna.create_study()
study.optimize(objective, n_trials=10)

###################################################################################################
# A drawback of this traditional interface is less flexibility.
# If ``objective`` requires an additional argument for ``objective``,
# you need to define your customized class for the objective as in
# `this FAQ section <../../faq.html#how-to-define-objective-functions-that-have-own-arguments>`_.
# That will make the codebase more complicated.
# The ask-and-tell interface provides a more flexible syntax to optimize hyper-parameters.
# The following example of ask-and-tell interface is equivalent to the previous code block.

study = optuna.create_study()

n_trials = 10
for _ in range(n_trials):
    trial = study.ask()  # `trial` is a `Trial` and not a `FrozenTrial`.

    C = trial.suggest_loguniform("C", 1e-7, 10.0)
    solver = trial.suggest_categorical("solver", ("lbfgs", "saga"))

    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)

    study.tell(trial, val_accuracy)   # tell the pair of trial and objective value

###################################################################################################
# The main difference is to use two methods: :func:`optuna.study.Study.ask`
# and :func:`optuna.study.Study.tell`.
# :func:`optuna.study.Study.ask` creates a trial that can sample hyper-parameters, and
# :func:`optuna.study.Study.tell` finishes the trial by passing ``trial`` and an objective value.
# You can apply optuna's hyper-parameter optimization to your original code easily
# without definition ``objective`` thanks to the ask-and-tell interface.
#
# If you make your optimization faster with a pruner, you need explicitly pass the state of trial
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
#    define study with hyperband pruner.
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
#    if pruned_trial:
#        study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # tell the pruned state
#    else:
#        score = clf.score(X_valid, y_valid)
#        study.tell(trial, score)  # tell objective value

###################################################################################################
# .. note::
#
#     :func:`optuna.study.Study.tell` method can take a trial number
#     rather than trial: ``study.tell(trial.number, y)`` is equivalent to ``study.tell(trial, y)``.


###################################################################################################
# .. _define-and-run:
#
# ---------------
# Define-and-Run
# ---------------
# The ask-and-tell interface supports both `define-by-run` and `define-and-run` APIs.
# This section shows the example of the `define-and-run` API
# since we already show the define-by-run example above.
#
# You need to define distributions for hyper-parameters before calling
# :func:`optuna.study.Study.ask` method for define-and-run API.
# For example,

distributions = {
    "C": optuna.distributions.LogUniformDistribution(1e-7, 10.0),
    "solver": optuna.distributions.CategoricalDistribution(("lbfgs", "saga")),
}

###################################################################################################
# You need to pass ``distributions`` to :func:`optuna.study.Study.ask` method at every call,
# and then, the retuned ``trial`` contains suggested hyper-parameters.

study = optuna.create_study()
n_trials = 10
for _ in range(n_trials):
    trial = study.ask(distributions)  # pass the pre-defined distributions.

    # two hyper-parameters are already sampled from the pre-defined distributions
    C = trial.params["C"]
    solver = trial.params["solver"]

    clf = LogisticRegression(C=C, solver=solver)
    clf.fit(X_train, y_train)
    val_accuracy = clf.score(X_test, y_test)

    study.tell(trial, val_accuracy)


###################################################################################################
# .. _batch-optimization:
#
# -------------------
# Batch Optimization
# -------------------
# The ask-and-tell interface enables us to optimize a batched objective.
# That can make a soft of optimization faster, especially, when the objective is evaluated by
# batch.
# For example, parallelizable evaluation, operation over vectors, etc.

###################################################################################################
# The following objective takes batched hyper-parameters ``xs`` instead of a single
# hyper-parameter and calculates the objective over the vector.

def batched_objective(xs: np.ndarray):
    return xs ** 2 + 1


###################################################################################################
# In the following example, the number of hyper-parameters in a batch is :math:`10`,
# and ``batched_objective`` is evaluated three times.
# Thus, the number of trials is :math:`30`.
# Note that you need to store either ``trial_ids`` or ``trial`` to call
# :func:`optuna.study.Study.tell` method after evaluations of the batched objective.

batch_size = 10
study = optuna.create_study()

for _ in range(3):

    # create batch
    trial_ids = []
    samples = []
    for _ in range(batch_size):
        trial = study.ask()
        trial_ids.append(trial.number)
        x = trial.suggest_int("x", -10, 10)
        samples.append(x)

    # evaluate batched objective
    samples = np.array(samples)
    objectives = batched_objective(samples)

    # finish all trials in the batch
    for trial_id, objective in zip(trial_ids, objectives):
        study.tell(trial_id, objective)
