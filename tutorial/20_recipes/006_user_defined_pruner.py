"""
.. _user_defined_pruner:

User-Defined Pruner
===================

In :mod:`optuna.pruners`, we described how an objective function can optionally include
calls to a pruning feature which allows Optuna to terminate an optimization
trial when intermediate results do not appear promising. In this document, we
describe how to implement your own pruner, i.e., a custom strategy for
determining when to stop a trial.

Overview of Pruning Interface
-----------------------------

The :func:`~optuna.study.create_study` constructor takes, as an optional
argument, a pruner inheriting from :class:`~optuna.pruners.BasePruner`. The
pruner should implement the abstract method
:func:`~optuna.pruners.BasePruner.prune`, which takes arguments for the
associated :class:`~optuna.study.Study` and :class:`~optuna.trial.Trial` and
returns a boolean value: :obj:`True` if the trial should be pruned and :obj:`False`
otherwise. Using the Study and Trial objects, you can access all other trials
through the :func:`~optuna.study.Study.get_trials` method and, and from a trial,
its reported intermediate values through the
:func:`~optuna.trial.FrozenTrial.intermediate_values` (a
dictionary which maps an integer ``step`` to a float value).

You can refer to the source code of the built-in Optuna pruners as templates for
building your own. In this document, for illustration, we describe the
construction and usage of a simple (but aggressive) pruner which prunes trials
that are in last place compared to completed trials at the same step.

.. note::
    Please refer to the documentation of :class:`~optuna.pruners.BasePruner` or,
    for example, :class:`~optuna.pruners.ThresholdPruner` or
    :class:`~optuna.pruners.PercentilePruner` for more robust examples of pruner
    implementation, including error checking and complex pruner-internal logic.

An Example: Implementing ``LastPlacePruner``
--------------------------------------------

We aim to optimize the ``loss`` and ``alpha`` hyperparameters for a stochastic
gradient descent classifier (``SGDClassifier``) run on the sklearn iris dataset. We
implement a pruner which terminates a trial at a certain step if it is in last
place compared to completed trials at the same step. We begin considering
pruning after a "warmup" of 1 training step and 5 completed trials. For
demonstration purposes, we :func:`print` a diagnostic message from ``prune`` when
it is about to return :obj:`True` (indicating pruning).

It may be important to note that the ``SGDClassifier`` score, as it is evaluated on
a holdout set, decreases with enough training steps due to overfitting. This
means that a trial could be pruned even if it had a favorable (high) value on a
previous training set. After pruning, Optuna will take the intermediate value
last reported as the value of the trial.

"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState


class LastPlacePruner(BasePruner):
    def __init__(self, warmup_steps, warmup_trials):
        self._warmup_steps = warmup_steps
        self._warmup_trials = warmup_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            this_score = trial.intermediate_values[step]

            # Get scores from other trials in the study reported at the same step
            completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            other_scores = [
                t.intermediate_values[step]
                for t in completed_trials
                if step in t.intermediate_values
            ]
            other_scores = sorted(other_scores)

            # Prune if this trial at this step has a lower value than all completed trials
            # at the same step. Note that steps will begin numbering at 0 in the objective
            # function definition below.
            if step >= self._warmup_steps and len(other_scores) > self._warmup_trials:
                if this_score < other_scores[0]:
                    print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                    return True

        return False


###################################################################################################
# Lastly, let's confirm the implementation is correct with the simple hyperparameter optimization.


def objective(trial):
    iris = load_iris()
    classes = np.unique(iris.target)
    X_train, X_valid, y_train, y_valid = train_test_split(
        iris.data, iris.target, train_size=100, test_size=50, random_state=0
    )

    loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "perceptron"])
    alpha = trial.suggest_float("alpha", 0.00001, 0.001, log=True)
    clf = SGDClassifier(loss=loss, alpha=alpha, random_state=0)
    score = 0

    for step in range(0, 5):
        clf.partial_fit(X_train, y_train, classes=classes)
        score = clf.score(X_valid, y_valid)

        trial.report(score, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return score


pruner = LastPlacePruner(warmup_steps=1, warmup_trials=5)
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=50)
