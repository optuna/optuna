"""
.. _optuna_callback:

Callback for Study.optimize
===========================

This tutorial showcases how to use & implement Optuna ``Callback`` for :func:`~optuna.study.Study.optimize`.
By a callback, we mean a callable that takes :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial` as arguments, and does some work.

Note that callbacks in this tutorial and ``*PruningCallback``'s of :mod:`optuna.integration` are completely different.
"""

from dataclasses import dataclass
import logging
import sys

import optuna

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

###################################################################################################
# Stop optimization upon condition is met.
# ----------------------------------------------------
#
# This example implements a stateful callback which stops the optimization if some trials are pruned in a row.


@dataclass
class StopWhenTrialKeepBeingPrunedCallback:

    threshold: int
    n_consequtive_pruned: int = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self.n_consequtive_pruned += 1
        else:
            self.n_consequtive_pruned = 0

        if self.n_consequtive_pruned >= self.threshold:
            study.stop()


def objective(trial):
    if trial.number > 4:
        raise optuna.TrialPruned

    return trial.suggest_float("x", 0, 1)


study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[StopWhenTrialKeepBeingPrunedCallback(2)])
