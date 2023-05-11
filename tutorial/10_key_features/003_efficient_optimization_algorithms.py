"""
.. _pruning:

Efficient Optimization Algorithms
=================================

Optuna enables efficient hyperparameter optimization by
adopting state-of-the-art algorithms for sampling hyperparameters and
pruning efficiently unpromising trials.

Sampling Algorithms
-------------------

Samplers basically continually narrow down the search space using the records of suggested parameter values and evaluated objective values,
leading to an optimal search space which giving off parameters leading to better objective values.
More detailed explanation of how samplers suggest parameters is in :class:`~optuna.samplers.BaseSampler`.

Optuna provides the following sampling algorithms:

- Grid Search implemented in :class:`~optuna.samplers.GridSampler`

- Random Search implemented in :class:`~optuna.samplers.RandomSampler`

- Tree-structured Parzen Estimator algorithm implemented in :class:`~optuna.samplers.TPESampler`

- CMA-ES based algorithm implemented in :class:`~optuna.samplers.CmaEsSampler`

- Algorithm to enable partial fixed parameters implemented in :class:`~optuna.samplers.PartialFixedSampler`

- Nondominated Sorting Genetic Algorithm II implemented in :class:`~optuna.samplers.NSGAIISampler`

- A Quasi Monte Carlo sampling algorithm implemented in :class:`~optuna.samplers.QMCSampler`

The default sampler is :class:`~optuna.samplers.TPESampler`.

Switching Samplers
------------------

"""

import optuna


###################################################################################################
# By default, Optuna uses :class:`~optuna.samplers.TPESampler` as follows.

study = optuna.create_study()
print(f"Sampler is {study.sampler.__class__.__name__}")

###################################################################################################
# If you want to use different samplers for example :class:`~optuna.samplers.RandomSampler`
# and :class:`~optuna.samplers.CmaEsSampler`,

study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")

study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")


###################################################################################################
# Pruning Algorithms
# ------------------
#
# ``Pruners`` automatically stop unpromising trials at the early stages of the training (a.k.a., automated early-stopping).
#
# Optuna provides the following pruning algorithms:
#
# - Median pruning algorithm implemented in :class:`~optuna.pruners.MedianPruner`
#
# - Non-pruning algorithm implemented in :class:`~optuna.pruners.NopPruner`
#
# - Algorithm to operate pruner with tolerance implemented in :class:`~optuna.pruners.PatientPruner`
#
# - Algorithm to prune specified percentile of trials implemented in :class:`~optuna.pruners.PercentilePruner`
#
# - Asynchronous Successive Halving algorithm implemented in :class:`~optuna.pruners.SuccessiveHalvingPruner`
#
# - Hyperband algorithm implemented in :class:`~optuna.pruners.HyperbandPruner`
#
# - Threshold pruning algorithm implemented in :class:`~optuna.pruners.ThresholdPruner`
#
# We use :class:`~optuna.pruners.MedianPruner` in most examples,
# though basically it is outperformed by :class:`~optuna.pruners.SuccessiveHalvingPruner` and
# :class:`~optuna.pruners.HyperbandPruner` as in `this benchmark result <https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako>`_.
#
#
# Activating Pruners
# ------------------
# To turn on the pruning feature, you need to call :func:`~optuna.trial.Trial.report` and :func:`~optuna.trial.Trial.should_prune` after each step of the iterative training.
# :func:`~optuna.trial.Trial.report` periodically monitors the intermediate objective values.
# :func:`~optuna.trial.Trial.should_prune` decides termination of the trial that does not meet a predefined condition.
#
# We would recommend using integration modules for major machine learning frameworks.
# Exclusive list is :mod:`~optuna.integration` and usecases are available in `~optuna/examples <https://github.com/optuna/optuna-examples/>`_.


import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection


def objective(trial):
    iris = sklearn.datasets.load_iris()
    classes = list(set(iris.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=0
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
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


###################################################################################################
# Set up the median stopping rule as the pruning condition.

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

###################################################################################################
# As you can see, several trials were pruned (stopped) before they finished all of the iterations.
# The format of message is ``"Trial <Trial Number> pruned."``.

###################################################################################################
# Which Sampler and Pruner Should be Used?
# ----------------------------------------
#
# From the benchmark results which are available at `optuna/optuna - wiki "Benchmarks with Kurobako" <https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako>`_, at least for not deep learning tasks, we would say that
#
# * For :class:`~optuna.samplers.RandomSampler`, :class:`~optuna.pruners.MedianPruner` is the best.
# * For :class:`~optuna.samplers.TPESampler`, :class:`~optuna.pruners.HyperbandPruner` is the best.
#
# However, note that the benchmark is not deep learning.
# For deep learning tasks,
# consult the below table.
# This table is from the `Ozaki et al., Hyperparameter Optimization Methods: Overview and Characteristics, in IEICE Trans, Vol.J103-D No.9 pp.615-631, 2020 <https://doi.org/10.14923/transinfj.2019JDR0003>`_ paper,
# which is written in Japanese.
#
# +---------------------------+-----------------------------------------+---------------------------------------------------------------+
# | Parallel Compute Resource | Categorical/Conditional Hyperparameters | Recommended Algorithms                                        |
# +===========================+=========================================+===============================================================+
# | Limited                   | No                                      | TPE. GP-EI if search space is low-dimensional and continuous. |
# +                           +-----------------------------------------+---------------------------------------------------------------+
# |                           | Yes                                     | TPE. GP-EI if search space is low-dimensional and continuous  |
# +---------------------------+-----------------------------------------+---------------------------------------------------------------+
# | Sufficient                | No                                      | CMA-ES, Random Search                                         |
# +                           +-----------------------------------------+---------------------------------------------------------------+
# |                           | Yes                                     | Random Search or Genetic Algorithm                            |
# +---------------------------+-----------------------------------------+---------------------------------------------------------------+
#

###################################################################################################
# Integration Modules for Pruning
# -------------------------------
# To implement pruning mechanism in much simpler forms, Optuna provides integration modules for the following libraries.
#
# For the complete list of Optuna's integration modules, see :mod:`~optuna.integration`.
#
# For example, :class:`~optuna.integration.XGBoostPruningCallback` introduces pruning without directly changing the logic of training iteration.
# (See also `example <https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py>`_ for the entire script.)
#
# .. code-block:: python
#
#         pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-error')
#         bst = xgb.train(param, dtrain, evals=[(dvalid, 'validation')], callbacks=[pruning_callback])
