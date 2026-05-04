"""
.. _ablation_study:

Ablation Study Becomes Easy with BruteForceSampler
===================================================

When conducting machine learning experiments, you often need to systematically evaluate all
combinations of techniques, hyperparameters, or components---a process known as an
``ablation study``.

A common approach is to write shell scripts with nested loops or use SLURM array jobs,
but these quickly become unwieldy when the search space has conditional structure
(e.g., "optimizer ``Adam`` has ``beta1`` and ``beta2``, but ``SGD`` has ``momentum``").
Frameworks like `Hydra <https://hydra.cc/>`_ support multirun sweeps with config overrides,
but handling conditional parameters requires additional boilerplate.

Optuna's :class:`~optuna.samplers.BruteForceSampler` solves this naturally: it exhaustively
enumerates all parameter combinations, including conditional (hierarchical) search spaces defined
via the define-by-run API. Combined with :class:`~optuna.storages.JournalStorage`
(or alternatively :class:`~optuna.storages.RDBStorage` with SQLite), it works seamlessly on HPC
clusters with shared filesystems, providing crash recovery and distributed execution out of the
box. Note that we focus only on :class:`~optuna.storages.JournalStorage` in this tutorial
for simplicity.

This tutorial walks through three scenarios:

- :ref:`basic-ablation-study`
- :ref:`conditional-search-space-ablation`
- :ref:`distributed-ablation-on-hpc`
"""

import optuna


###################################################################################################
# .. _basic-ablation-study:
#
# ---------------------
# Basic Ablation Study
# ---------------------
#
# Suppose you want to compare three optimizers and two learning rate schedules.
# Define an objective function using ``suggest_categorical`` and ``suggest_float``
# (with a finite ``step``), and let :class:`~optuna.samplers.BruteForceSampler` try every
# combination.


def objective(trial: optuna.Trial) -> float:
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    # If you would like to work on the log scale, you could also do like:
    # lr = 10**trial.suggest_float("lr_exponent", -6, -3, step=0.5)
    lr = trial.suggest_float("lr", 0.001, 0.01, step=0.001)
    lr_schedule = trial.suggest_categorical("lr_schedule", ["constant", "cosine"])

    # In a real experiment, you would train a model here and return the metric.
    # For demonstration, we use a mock score.
    mock_scores = {"Adam": 0.9, "SGD": 0.85, "RMSprop": 0.88}
    score = mock_scores[optimizer] + lr * 10 + (0.01 if lr_schedule == "cosine" else 0.0)
    return score


sampler = optuna.samplers.BruteForceSampler()
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective)

###################################################################################################
# The study automatically stops once every combination has been evaluated.
# You can inspect results with a DataFrame:

print(f"Total trials: {len(study.trials)}")
for trial in study.trials[:5]:
    print(f"  Trial {trial.number}: {trial.params} -> {trial.value}")

###################################################################################################
# .. note::
#    :class:`~optuna.samplers.BruteForceSampler` requires all continuous parameters to have
#    a finite ``step``. Using ``suggest_float`` without ``step`` will raise an error because
#    the search space would be infinite.

###################################################################################################
# .. _conditional-search-space-ablation:
#
# ----------------------------------------
# Conditional (Hierarchical) Search Space
# ----------------------------------------
#
# A key advantage of :class:`~optuna.samplers.BruteForceSampler` over simple grid search
# approaches is its support for conditional search spaces via Optuna's define-by-run API.
# For example, different optimizers may have different hyperparameters:


def objective_conditional(trial: optuna.Trial) -> float:
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    if optimizer == "Adam":
        beta1 = trial.suggest_categorical("beta1", [0.9, 0.95])
        beta2 = trial.suggest_categorical("beta2", [0.999, 0.9999])
        config = f"Adam(beta1={beta1}, beta2={beta2})"
    else:
        momentum = trial.suggest_categorical("momentum", [0.0, 0.9, 0.99])
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        config = f"SGD(momentum={momentum}, nesterov={nesterov})"

    lr = trial.suggest_float("lr", 0.001, 0.01, step=0.001)

    # Replace this with your actual training and evaluation code.
    mock_score = hash(config) % 100 / 100 + lr
    return mock_score


study_conditional = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.BruteForceSampler()
)
study_conditional.optimize(objective_conditional)

###################################################################################################
# The sampler explores all valid paths through the conditional search space:
# 4 combinations for Adam (2 beta1 x 2 beta2) and 6 for SGD (3 momentum x 2 nesterov),
# each combined with 10 learning rate values, totaling (4 + 6) x 10 = 100 trials.

print(f"Total trials: {len(study_conditional.trials)}")

###################################################################################################
# With shell scripts or array jobs, you would need to manually enumerate these conditional
# branches and compute the correct array indices. The define-by-run API makes this trivial.

###################################################################################################
# .. _distributed-ablation-on-hpc:
#
# -------------------------------------------
# Distributed Ablation Study on HPC Clusters
# -------------------------------------------
#
# On an HPC cluster with a shared filesystem (e.g., NFS or Lustre),
# :class:`~optuna.storages.JournalStorage` enables multiple worker processes to collaborate
# on the same ablation study without setting up a database server.
#
# The following snippet shows a self-contained optimization script. Save it as a Python file
# and launch it from multiple nodes or SLURM array jobs---each process will pick up
# unevaluated combinations automatically.
#
# .. code-block:: python
#
#     import optuna
#
#
#     def objective(trial: optuna.Trial) -> float:
#         optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
#         lr = trial.suggest_float("lr", 0.001, 0.01, step=0.001)
#         # Train your model and return the metric.
#         ...
#
#
#     # Use a file path on the shared filesystem.
#     storage = optuna.storages.JournalStorage(
#         optuna.storages.journal.JournalFileBackend("/shared/nfs/ablation_journal.log"),
#     )
#
#     study = optuna.create_study(
#         study_name="my-ablation",
#         storage=storage,
#         direction="maximize",
#         sampler=optuna.samplers.BruteForceSampler(),
#         load_if_exists=True,  # All workers join the same study.
#     )
#     study.optimize(objective)
#
# With a SLURM job array, the submission script would be:
#
# .. code-block:: bash
#
#     #!/bin/bash
#     #SBATCH --job-name=ablation
#     #SBATCH --array=0-7
#     #SBATCH --ntasks=1
#
#     python run_ablation.py
#
# Each array task runs the same script. :class:`~optuna.storages.JournalStorage` coordinates
# the work: each worker picks up unevaluated parameter combinations, and the study
# automatically stops once every combination has been evaluated.

###################################################################################################
# Since :class:`~optuna.storages.JournalStorage` replays its log file on startup, it handles
# crash recovery naturally. If a worker fails, simply relaunch it---it will skip already
# completed trials and resume from where it left off.
#
# .. tip::
#    Use ``load_if_exists=True`` in :func:`optuna.create_study` so that all workers join the
#    same study instead of raising an error when the study already exists.

###################################################################################################
# .. seealso::
#    - :class:`~optuna.samplers.BruteForceSampler` for API details.
#    - :ref:`journal_storage` for more on :class:`~optuna.storages.JournalStorage`.
