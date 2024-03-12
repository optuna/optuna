"""
.. _wilcoxon_pruner:

Early-stopping independent evaluations by Wilcoxon pruner
============================================================

This tutorial showcases Optuna's `WilcoxonPruner <https://optuna.readthedocs.io/en/latest/reference/generated/optuna.pruners.WilcoxonPruner.html>`_.
This pruner is effective for objective functions that averages multiple evaluations.

We solve `Traveling Salesman Problem (TSP) <https://en.wikipedia.org/w/index.php?title=Travelling_salesman_problem&oldid=1211575788>`_
by `Simulated Annealing (SA) <https://en.wikipedia.org/w/index.php?title=Simulated_annealing&oldid=1187355062>`_.

Overview: Solving Traveling Salesman Problem with Simulated Annealing
----------------------------------------------------------------------------

Traveling Salesman Problem (TSP) is a classic problem in combinatorial optimization
that involves finding the shortest possible route for a salesman
who needs to visit a set of cities, each exactly once, and return to the starting city.
TSP has been extensively studied in fields such as mathematics, computer science,
and operations research, and has numerous practical applications in logistics,
manufacturing, and DNA sequencing, among others.
The problem is classified as NP-hard, so approximation algorithms or
heuristic methods are commonly employed for larger instances.

One simple heuristic method applicable to TSP is simulated annealing (SA).
SA starts with an initial solution (it can be constructed by a simpler heuristic
like greedy method), and it randomly checks the neighborhood (defined later)
of the solution. If a neighbor is better, the solution is updated to the neighbor.
If the neighbor is worse, SA still updates the solution to the neighbor with
probability :math:`e^{-\Delta c / T}`, where
:math:`\Delta c (> 0)` is the difference of
the cost (sum of the distance) between the new solution and the old one and
:math:`T` is a parameter called "temperature". The temperature controls
how much worsening of the solution is tolerated to escape from the local minimum
(high means more tolerant). If the temperature is too low, SA will quickly
fall into a local minimum; if the temperature is too high, SA will be like
a random walk and the optimization will be inefficient. Typically, we set a
"temperature schedule" that starts from a high temperature and gradually
decreases to zero.

There are several ways to define neighborhood for TSP, but we use a
simple neighborhood called `2-opt <https://en.wikipedia.org/w/index.php?title=2-opt&oldid=1194969927>`_. 2-opt neighbor chooses a path in
the current solution and reverses the visiting order in the path.
For example, if the initial solution is `a→b→c→d→e→a`, `a→d→c→b→e→a` is
a 2-opt neighbor (the path from `b` to `d` is reversed).
This neighborhood is good because computing the difference of the cost
can be done in constant time (we only need to care about the start
and the end of the chosen path).

Main Tutorial: Tuning SA Parameters for TSP
====================================================

First, let's import some packages and define the parameters setting of SA
and the cost function of TSP.
"""  # NOQA

from dataclasses import dataclass
import math

import numpy as np
import optuna
import plotly.graph_objects as go
from numpy.linalg import norm


@dataclass
class SAOptions:
    max_iter: int = 10000
    T0: float = 1.0
    alpha: float = 2.0
    patience: int = 50


def tsp_cost(vertices: np.ndarray, idxs: np.ndarray) -> float:
    return norm(vertices[idxs] - vertices[np.roll(idxs, 1)], axis=-1).sum()


###################################################################################################
# Greedy solution for initial guess.


def tsp_greedy(vertices: np.ndarray) -> np.ndarray:
    idxs = [0]
    for _ in range(len(vertices) - 1):
        dists_from_last = norm(vertices[idxs[-1], None] - vertices, axis=-1)
        dists_from_last[idxs] = np.inf
        idxs.append(np.argmin(dists_from_last))
    return np.array(idxs)


###################################################################################################
# .. note::
#     For simplicity of implementation, we use SA with the 2-opt neighborhood to solve TSP,
#     but note that this is far from the "best" way to solve TSP. There are significantly more
#     advanced methods than this method.


###################################################################################################
# The implementation of SA with 2-opt neighborhood is following.


def tsp_simulated_annealing(vertices: np.ndarray, options: SAOptions) -> np.ndarray:

    def temperature(t: float):
        assert 0.0 <= t and t <= 1.0
        return options.T0 * (1 - t) ** options.alpha

    N = len(vertices)

    idxs = tsp_greedy(vertices)
    cost = tsp_cost(vertices, idxs)
    best_idxs = idxs.copy()
    best_cost = cost
    remaining_patience = options.patience

    for iter in range(options.max_iter):

        i = np.random.randint(0, N)
        j = (i + 2 + np.random.randint(0, N - 3)) % N
        i, j = min(i, j), max(i, j)
        # Reverse the order of vertices between range [i+1, j].

        # cost difference by 2-opt reversal
        delta_cost = (
            -norm(vertices[idxs[(i + 1) % N]] - vertices[idxs[i]])
            - norm(vertices[idxs[j]] - vertices[idxs[(j + 1) % N]])
            + norm(vertices[idxs[i]] - vertices[idxs[j]])
            + norm(vertices[idxs[(i + 1) % N]] - vertices[idxs[(j + 1) % N]])
        )
        temp = temperature(iter / options.max_iter)
        if delta_cost <= 0.0 or np.random.random() < math.exp(-delta_cost / temp):
            # accept the 2-opt reversal
            cost += delta_cost
            idxs[i + 1 : j + 1] = idxs[i + 1 : j + 1][::-1]
            if cost < best_cost:
                best_idxs[:] = idxs
                best_cost = cost
                remaining_patience = options.patience

        if cost > best_cost:
            # If the best solution is not updated for "patience" iteratoins,
            # restart from the best solution.
            remaining_patience -= 1
            if remaining_patience == 0:
                idxs[:] = best_idxs
                cost = best_cost
                remaining_patience = options.patience

    return best_idxs


###################################################################################################
# We make a random dataset of TSP.


def make_dataset(num_vertex: int, num_problem: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.random((num_problem, num_vertex, 2))


dataset = make_dataset(
    num_vertex=100,
    num_problem=50,
)

N_TRIALS = 50


###################################################################################################
# We set a very small number of SA iterations for demonstration purpose.
# In practice, you should set a larger number of iterations (e.g., 1000000).


N_SA_ITER = 10000
count = 0


###################################################################################################
# We counts the number of evaluation to know how many problems is pruned.


num_evaluation = 0


###################################################################################################
# In this tutorial, we optimize three parameters: ``T0``, ``alpha``, and ``patience``.
#
# ``T0`` and ``alpha`` defining the temperature schedule
# ---------------------------------------------------------------------------------------
#
# In simulated annealing, it is important to determine a good temperature scheduling, but
# there is no "silver schedule" that is good for all problems, so we must tune the schedule
# for this problem.
# This code parametrizes the temperature as a monomial function ``T0 * (1 - t) ** alpha``, where
# `t` progresses from 0 to 1. We try to optimize the two parameters ``T0`` and ``alpha``.
#
# ``patience``
# -----------------------------
#
# This parameter specifies a threshold of how many iterations we allow the annealing process
# continue without updating the best value. Practically, simulated annealing often drives
# the solution far away from the current best solution, and rolling back to the best solution
# periodically often improves optimization efficiency a lot. However, if the rollback happens
# too often, the optimization may get stuck in a local optimum, so we must tune the threshold
# to a sensible amount.
#
# .. note::
#     Some samplers, including the default ``TPESampler``, currently cannot utilize the
#     information of pruned trials effectively (especially when the last intermediate value
#     is not the best approximation to the final objective function).
#     As a workaround for this issue, you can return an estimation of the final value
#     (e.g., the average of all evaluated values) when ``trial.should_prune()`` returns ``True``,
#     instead of `raise optuna.TrialPruned()`.
#     This will improve the sampler performance.


###################################################################################################
# We define the objective function to be optimized as follows.
# We early stop the evaluation by using the pruner.


def objective(trial: optuna.Trial) -> float:
    global num_evaluation
    options = SAOptions(
        max_iter=N_SA_ITER,
        T0=trial.suggest_float("T0", 0.01, 10.0, log=True),
        alpha=trial.suggest_float("alpha", 1.0, 10.0, log=True),
        patience=trial.suggest_int("patience", 10, 1000, log=True),
    )
    results = []

    # For best results, shuffle the evaluation order in each trial.
    ordering = np.random.permutation(len(dataset))
    for i in ordering:
        num_evaluation += 1
        result_idxs = tsp_simulated_annealing(vertices=dataset[i], options=options)
        result_cost = tsp_cost(dataset[i], result_idxs)
        results.append(result_cost)

        trial.report(result_cost, i)
        if trial.should_prune():
            # raise optuna.TrialPruned()  # This is a standard logic of pruning in Optuna.

            # Return the current predicted value when pruned.
            # This is a workaround for the problem that
            # current TPE sampler cannot utilize pruned trials effectively.
            return sum(results) / len(results)

    return sum(results) / len(results)


###################################################################################################
# We use ``TPESampler`` with ``WilcoxonPruner``.


np.random.seed(0)
sampler = optuna.samplers.TPESampler(seed=1)
pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.enqueue_trial({"T0": 1.0, "alpha": 2.0, "patience": 50})  # default params
study.optimize(objective, n_trials=N_TRIALS)


###################################################################################################
# We can show the optimization results as:


print(f"The number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value} (params: {study.best_params})")
print(f"Number of evaluation: {num_evaluation} / {len(dataset) * N_TRIALS}")


###################################################################################################
# Visualize the optimization history.
# Note that this plot shows both completed and pruned trials in same ways.


optuna.visualization.plot_optimization_history(study)


###################################################################################################
# Visualize the number of evaluations in each trial.


x_values = [x for x in range(len(study.trials)) if x != study.best_trial.number]
y_values = [
    len(t.intermediate_values) for t in study.trials if t.number != study.best_trial.number
]
best_trial_y = [len(study.best_trial.intermediate_values)]
best_trial_x = [study.best_trial.number]
fig = go.Figure()
fig.add_trace(go.Bar(x=x_values, y=y_values, name="Evaluations"))
fig.add_trace(go.Bar(x=best_trial_x, y=best_trial_y, name="Best Trial", marker_color="red"))
fig.update_layout(
    title="Number of evaluations in each trial",
    xaxis_title="Trial number",
    yaxis_title="Number of evaluations before pruned",
)
fig


###################################################################################################
# Visualize the greedy solution (used by initial guess) of a TSP problem.


d = dataset[0]
result_idxs = tsp_greedy(d)
result_idxs = np.append(result_idxs, result_idxs[0])
fig = go.Figure()
fig.add_trace(go.Scatter(x=d[result_idxs, 0], y=d[result_idxs, 1], mode="lines+markers"))
fig.update_layout(
    title=f"greedy solution (initial guess),  cost: {tsp_cost(d, result_idxs):.3f}",
    xaxis=dict(scaleanchor="y", scaleratio=1),
)
fig


###################################################################################################
# Visualize the solution found by ``tsp_simulated_annealing`` of the same TSP problem.


params = study.best_params
options = SAOptions(
    max_iter=N_SA_ITER,
    patience=params["patience"],
    T0=params["T0"],
    alpha=params["alpha"],
)
result_idxs = tsp_simulated_annealing(d, options)
result_idxs = np.append(result_idxs, result_idxs[0])
fig = go.Figure()
fig.add_trace(go.Scatter(x=d[result_idxs, 0], y=d[result_idxs, 1], mode="lines+markers"))
fig.update_layout(
    title=f"n_iter: {options.max_iter}, cost: {tsp_cost(d, result_idxs):.3f}",
    xaxis=dict(scaleanchor="y", scaleratio=1),
)
fig
