"""
.. _wilcoxon_pruner:

Early-stopping independent evaluations by WilcoxonPruner
============================================================

This tutorial showcases Optuna's WilcoxonPruner.
This pruner is effective for objective functions that averages multiple evaluations.

We solve Traveling Salesman Problem (TSP) by Simulated Annealing (SA).

Overview of Traveling Salesman Problem
--------------------------------------

Traveling Salesman Problem (TSP) is a classic problem in combinatorial optimization
that involves finding the shortest possible route for a salesman
who needs to visit a set of cities, each exactly once, and return to the starting city.
The problem is classified as NP-hard, indicating that it is extremely challenging
and that no efficient algorithm is known to solve all instances of the problem
within a reasonable amount of time.

TSP has been extensively studied in fields such as mathematics, computer science,
and operations research, and has numerous practical applications in logistics,
manufacturing, and DNA sequencing, among others.
Exact solutions can be obtained for small instances; however,
due to the computational complexity involved, approximation algorithms or
heuristic methods are commonly employed for larger instances.

Overview of Simulated Annealing
-------------------------------

Simulated Annealing (SA) is a probabilistic optimization algorithm used to find
the global optimum of a given function.
Inspired by the physical process of annealing in metallurgy,
where materials such as metal or glass are heated to a high temperature
and then cooled slowly to remove defects and reduce energy states,
the algorithm mimics this process to search for solutions in the problem space.

The algorithm starts with an initial solution and then moves to
a neighboring solution with a certain probability that depends on the
difference in the energy states (or costs) of the solutions and
a global parameter called "temperature". At high temperatures,
the algorithm is more likely to accept worse solutions,
allowing it to explore the solution space more freely and
avoid getting stuck in local optima.
As the temperature decreases according to a cooling schedule,
the algorithm becomes more conservative, accepting only solutions
that improve the objective function or those that do not significantly worsen it.

This method allows the SA algorithm to balance exploration and exploitation,
making it effective for solving complex optimization problems where
the solution space is large and potentially rugged with many local optima.
However, determining an effective cooling schedule is problem-dependent and
can be challenging. There is no one-size-fits-all approach, and
the optimal schedule may vary significantly between different types of problems.
Finding a good cooling schedule is an integral part of successfully applying SA.

Main Tutorial: Tuning SA Parameters for Solving TSP
====================================================
"""

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
#     The following `tsp_simulated_annealing` function can be acceralated by `numba`.
# .. note::
#     For simplicity of implementation, we use SA with the 2-opt neighborhood to solve TSP,
#     but note that this is far from the "best" way to solve TSP. There are significantly more
#     advanced methods than this method.


def tsp_simulated_annealing(vertices: np.ndarray, options: SAOptions) -> np.ndarray:

    def temperature(t: float):
        # t: 0 ... 1
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
# In practice, you should set a larger number of iterations.


N_SA_ITER = 10000
count = 0


###################################################################################################
# We counts the number of evaluation to know how many problems is pruned.


num_evaluation = 0


###################################################################################################
# In this tutorial, we optimize three parameters: `T0`, `alpha`, and `patience`.
#
# `T0` and `alpha` defining the temperature schedule
# ---------------------------------------------------------------------------------------
#
# In simulated annealing, it is important to determine a good temperature scheduling, but
# there is no "silver schedule" that is good for all problems, so we must tune the schedule
# for this problem.
# This code parametrizes the temperature as a monomial function `T0 * (1 - t) ** alpha`, where
# `t` progresses from 0 to 1. We try to optimize the two parameters `T0` and `alpha`.
#
# `patience`
# -----------------------------
#
# In this specific context, `patience` refers to the mechanism of reverting to the best solution
# found so far after a certain number of iterations without improvement. This concept is
# somewhat akin to a "reset" or "rollback" function, where if the algorithm hasn't found a
# better solution within the defined `patience` threshold (a set number of iterations),
# it will revert to the best solution it has encountered. This strategy can prevent the
# algorithm from wandering too far from promising regions of the solution space and can
# help in maintaining a focus on refining the best solutions found, rather than continuing
# to explore less promising paths.
#
# .. note::
#     As an advanced workaround, if `trial.should_prune()` returns `True`,
#     you can return an estimation of the final value (e.g., the average of all evaluated values)
#     instead of `raise optuna.TrialPruned()`.
#     Some algorithms including `TPESampler` internally split trials into below (good) and above (bad),
#     and pruned trial will always be classified as above.
#     However, there are some trials that are slightly worse than the best trial and will be pruned,
#     but they should be classified as below (e.g., top 10%).
#     This workaround provides beneficial information about such trials to these algorithms.


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
            # raise optuna.TrialPruned()

            # Return the current predicted value when pruned.
            # This is a workaround for the problem that
            # current TPE sampler cannot utilize pruned trials effectively.
            return sum(results) / len(results)

    return sum(results) / len(results)


###################################################################################################
# We use `TPESampler` with `WilcoxonPruner`.


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
# Visualize the solution found by `tsp_simulated_annealing` of the same TSP problem.


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
