"""
.. _wilcoxon_pruner:

Wilcoxon Pruner
===============

This tutorial showcases Optuna's wilcoxon pruner.
This pruner is effective for objective functions that averages multiple evaluations.

We solve Traveling Salesman Problem (TSP) by Simulated Annealing (SA).

Overview of Traveling Salesman Problem
-----------------------------

Traveling Salesman Problem (TSP) is a classic problem in combinatorial optimization
that involves finding the shortest possible route for a salesman
who needs to visit a set of cities, each exactly once, and return to the starting city.
TThe problem is classified as NP-hard, indicating that it is extremely challenging
and that no efficient algorithm is known to solve all instances of the problem
within a reasonable amount of time.

TSP has been extensively studied in fields such as mathematics, computer science,
and operations research, and has numerous practical applications in logistics,
manufacturing, and DNA sequencing, among others.
Exact solutions can be obtained for small instances; however,
due to the computational complexity involved, approximation algorithms or
heuristic methods are commonly employed for larger instances.

Overview of Simulated Annealing
-----------------------------

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

"""

import math
from typing import NamedTuple

import numpy as np
import optuna
from numpy.linalg import norm


class SAOptions(NamedTuple):
    max_iter: int = 1000
    T0: float = 1.0
    alpha: float = 1.0
    patience: int = 300


###################################################################################################
# .. note::
#     The following `simulated_annealing` function can be acceralated by `numba`.


def simulated_annealing(vertices, initial_idxs, options: SAOptions):

    def temperature(t: float):
        # t: 0 ... 1
        return options.T0 * (1 - t) ** options.alpha

    idxs = initial_idxs.copy()
    N = len(vertices)
    assert len(idxs) == N

    cost = sum([norm(vertices[idxs[i]] - vertices[idxs[(i + 1) % N]]) for i in range(N)])
    best_idxs = idxs.copy()
    best_cost = cost

    remaining_patience = options.patience
    np.random.seed(11111)

    for iter in range(options.max_iter):
        i = np.random.randint(0, N)
        j = (i + 2 + np.random.randint(0, N - 3)) % N
        i, j = min(i, j), max(i, j)
        delta_cost = (
            -norm(vertices[idxs[(i + 1) % N]] - vertices[idxs[i]])
            - norm(vertices[idxs[j]] - vertices[idxs[(j + 1) % N]])
            + norm(vertices[idxs[i]] - vertices[idxs[j]])
            + norm(vertices[idxs[(i + 1) % N]] - vertices[idxs[(j + 1) % N]])
        )
        temp = temperature(iter / options.max_iter)

        if delta_cost <= 0.0 or np.random.rand() < math.exp(-delta_cost / temp):
            cost += delta_cost
            idxs[i + 1 : j + 1] = idxs[i + 1 : j + 1][::-1]
            if cost < best_cost:
                best_idxs[:] = idxs
                best_cost = cost

        if cost >= best_cost:
            remaining_patience -= 1
            if remaining_patience == 0:
                idxs[:] = best_idxs
                cost = best_cost
                remaining_patience = options.patience

    return best_idxs


###################################################################################################
# We make a random dataset of TSP.


def make_dataset(num_vertex, num_problem, seed):
    rng = np.random.default_rng(seed=seed)
    dataset = []
    for _ in range(num_problem):
        dataset.append(
            {
                "vertices": rng.random((num_vertex, 2)),
                "idxs": rng.permutation(num_vertex),
            }
        )
    return dataset


NUM_PROBLEM = 20
dataset = make_dataset(200, NUM_PROBLEM, seed=33333)


###################################################################################################
# In each trial, it is recommended to shuffle the order in which data is processed.
# We make pseudo random number generator here.


rng = np.random.default_rng(seed=44444)


###################################################################################################
# We counts the number of evaluation to know how many problems is pruned.


num_evaluation = 0


###################################################################################################
# In this tutorial, we optimize three parameters: T0, alpha, and patience.
#
# T0 (Initial Temperature)
# -----------------------------
#
# In Simulated Annealing, the concept of "temperature" is an analogy to control the randomness
# of the search process. The initial temperature, denoted as T0, sets the starting level of
# this temperature. A higher initial temperature allows the algorithm to explore a wider range of
# solutions and to accept worse solutions with higher probability, facilitating escape from local
# optima. As the algorithm progresses, the temperature is gradually decreased, leading to a more
# refined search around promising areas,
# ultimately aiming for convergence towards an optimal solution.
#
# Alpha (Cooling Rate)
# -----------------------------
#
# Alpha is a parameter that dictates the rate at which the temperature is decreased in each
# iteration of the algorithm. It is typically a value between 0 and 1, and
# it's used to multiply the current temperature to obtain the temperature for the next iteration,
# following the formula T = alpha * T. A smaller alpha value results in a quicker decrease
# in temperature, making the algorithm converge faster but potentially missing broader exploration.
#
# Patience
# -----------------------------
# In this specific context, patience refers to the mechanism of reverting to the best solution
# found so far after a certain number of iterations without improvement. This concept is
# somewhat akin to a "reset" or "rollback" function, where if the algorithm hasn't found a
# better solution within the defined 'patience' threshold (a set number of iterations),
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


def objective(trial):
    global num_evaluation
    patience = trial.suggest_int("patience", 10, 1000, log=True)
    T0 = trial.suggest_float("T0", 0.1, 10.0, log=True)
    alpha = trial.suggest_float("alpha", 1.1, 10.0, log=True)
    options = SAOptions(max_iter=10000, patience=patience, T0=T0, alpha=alpha)
    ordering = rng.permutation(range(len(dataset)))
    results = []
    for i in ordering:
        num_evaluation += 1
        d = dataset[i]
        result_idxs = simulated_annealing(d["vertices"], d["idxs"], options)
        result_cost = 0.0
        n = len(d["vertices"])
        for j in range(n):
            result_cost += norm(
                d["vertices"][result_idxs[j]] - d["vertices"][result_idxs[(j + 1) % n]]
            )
        results.append(result_cost)

        trial.report(result_cost, i)
        if trial.should_prune():
            return sum(results) / len(results)  # It is the advanced workaround.
            # raise optuna.TrialPruned()

    return sum(results) / len(results)


###################################################################################################
# We use `TPESampler` with `WilcoxonPruner`.


NUM_TRIAL = 100
sampler = optuna.samplers.TPESampler(seed=55555)
pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.05)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.enqueue_trial({"patience": 300, "T0": 1.0, "alpha": 1.8})  # default params
study.optimize(objective, n_trials=NUM_TRIAL)


###################################################################################################
# We can show the optimization results as:


print(f"The number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value} (params: {study.best_params})")
print(f"Number of evaluation: {num_evaluation} / {NUM_PROBLEM * NUM_TRIAL}")


###################################################################################################
# Visualize the optimization history. See :func:`~optuna.visualization.plot_optimization_history` for the details.


optuna.visualization.plot_optimization_history(study)
