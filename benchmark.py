import csv
import os

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.problems.many import WFG1
from pymoo.problems.many import WFG2
from pymoo.problems.many import WFG3
from pymoo.problems.many import WFG4
from pymoo.problems.many import WFG5
from pymoo.problems.many import WFG6
from pymoo.problems.many import WFG7
from pymoo.problems.many import WFG8
from pymoo.problems.many import WFG9

import optuna
from optuna.trial import Trial


def optimize_and_evaluate(study, objective, ind, n_obj, laps, step_per_lap):
    HVs = []
    for _ in range(laps):
        study.optimize(objective, n_trials=step_per_lap)

        pareto_front = study.best_trials
        pareto_front_array = []
        nadir_point = np.array([-np.inf] * n_obj)
        for trial in pareto_front:
            pareto_front_array.append(trial.values)
        pareto_front_array = np.array(pareto_front_array)
        nadir_point = np.max(pareto_front_array, axis=0)
        pareto_front_array /= nadir_point
        HVs.append(ind(pareto_front_array))

    return np.array(HVs)


def run_nsga_ii(objective, ind, n_obj, laps, step_per_lap, iters):
    hv_iters_ii = []
    for j in range(iters):
        sampler_ii = optuna.samplers.NSGAIISampler()
        study = optuna.create_study(directions=["minimize"] * n_obj, sampler=sampler_ii)
        hv_laps = optimize_and_evaluate(study, objective, ind, n_obj, laps, step_per_lap)
        hv_iters_ii.append(hv_laps)
    return hv_iters_ii


def run_nsga_iii(objective, ind, n_obj, laps, step_per_lap, iters):
    hv_iters_iii = []
    reference_points = optuna.samplers.nsgaiii.generate_default_reference_point(n_obj, 5)
    for j in range(iters):
        sampler_iii = optuna.samplers.NSGAIIISampler(reference_points)
        study = optuna.create_study(directions=["minimize"] * n_obj, sampler=sampler_iii)
        hv_laps = optimize_and_evaluate(study, objective, ind, n_obj, laps, step_per_lap)
        hv_iters_iii.append(hv_laps)
    return hv_iters_iii


def save_csv(filepath, hv_iters):
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(hv_iters)


def main():
    os.makedirs("results", exist_ok=True)

    iters = 10
    laps = 40
    step_per_lap = 50
    problems = [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]
    dimensions = [3, 5, 8, 10]

    for n_obj in dimensions:
        k, l = 2 * (n_obj - 1), 20
        n_var = k + l

        reference_point_hv = [1.1] * n_obj
        ind = HV(ref_point=reference_point_hv)

        for i, prob in enumerate(problems):
            # Define objective function
            problem = prob(n_var, n_obj, k=k, l=l)

            def objective(trial):
                x = np.array([trial.suggest_float(f"x{i}", 0, 1) for i in range(n_var)])
                return list(problem.evaluate(x))

            csv_path_ii = f"results/WFG{i+1}_M_{n_obj}_NSGA-II.csv"
            if not os.path.exists(csv_path_ii):
                # Run NSGA-II
                hv_iters_ii = run_nsga_ii(objective, ind, n_obj, laps, step_per_lap, iters)
                # Save Result
                save_csv(csv_path_ii, hv_iters_ii)

            csv_path_iii = f"results/WFG{i+1}_M_{n_obj}_NSGA-III.csv"
            if not os.path.exists(csv_path_iii):
                # Run NSGA-III
                hv_iters_iii = run_nsga_iii(objective, ind, n_obj, laps, step_per_lap, iters)
                # Save Result
                save_csv(csv_path_iii, hv_iters_iii)

            # Draw graph
            # num_trials_axis = [step_per_lap * (j + 1) for j in range(laps)]
            # hv_iters_ii_ave = np.mean(np.loadtxt(csv_path_ii, delimiter=','), axis=0)
            # hv_iters_iii_ave = np.mean(np.loadtxt(csv_path_iii, delimiter=','), axis=0)

            # title = f"WFG{i+1}: {n_obj} objective, {n_var} variable, {iters}-average"
            # png_path = f"results/WFG{i+1}_M_{n_obj}.png"
            # plt.plot(num_trials_axis, hv_iters_ii_ave, label="NSGA-II")
            # plt.plot(num_trials_axis, hv_iters_iii_ave, label="NSGA-III")
            # plt.title(title)
            # plt.xlabel("trials")
            # plt.ylabel("hyper volume of pareto front")
            # plt.legend()
            # plt.savefig(png_path)
            # plt.show()


if __name__ == "__main__":
    main()
