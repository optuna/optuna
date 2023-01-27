import matplotlib.pyplot as plt
import pandas as pd


for n_obj in [5, 8, 10]:
    for i, m in zip([1, 2], [2, 6]):

        k, l = 2 * (n_obj - 1), 20
        n_var = k + l
        step_per_lap = 50
        data_dir = "results"

        df2 = pd.read_csv(f"{data_dir}/WFG{i}_M_{n_obj}_NSGA-II.csv", header=None)
        laps = len(df2.columns)
        iters = len(df2)
        num_trials_axis = [step_per_lap * (j + 1) for j in range(laps)]
        plt.plot(num_trials_axis, df2.agg("mean"), label="NSGA-II")

        df3 = pd.read_csv(f"{data_dir}/WFG{i}_M_{n_obj}_NSGA-III.csv", header=None)
        laps = len(df3.columns)
        iters = len(df3)
        num_trials_axis = [step_per_lap * (j + 1) for j in range(laps)]
        plt.plot(num_trials_axis, df3.agg("mean"), label="NSGA-III")

        title = f"WFG{m}: {n_obj} objective, {n_var} variable, {iters}-average"
        png_path = f"{data_dir}/WFG{m}_M_{n_obj}.png"
        # plt.plot(num_trials_axis, hv_iters_ii_ave, label="NSGA-II")
        # plt.plot(num_trials_axis, hv_iters_iii_ave, label="NSGA-III")
        plt.title(title)
        plt.xlabel("trials")
        plt.ylabel("hyper volume of pareto front")
        plt.legend()
        plt.savefig(png_path)
        plt.close()
