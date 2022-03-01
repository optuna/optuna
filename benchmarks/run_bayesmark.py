import argparse
import json
import os
import subprocess
from typing import Dict
from typing import List

from matplotlib import cm
from matplotlib import colors
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from xarray import Dataset

import bayesmark.constants as cc
from bayesmark.serialize import XRSerializer


_DB = "bo_optuna_run"


def run_benchmark(args: argparse.Namespace) -> None:

    sampler_list = args.sampler_list.split()
    sampler_kwargs_list = args.sampler_kwargs_list.split()
    pruner_list = args.pruner_list.split()
    pruner_kwargs_list = args.pruner_kwargs_list.split()

    config = dict()
    for sampler, sampler_kwargs in zip(sampler_list, sampler_kwargs_list):
        for pruner, pruner_kwargs in zip(pruner_list, pruner_kwargs_list):
            optimizer_name = f"{sampler}-{pruner}-Optuna"
            optimizer_kwargs = {
                "sampler": sampler,
                "sampler_kwargs": json.loads(sampler_kwargs),
                "pruner": pruner,
                "pruner_kwargs": json.loads(pruner_kwargs),
            }
            # We need to dynamically generate config.json sice sampler pruner combos (solvers)
            # are parametrized by user. Following sample config schema.
            # https://github.com/uber/bayesmark/blob/master/example_opt_root/config.json
            config[optimizer_name] = ["optuna_optimizer.py", optimizer_kwargs]

    with open(os.path.join("benchmarks", "bayesmark", "config.json"), "w") as file:
        json.dump(config, file, indent=4)

    samplers = " ".join(config.keys())
    metric = "nll" if args.dataset in ["breast", "iris", "wine", "digits"] else "mse"
    cmd = (
        f"bayesmark-launch -n {args.budget} -r {args.repeat} "
        f"-dir runs -b {_DB} "
        f"-o RandomSearch {samplers} "
        f"-c {args.model} -d {args.dataset} "
        f"-m {metric} --opt-root benchmarks/bayesmark"
    )
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-agg -dir runs -b {_DB}"
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-anal -dir runs -b {_DB}"
    subprocess.run(cmd, shell=True)


def make_plots(args: argparse.Namespace) -> None:

    # https://github.com/uber/bayesmark/blob/master/notebooks/plot_test_case.ipynb
    db_root = os.path.abspath("runs")
    summary, _ = XRSerializer.load_derived(db_root, db=_DB, key=cc.PERF_RESULTS)

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    for benchmark in summary.coords["function"].values:
        for metric, ax in zip(["mean", "median"], axs):
            make_plot(summary, ax, benchmark, metric)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(benchmark)
    fig.savefig(os.path.join("plots", f"optuna-{args.dataset}-{args.model}-sumamry.png"))


def make_plot(summary: Dataset, ax: Axes, func: str, metric: str) -> None:

    color = build_color_dict(summary.coords["optimizer"].values.tolist())
    optimizers = summary.coords["optimizer"].values

    for optimizer in optimizers:
        curr_ds = summary.sel(
            {"function": func, "optimizer": optimizer, "objective": cc.VISIBLE_TO_OPT}
        )

        ax.fill_between(
            curr_ds.coords[cc.ITER].values,
            curr_ds[f"{metric} LB"].values,
            curr_ds[f"{metric} UB"].values,
            color=color[optimizer],
            alpha=0.5,
        )
        ax.plot(
            curr_ds.coords["iter"].values,
            curr_ds[metric].values,
            color=color[optimizer],
            label=optimizer,
        )

    ax.set_xlabel("Budget", fontsize=10)
    ax.set_ylabel(f"{metric.capitalize()} score", fontsize=10)
    ax.grid(alpha=0.2)
    ax.label_outer()


def build_color_dict(names: List[str]) -> Dict[str, np.ndarray]:

    norm = colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(norm, cm.tab20)
    color_dict = m.to_rgba(np.linspace(0, 1, len(names)))
    color_dict = dict(zip(names, color_dict))

    return color_dict


def partial_report(args: argparse.Namespace) -> None:

    db_root = os.path.abspath("runs")
    summary, _ = XRSerializer.load_derived(db_root, db=_DB, key=cc.MEAN_SCORE)

    # Following bayesmark way of constructing leaderboard.
    # https://github.com/uber/bayesmark/blob/8c420e935718f0d6867153b781e58943ecaf2338/bayesmark/experiment_analysis.py#L324-L328
    scores = summary["mean"].sel({"objective": cc.VISIBLE_TO_OPT}, drop=True)[{"iter": -1}]
    leaderboard = (100 * (1 - scores)).to_series().to_dict()
    sorted_lb = {k: v for k, v in sorted(leaderboard.items(), key=lambda item: item[1])}

    filename = f"{args.dataset}-{args.model}-partial-report.json"
    with open(os.path.join("partial", filename), "w") as file:
        json.dump(sorted_lb, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--model", type=str, default="kNN")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--sampler-list", type=str, default="TPESampler CmaEsSampler")
    parser.add_argument(
        "--sampler-kwargs-list",
        type=str,
        default='{"multivariate":true,"constant_liar":true} {}',
    )
    parser.add_argument("--pruner-list", type=str, default="NopPruner")
    parser.add_argument("--pruner-kwargs-list", type=str, default="{}")

    args = parser.parse_args()
    run_benchmark(args)
    make_plots(args)
    partial_report(args)
