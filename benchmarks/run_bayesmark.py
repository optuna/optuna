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
import pandas as pd
from xarray import Dataset


_DB = "bo_optuna_run"


def run_benchmark(args: argparse.Namespace) -> None:
    sampler_list = args.sampler_list.split()
    sampler_kwargs_list = args.sampler_kwargs_list.split()
    pruner_list = args.pruner_list.split()
    pruner_kwargs_list = args.pruner_kwargs_list.split()

    if len(sampler_list) != len(sampler_kwargs_list):
        raise ValueError(
            "The number of samplers does not match the given keyword arguments. \n"
            f"sampler_list: {sampler_list}, sampler_kwargs_list: {sampler_kwargs_list}."
        )

    if len(pruner_list) != len(pruner_kwargs_list):
        raise ValueError(
            "The number of pruners does not match the given keyword arguments. \n"
            f"pruner_list: {pruner_list}, pruner_keyword_arguments: {pruner_kwargs_list}."
        )

    config = dict()
    for i, (sampler, sampler_kwargs) in enumerate(zip(sampler_list, sampler_kwargs_list)):
        sampler_name = sampler
        if sampler_list.count(sampler) > 1:
            sampler_name += f"_{sampler_list[:i].count(sampler)}"
        for j, (pruner, pruner_kwargs) in enumerate(zip(pruner_list, pruner_kwargs_list)):
            pruner_name = pruner
            if pruner_list.count(pruner) > 1:
                pruner_name += f"_{pruner_list[:j].count(pruner)}"
            optimizer_name = f"{args.name_prefix}_{sampler_name}_{pruner_name}"
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
        f"bayesmark-launch -n {args.budget} -r {args.n_runs} "
        f"-dir runs -b {_DB} "
        f"-o {samplers} "
        f"-c {args.model} -d {args.dataset} "
        f"-m {metric} --opt-root benchmarks/bayesmark"
    )
    subprocess.run(cmd, shell=True)


def make_plots(args: argparse.Namespace) -> None:
    filename = f"{args.dataset}-{args.model}-partial-report.json"
    df = pd.read_json(os.path.join("partial", filename))
    df["best_value"] = df.groupby(["opt", "uuid"]).generalization.cummin()
    summaries = (
        df.groupby(["opt", "iter"])
        .best_value.agg(["mean", "std"])
        .rename(columns={"mean": "best_mean", "std": "best_std"})
        .reset_index()
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    warmup = json.loads(args.plot_warmup)
    metric = df.metric[0]
    color_lookup = build_color_dict(sorted(df["opt"].unique()))

    for optimizer, summary in summaries.groupby("opt"):
        color = color_lookup[optimizer]
        make_plot(summary, ax, optimizer, metric, warmup, color)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(f"Bayesmark-{args.dataset.capitalize()}-{args.model}")
    fig.savefig(os.path.join("plots", f"optuna-{args.dataset}-{args.model}-sumamry.png"))


def make_plot(
    summary: pd.DataFrame,
    ax: Axes,
    optimizer: str,
    metric: str,
    plot_warmup: bool,
    color: np.ndarray,
) -> None:
    start = 0 if plot_warmup else 10
    if len(summary.best_mean) <= start:
        return

    ax.fill_between(
        np.arange(len(summary.best_mean))[start:],
        (summary.best_mean - summary.best_std)[start:],
        (summary.best_mean + summary.best_std)[start:],
        color=color,
        alpha=0.25,
        step="mid",
    )

    ax.plot(
        np.arange(len(summary.best_mean))[start:],
        summary.best_mean[start:],
        color=color,
        label=optimizer,
        drawstyle="steps-mid",
    )

    ax.set_xlabel("Budget", fontsize=10)
    ax.set_ylabel(f"Validation {metric.upper()}", fontsize=10)
    ax.grid(alpha=0.2)


def build_color_dict(names: List[str]) -> Dict[str, np.ndarray]:
    norm = colors.Normalize(vmin=0, vmax=1)
    m = cm.ScalarMappable(norm, cm.tab20)
    color_dict = m.to_rgba(np.linspace(0, 1, len(names)))
    color_dict = dict(zip(names, color_dict))

    return color_dict


def partial_report(args: argparse.Namespace) -> None:
    eval_path = os.path.join("runs", _DB, "eval")
    time_path = os.path.join("runs", _DB, "time")
    studies = os.listdir(eval_path)
    summaries: List[pd.DataFrame] = []

    for study in studies:
        table_buffer: List[pd.DataFrame] = []
        column_buffer: List[str] = []
        for path in [eval_path, time_path]:
            with open(os.path.join(path, study), "r") as file:
                data = json.load(file)
                df = Dataset.from_dict(data["data"]).to_dataframe()
                df = df.droplevel("suggestion")

            for argument, meatadata in data["meta"]["args"].items():
                colname = argument[2:] if argument.startswith("--") else argument
                if colname not in column_buffer:
                    df[colname] = meatadata
                    column_buffer.append(colname)

            table_buffer.append(df)

        summary = pd.merge(*table_buffer, left_index=True, right_index=True)
        summaries.append(summary.reset_index())

    filename = f"{args.dataset}-{args.model}-partial-report.json"
    pd.concat(summaries).reset_index(drop=True).to_json(os.path.join("partial", filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iris")
    parser.add_argument("--model", type=str, default="kNN")
    parser.add_argument("--name-prefix", type=str, default="")
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--sampler-list", type=str, default="TPESampler CmaEsSampler")
    parser.add_argument(
        "--sampler-kwargs-list",
        type=str,
        default='{"multivariate":true,"constant_liar":true} {}',
    )
    parser.add_argument("--pruner-list", type=str, default="NopPruner")
    parser.add_argument("--pruner-kwargs-list", type=str, default="{}")
    parser.add_argument("--plot-warmup", type=str, default="true")

    args = parser.parse_args()
    os.makedirs("runs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("partial", exist_ok=True)

    run_benchmark(args)
    partial_report(args)
    make_plots(args)
