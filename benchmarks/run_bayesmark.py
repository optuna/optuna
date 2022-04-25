import argparse
import copy
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

import xarray as xr
from xarray import Dataset

import bayesmark.constants as cc
from bayesmark.serialize import XRSerializer
from bayesmark.experiment_aggregate import validate_agg_perf
import bayesmark.xr_util as xru


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
    print("command:", cmd)
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-agg -dir runs -b {_DB}"
    print("command:", cmd)
    subprocess.run(cmd, shell=True)

    cmd = f"bayesmark-anal -dir runs -b {_DB}"
    print("command:", cmd)
    subprocess.run(cmd, shell=True)


def make_kurobako_results(args: argparse.Namespace):
    db_root = os.path.abspath("runs")
    perf_ds, meta = XRSerializer.load_derived(db_root, db=_DB, key=cc.EVAL_RESULTS)

    template_result = {
        "start_time": "2022-04-24T14:59:47.372925+09:00",  # dummy
        "end_time": "2022-04-24T14:59:48.138886+09:00",  # dummy
        "seed": 0,  # dummy
        "budget": len(perf_ds.coords[cc.ITER].values),
        "concurrency": len(perf_ds.coords[cc.SUGGEST].values),
        "scheduling": "RANDOM",  # dummy
        "solver": {
            "recipe": {
                'name': None,
                'optuna': {
                    'loglevel': 'debug',  # dummy
                    'pruner': 'NopPruner',  # dummy
                    'pruner_kwargs': '{}',  # dummy
                    'sampler': 'RandomSampler',  # dummy
                    'sampler_kwargs': '{}'  # dummy
                },
            },
            'spec': {
                'attrs': {
                    'github': 'https://github.com/optuna/optuna',
                    'paper': 'Akiba, Takuya, et al. "Optuna: A next-generation '
                             'hyperparameter optimization framework." '
                             'Proceedings of the 25th ACM SIGKDD International '
                             'Conference on Knowledge Discovery & Data Mining. '
                             'ACM, 2019.',
                    'version': 'optuna=3.0.0b0.dev0, kurobako-py=0.2.0'},
                'capabilities': ['UNIFORM_CONTINUOUS',
                                      'UNIFORM_DISCRETE',
                                      'LOG_UNIFORM_CONTINUOUS',
                                      'LOG_UNIFORM_DISCRETE',
                                      'CATEGORICAL',
                                      'CONDITIONAL',
                                      'MULTI_OBJECTIVE',
                                      'CONCURRENT'],
                'name': '_RandomSampler_NopPruner'
            }
        },
        "problem": {
            'recipe': {
                'hpobench': {
                    'dataset': './fcnet_tabular_benchmarks/fcnet_slice_localization_data.hdf5'
                }
            },
            'spec': {
                'attrs': {
                    'github': 'https://github.com/automl/nas_benchmarks',
                    'paper': 'Klein, Aaron, and Frank Hutter. "Tabular '
                             'Benchmarks for Joint Architecture and '
                             'Hyperparameter Optimization." arXiv preprint '
                             'arXiv:1905.04970 (2019).',
                    'version': 'kurobako_problems=0.1.13'
                },
                'name': 'HPO-Bench-Slice',
                'params_domain': [
                    {
                        'distribution': 'UNIFORM',
                        'name': 'activation_fn_1',
                        'range': {
                            'choices': ['tanh', 'relu'],
                            'type': 'CATEGORICAL'
                        }
                    },
                    {'distribution': 'UNIFORM',
                     'name': 'activation_fn_2',
                     'range': {'choices': ['tanh', 'relu'],
                               'type': 'CATEGORICAL'}},
                    {'distribution': 'UNIFORM',
                     'name': 'batch_size',
                     'range': {'high': 4,
                               'low': 0,
                               'type': 'DISCRETE'}},
                    {'distribution': 'UNIFORM',
                     'name': 'dropout_1',
                     'range': {'high': 3,
                               'low': 0,
                               'type': 'DISCRETE'}},
                    {'distribution': 'UNIFORM',
                     'name': 'dropout_2',
                     'range': {'high': 3,
                               'low': 0,
                               'type': 'DISCRETE'}},
                    {'distribution': 'UNIFORM',
                     'name': 'init_lr',
                     'range': {'high': 6,
                               'low': 0,
                               'type': 'DISCRETE'}},
                    {'distribution': 'UNIFORM',
                     'name': 'lr_schedule',
                     'range': {'choices': ['cosine', 'const'],
                               'type': 'CATEGORICAL'}},
                    {'distribution': 'UNIFORM',
                     'name': 'n_units_1',
                     'range': {'high': 6,
                               'low': 0,
                               'type': 'DISCRETE'}},
                    {'distribution': 'UNIFORM',
                     'name': 'n_units_2',
                     'range': {'high': 6,
                               'low': 0,
                               'type': 'DISCRETE'}}],
                'steps': 100,
                'values_domain': [
                    {
                        'distribution': 'UNIFORM',
                        'name': 'Validation MSE',
                        'range':
                            {'low': 0.0, 'type': 'CONTINUOUS'}
                    }
                ]
            }
        },
        "trials": []
    }

    template_trial = {
        "evaluations": [
            {
                'ask_elapsed': 0.002855974,
                'end_step': 300,
                'evaluate_elapsed': 0.014396439,
                'start_step': 200,
                'tell_elapsed': 0.001346957,
                "values": [],
            }
        ],
        'params': [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 4.0, 3.0],
        'thread_id': 0,
    }

    results = []
    perf_da = perf_ds[cc.VISIBLE_TO_OPT]

    validate_agg_perf(perf_da, min_trial=1)
    for func_name in perf_da.coords[cc.TEST_CASE].values:
        for method_name in perf_da.coords[cc.METHOD].values:
            for study_id in perf_da.coords[cc.TRIAL].values:
                result = copy.deepcopy(template_result)
                result["solver"]["recipe"]["name"] = method_name
                result["problem"]["spec"]["name"] = func_name
                for iter in perf_da.coords[cc.ITER].values:
                    assert len(perf_ds.coords[cc.SUGGEST].values) == 1
                    curr_perf_value = perf_da.sel(
                        {
                            cc.METHOD: method_name,
                            cc.TEST_CASE: func_name,
                            cc.TRIAL: study_id,
                            cc.ITER: iter,
                        },
                        drop=True)
                    trial = copy.deepcopy(template_trial)
                    trial["evaluations"][0]["values"].append(curr_perf_value[0].item())
                    result["trials"].append(trial)
                results.append(result)

    encoder = json.JSONEncoder()
    with open("./bayesmark_results.json", "w") as f:
        for res in results:
            json_str = encoder.encode(res)
            f.write(json_str)
            f.write("\n")

def make_plots(args: argparse.Namespace) -> None:

    # https://github.com/uber/bayesmark/blob/master/notebooks/plot_test_case.ipynb
    db_root = os.path.abspath("runs")
    summary, _ = XRSerializer.load_derived(db_root, db=_DB, key=cc.PERF_RESULTS)
    plot_warmup = json.loads(args.plot_warmup)

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots()

    for benchmark in summary.coords["function"].values:
        for metric, ax in zip(["mean", "median"], axs):
            make_plot(summary, ax, benchmark, metric, plot_warmup)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(benchmark)
    fig.savefig(os.path.join("plots", f"optuna-{args.dataset}-{args.model}-sumamry.png"))


def make_plot(summary: Dataset, ax: Axes, func: str, metric: str, plot_warmup: bool) -> None:

    color = build_color_dict(summary.coords["optimizer"].values.tolist())
    optimizers = summary.coords["optimizer"].values
    start = 0 if plot_warmup else 10

    for optimizer in optimizers:
        curr_ds = summary.sel(
            {"function": func, "optimizer": optimizer, "objective": cc.VISIBLE_TO_OPT}
        )

        if len(curr_ds.coords[cc.ITER].values) <= start:
            # Not enough trials to make a plot.
            continue

        ax.fill_between(
            curr_ds.coords[cc.ITER].values[start:],
            curr_ds[f"{metric} LB"].values[start:],
            curr_ds[f"{metric} UB"].values[start:],
            color=color[optimizer],
            alpha=0.5,
        )
        ax.plot(
            curr_ds.coords["iter"].values[start:],
            curr_ds[metric].values[start:],
            color=color[optimizer],
            label=optimizer,
        )

    ax.set_xlabel("Budget", fontsize=10)
    ax.set_ylabel(f"{metric.capitalize()} score", fontsize=10)
    ax.grid(alpha=0.2)


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
    sorted_lb = {k: v for k, v in sorted(leaderboard.items(), key=lambda i: i[1], reverse=True)}

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
    parser.add_argument("--plot-warmup", type=str, default="true")

    args = parser.parse_args()
    os.makedirs("runs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("partial", exist_ok=True)

    #run_benchmark(args)
    make_kurobako_results(args)
    #make_plots(args)
    #partial_report(args)
