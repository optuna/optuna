from __future__ import annotations

import argparse
import os
import subprocess


def run(args: argparse.Namespace) -> None:
    kurobako_cmd = os.path.join(args.path_to_kurobako, "kurobako")
    subprocess.run(f"{kurobako_cmd} --version", shell=True)

    if not (os.path.exists(args.data_dir) and os.path.isdir(args.data_dir)):
        raise ValueError(f"Data directory {args.data_dir} cannot be found.")

    os.makedirs(args.out_dir, exist_ok=True)
    study_json_filename = os.path.join(args.out_dir, "studies.json")
    solvers_filename = os.path.join(args.out_dir, "solvers.json")
    problems_filename = os.path.join(args.out_dir, "problems.json")

    # Ensure all files are empty.
    for filename in [study_json_filename, solvers_filename, problems_filename]:
        with open(filename, "w"):
            pass

    # Create ZDT problems
    cmd = f"{kurobako_cmd} problem-suite zdt | tee -a {problems_filename}"
    subprocess.run(cmd, shell=True)

    # Create WFG 1~9 problem
    for n_wfg in range(1, 10):
        if n_wfg == 8:
            n_dim = 3
            k = 2
        elif n_wfg in (7, 9):
            n_dim = 2
            k = 1
        else:
            n_dim = 10
            k = 2
        n_objective = 2

        python_command = f"benchmarks/kurobako/problems/wfg/problem.py \
            {n_wfg} {n_dim} {n_objective} {k}"
        cmd = (
            f"{kurobako_cmd} problem command python {python_command}"
            f"| tee -a {problems_filename}"
        )
        subprocess.run(cmd, shell=True)

    # Create NAS bench problem(A) (for Multi-Objective Settings).
    dataset = os.path.join(args.data_dir, "nasbench_full.bin")
    cmd = (
        f'{kurobako_cmd} problem nasbench "{dataset}" '
        f"--metrics params accuracy | tee -a {problems_filename}"
    )
    subprocess.run(cmd, shell=True)

    # Create solvers.
    sampler_list = args.sampler_list.split()
    sampler_kwargs_list = args.sampler_kwargs_list.split()

    if len(sampler_list) != len(sampler_kwargs_list):
        raise ValueError(
            "The number of samplers does not match the given keyword arguments. \n"
            f"sampler_list: {sampler_list}, sampler_kwargs_list: {sampler_kwargs_list}."
        )

    for i, (sampler, sampler_kwargs) in enumerate(zip(sampler_list, sampler_kwargs_list)):
        sampler_name = sampler
        if sampler_list.count(sampler) > 1:
            sampler_name += f"_{sampler_list[:i].count(sampler)}"
        name = f"{args.name_prefix}_{sampler_name}"
        python_command = f"{args.path_to_create_study} {sampler} {sampler_kwargs}"
        cmd = (
            f"{kurobako_cmd} solver --name {name} command python3 {python_command}"
            f"| tee -a {solvers_filename}"
        )
        subprocess.run(cmd, shell=True)

    # Create study.
    cmd = (
        f"{kurobako_cmd} studies --budget {args.budget} "
        f"--solvers $(cat {solvers_filename}) --problems $(cat {problems_filename}) "
        f"--repeats {args.n_runs} --seed {args.seed} --concurrency {args.n_concurrency} "
        f"> {study_json_filename}"
    )
    subprocess.run(cmd, shell=True)

    result_filename = os.path.join(args.out_dir, "results.json")
    cmd = (
        f"cat {study_json_filename} | {kurobako_cmd} run --parallelism {args.n_jobs} -q "
        f"> {result_filename}"
    )
    subprocess.run(cmd, shell=True)

    # Report.
    report_filename = os.path.join(args.out_dir, "report.md")
    cmd = f"cat {result_filename} | {kurobako_cmd} report > {report_filename}"
    subprocess.run(cmd, shell=True)

    # Plot pareto-front.
    plot_args: dict[str, dict[str, int | float]]
    plot_args = {
        "NASBench": {"xmin": 0, "xmax": 25000000, "ymin": 0, "ymax": 0.2},
        "ZDT1": {"xmin": 0, "xmax": 1, "ymin": 1, "ymax": 7},
        "ZDT2": {"xmin": 0, "xmax": 1, "ymin": 2, "ymax": 7},
        "ZDT3": {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 7},
        "ZDT4": {"xmin": 0, "xmax": 1, "ymin": 20, "ymax": 250},
        "ZDT5": {"xmin": 8, "xmax": 24, "ymin": 1, "ymax": 6},
        "ZDT6": {"xmin": 0.2, "xmax": 1, "ymin": 5, "ymax": 10},
        "WFG1": {"xmin": 2.7, "xmax": 3.05, "ymin": 4.7, "ymax": 5.05},
        "WFG2": {"xmin": 2.0, "xmax": 2.8, "ymin": 3.0, "ymax": 4.8},
        "WFG3": {"xmin": 2.0, "xmax": 2.8, "ymin": 3.0, "ymax": 4.8},
        "WFG4": {"xmin": 2.0, "xmax": 3.0, "ymin": 0.0, "ymax": 3.6},
        "WFG5": {"xmin": 2.0, "xmax": 3.0, "ymin": 2.5, "ymax": 5.0},
        "WFG6": {"xmin": 2.0, "xmax": 3.0, "ymin": 3.4, "ymax": 5.0},
        "WFG7": {"xmin": 2.0, "xmax": 3.0, "ymin": 4.0, "ymax": 5.0},
        "WFG8": {"xmin": 2.0, "xmax": 3.0, "ymin": 3.4, "ymax": 5.0},
        "WFG9": {"xmin": 2.0, "xmax": 3.0, "ymin": 0.0, "ymax": 5.0},
    }

    for problem_name, plot_arg in plot_args.items():
        xmin, xmax = plot_arg["xmin"], plot_arg["xmax"]
        ymin, ymax = plot_arg["ymin"], plot_arg["ymax"]
        cmd = (
            f"cat {result_filename} | grep {problem_name} | "
            f"{kurobako_cmd} plot pareto-front -o {args.out_dir} "
            f"--xmin {xmin} --xmax {xmax} --ymin {ymin} --ymax {ymax}"
        )
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-kurobako", type=str, default="")
    parser.add_argument(
        "--path-to-create-study", type=str, default="benchmarks/kurobako/mo_create_study.py"
    )
    parser.add_argument("--name-prefix", type=str, default="")
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-concurrency", type=int, default=1)
    parser.add_argument(
        "--sampler-list",
        type=str,
        default="RandomSampler TPESampler NSGAIISampler",
    )
    parser.add_argument(
        "--sampler-kwargs-list",
        type=str,
        default=r"{} {\"multivariate\":true\,\"constant_liar\":true} {\"population_size\":20}",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="out")
    args = parser.parse_args()

    run(args)
