import argparse
import os
import subprocess


def run(args: argparse.Namespace) -> None:
    kurobako_cmd = os.path.join(args.path_to_kurobako, "kurobako")
    subprocess.run(f"{kurobako_cmd} --version", shell=True)

    os.makedirs(args.out_dir, exist_ok=True)
    study_json_filename = os.path.join(args.out_dir, "studies.json")
    solvers_filename = os.path.join(args.out_dir, "solvers.json")
    problems_filename = os.path.join(args.out_dir, "problems.json")

    # Ensure all files are empty.
    for filename in [study_json_filename, solvers_filename, problems_filename]:
        with open(filename, "w"):
            pass

    searchspace_datasets = [
        "nasbench201 cifar10",
        "nasbench201 cifar100",
        "nasbench201 ImageNet16-120",
    ]

    for searchspace_dataset in searchspace_datasets:
        python_command = f"benchmarks/naslib/problem.py {searchspace_dataset}"
        cmd = (
            f"{kurobako_cmd} problem command python3 {python_command}"
            f"| tee -a {problems_filename}"
        )
        subprocess.run(cmd, shell=True)

    # Create solvers.
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
            f"pruner_list: {pruner_list}, pruner_kwargs_list: {pruner_kwargs_list}."
        )

    for i, (sampler, sampler_kwargs) in enumerate(zip(sampler_list, sampler_kwargs_list)):
        sampler_name = sampler
        if sampler_list.count(sampler) > 1:
            sampler_name += f"_{sampler_list[:i].count(sampler)}"
        for j, (pruner, pruner_kwargs) in enumerate(zip(pruner_list, pruner_kwargs_list)):
            pruner_name = pruner
            if pruner_list.count(pruner) > 1:
                pruner_name += f"_{pruner_list[:j].count(pruner)}"
            name = f"{args.name_prefix}_{sampler_name}_{pruner_name}"
            cmd = (
                f"{kurobako_cmd} solver --name {name} optuna --loglevel debug "
                f"--sampler {sampler} --sampler-kwargs {sampler_kwargs} "
                f"--pruner {pruner} --pruner-kwargs {pruner_kwargs} "
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

    cmd = (
        f"cat {result_filename} | {kurobako_cmd} plot curve --errorbar -o {args.out_dir} --xmin 10"
    )
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-kurobako", type=str, default="")
    parser.add_argument("--name-prefix", type=str, default="")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-concurrency", type=int, default=1)
    parser.add_argument("--sampler-list", type=str, default="RandomSampler TPESampler")
    parser.add_argument(
        "--sampler-kwargs-list",
        type=str,
        default=r"{} {\"multivariate\":true\,\"constant_liar\":true}",
    )
    parser.add_argument("--pruner-list", type=str, default="NopPruner")
    parser.add_argument("--pruner-kwargs-list", type=str, default="{}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="out")
    args = parser.parse_args()

    run(args)
