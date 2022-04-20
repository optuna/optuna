import argparse
import os
import subprocess


def run(args: argparse.Namespace) -> None:
    kurobako_cmd = os.path.join(args.path_to_kurobako, "kurobako")
    subprocess.run(f"{kurobako_cmd} --version", shell=True)

    if not (os.path.exists(args.data_dir) and os.path.isdir(args.data_dir)):
        raise ValueError(f"Data directory {args.data_dir} cannot be found.")

    os.makedirs(args.out_dir, exist_ok=True)
    study_json_fn = os.path.join(args.out_dir, "studies.json")
    solvers_filename = os.path.join(args.out_dir, "solvers.json")
    problems_filename = os.path.join(args.out_dir, "problems.json")

    # Ensure all files are empty.
    for filename in [study_json_fn, solvers_filename, problems_filename]:
        with open(filename, "w"):
            pass

    # Create HPO bench problem.
    datasets = [
        "fcnet_tabular_benchmarks/fcnet_naval_propulsion_data.hdf5",
        "fcnet_tabular_benchmarks/fcnet_parkinsons_telemonitoring_data.hdf5",
        "fcnet_tabular_benchmarks/fcnet_protein_structure_data.hdf5",
        "fcnet_tabular_benchmarks/fcnet_slice_localization_data.hdf5",
    ]
    for dataset in datasets:
        dataset = os.path.join(args.data_dir, dataset)
        cmd = f'{kurobako_cmd} problem hpobench "{dataset}" | tee -a {problems_filename}'
        subprocess.run(cmd, shell=True)

    # Create NAS bench problem.
    dataset = os.path.join(args.data_dir, "nasbench_full.bin")
    cmd = f'{kurobako_cmd} problem nasbench "{dataset}" | tee -a {problems_filename}'
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
            f"pruner_list: {pruner_list}, pruner_keyword_arguments: {pruner_kwargs_list}."
        )

    for sampler, sampler_kwargs in zip(sampler_list, sampler_kwargs_list):
        for pruner, pruner_kwargs in zip(pruner_list, pruner_kwargs_list):
            name = f"{args.name_prefix}_{sampler}_{pruner}"
            cmd = (
                f"{kurobako_cmd} solver --name {name} optuna --loglevel debug "
                f"--sampler {sampler} --sampler-kwargs {sampler_kwargs} "
                f"--pruner {pruner} --pruner-kwargs {pruner_kwargs} "
                f"| tee -a {solvers_filename}"
            )
            subprocess.run(cmd, shell=True)

    # Create study.
    cmd = (
        f"{kurobako_cmd} studies --budget 80 "
        f"--solvers $(cat {solvers_filename}) --problems $(cat {problems_filename}) "
        f"--repeats {args.n_runs} --seed {args.seed} "
        f"> {study_json_fn}"
    )
    subprocess.run(cmd, shell=True)

    result_filename = os.path.join(args.out_dir, "results.json")
    cmd = (
        f"cat {study_json_fn} | {kurobako_cmd} run --parallelism {args.n_jobs} "
        f"> {result_filename}"
    )
    subprocess.run(cmd, shell=True)

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
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--sampler-list", type=str, default="RandomSampler TPESampler")
    parser.add_argument(
        "--sampler-kwargs-list",
        type=str,
        default=r"{} {\"multivariate\":true\,\"constant_liar\":true}",
    )
    parser.add_argument("--pruner-list", type=str, default="NopPruner")
    parser.add_argument("--pruner-kwargs-list", type=str, default="{}")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="out")
    args = parser.parse_args()

    run(args)
