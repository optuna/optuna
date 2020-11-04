import argparse

import os
import json
import subprocess
from typing import List
from typing import Tuple

import optuna


HPOBENCH: str = "hpobench"
HPOBENCH_PROTEIN: str = "hpobenchprotein"
NASBENCH: str = "nasbench"
SIGOPT: str = "sigopt"
SOLVERS_JSON_FN: str = "solvers.json"
PROBLEMS_JSON_FN: str = "problems.json"
HPOBENCH_FNS: Tuple[str, str, str, str] = (
    "fcnet_tabular_benchmarks/fcnet_naval_propulsion_data.hdf5",
    "fcnet_tabular_benchmarks/fcnet_parkinsons_telemonitoring_data.hdf5",
    "fcnet_tabular_benchmarks/fcnet_protein_structure_data.hdf5",
    "fcnet_tabular_benchmarks/fcnet_slice_localization_data.hdf5",
)
NASBENCH_FN: str = "nasbench_full.bin"
SIGOPT_FNS: Tuple[str, str] = (
    "rosenbrock-log",
    "six-hump-camel",
)


def _get_invalid_problems(problem_list: List[str]) -> List[str]:
    return [
        problem
        for problem in problem_list
        if problem not in (HPOBENCH, HPOBENCH_PROTEIN, NASBENCH, SIGOPT)
    ]


def _get_invalid_samplers(sampler_list: List[str]) -> List[str]:
    return [
        sampler
        for sampler in sampler_list
        if not (hasattr(optuna.samplers, sampler) or hasattr(optuna.integration, sampler))
    ]


def _get_invalid_pruners(pruner_list: List[str]) -> List[str]:
    return [pruner for pruner in pruner_list if not hasattr(optuna.pruners, pruner)]


def _run_kurobako(args: argparse.Namespace) -> None:

    if not (os.path.exists(args.input_dir) and os.path.isdir(args.input_dir)):
        raise RuntimeError(
            f"Input directory `{args.input_dir}` does not conform to the requirements."
        )

    _invalid_problems = _get_invalid_problems(args.problem_list)
    if _invalid_problems:
        raise RuntimeError(f"Invalid problems {_invalid_problems} are included in `problem_list`")

    _invalid_samplers = _get_invalid_samplers(args.sampler_list)
    if _invalid_samplers:
        raise RuntimeError(f"Invalid samplers {_invalid_samplers} are included in `sampler_list`")

    _invalid_pruners = _get_invalid_pruners(args.pruner_list)
    if _invalid_pruners:
        raise RuntimeError(f"Invalid pruners {_invalid_pruners} are included in `pruner_list`")

    studies_directory = os.path.join(args.output_dir, "studies", args.name)
    if not os.path.exists(studies_directory):
        os.makedirs(studies_directory)

    study_json_fn = os.path.join(studies_directory, f"studies_{args.name}.json")

    solvers_fn = os.path.join(args.output_dir, SOLVERS_JSON_FN)
    subprocess.check_call(f"echo -n >| {solvers_fn}", shell=True)
    problems_fn = os.path.join(args.output_dir, PROBLEMS_JSON_FN)
    subprocess.check_call(f"echo -n >| {problems_fn}", shell=True)

    with open(study_json_fn, "w") as f:
        for problem in args.problem_list:
            dataset_list: List[str] = []
            if problem == HPOBENCH:
                dataset_list = [os.path.join(args.input_dir, fn) for fn in HPOBENCH_FNS]
            elif problem == HPOBENCH_PROTEIN:
                dataset_list = [os.path.join(args.input_dir, HPOBENCH_FNS[2])]
            elif problem == NASBENCH:
                dataset_list = [os.path.join(args.input_dir, NASBENCH_FN)]
            elif problem == SIGOPT:
                dataset_list = list(SIGOPT_FNS)
            else:
                raise RuntimeError(f"Invalid problem `{problem}`")
            for dataset in dataset_list:
                cmd = f""" {args.kurobako_path}kurobako problem {problem} "{dataset}" | tee -a {problems_fn} """
                print(cmd)
                subprocess.check_call(cmd, shell=True)

            # with open(problems_fn, "w") as f:
            for sampler in args.sampler_list:
                sampler_kwargs = "{}"
                if sampler == "SkoptSampler":
                    sampler_kwargs = """ {"skopt_kwargs": {"base_estimator": "GP"}} """
                    sampler_kwargs = json.dumps(
                        {
                            "skopt_kwargs": {
                                "base_estimator": "GP",
                            },
                        }
                    )

                for pruner in args.pruner_list:
                    pruner_kwargs = "{}"
                    if pruner == "HyperbandPruner":
                        pruner_kwargs = json.dumps(
                            {
                                "min_resource": args.min_resource,
                                "reduction_factor": args.reduction_factor,
                            }
                        )
                    elif pruner == "MedianPruner":
                        pruner_kwargs = json.dumps(
                            {
                                "warmup_steps": args.n_warmup_steps,
                            }
                        )
                    else:
                        pass

                    cmd = f""" {args.kurobako_path}kurobako solver --name "{sampler}-{pruner}" optuna --loglevel debug --sampler "{sampler}" --sampler-kwargs "{sampler_kwargs}" --pruner "{pruner}" --pruner-kwargs "{pruner_kwargs}" | tee -a {solvers_fn} """  # NOQA
                    print(cmd)
                    subprocess.check_call(cmd, shell=True)

                cmd = """{}kurobako studies --budget {} --solvers $(cat {}) --problems $(cat {}) --repeats {} --seed {} >> {}""".format(  # NOQA
                    args.kurobako_path,
                    args.budget,
                    solvers_fn,
                    problems_fn,
                    args.n_run,
                    args.seed,
                    study_json_fn,
                )
                subprocess.check_call(cmd, shell=True)

    results_directory = os.path.join(args.output_dir, "results", args.name)
    os.makedirs(results_directory, exist_ok=True)
    result_json_fn = os.path.join(results_directory, f"results_{args.name}.json")
    cmd = "cat {} | {} run --parallelism {} > {}".format(
        study_json_fn, f"{args.kurobako_path}kurobako", args.n_jobs, result_json_fn
    )
    subprocess.check_call(cmd, shell=True)

    reports_directory = os.path.join(args.output_dir, "report", args.name)
    os.makedirs(reports_directory, exist_ok=True)
    cmd = "cat {} | {} report > {}".format(
        result_json_fn,
        f"{args.kurobako_path}kurobako",
        os.path.join(reports_directory, f"report_{args.name}.md"),
    )
    subprocess.check_call(cmd, shell=True)

    if args.plot:
        cmd = "cd {}; cat {} | {} plot curve --errorbar".format(
            results_directory,
            os.path.basename(result_json_fn),
            f"{args.kurobako_path}kurobako",
        )

        subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script for benchmark with [Kurobako](https://github.com/sile/kurobako)"
    )

    parser.add_argument("--name", type=str, default="kurobako-bench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--n-run", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=10)

    parser.add_argument("--kurobako-path", type=str, default="")
    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="result")

    parser.add_argument("--min-resource", type=int, default=1)
    parser.add_argument("--min-resource-hpo", type=int, default=1)
    parser.add_argument("--min-resource-nas", type=int, default=1)
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--n-warmup-steps", type=int, default=40)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--budget-hpo", type=int, default=80)
    parser.add_argument("--budget-nas", type=int, default=80)

    parser.add_argument(
        "--sampler-list",
        type=str,
        nargs="*",
        default=[
            "RandomSampler",
            "TPESampler",
        ],
    )
    parser.add_argument(
        "--pruner-list",
        type=str,
        nargs="*",
        default=[
            "MedianPruner",
            "HyperbandPruner",
            "NopPruner",
        ],
    )
    parser.add_argument(
        "--problem-list",
        type=str,
        nargs="*",
        default=[
            "hpobench",
            "nasbench",
        ],
    )

    args = parser.parse_args()
    _run_kurobako(args)
