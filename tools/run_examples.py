import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


EXAMPLES_DIR = Path(__file__).parents[1] / Path("examples")
ALL_EXAMPLES: List[str] = [str(p) for p in EXAMPLES_DIR.glob("**/*.py")]
# TODO (crcrpar): Allow fastaiv2 example once https://forums.fast.ai/t/ganlearner-error-no-implementation-found-on-types-that-implement-invisibletensor/83451/7 gets resolved.  # NOQA
_DEFAULT_DENY_LIST: List[str] = [
    "fastai",
    "hydra/simple",
    "pytorch_distributed_",
    "rapids_",
    "allennlp/subsample_dataset_reader",
]
VER_TO_DENY_LIST: Dict[Tuple[int, int], List[str]] = {
    (3, 6): _DEFAULT_DENY_LIST + ["botorch_"],
    (3, 7): _DEFAULT_DENY_LIST,
    (3, 8): _DEFAULT_DENY_LIST + ["keras_", "tensorboard_", "tensorflow_", "tfkeras_"],
}


def _get_distributed_examples(files: List[str]) -> List[str]:
    return [f for f in files if "distributed" in f]


def _get_ipynb_examples(files: Optional[List[str]] = None) -> List[str]:
    if files is None:
        return [str(p) for p in EXAMPLES_DIR.glob("**/*.ipynb")]
    else:
        return [f for f in files if "ipynb" in f]


def _get_multinode_examples(files: List[str]) -> List[str]:
    return [f for f in files if "chainermn" in f]


def _get_hydra_examples(files: List[str]) -> List[str]:
    return [f for f in files if "hydra" in f]


class ExampleFiles:
    def __init__(self, files: Optional[List[str]] = None) -> None:
        _all_examples = ALL_EXAMPLES
        deny_list = VER_TO_DENY_LIST[sys.version_info[:2]]
        if files is not None:
            _all_examples = [f for f in files if f in _all_examples]

        all_examples = []
        for f in _all_examples:
            if not any(d in f for d in deny_list):
                all_examples.append(f)

        self.distributed_examples = _get_distributed_examples(all_examples)
        self.ipynb_examples = _get_ipynb_examples(all_examples)
        self.multinode_examples = _get_multinode_examples(all_examples)
        self.hydra_examples = _get_hydra_examples(all_examples)
        self.vanilla_examples = list(
            set(all_examples)
            - set(self.distributed_examples)
            - set(self.multinode_examples)
            - set(self.hydra_examples)
        )


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--selected-files", nargs="*", help="Files to be executed.")
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-ipynb", action="store_true")
    parser.add_argument("--skip-multinode", action="store_true")
    parser.add_argument("--skip-hydra", action="store_true")
    parser.add_argument("--on-githubactions", action="store_true")
    return parser.parse_args()


def run_command(
    cmd: str,
    environment_variables: Optional[Dict[str, str]] = None,
):
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, check=True, env=environment_variables)


def has_pruning_option(filename: str):
    return "--pruning" in open(filename).read()


def is_denied(filename: str, deny_list: List[str]) -> bool:
    for d in deny_list:
        if d in filename:
            return True
    return False


def run_vanilla_examples(
    examples: List[str], environment_variables: Optional[Dict[str, str]] = None
):
    if not examples:
        print("No vanilla examples to run")
        return True
    print(f"Run {len(examples)} vanilla examples")
    for i, filename in enumerate(examples):
        print(f"# {i}th example: {filename}")
        cmd = f"python {filename} > /dev/null"
        run_command(cmd, environment_variables)

        if has_pruning_option(filename):
            cmd = f"python {filename} --pruning > /dev/null"
            run_command(cmd, environment_variables)
    else:
        return True


def run_ipynb_examples(
    ipynb_examples: List[str], environment_variables: Optional[Dict[str, str]] = None
):
    if not ipynb_examples:
        print("No ipynb examples to run")
        return True
    print(f"Run {len(ipynb_examples)} ipynb examples")
    for i, filename in enumerate(ipynb_examples):
        print(f"# {i}th ipynb example: {filename}")
        cmd = f"pytest --nbval-lax {filename} > /dev/null"
        run_command(cmd, environment_variables)
    else:
        return True


def run_multinode_examples(
    multinode_examples, environment_variables: Optional[Dict[str, str]] = None
):
    if not multinode_examples:
        print("No multinode examples to run")
        return True
    print(f"Run {len(multinode_examples)} multinode examples")
    for i, filename in enumerate(multinode_examples):
        print(f"# {i}th multinode example: {filename}")
        storage_url = "sqlite:///example.db"
        study_name = subprocess.check_output(
            f"optuna create-study --storage {storage_url}", shell=True
        )
        run_command(
            f"mpirun -n 2 -- python {filename} ${study_name} ${storage_url} > /dev/null",
            environment_variables,
        )
    if "examples/pytorch/pytorch_distributed_simple.py" in multinode_examples:
        run_command(
            "mpirun -n 2 -- python examples/pytorch/pytorch_distributed_simple.py",
            environment_variables,
        )
    return True


def run_hydra_examples(hydra_examples, environment_variables: Optional[Dict[str, str]] = None):
    if not hydra_examples:
        print("No hydra examples to run")
        return True
    print(f"Run {len(hydra_examples)} hydra examples")
    for i, filename in enumerate(hydra_examples):
        print(f"# {i}th hydra example: {filename}")
        run_command(f"python {filename} --multirun > /dev/null", environment_variables)
    else:
        return True


if __name__ == "__main__":
    args = parse_args()
    py_version = sys.version_info[:2]
    print(args.selected_files)
    print(f"Python Version: {py_version}")
    deny_list = VER_TO_DENY_LIST[py_version]

    example_files = ExampleFiles(args.selected_files)

    env = os.environ

    if not args.skip_vanilla:
        run_vanilla_examples(example_files.vanilla_examples, env)
    if not args.skip_ipynb:
        run_ipynb_examples(example_files.ipynb_examples, env)
    if not args.skip_multinode:
        run_multinode_examples(example_files.multinode_examples, env)
    if not args.skip_hydra:
        run_hydra_examples(example_files.hydra_examples, env)
