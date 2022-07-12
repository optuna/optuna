import sys
from typing import Any
from typing import List

from kurobako import problem
from naslib.utils import get_dataset_api


op_names = [
    "skip_connect",
    "none",
    "nor_conv_3x3",
    "nor_conv_1x1",
    "avg_pool_3x3",
]
edge_num = 4 * 3 // 2
max_epoch = 199

prune_start_epoch = 10
prune_epoch_step = 10


class NASLibProblemFactory(problem.ProblemFactory):
    def __init__(self, dataset: str) -> None:
        """Creates ProblemFactory for NASBench201.

        Args:
            dataset:
                Accepts one of "cifar10", "cifar100" or "ImageNet16-120".
        """
        self._dataset = dataset
        if dataset == "cifar10":
            # Set name used in dataset API.
            self._dataset = "cifar10-valid"
        self._dataset_api = get_dataset_api("nasbench201", dataset)

    def specification(self) -> problem.ProblemSpec:

        params = [
            problem.Var(f"x{i}", problem.CategoricalRange(op_names)) for i in range(edge_num)
        ]
        return problem.ProblemSpec(
            name=f"NASBench201-{self._dataset}",
            params=params,
            values=[problem.Var("value")],
            steps=list(range(prune_start_epoch, max_epoch, prune_epoch_step)) + [max_epoch],
        )

    def create_problem(self, seed: int) -> problem.Problem:
        return NASLibProblem(self._dataset, self._dataset_api)


class NASLibProblem(problem.Problem):
    def __init__(self, dataset: str, dataset_api: Any) -> None:
        super().__init__()
        self._dataset = dataset
        self._dataset_api = dataset_api

    def create_evaluator(self, params: List[float]) -> problem.Evaluator:
        ops = [op_names[int(x)] for x in params]
        arch_str = "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(*ops)
        return NASLibEvaluator(
            self._dataset_api["nb201_data"][arch_str][self._dataset]["eval_acc1es"]
        )


class NASLibEvaluator(problem.Evaluator):
    def __init__(self, learning_curve: List[float]) -> None:
        self._current_step = 0
        self._lc = learning_curve

    def current_step(self) -> int:
        return self._current_step

    def evaluate(self, next_step: int) -> List[float]:
        self._current_step = next_step
        return [-self._lc[next_step]]


if __name__ == "__main__":

    if len(sys.argv) < 1 + 2:
        print("Usage: python3 nas_bench_suite/problems.py <search_space> <dataset>")
        print("Example: python3 nas_bench_suite/problems.py nasbench201 cifar10")
        exit(1)

    search_space_name = sys.argv[1]
    # We currently do not support other benchmarks.
    assert search_space_name == "nasbench201"
    dataset = sys.argv[2]
    runner = problem.ProblemRunner(NASLibProblemFactory(dataset))
    runner.run()
