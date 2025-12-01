import time

import numpy as np

from optuna._hypervolume import compute_hypervolume
from optuna.study._multi_objective import _is_pareto_front


def benchmark_pareto_front_4d_random() -> None:
    print("Benchmarking _is_pareto_front in 4D (Random)...")
    n_trials_list = [1000, 5000, 10000]
    for n in n_trials_list:
        data = np.random.rand(n, 4)
        start = time.time()
        _is_pareto_front(data, assume_unique_lexsorted=False)
        end = time.time()
        print(f"n={n}: {end - start:.4f} seconds")


def benchmark_pareto_front_4d_worst_case() -> None:
    print("\nBenchmarking _is_pareto_front in 4D (Worst Case - Sphere)...")
    n_trials_list = [1000, 5000, 10000]
    for n in n_trials_list:
        # Generate points on the positive octant of a 4D sphere
        data = np.abs(np.random.normal(size=(n, 4)))
        data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

        start = time.time()
        mask = _is_pareto_front(data, assume_unique_lexsorted=False)
        end = time.time()
        print(f"n={n}: {end - start:.4f} seconds (Pareto count: {np.sum(mask)})")


def benchmark_hypervolume_3d_stress() -> None:
    print("\nBenchmarking compute_hypervolume in 3D (Stress Test)...")
    n_trials_list = [2000, 5000, 10000]
    reference_point = np.array([2.0, 2.0, 2.0])
    for n in n_trials_list:
        # Generate points on positive octant of sphere
        data = np.abs(np.random.normal(size=(n, 3)))
        data = data / np.linalg.norm(data, axis=1)[:, np.newaxis]

        try:
            start = time.time()
            compute_hypervolume(data, reference_point)
            end = time.time()
            print(f"n={n}: {end - start:.4f} seconds")
        except Exception as e:
            print(f"n={n}: Failed with {e}")


if __name__ == "__main__":
    benchmark_pareto_front_4d_random()
    benchmark_pareto_front_4d_worst_case()
    benchmark_hypervolume_3d_stress()
