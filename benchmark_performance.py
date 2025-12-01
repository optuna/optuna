
import time
import numpy as np
import optuna
from optuna.study._multi_objective import _is_pareto_front
from optuna._hypervolume import compute_hypervolume

def benchmark_hypervolume_3d_stress():
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
    benchmark_hypervolume_3d_stress()
