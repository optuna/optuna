
import numpy as np
import optuna
from optuna.study._multi_objective import _is_pareto_front
from optuna._hypervolume import compute_hypervolume
from optuna._hypervolume.wfg import _compute_hv

def naive_is_pareto_front(loss_values):
    n = len(loss_values)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(loss_values[j] <= loss_values[i]) and np.any(loss_values[j] < loss_values[i]):
                is_pareto[i] = False
                break
    return is_pareto

def verify_pareto_front():
    print("Verifying _is_pareto_front...")
    for n in [10, 100, 500]:
        for d in [2, 3, 4, 5]:
            data = np.random.rand(n, d)
            # Ensure uniqueness for strict comparison
            data = np.unique(data, axis=0)
            
            # Naive
            expected = naive_is_pareto_front(data)
            
            # Optimized
            # Note: _is_pareto_front requires unique lexsorted if assume_unique_lexsorted=True
            # But the public API handles it.
            actual = _is_pareto_front(data, assume_unique_lexsorted=False)
            
            if not np.array_equal(expected, actual):
                print(f"FAILED for n={n}, d={d}")
                return False
    print("PASSED _is_pareto_front verification.")
    return True

def verify_hypervolume_3d():
    print("\nVerifying compute_hypervolume (3D)...")
    for n in [10, 50, 100]:
        reference_point = np.array([2.0, 2.0, 2.0])
        data = np.random.rand(n, 3)
        
        # Use the generic WFG implementation (which uses recursion) as ground truth
        # We need to sort and filter for _compute_hv
        unique_data = np.unique(data, axis=0)
        on_front = _is_pareto_front(unique_data, assume_unique_lexsorted=False)
        pareto_sols = unique_data[on_front]
        # Sort by first objective for _compute_hv
        pareto_sols = pareto_sols[pareto_sols[:, 0].argsort()]
        
        expected = _compute_hv(pareto_sols, reference_point)
        
        # Optimized 3D
        actual = compute_hypervolume(data, reference_point)
        
        if not np.isclose(expected, actual):
            print(f"FAILED for n={n}. Expected {expected}, got {actual}")
            return False
            
    print("PASSED compute_hypervolume (3D) verification.")
    return True

if __name__ == "__main__":
    if verify_pareto_front() and verify_hypervolume_3d():
        print("\nAll verifications PASSED!")
    else:
        print("\nVerification FAILED!")
        exit(1)
