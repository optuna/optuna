from __future__ import annotations

import numpy as np

from optuna._gp import acqf
from optuna._gp.search_space import sample_normalized_params


def optimize_acqf_sample(
    acqf_params: acqf.AcquisitionFunctionParams, n_samples: int = 2048, seed: int | None = None
) -> tuple[np.ndarray, float]:
    # Normalized parameter values are sampled.
    xs = sample_normalized_params(n_samples, acqf_params.search_space, seed=seed)
    res = acqf.eval_acqf_no_grad(acqf_params, xs)
    best_i = np.argmax(res)
    return xs[best_i, :], res[best_i]
