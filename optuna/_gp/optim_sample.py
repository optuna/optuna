from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optuna._gp.acqf import BaseAcquisitionFunc


def optimize_acqf_sample(
    acqf: "BaseAcquisitionFunc",
    *,
    n_samples: int = 2048,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:
    # Normalized parameter values are sampled.
    xs = acqf.search_space.sample_normalized_params(n_samples, rng=rng)
    res = acqf.eval_acqf_no_grad(xs)

    best_i = np.argmax(res)
    return xs[best_i, :], res[best_i]
