from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

from packaging.version import Version


if TYPE_CHECKING:
    from collections.abc import Generator

    import scipy
    import torch
else:
    from optuna import _LazyImport

    scipy = _LazyImport("scipy")
    torch = _LazyImport("torch")


@contextmanager
def limit_threads_in_optimization() -> Generator[None, None, None]:
    """
    Context manager to limit threading to resolve a thread oversubscription issue.

    On Linux, ``GPSampler`` can slow down dramatically because NumPy/SciPy routines spawn OpenBLAS
    threads while PyTorch spawns OpenMP (libgomp) threads, and the two thread pools compete for the
    same cores. The GP fit and the acquisition-function optimization issue many small NumPy/PyTorch
    calls, where this contention dominates the runtime rather than the actual computation.

    Two knobs mitigate this:

    1. ``torch.set_num_threads(1)``: always applied. Limiting PyTorch's intra-op threads is
       the dominant fix and helps regardless of the SciPy version (benchmarked ~6-12x
       speedup on a 20-core Linux machine). It is safe and reversible. This addresses the
       PyTorch/OpenMP side of the oversubscription.
    2. ``OPENBLAS_NUM_THREADS=1``: applied only for SciPy v1.15+. SciPy v1.15.0 switched its
       optimizer backend so that OpenBLAS threads now contend with PyTorch's; on older SciPy
       this contention does not surface, so limiting OpenBLAS threads only helps on v1.15+.
       Reference: https://github.com/scipy/scipy/issues/22438

    ``torch.set_num_interop_threads()`` is intentionally NOT used: it can be called only
    once per process, so a library cannot rely on it, and combining it with the above would
    risk an inconsistency between the Python API and the underlying C library state.

    TODO: Watch SciPy and Torch updates for threading behavior changes.
    TODO: Benchmark optimization speed for new releases.
    """
    limit_openblas = Version(scipy.__version__) >= Version("1.15.0")
    old_openblas_val = os.environ.get("OPENBLAS_NUM_THREADS")
    if limit_openblas:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    old_torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old_torch_threads)
        if limit_openblas:
            if old_openblas_val is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = old_openblas_val
