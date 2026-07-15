from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING

from packaging.version import Version


if TYPE_CHECKING:
    from typing import Generator

    import scipy
    import torch
else:
    from optuna import _LazyImport

    scipy = _LazyImport("scipy")
    torch = _LazyImport("torch")


@contextmanager
def limit_threads_in_optimization() -> Generator[None, None, None]:
    """
    Context manager to limit threading for L-BFGS-B optimization.

    L-BFGS-B is inherently sequential and performs poorly with excessive threading.
    This context manager addresses two distinct threading issues on Linux:

    Issue 1: Torch and OpenMP (libgomp)
    ====================================
    Torch uses OpenMP (libgomp) which creates excessive threads by default on Linux.
    This causes severe performance degradation due to context switching.
    Setting torch.set_num_threads(1) mitigates significant threading overhead and provides
    effective performance improvement (~30-50%).

    Issue 2: SciPy v1.15+ and OpenBLAS
    ===================================
    SciPy v1.15.0 changed L-BFGS-B implementation from Fortran to OpenBLAS.
    OpenBLAS also creates excessive threads by default, causing performance degradation.
    Solution: Limit OPENBLAS_NUM_THREADS to 1 (SciPy v1.15+ only).
    Reference: https://github.com/scipy/scipy/issues/22438

    Implementation:
    ===============
    1. torch.set_num_threads(1): Always applied (safe, reversible)
    2. OPENBLAS_NUM_THREADS=1: Applied only for SciPy v1.15+ (when OpenBLAS is used)

    Note: torch.set_num_interop_threads() is NOT used to avoid internal inconsistency
    between Python API and C library state.

    TODO: Watch SciPy and Torch updates for threading behavior changes.
    TODO: Benchmark optimization speed for new releases.
    NOTE: fmin_l_bfgs_b in optim_mixed.py seems unaffected by this limitation.
    """
    old_openblas_val = os.environ.get("OPENBLAS_NUM_THREADS")
    old_torch_threads = None

    # Limit OPENBLAS threads for SciPy v1.15+
    if Version(scipy.__version__) >= Version("1.15.0"):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    try:
        # Limit torch threads to avoid performance degradation
        old_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
    finally:
        # Restore OPENBLAS_NUM_THREADS if modified
        if Version(scipy.__version__) >= Version("1.15.0"):
            if old_openblas_val is None:
                os.environ.pop("OPENBLAS_NUM_THREADS", None)
            else:
                os.environ["OPENBLAS_NUM_THREADS"] = old_openblas_val

        # Restore torch threads
        if old_torch_threads is not None:
            try:
                torch.set_num_threads(old_torch_threads)
            except Exception:
                pass
