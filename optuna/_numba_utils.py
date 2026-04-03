"""Utilities for optional numba JIT compilation support.

When numba is installed, performance-critical numerical functions can be
JIT-compiled for significant speedups. When numba is not available, all
code falls back to the pure NumPy/Python implementations transparently.

Install numba for acceleration::

    pip install optuna[numba]
"""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import TypeVar


F = TypeVar("F", bound=Callable[..., Any])

try:
    import numba  # type: ignore[import-untyped]

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None  # type: ignore[assignment]


def njit(**kwargs: Any) -> Callable[[F], F]:
    """Decorator that applies ``numba.njit`` when numba is available, otherwise is a no-op."""
    if HAS_NUMBA:
        return numba.njit(**kwargs)  # type: ignore[no-any-return]

    def _passthrough(fn: F) -> F:
        return fn

    return _passthrough


def numba_vectorize(signatures: list[str] | None = None, **kwargs: Any) -> Callable[[F], F]:
    """Decorator that applies ``numba.vectorize`` when available, otherwise is a no-op."""
    if HAS_NUMBA:
        return numba.vectorize(signatures, **kwargs)  # type: ignore[no-any-return]

    def _passthrough(fn: F) -> F:
        return fn

    return _passthrough
