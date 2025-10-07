from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna._imports import try_import
from optuna.logging import get_logger


with try_import() as _greenlet_imports:
    from greenlet import greenlet

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import scipy.optimize as so

else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")


if not _greenlet_imports.is_successful():
    _logger = get_logger(__name__)
    _logger.warning(
        "The 'greenlet' package is unavailable, falling back to sequential L-BFGS-B optimization. "
        "This may lead to slower suggestions."
    )


def _batched_lbfgsb(
    func_and_grad: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,
    args_tuple: tuple[Any, ...],
    bounds: list[tuple[float, float]] | None,
    m: int,
    factr: float,
    pgtol: float,
    max_evals: int,
    max_iters: int,
    max_line_search: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x0_batched.ndim != 2:
        raise ValueError("x0_batched must be 2D array.")
    batch_size = len(x0_batched)
    xs_opt = np.empty_like(x0_batched)
    fvals_opt = np.empty(batch_size, dtype=float)
    n_iterations = np.empty(batch_size, dtype=int)

    def run(i: int) -> None:
        def _func_and_grad(x: np.ndarray, *args: Any) -> tuple[float, np.ndarray]:
            fval, grad = greenlet.getcurrent().parent.switch(x, args)
            # NOTE(nabenabe): copy is necessary to convert grad to writable.
            return float(fval), grad.copy()

        x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=_func_and_grad,
            x0=x0_batched[i],
            args=tuple(arg[i] for arg in args_tuple),
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            maxfun=max_evals,
            maxiter=max_iters,
            maxls=max_line_search,
        )
        xs_opt[i] = x_opt
        fvals_opt[i] = fval_opt
        n_iterations[i] = info["nit"]

    greenlets = [greenlet(run) for _ in range(batch_size)]
    x_and_argsT_pairs = [gl.switch(i) for i, gl in enumerate(greenlets)]

    while x_and_argsT_pairs:
        x_batched_list = [pair[0] for pair in x_and_argsT_pairs if pair is not None]
        args_tuple_transposed = [pair[1] for pair in x_and_argsT_pairs if pair is not None]
        x_batched = np.array(x_batched_list)
        current_args_tuple = tuple(zip(*args_tuple_transposed))
        fvals, grads = func_and_grad(x_batched, *current_args_tuple)
        results = [gl.switch((fvals[i], grads[i])) for i, gl in enumerate(greenlets)]
        live_pairs_and_greenlets = [
            (pair, gl) for pair, gl in zip(results, greenlets) if pair is not None
        ]
        if not live_pairs_and_greenlets:
            break
        x_and_argsT_pairs, greenlets = map(list, zip(*live_pairs_and_greenlets))

    return xs_opt, fvals_opt, n_iterations


def batched_lbfgsb(
    func_and_grad: Callable[..., tuple[np.ndarray, np.ndarray]],
    x0_batched: np.ndarray,  # shape (batch_size, dim)
    args_tuple: tuple[Any, ...] = (),
    bounds: list[tuple[float, float]] | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Batched L-BFGS-B optimization with/without greenlet.
    - `func_and_grad` is expected to take a 2D array as the first argument and return a tuple of
      a 1D array of function values and a 2D array of gradients.
    - `x0_batched` is a 2D array where each row is an initial point for optimization.
    - `args_tuple` is a tuple of additional arguments to pass to `func_and_grad`. e.g., if
      `args_tuple` is `([alpha1, ..., alphaB], [beta1, ..., betaB])`, then
      `func_and_grad` is called as
      `func_and_grad(x0_batched, [alpha1, ..., alphaB], [beta1, ..., betaB])`. Note that each
      argument in `args_tuple` is expected to be a list of length `B` (batch size).
    """
    if x0_batched.ndim != 2:
        raise ValueError("x0_batched must be 2D array.")

    if _greenlet_imports.is_successful() and len(x0_batched) > 1:
        # NOTE(Kaichi-Irie): when batch size is 1, using greenlet causes context-switch overhead.
        xs_opt, fvals_opt, n_iterations = _batched_lbfgsb(
            func_and_grad=func_and_grad,
            x0_batched=x0_batched,
            args_tuple=args_tuple,
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            max_evals=max_evals,
            max_iters=max_iters,
            max_line_search=max_line_search,
        )

    # fall back to sequential optimization if greenlet is not available.
    else:

        def _func_and_grad_wrapper(x_1d: np.ndarray, *args_1d: Any) -> tuple[float, np.ndarray]:
            if x_1d.ndim != 1:
                raise ValueError("x must be 1D array.")
            args_2d = ([arg] for arg in args_1d)
            x_2d = x_1d[None, :]  # (dim,) -> (1, dim)
            fval, grad = func_and_grad(x_2d, *args_2d)
            return fval.item(), grad[0].copy()

        xs_opt = np.empty_like(x0_batched)
        fvals_opt = np.empty(x0_batched.shape[0], dtype=float)
        n_iterations = np.empty(x0_batched.shape[0], dtype=int)
        for i, x0 in enumerate(x0_batched):
            xs_opt[i], fvals_opt[i], info = so.fmin_l_bfgs_b(
                func=_func_and_grad_wrapper,
                x0=x0,
                args=tuple(arg[i] for arg in args_tuple),
                bounds=bounds,
                m=m,
                factr=factr,
                pgtol=pgtol,
                maxfun=max_evals,
                maxiter=max_iters,
                maxls=max_line_search,
            )
            n_iterations[i] = info["nit"]

    return xs_opt, fvals_opt, n_iterations
