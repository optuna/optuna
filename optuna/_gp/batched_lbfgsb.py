from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as so

from optuna._imports import try_import


with try_import() as _imports:
    from greenlet import greenlet

if TYPE_CHECKING:
    from typing import Protocol

    class FuncAndGrad(Protocol):
        def __call__(
            self, x: np.ndarray, unconverged_batch_indices: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            raise NotImplementedError


def _batched_lbfgsb(
    func_and_grad: FuncAndGrad,
    x0_batched: np.ndarray,
    bounds: np.ndarray | None,
    m: int,
    factr: float,
    pgtol: float,
    max_evals: int,
    max_iters: int,
    max_line_search: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert x0_batched.ndim == 2
    batch_size, dimension = x0_batched.shape
    assert bounds.shape == (dimension, 2)

    x_opts = [None] * batch_size
    fval_opts = [None] * batch_size
    n_iterations = [None] * batch_size

    def run(i: int) -> None:
        # This wrapper will be passed to fmin_l_bfgs_b.
        # It receives the current point `x` and additional arguments `args`.
        def func(x: np.ndarray, *args) -> tuple[float, np.ndarray]:
            assert x.shape == (dimension,)
            # Pass the current point `x` and its original index `i` to the parent greenlet.
            y, grad = greenlet.getcurrent().parent.switch((x, i))
            assert grad.shape == (dimension,)
            return y, grad

        x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=func,
            x0=x0_batched[i],
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            maxfun=max_evals,
            maxiter=max_iters,
            maxls=max_line_search,
            iprint=-1,
        )
        x_opts[i] = x_opt
        fval_opts[i] = fval_opt
        n_iterations[i] = info["nit"]

    # Collect initial requests
    requests = []
    greenlets = []
    for i in range(batch_size):
        gl = greenlet(run)
        # The first switch starts the greenlet and passes the index `i`.
        # It returns a tuple (x, original_index)
        req = gl.switch(i)
        if req is None:  # The greenlet finished without requesting evaluation.
            continue
        requests.append(req)
        greenlets.append(gl)

    while len(requests) > 0:
        # Unzip requests into points and their original indices
        xs, unconverged_indices = zip(*requests)
        unconverged_indices = np.array(unconverged_indices)

        # Batch evaluation
        ys, grads = func_and_grad(
            np.stack(xs),
            unconverged_indices,
        )
        assert ys.shape == (len(xs),)
        assert grads.shape == (len(xs), dimension)

        # Distribute results and collect next requests
        requests_next = []
        greenlets_next = []
        for j, gl in enumerate(greenlets):
            # Pass the evaluation result (y, grad) to the waiting greenlet.
            req = gl.switch((ys[j], grads[j]))
            if req is None:  # The greenlet has finished.
                continue
            requests_next.append(req)
            greenlets_next.append(gl)
        requests = requests_next
        greenlets = greenlets_next

    x_opts = np.array(x_opts)
    fval_opts = np.array(fval_opts)
    n_iterations = np.array(n_iterations)
    return x_opts, fval_opts, n_iterations


def batched_lbfgsb(
    func_and_grad: FuncAndGrad,
    x0_batched: np.ndarray,
    bounds: np.ndarray | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def func_and_grad_1D_wrapper(
        scaled_x: np.ndarray, batch_index: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """A wrapper for `func_and_grad` to handle 1D inputs.

        This is used as a fallback to sequential optimization when the batched
        L-BFGS-B is not available. It adapts the batched `func_and_grad` for
        use with optimizers like `scipy.optimize.fmin_l_bfgs_b` that expect
        a 1D input array.
        """
        assert scaled_x.ndim == 1
        unconverged_batch_indices = np.array([batch_index])
        fval, grad = func_and_grad(scaled_x[None], unconverged_batch_indices)
        return fval.item(), grad.ravel()

    if _imports.is_successful():
        scaled_cont_x_opts, neg_fval_opts, n_iterations = _batched_lbfgsb(
            func_and_grad=func_and_grad,
            x0_batched=x0_batched,
            bounds=bounds,
            m=m,
            factr=factr,
            pgtol=pgtol,
            max_evals=max_evals,
            max_iters=max_iters,
            max_line_search=max_line_search,
        )

    # fallback to sequential optimization if SciPy version is not supported
    else:
        scaled_cont_x_opts, neg_fval_opts, n_iterations = [], [], []
        for batch_index, x0 in enumerate(x0_batched):
            scaled_cont_x_opt, neg_fval_opt, info = so.fmin_l_bfgs_b(
                func=func_and_grad_1D_wrapper,
                x0=x0,
                args=(batch_index,),
                bounds=bounds,
                m=m,
                factr=factr,
                pgtol=pgtol,
                maxfun=max_evals,
                maxiter=max_iters,
                maxls=max_line_search,
            )
            scaled_cont_x_opts.append(scaled_cont_x_opt)
            neg_fval_opts.append(neg_fval_opt)
            n_iterations.append(info["nit"])
        scaled_cont_x_opts = np.array(scaled_cont_x_opts)
        neg_fval_opts = np.array(neg_fval_opts)
        n_iterations = np.array(n_iterations)

    return scaled_cont_x_opts, neg_fval_opts, n_iterations
