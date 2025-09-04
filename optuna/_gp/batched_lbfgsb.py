# %%

import numpy as np
import scipy.optimize as so
from greenlet import greenlet


def batched_lbfgsb(
    func_and_grad,
    x0_batched: np.ndarray,
    bounds: np.ndarray,
    pgtol: float,
    max_iters: int,
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
            pgtol=pgtol,
            maxiter=max_iters,
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
