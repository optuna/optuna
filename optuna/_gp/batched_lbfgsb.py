"""
Please note that this is a port of the L-BFGS-B iplementation from SciPy and this implementation
may not work in future due to the dependency on the SciPy's internal API.

Heavily modified to adapt to Optuna by Shuhei Watanabe (2025) <shuhei.watanabe.utokyo@gmail.com>

License for the Python wrapper
==============================

Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from packaging.version import Version
from scipy import __version__ as scipy_version
from scipy.optimize import _lbfgsb as scipy_lbfgsb

if TYPE_CHECKING:
    from typing import Any, Protocol

    class FuncAndGrad(Protocol):
        def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raise NotImplementedError


status_messages = {
    0: "START",
    1: "NEW_X",
    2: "RESTART",
    3: "FG",
    4: "CONVERGENCE",
    5: "STOP",
    6: "WARNING",
    7: "ERROR",
    8: "ABNORMAL",
}
task_messages = {
    0: "",
    301: "",
    302: "",
    401: "NORM OF PROJECTED GRADIENT <= PGTOL",
    402: "RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    501: "CPU EXCEEDING THE TIME LIMIT",
    502: "TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    503: "PROJECTED GRADIENT IS SUFFICIENTLY SMALL",
    504: "TOTAL NO. OF ITERATIONS REACHED LIMIT",
    505: "CALLBACK REQUESTED HALT",
    601: "ROUNDING ERRORS PREVENT PROGRESS",
    602: "STP = STPMAX",
    603: "STP = STPMIN",
    604: "XTOL TEST SATISFIED",
    701: "NO FEASIBLE SOLUTION",
    702: "FACTR < 0",
    703: "FTOL < 0",
    704: "GTOL < 0",
    705: "XTOL < 0",
    706: "STP < STPMIN",
    707: "STP > STPMAX",
    708: "STPMIN < 0",
    709: "STPMAX < STPMIN",
    710: "INITIAL G >= 0",
    711: "M <= 0",
    712: "N <= 0",
    713: "INVALID NBD",
}
_is_lbfgsb_fortran = Version(scipy_version) < Version("1.15.0")


class _DataConstantInPython:
    def __init__(
        self,
        batch_size: int,
        bounds: np.ndarray,
        m: int,
        factr: float,
        pgtol: float,
        maxls: int,
    ) -> None:
        # See https://github.com/scipy/scipy/blob/v1.15.0/scipy/optimize/__lbfgsb.c
        self._csave: np.ndarray | None
        self._ln_task: np.ndarray | None
        int_type = scipy_lbfgsb.types.intvar.dtype if _is_lbfgsb_fortran else np.int32
        if _is_lbfgsb_fortran:
            self._ln_task = None
            self._csave = np.zeros((batch_size, 1), dtype="S60")
        else:
            self._ln_task = np.zeros((batch_size, 2), dtype=int_type)
            self._csave = None

        self._m = m
        self._factr = factr
        self._pgtol = pgtol
        self._maxls = maxls
        dim = len(bounds[0])
        self._l = np.where(np.isinf(bounds[0]), 0.0, bounds[0])  # Lower bounds
        self._u = np.where(np.isinf(bounds[1]), 0.0, bounds[1])  # Upper bounds
        bounds_map = {
            (False, False): 0,
            (True, False): 1,
            (True, True): 2,
            (False, True): 3,
        }
        self._nbd = np.array(
            [bounds_map[if1, if2] for if1, if2 in zip(*np.isfinite(bounds).tolist())],
            dtype=int_type,
        )
        self._wa = np.zeros(
            (batch_size, 2 * m * dim + 5 * dim + 11 * m**2 + 8 * m), dtype=np.float64
        )
        self._iwa = np.zeros((batch_size, 3 * dim), dtype=int_type)
        self._lsave = np.zeros((batch_size, 4), dtype=int_type)
        self._isave = np.zeros((batch_size, 44), dtype=int_type)
        self._dsave = np.zeros((batch_size, 29), dtype=np.float64)

    def lbfgsb_args(
        self,
        batch_id: int,
        task_status: np.ndarray,
        x: np.ndarray,
        f: np.ndarray,
        g: np.ndarray,
    ) -> Any:
        return (
            self._m,
            x,
            self._l,
            self._u,
            self._nbd,
            f,
            g,
            self._factr,
            self._pgtol,
            self._wa[batch_id],
            self._iwa[batch_id],
            task_status,
            *((-1,) if _is_lbfgsb_fortran else ()),
            *((self._csave[batch_id],) if self._csave is not None else ()),
            self._lsave[batch_id],
            self._isave[batch_id],
            self._dsave[batch_id],
            self._maxls,
            *((self._ln_task[batch_id],) if self._ln_task is not None else ()),
        )


class _TaskStatusManager:
    def __init__(self, batch_size: int, max_iters: int, max_evals: int) -> None:
        if _is_lbfgsb_fortran:
            self.task_status = np.full((batch_size, 1), status_messages[0], dtype="S60")
        else:
            self.task_status = np.zeros((batch_size, 2), dtype=np.int32)

        self.is_batch_terminated = np.zeros(batch_size, dtype=bool)
        self._max_iters = max_iters
        self._max_evals = max_evals
        self._batch_size = batch_size
        self._n_iterations = np.zeros(batch_size, dtype=int)
        self._n_evals = np.zeros(batch_size, dtype=int)

    def _update_task_status(self, batch_id: int, status_id: int, task_id: int) -> None:
        if _is_lbfgsb_fortran:
            self.task_status[batch_id] = (
                f"{status_messages[status_id]}: {task_messages[task_id]}"
            )
        else:
            self.task_status[batch_id] = [status_id, task_id]

    def _judge_status(self, batch_id: int, status_id: int) -> bool:
        if _is_lbfgsb_fortran:
            expected_msg = status_messages[status_id].encode()
            return self.task_status[batch_id].tobytes().startswith(expected_msg)
        else:
            return bool(self.task_status[batch_id, 0] == status_id)

    def reach_iter_limit(self, batch_id: int) -> bool:
        if reach_limit := self._n_iterations[batch_id] >= self._max_iters:
            self.is_batch_terminated[batch_id] = True
            self._update_task_status(batch_id, 5, 504)
        return reach_limit

    def reach_eval_limit(self, batch_id: int) -> bool:
        if reach_limit := self._n_evals[batch_id] > self._max_evals:
            self.is_batch_terminated[batch_id] = True
            self._update_task_status(batch_id, 5, 502)
        return reach_limit

    def should_evaluate(self, batch_id: int) -> bool:
        if should_evaluate := self._judge_status(batch_id, status_id=3):
            self._n_evals[batch_id] += 1
        return should_evaluate

    def should_terminate_batch(self, batch_id: int) -> bool:
        b = batch_id
        if self.is_batch_terminated[b]:
            return True
        elif self._judge_status(b, status_id=1):  # New parameter suggested.
            self._n_iterations[b] += 1  # This timing follows SciPy.
            self.is_batch_terminated[b] = self.reach_iter_limit(
                b
            ) or self.reach_eval_limit(b)
        elif not self._judge_status(b, status_id=0) and not self._judge_status(
            b, status_id=3
        ):
            # 0: Start, 3: Next function evaluation.
            self.is_batch_terminated[b] = True
        return self.is_batch_terminated[b]

    @property
    def info(self) -> dict[str, list[bool] | list[int] | list[str]]:
        messages = [
            (
                ts.tobytes().decode().rstrip("\x00")
                if _is_lbfgsb_fortran
                else f"{status_messages[ts[0]]}: {task_messages[ts[1]]}"
            )
            for ts in self.task_status
        ]
        return {
            "is_converged": [
                self._judge_status(b, status_id=4) for b in range(self._batch_size)
            ],
            "n_iterations": self._n_iterations.tolist(),
            "n_evals": self._n_evals.tolist(),
            "messages": messages,
        }


def batched_lbfgsb(
    func_and_grad: FuncAndGrad,
    x0: np.ndarray,
    bounds: np.ndarray | None = None,
    m: int = 10,
    factr: float = 1e7,
    pgtol: float = 1e-5,
    max_evals: int = 15000,
    max_iters: int = 15000,
    max_line_search: int = 20,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[bool] | list[int] | list[str]]]:
    """
    Minimize a batched function of one or more variables using the L-BFGS-B algorithm.

    Args:
        x0:
            Initial guess with the shape of (batch_size, dimension).
        bounds:
            The lower and upper bounds of each parameter with the shape of (dimension, 2).
            ``(min, max)`` pairs for each element in ``x``, defining the bounds on that parameter.
            Use None or +-inf when there is no bound in that direction.
        m:
            The maximum number of variable metric corrections used to define the limited memory
            matrix. (The limited memory BFGS method does not store the full Hessian but uses this
            many terms in an approximation to it.)
        factr:
            The iteration stops when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
            where ``eps`` is the machine precision, which is automatically generated by the code.
            Typical values for `factr` are: 1e12 for low accuracy; 1e7 for moderate accuracy; 10.0
            for extremely high accuracy.
        pgtol:
            The iteration will stop when ``max{|proj g_i | i = 1, ..., n} <= pgtol`` where
            ``proj g_i`` is the i-th component of the projected gradient.
        max_evals:
            Maximum number of function evaluations before minimization terminates.
        max_iters:
            Maximum number of algorithm iterations.
        max_line_search:
            Maximum number of line search steps (per iteration). Default is 20.

    Returns:
        A tuple containing:
        - The optimized parameters with the shape of (batch_size, dimension).
        - The function values at the optimized parameters with the shape of (batch_size,).
        - A dictionary containing convergence information, including:
            - `is_converged`: A list of booleans indicating whether each batch converged.
            - `n_iterations`: A list of the number of iterations for each batch.
            - `n_evals`: A list of the number of function evaluations for each batch.
            - `messages`: A list of messages indicating the status of each batch.

    Notes:
        SciPy uses a C-translated and modified version of the Fortran code, L-BFGS-B v3.0
        (released April 25, 2011, BSD-3 licensed). Original Fortran version was written by
        Ciyou Zhu, Richard Byrd, Jorge Nocedal and, Jose Luis Morales.

    References:
        * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained
        Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5,
        pp. 1190-1208.
        * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B, FORTRAN routines
        for large scale bound constrained optimization (1997), ACM Transactions on
        Mathematical Software, 23, 4, pp. 550 - 560.
        * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN
        routines for large scale bound constrained optimization (2011), ACM Transactions on
        Mathematical Software, 38, 1.
    """
    batched_x = x0.reshape(-1, (original_x_shape := x0.shape)[-1]).copy()
    b_indices = np.arange((batch_size := len(batched_x)), dtype=int)
    bounds = (
        np.array([[-np.inf, np.inf]] * x0.shape[-1]).T if bounds is None else bounds.T
    )
    data = _DataConstantInPython(batch_size, bounds, m, factr, pgtol, max_line_search)
    f_vals = np.zeros(batch_size, dtype=np.float64)
    grads = np.zeros_like(batched_x, dtype=np.float64)
    tm = _TaskStatusManager(batch_size, max_iters=max_iters, max_evals=max_evals)
    while (batch_indices := b_indices[~tm.is_batch_terminated]).size:
        f_vals[batch_indices], grads[batch_indices] = func_and_grad(
            batched_x[batch_indices]
        )
        for b in batch_indices:
            lbfgsb_args = data.lbfgsb_args(
                b, tm.task_status[b], batched_x[b], f_vals[b], grads[b]
            )
            while not tm.should_terminate_batch(b):
                scipy_lbfgsb.setulb(
                    *lbfgsb_args
                )  # x,f,g,task_status will be updated inplace.
                if tm.should_evaluate(b):
                    break
    return (
        batched_x.reshape(original_x_shape),
        f_vals.reshape(original_x_shape[:-1]),
        tm.info,
    )
