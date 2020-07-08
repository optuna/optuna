from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import optimize as scipy_optimize

from optuna.samplers._gp.optimizer.base import BaseOptimizer


class ScipyOptimizer(BaseOptimizer):
    """ An acquisition function optimizer which uses the `scipy.optimize`.

    .. note::
        The default optimization algorithm is L-BFGS. Please see the original paper:
        [*] D. C. Liu and J. Nocedal. On the limited memory BFGS method for large scale
        optimization. Mathematical Programming, 45(1):503–528, Aug 1989.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        maxiter: int = 1000,
        method: str = 'L-BFGS-B',
        n_samples_for_anchor: int = 1000,
        n_anchor: int = 5,
    ):

        self._bounds = bounds
        self._maxiter = maxiter
        self._method = method
        self._n_samples_for_anchor = n_samples_for_anchor
        self._n_anchor = n_anchor

    def optimize(
            self, f: Callable[[Any], Any], df: Optional[Callable[[Any], Any]] = None
    ) -> np.ndarray:

        # Change the optimization problem from maximization to minimization for scipy.optimize.
        obj = lambda x: - f(x)
        if df is not None:
            dobj = lambda x: - df(x)
        else:
            dobj = None

        anchor_points = self._suggest_anchor_points(f=obj)
        optimized_points = []
        for i, a in enumerate(anchor_points):
            res = self._optimize_with_x0(a, f=obj, df=dobj)
            optimized_points.append(res)

        x_max, _ = min(optimized_points, key=lambda x: x[1])
        return x_max

    def _suggest_anchor_points(self, f: Callable[[Any], Any]) -> np.ndarray:
        """Suggests the anchor points (initial design points for the acquisition optimizer).

        The suggestion algorithm is based on the objective function values in ascending order.

        Args:
            f:
                A objective function.

        Returns:
            Suggested anchor points. The shape is (n_anchor, input_dim).
        """

        candidate_points = np.zeros((self._n_samples_for_anchor, len(self._bounds)))
        for i in range(len(self._bounds)):
            candidate_points[:, i] = np.random.uniform(low=self._bounds[i][0], high=self._bounds[i][1], size=self._n_samples_for_anchor)
        assert candidate_points.ndim == 2
        assert candidate_points.shape[0] == self._n_samples_for_anchor

        scores = f(candidate_points).flatten()
        assert scores.ndim == 1

        anchor_points = candidate_points[np.argsort(scores)[:min(len(scores), self._n_anchor)], :]
        assert anchor_points.ndim == 2
        assert anchor_points.shape[0] == self._n_anchor or anchor_points.shape[0] == len(scores)

        return anchor_points

    def _optimize_with_x0(self, x0: np.ndarray, f: Callable[[Any], Any], df: Optional[Callable[[Any], Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize (minimize) the given objective function f.

        Args:
            x0:
                An initial point for optimization. The shape is (input_dim,).
            f:
                A objective function.
            df:
                A gradient of the objective function.

        Returns:
            The tuple of the optimized point and the objective value.
        """

        res = scipy_optimize.minimize(fun=f, x0=x0, method=self._method, jac=df, bounds=self._bounds, options={'maxiter': self._maxiter})

        if res.success:
            x = np.atleast_2d(res.x)
            fx = np.atleast_2d(res.fun)
        else:
            x = np.atleast_2d(x0)
            fx = np.atleast_2d(f(x0))

        return x, fx
