from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer
from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so

    from optuna._gp.acqf import BaseAcquisitionFunc
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")

_logger = get_logger(__name__)


def _gradient_ascent(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, float, bool]:
    """
    This function optimizes the acquisition function using preconditioning.
    Preconditioning equalizes the variances caused by each parameter and
    speeds up the convergence.

    In Optuna, acquisition functions use Matern 5/2 kernel, which is a function of `x / l`
    where `x` is `normalized_params` and `l` is the corresponding lengthscales.
    Then acquisition functions are a function of `x / l`, i.e. `f(x / l)`.
    As `l` has different values for each param, it makes the function ill-conditioned.
    By transforming `x / l` to `zl / l = z`, the function becomes `f(z)` and has
    equal variances w.r.t. `z`.
    So optimization w.r.t. `z` instead of `x` is the preconditioning here and
    speeds up the convergence.
    As the domain of `x` is [0, 1], that of `z` becomes [0, 1/l].
    """
    if len(continuous_indices) == 0:
        return initial_params, initial_fval, False
    normalized_params = initial_params.copy()

    def negative_acqf_with_grad(scaled_x: np.ndarray) -> tuple[float, np.ndarray]:
        # Scale back to the original domain, i.e. [0, 1], from [0, 1/s].
        normalized_params[continuous_indices] = scaled_x * lengthscales
        (fval, grad) = acqf.eval_acqf_with_grad(normalized_params)
        # Flip sign because scipy minimizes functions.
        # Let the scaled acqf be g(x) and the acqf be f(sx), then dg/dx = df/dx * s.
        return -fval, -grad[continuous_indices] * lengthscales

    with single_blas_thread_if_scipy_v1_15_or_newer():
        scaled_cont_x_opt, neg_fval_opt, info = so.fmin_l_bfgs_b(
            func=negative_acqf_with_grad,
            x0=normalized_params[continuous_indices] / lengthscales,
            bounds=[(0, 1 / s) for s in lengthscales],
            pgtol=math.sqrt(tol),
            maxiter=200,
        )

    if -neg_fval_opt > initial_fval and info["nit"] > 0:  # Improved.
        # `nit` is the number of iterations.
        normalized_params[continuous_indices] = scaled_cont_x_opt * lengthscales
        return normalized_params, -neg_fval_opt, True

    return initial_params, initial_fval, False  # No improvement.


def _exhaustive_search(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    choices_except_current = choices[choices != initial_params[param_idx]]

    all_params = np.repeat(initial_params[None, :], len(choices_except_current), axis=0)
    all_params[:, param_idx] = choices_except_current
    fvals = acqf.eval_acqf_no_grad(all_params)
    best_idx = np.argmax(fvals)

    if fvals[best_idx] > initial_fval:  # Improved.
        return all_params[best_idx, :], fvals[best_idx], True

    return initial_params, initial_fval, False  # No improvement.


def _discrete_line_search(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    grids: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, float, bool]:
    if len(grids) == 1:
        # Do not optimize anything when there's only one choice.
        return initial_params, initial_fval, False

    def find_nearest_index(x: float) -> int:
        i = int(np.clip(np.searchsorted(grids, x), 1, len(grids) - 1))
        return i - 1 if abs(x - grids[i - 1]) < abs(x - grids[i]) else i

    current_choice_i = find_nearest_index(initial_params[param_idx])
    assert np.isclose(initial_params[param_idx], grids[current_choice_i])

    negative_fval_cache = {current_choice_i: -initial_fval}

    normalized_params = initial_params.copy()

    def negative_acqf_with_cache(i: int) -> float:
        # Function value at choices[i].
        cache_val = negative_fval_cache.get(i)
        if cache_val is not None:
            return cache_val
        normalized_params[param_idx] = grids[i]

        # Flip sign because scipy minimizes functions.
        negval = -float(acqf.eval_acqf_no_grad(normalized_params))
        negative_fval_cache[i] = negval
        return negval

    def interpolated_negative_acqf(x: float) -> float:
        if x < grids[0] or x > grids[-1]:
            return np.inf
        right = int(np.clip(np.searchsorted(grids, x), 1, len(grids) - 1))
        left = right - 1
        neg_acqf_left, neg_acqf_right = negative_acqf_with_cache(left), negative_acqf_with_cache(
            right
        )
        w_left = (grids[right] - x) / (grids[right] - grids[left])
        w_right = 1.0 - w_left
        return w_left * neg_acqf_left + w_right * neg_acqf_right

    EPS = 1e-12
    res = so.minimize_scalar(
        interpolated_negative_acqf,
        # The values of this bracket are (inf, -fval, inf).
        # This trivially satisfies the bracket condition if fval is finite.
        bracket=(grids[0] - EPS, grids[current_choice_i], grids[-1] + EPS),
        method="brent",
        tol=xtol,
    )
    opt_idx = find_nearest_index(res.x)
    fval_opt = -negative_acqf_with_cache(opt_idx)

    # We check both conditions because of numerical errors.
    if opt_idx != current_choice_i and fval_opt > initial_fval:
        normalized_params[param_idx] = grids[opt_idx]
        return normalized_params, fval_opt, True

    return initial_params, initial_fval, False  # No improvement.


def _local_search_discrete(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, float, bool]:

    # If the number of possible parameter values is small, we just perform an exhaustive search.
    # This is faster and better than the line search.
    MAX_INT_EXHAUSTIVE_SEARCH_PARAMS = 16

    is_categorical = acqf.search_space.is_categorical[param_idx]
    if is_categorical or len(choices) <= MAX_INT_EXHAUSTIVE_SEARCH_PARAMS:
        return _exhaustive_search(acqf, initial_params, initial_fval, param_idx, choices)
    else:
        return _discrete_line_search(acqf, initial_params, initial_fval, param_idx, choices, xtol)


def local_search_mixed(
    acqf: BaseAcquisitionFunc,
    initial_normalized_params: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    continuous_indices = acqf.search_space.continuous_indices

    # This is a technique for speeding up optimization.
    # We use an isotropic kernel, so scaling the gradient will make
    # the hessian better-conditioned.
    # NOTE: Ideally, separating lengthscales should be used for the constraint functions,
    # but for simplicity, the ones from the objective function are being reused.
    # TODO(kAIto47802): Think of a better way to handle this.
    lengthscales = acqf.length_scales[continuous_indices]

    choices_of_discrete_params = acqf.search_space.get_choices_of_discrete_params()

    discrete_xtols = [
        # Terminate discrete optimizations once the change in x becomes smaller than this.
        # Basically, if the change is smaller than min(dx) / 4, it is useless to see more details.
        np.min(np.diff(choices), initial=np.inf) / 4
        for choices in choices_of_discrete_params
    ]

    best_normalized_params = initial_normalized_params.copy()
    best_fval = float(acqf.eval_acqf_no_grad(best_normalized_params))

    CONTINUOUS = -1
    last_changed_param: int | None = None

    for _ in range(max_iter):
        if last_changed_param == CONTINUOUS:
            # Parameters not changed since last time.
            return best_normalized_params, best_fval
        (best_normalized_params, best_fval, updated) = _gradient_ascent(
            acqf,
            best_normalized_params,
            best_fval,
            continuous_indices,
            lengthscales,
            tol,
        )
        if updated:
            last_changed_param = CONTINUOUS

        for i, choices, xtol in zip(
            acqf.search_space.discrete_indices, choices_of_discrete_params, discrete_xtols
        ):
            if last_changed_param == i:
                # Parameters not changed since last time.
                return best_normalized_params, best_fval
            (best_normalized_params, best_fval, updated) = _local_search_discrete(
                acqf, best_normalized_params, best_fval, i, choices, xtol
            )
            if updated:
                last_changed_param = i

        if last_changed_param is None:
            # Parameters not changed from the beginning.
            return best_normalized_params, best_fval

    _logger.warning("local_search_mixed: Local search did not converge.")
    return best_normalized_params, best_fval


def optimize_acqf_mixed(
    acqf: BaseAcquisitionFunc,
    *,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:

    rng = rng or np.random.RandomState()

    if warmstart_normalized_params_array is None:
        warmstart_normalized_params_array = np.empty((0, acqf.search_space.dim))

    assert (
        len(warmstart_normalized_params_array) <= n_local_search - 1
    ), "We must choose at least 1 best sampled point + given_initial_xs as start points."

    sampled_xs = acqf.search_space.sample_normalized_params(n_preliminary_samples, rng=rng)

    # Evaluate all values at initial samples
    f_vals = acqf.eval_acqf_no_grad(sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)

    # TODO(nabenabe): Benchmark the BoTorch roulette selection as well.
    # https://github.com/pytorch/botorch/blob/v0.14.0/botorch/optim/initializers.py#L942
    # We use a modified roulette wheel selection to pick the initial param for each local search.
    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0  # We already picked the best param, so remove it from roulette.
    probs /= probs.sum()
    n_non_zero_probs_improvement = int(np.count_nonzero(probs > 0.0))
    # n_additional_warmstart becomes smaller when study starts to converge.
    n_additional_warmstart = min(
        n_local_search - len(warmstart_normalized_params_array) - 1, n_non_zero_probs_improvement
    )
    if n_additional_warmstart == n_non_zero_probs_improvement:
        _logger.warning("Study already converged, so the number of local search is reduced.")
    chosen_idxs = np.array([max_i])
    if n_additional_warmstart > 0:
        additional_idxs = rng.choice(
            len(sampled_xs), size=n_additional_warmstart, replace=False, p=probs
        )
        chosen_idxs = np.append(chosen_idxs, additional_idxs)

    best_x = sampled_xs[max_i, :]
    best_f = float(f_vals[max_i])

    for x_warmstart in np.vstack([sampled_xs[chosen_idxs, :], warmstart_normalized_params_array]):
        x, f = local_search_mixed(acqf, x_warmstart, tol=tol)
        if f > best_f:
            best_x = x
            best_f = f

    return best_x, best_f
