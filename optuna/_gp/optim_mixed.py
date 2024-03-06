from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.acqf import AcquisitionFunctionParams
from optuna._gp.acqf import eval_acqf_no_grad
from optuna._gp.acqf import eval_acqf_with_grad
from optuna._gp.search_space import normalize_one_param
from optuna._gp.search_space import sample_normalized_params
from optuna._gp.search_space import ScaleType
from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")

_logger = get_logger(__name__)


def _local_search_continuous(
    acqf_params: AcquisitionFunctionParams,
    initial_params: np.ndarray,
    initial_fval: float,
    continuous_params: np.ndarray,
    continuous_param_scale: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, float] | None:
    if len(continuous_params) == 0:
        return None
    normalized_params = initial_params.copy()

    def negfun_continuous_with_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        normalized_params[continuous_params] = x * continuous_param_scale
        (fval, grad) = eval_acqf_with_grad(acqf_params, normalized_params)
        # Flip sign because scipy minimizes functions.
        return (-fval, -grad[continuous_params] * continuous_param_scale)

    x_opt, fval_opt, info = so.fmin_l_bfgs_b(
        func=negfun_continuous_with_grad,
        x0=normalized_params[continuous_params] / continuous_param_scale,
        bounds=[(0, 1 / s) for s in continuous_param_scale],
        pgtol=math.sqrt(tol),
        maxiter=200,
    )

    if info["warnflag"] == 2:
        fval_opt = negfun_continuous_with_grad(x_opt)[0]

    if -fval_opt < initial_fval or info["nit"] == 0:
        # Return None if the optimization did not improve the value.
        # `nit` is the number of iterations.
        return None  # 
    else:
        normalized_params[continuous_params] = x_opt * continuous_param_scale
        return (normalized_params, -fval_opt)


def _local_search_discrete_exhaustive_search(
    acqf_params: AcquisitionFunctionParams,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    choices_except_current = choices[choices != initial_params[param_idx]]

    all_params = np.repeat(initial_params[None, :], len(choices_except_current), axis=0)
    all_params[:, param_idx] = choices_except_current
    fvals = eval_acqf_no_grad(acqf_params, all_params)
    best_idx = np.argmax(fvals)

    if fvals[best_idx] > initial_fval:
        return (all_params[best_idx, :], fvals[best_idx])
    else:
        return None


def _local_search_discrete_line_search(
    acqf_params: AcquisitionFunctionParams,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, float] | None:
    if len(choices) == 1:
        # Do not optimize anything when there's only one choice.
        return None

    def get_rounded_index(x: float) -> int:
        i = int(np.clip(np.searchsorted(choices, x), 1, len(choices) - 1))
        return i - 1 if abs(x - choices[i - 1]) < abs(x - choices[i]) else i

    current_choice_i = get_rounded_index(initial_params[param_idx])
    assert initial_params[param_idx] == choices[current_choice_i]

    negval_cache = {current_choice_i: -initial_fval}

    normalized_params = initial_params.copy()

    def inegfun_cached(i: int) -> float:
        nonlocal negval_cache, normalized_params
        # Function value at choices[i].
        cache_val = negval_cache.get(i)
        if cache_val is not None:
            return cache_val
        normalized_params[param_idx] = choices[i]

        # Flip sign because scipy minimizes functions.
        negval = -float(eval_acqf_no_grad(acqf_params, normalized_params))
        negval_cache[i] = negval
        return negval

    def negfun_interpolated(x: float) -> float:
        if x < choices[0] or x > choices[-1]:
            return np.inf
        i1 = int(np.clip(np.searchsorted(choices, x), 1, len(choices) - 1))
        i0 = i1 - 1

        f0, f1 = inegfun_cached(i0), inegfun_cached(i1)

        w0 = (choices[i1] - x) / (choices[i1] - choices[i0])
        w1 = 1.0 - w0

        return w0 * f0 + w1 * f1

    EPS = 1e-12
    res = so.minimize_scalar(
        negfun_interpolated,
        # The values of this bracket are (inf, -fval, inf).
        # This trivially satisfies the bracket condition if fval is finite.
        bracket=(choices[0] - EPS, choices[current_choice_i], choices[-1] + EPS),
        method="brent",
        tol=xtol,
    )
    i_star = get_rounded_index(res.x)
    fval_new = -inegfun_cached(i_star)

    # We check both conditions because of numerical errors.
    if i_star != current_choice_i and fval_new > initial_fval:
        normalized_params[param_idx] = choices[i_star]
        return (normalized_params, fval_new)
    else:
        return None


# If the number of possible parameter values is small, we just perform an exhaustive search.
# This is faster and better than the line search.
MAX_INT_EXHAUSTIVE_SEARCH_PARAMS = 16


def _local_search_discrete(
    acqf_params: AcquisitionFunctionParams,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, float] | None:
    scale_type = acqf_params.search_space.scale_types[param_idx]
    if scale_type == ScaleType.CATEGORICAL or len(choices) <= MAX_INT_EXHAUSTIVE_SEARCH_PARAMS:
        return _local_search_discrete_exhaustive_search(
            acqf_params, initial_params, initial_fval, param_idx, choices
        )
    else:
        return _local_search_discrete_line_search(
            acqf_params, initial_params, initial_fval, param_idx, choices, xtol
        )


def local_search_mixed(
    acqf_params: AcquisitionFunctionParams,
    initial_normalized_params: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> tuple[np.ndarray, float]:
    scale_types = acqf_params.search_space.scale_types
    bounds = acqf_params.search_space.bounds
    steps = acqf_params.search_space.steps

    continuous_params = np.where(steps == 0.0)[0]

    inverse_squared_lengthscales = (
        acqf_params.kernel_params.inverse_squared_lengthscales.detach().numpy()
    )
    # This is a technique for speeding up optimization.
    # We use an isotropic kernel, so scaling the gradient will make
    # the hessian better-conditioned.
    continuous_param_scale = 1 / np.sqrt(inverse_squared_lengthscales[continuous_params])

    noncontinuous_params = np.where(steps > 0)[0]
    noncontinuous_param_choices = [
        (
            np.arange(bounds[i, 1])
            if scale_types[i] == ScaleType.CATEGORICAL
            else normalize_one_param(
                param_value=np.arange(bounds[i, 0], bounds[i, 1] + 0.5 * steps[i], steps[i]),
                scale_type=ScaleType(scale_types[i]),
                bounds=(bounds[i, 0], bounds[i, 1]),
                step=steps[i],
            )
        )
        for i in noncontinuous_params
    ]

    noncontinuous_paramwise_xtol = [
        # Enough to discriminate between two choices.
        np.min(np.diff(choices), initial=np.inf) / 4
        for choices in noncontinuous_param_choices
    ]

    normalized_params = initial_normalized_params.copy()
    fval = float(eval_acqf_no_grad(acqf_params, normalized_params))

    NEVER = -1
    CONTINUOUS = -2
    last_changed_param = NEVER

    for _ in range(max_iter):
        if last_changed_param == CONTINUOUS:
            # Parameters not changed since last time.
            return (normalized_params, fval)
        res = _local_search_continuous(
            acqf_params,
            normalized_params,
            fval,
            continuous_params,
            continuous_param_scale,
            tol,
        )
        if res is not None:
            normalized_params, fval = res
            last_changed_param = CONTINUOUS

        for i, choices, xtol in zip(
            noncontinuous_params, noncontinuous_param_choices, noncontinuous_paramwise_xtol
        ):
            if last_changed_param == i:
                # Parameters not changed since last time.
                return (normalized_params, fval)
            res = _local_search_discrete(acqf_params, normalized_params, fval, i, choices, xtol)
            if res is not None:
                normalized_params, fval = res
                last_changed_param = i

        if last_changed_param == NEVER:
            # Parameters not changed from the beginning.
            return (normalized_params, fval)

    _logger.warn("local_search_mixed: Local search did not converge.")
    return (normalized_params, fval)


def optimize_acqf_mixed(
    acqf_params: AcquisitionFunctionParams,
    *,
    given_initial_xs: np.ndarray | None = None,
    n_additional_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, float]:

    rng = rng or np.random.RandomState()

    dim = acqf_params.search_space.scale_types.shape[0]
    if given_initial_xs is None:
        given_initial_xs = np.empty((0, dim))

    assert (
        len(given_initial_xs) <= n_local_search - 1
    ), "We must choose at least 1 best sampled point + given_initial_xs as start points."

    sampled_xs = sample_normalized_params(n_additional_samples, acqf_params.search_space, rng=rng)

    # Evaluate all values at initial samples
    f_vals = eval_acqf_no_grad(acqf_params, sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    max_i = np.argmax(f_vals)

    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0
    probs /= probs.sum()
    remaining_idxs = rng.choice(
        len(sampled_xs),
        size=n_local_search - len(given_initial_xs) - 1,
        replace=False,
        p=probs,
    )

    best_x = sampled_xs[max_i, :]
    best_f = float(f_vals[max_i])

    for x_guess in np.vstack(
        [sampled_xs[max_i, :], sampled_xs[remaining_idxs, :], given_initial_xs]
    ):
        x, f = local_search_mixed(acqf_params, x_guess, tol=tol)
        if f > best_f:
            best_x = x
            best_f = f

    return best_x, best_f
