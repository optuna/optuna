"""Microbenchmarks comparing numba-accelerated vs pure-Python/NumPy paths.

Run: python3 benchmarks/bench_numba.py

Each benchmark runs the function with numba enabled, then with HAS_NUMBA
monkeypatched to False (forcing the fallback), and reports timings + speedup.
"""
from __future__ import annotations

import time
from unittest import mock

import numpy as np


def _time_fn(fn, *args, warmup: int = 2, repeats: int = 10, **kwargs) -> float:
    """Time a function, returning median wall-clock seconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


_MODULES_USING_NUMBA = [
    "optuna._numba_utils",
    "optuna.samplers._tpe._erf",
    "optuna.samplers._tpe._truncnorm",
    "optuna._hypervolume.wfg",
    "optuna.study._multi_objective",
]


def _patch_no_numba():
    """Context manager that disables numba in all optuna modules."""
    import importlib

    patches = {}
    for mod_name in _MODULES_USING_NUMBA:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "HAS_NUMBA"):
                patches[mod_name + ".HAS_NUMBA"] = False
        except ImportError:
            pass
    return mock.patch.multiple("", **{}) if not patches else _multi_patch(patches)


class _multi_patch:
    """Apply multiple monkeypatches as a context manager."""

    def __init__(self, patches: dict[str, object]):
        self._patchers = [mock.patch(target, value) for target, value in patches.items()]

    def __enter__(self):
        for p in self._patchers:
            p.start()

    def __exit__(self, *args):
        for p in self._patchers:
            p.stop()


def _run_bench(name: str, fn_numba, fn_fallback, *args, warmup: int = 3, repeats: int = 20):
    """Run a single benchmark comparing numba vs fallback."""
    t_numba = _time_fn(fn_numba, *args, warmup=warmup, repeats=repeats)
    t_fallback = _time_fn(fn_fallback, *args, warmup=warmup, repeats=repeats)
    speedup = t_fallback / t_numba if t_numba > 0 else float("inf")
    print(f"  {name:<50s}  numba: {t_numba*1e3:8.3f} ms  fallback: {t_fallback*1e3:8.3f} ms  speedup: {speedup:6.2f}x")
    return t_numba, t_fallback


# =============================================================================
# BENCHMARK 1: erf()
# =============================================================================
def bench_erf():
    print("\n=== erf() ===")
    from optuna.samplers._tpe._erf import _erf_array_numba, _erf_right_non_big
    import math

    for size in [100, 2_000, 10_000]:
        x = np.linspace(-5, 5, size)

        def fn_numba(x=x):
            return _erf_array_numba(np.asarray(x, dtype=np.float64))

        def fn_fallback_small(x=x):
            return np.asarray([math.erf(v) for v in x.ravel()]).reshape(x.shape)

        def fn_fallback_large(x=x):
            a = np.abs(x).ravel()
            is_not_nan = ~np.isnan(a)
            out = np.where(is_not_nan, 1.0, np.nan)
            non_big = np.nonzero(is_not_nan & (a < 6))[0]
            out[non_big] = _erf_right_non_big(a[non_big])
            return np.sign(x) * out.reshape(x.shape)

        fallback = fn_fallback_small if size < 2000 else fn_fallback_large
        _run_bench(f"erf (n={size})", fn_numba, fallback)


# =============================================================================
# BENCHMARK 2: _log_ndtr()
# =============================================================================
def bench_log_ndtr():
    print("\n=== _log_ndtr() ===")
    from optuna.samplers._tpe._truncnorm import (
        _log_ndtr_array_numba,
        _log_ndtr_single,
    )

    for size in [100, 1_000, 5_000]:
        a = np.linspace(-30, 10, size)

        def fn_numba(a=a):
            return _log_ndtr_array_numba(np.asarray(a, dtype=np.float64))

        def fn_fallback(a=a):
            return np.frompyfunc(_log_ndtr_single, 1, 1)(a).astype(float)

        _run_bench(f"_log_ndtr (n={size})", fn_numba, fn_fallback)


# =============================================================================
# BENCHMARK 3: _ndtri_exp()
# =============================================================================
def bench_ndtri_exp():
    print("\n=== _ndtri_exp() ===")
    from optuna.samplers._tpe._truncnorm import (
        _log_ndtr,
        _ndtri_exp_numba,
    )

    for size in [100, 1_000, 5_000]:
        x_orig = np.linspace(-5, 5, size)
        y = _log_ndtr(x_orig)

        def fn_numba(y=y):
            return _ndtri_exp_numba(y)

        def fn_fallback(y=y):
            # Force the pure-Python fallback path
            from optuna.samplers._tpe._truncnorm import (
                _log_ndtr,
                _ndtri_exp_approx_C,
                _norm_pdf_logC,
            )

            z = y.copy()
            flipped = y > -1e-2
            z[flipped] = np.log(-np.expm1(y[flipped]))
            x = np.empty_like(y)
            small = np.nonzero(z < -5)[0]
            if small.size:
                x[small] = -np.sqrt(-2.0 * (z[small] + _norm_pdf_logC))
            moderate = np.nonzero(z >= -5)[0]
            if moderate.size:
                x[moderate] = -_ndtri_exp_approx_C * np.log(np.expm1(-z[moderate]))
            for _ in range(100):
                with mock.patch("optuna.samplers._tpe._truncnorm.HAS_NUMBA", False):
                    log_ndtr_x = _log_ndtr(x)
                log_norm_pdf_x = -0.5 * x**2 - _norm_pdf_logC
                dx = (log_ndtr_x - z) * np.exp(log_ndtr_x - log_norm_pdf_x)
                x -= dx
                if np.all(np.abs(dx) < 1e-8 * np.abs(x)):
                    break
            x[flipped] *= -1
            return x

        _run_bench(f"_ndtri_exp (n={size})", fn_numba, fn_fallback)


# =============================================================================
# BENCHMARK 4: Pareto front detection
# =============================================================================
def bench_pareto_front():
    print("\n=== _is_pareto_front_nd() ===")
    from optuna.study._multi_objective import _is_pareto_front_nd, _is_pareto_front_nd_numba

    for n_points in [50, 200, 500]:
        for n_obj in [3, 5]:
            rng = np.random.RandomState(42)
            vals = rng.rand(n_points, n_obj)
            vals = vals[np.lexsort(vals.T[::-1])]
            keyed = np.column_stack([np.arange(len(vals), dtype=float), vals])

            def fn_numba(v=keyed):
                return _is_pareto_front_nd_numba(v)

            def fn_fallback(v=keyed):
                with mock.patch("optuna.study._multi_objective.HAS_NUMBA", False):
                    return _is_pareto_front_nd(v)

            _run_bench(f"pareto_front (n={n_points}, d={n_obj})", fn_numba, fn_fallback)


# =============================================================================
# BENCHMARK 5: WFG hypervolume (4+ dimensions)
# =============================================================================
def bench_hypervolume():
    print("\n=== WFG hypervolume (4D+) ===")
    from optuna._hypervolume.wfg import _compute_hv, _compute_hv_numba
    from optuna.study._multi_objective import _is_pareto_front

    for n_points, n_dim in [(5, 4), (10, 4), (8, 5), (10, 6)]:
        rng = np.random.RandomState(42)
        vals = rng.rand(n_points, n_dim)
        ref = np.ones(n_dim) * 2.0
        on_front = _is_pareto_front(vals, assume_unique_lexsorted=False)
        pareto = vals[on_front]
        pareto = pareto[np.lexsort(pareto.T[::-1])]

        def fn_numba(p=pareto, r=ref):
            return _compute_hv_numba(p, r)

        def fn_fallback(p=pareto, r=ref):
            return _compute_hv(p, r)

        _run_bench(
            f"hypervolume (n={len(pareto)}, d={n_dim})",
            fn_numba, fn_fallback,
            warmup=2, repeats=10,
        )


# =============================================================================
# BENCHMARK 6: End-to-end TPE sampling (the real test)
# =============================================================================
def bench_tpe_sampling():
    print("\n=== End-to-end TPE suggest (single-objective) ===")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for n_completed in [50, 200, 500, 1000]:
        # Pre-populate a study with n_completed trials.
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
        rng = np.random.RandomState(42)
        for i in range(n_completed):
            trial = study.ask()
            trial.suggest_float("x0", -5, 5)
            trial.suggest_float("x1", -5, 5)
            trial.suggest_float("x2", -5, 5)
            trial.suggest_int("x3", 0, 100)
            trial.suggest_categorical("x4", ["a", "b", "c", "d"])
            study.tell(trial, rng.randn())

        def fn_with_numba(study=study):
            trial = study.ask()
            trial.suggest_float("x0", -5, 5)
            trial.suggest_float("x1", -5, 5)
            trial.suggest_float("x2", -5, 5)
            trial.suggest_int("x3", 0, 100)
            trial.suggest_categorical("x4", ["a", "b", "c", "d"])
            study.tell(trial, 0.0)

        def fn_without_numba(study=study):
            with _patch_no_numba():
                trial = study.ask()
                trial.suggest_float("x0", -5, 5)
                trial.suggest_float("x1", -5, 5)
                trial.suggest_float("x2", -5, 5)
                trial.suggest_int("x3", 0, 100)
                trial.suggest_categorical("x4", ["a", "b", "c", "d"])
                study.tell(trial, 0.0)

        _run_bench(
            f"TPE suggest (completed={n_completed}, 5 params)",
            fn_with_numba, fn_without_numba,
            warmup=3, repeats=15,
        )


# =============================================================================
# BENCHMARK 7: End-to-end multi-objective TPE
# =============================================================================
def bench_tpe_multi_objective():
    print("\n=== End-to-end TPE suggest (multi-objective, 2 objectives) ===")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for n_completed in [50, 200, 500]:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        rng = np.random.RandomState(42)
        for i in range(n_completed):
            trial = study.ask()
            trial.suggest_float("x0", -5, 5)
            trial.suggest_float("x1", -5, 5)
            trial.suggest_float("x2", -5, 5)
            study.tell(trial, [rng.randn(), rng.randn()])

        def fn_with_numba(study=study):
            trial = study.ask()
            trial.suggest_float("x0", -5, 5)
            trial.suggest_float("x1", -5, 5)
            trial.suggest_float("x2", -5, 5)
            study.tell(trial, [0.0, 0.0])

        def fn_without_numba(study=study):
            with _patch_no_numba():
                trial = study.ask()
                trial.suggest_float("x0", -5, 5)
                trial.suggest_float("x1", -5, 5)
                trial.suggest_float("x2", -5, 5)
                study.tell(trial, [0.0, 0.0])

        _run_bench(
            f"Multi-obj TPE (completed={n_completed}, 3 params, 2 obj)",
            fn_with_numba, fn_without_numba,
            warmup=3, repeats=10,
        )


# =============================================================================
# BENCHMARK 8: HSSP 2D (potential numba target)
# =============================================================================
def bench_hssp_2d():
    print("\n=== HSSP 2D (NOT numba-accelerated — potential target) ===")
    from optuna._hypervolume.hssp import _solve_hssp_2d

    for n_points, subset in [(20, 10), (50, 25), (100, 50)]:
        rng = np.random.RandomState(42)
        vals = rng.rand(n_points, 2)
        vals = vals[np.lexsort(vals.T[::-1])]
        indices = np.arange(n_points)
        ref = np.array([2.0, 2.0])

        def fn(v=vals, idx=indices, s=subset, r=ref):
            return _solve_hssp_2d(v, idx, s, r)

        t = _time_fn(fn, warmup=3, repeats=20)
        print(f"  {'hssp_2d (n=' + str(n_points) + ', k=' + str(subset) + ')':<50s}  time: {t*1e3:8.3f} ms  (no numba version yet)")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    from optuna._numba_utils import HAS_NUMBA

    print(f"HAS_NUMBA = {HAS_NUMBA}")
    print("=" * 100)

    bench_erf()
    bench_log_ndtr()
    bench_ndtri_exp()
    bench_pareto_front()
    bench_hypervolume()
    bench_hssp_2d()
    bench_tpe_sampling()
    bench_tpe_multi_objective()

    print("\n" + "=" * 100)
    print("DONE")
