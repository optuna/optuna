"""Tests verifying that the optional numba acceleration paths produce correct results.

Both the numba-accelerated and pure-Python/NumPy fallback paths are tested by
monkeypatching HAS_NUMBA to False, forcing the fallback even when numba is installed.
"""
from __future__ import annotations

import importlib
import math
from unittest import mock

import numpy as np
import pytest

from optuna._numba_utils import HAS_NUMBA
from optuna._numba_utils import njit
from optuna._numba_utils import numba_vectorize


# ---------------------------------------------------------------------------
# Fixture that parametrizes every test to run in both "numba" and "no-numba" modes.
# In "no-numba" mode we patch HAS_NUMBA=False in every module that checks it,
# so the runtime dispatchers fall through to the pure-Python fallback.
# ---------------------------------------------------------------------------

_MODULES_USING_NUMBA = [
    "optuna._numba_utils",
    "optuna.samplers._tpe._erf",
    "optuna.samplers._tpe._truncnorm",
    "optuna._hypervolume.wfg",
    "optuna.study._multi_objective",
]


@pytest.fixture(params=["numba", "no-numba"])
def numba_mode(request: pytest.FixtureRequest) -> str:  # type: ignore[type-arg]
    if request.param == "numba" and not HAS_NUMBA:
        pytest.skip("numba is not installed")
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def _patch_numba(numba_mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """When numba_mode is 'no-numba', patch HAS_NUMBA=False in all relevant modules."""
    if numba_mode == "no-numba":
        for mod_name in _MODULES_USING_NUMBA:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            monkeypatch.setattr(mod, "HAS_NUMBA", False)


# ---------------------------------------------------------------------------
# Tests for the decorator utilities themselves
# ---------------------------------------------------------------------------


def test_njit_passthrough_when_no_numba(numba_mode: str) -> None:
    """When numba is absent the njit decorator must be a no-op."""
    if numba_mode == "numba":
        pytest.skip("this test only validates the no-numba passthrough")

    # Build a fresh decorator with HAS_NUMBA=False
    with mock.patch("optuna._numba_utils.HAS_NUMBA", False):
        from optuna._numba_utils import njit as njit_patched

        @njit_patched(cache=True)
        def _add(a: float, b: float) -> float:
            return a + b

    assert _add(1.0, 2.0) == 3.0


def test_numba_vectorize_passthrough_when_no_numba(numba_mode: str) -> None:
    if numba_mode == "numba":
        pytest.skip("this test only validates the no-numba passthrough")

    with mock.patch("optuna._numba_utils.HAS_NUMBA", False):
        from optuna._numba_utils import numba_vectorize as vec_patched

        @vec_patched()
        def _double(x: float) -> float:
            return x * 2.0

    assert _double(3.0) == 6.0


# ---------------------------------------------------------------------------
# Tests for _erf
# ---------------------------------------------------------------------------


class TestErf:
    """Test erf against math.erf in both numba and fallback modes."""

    def test_small_array(self, numba_mode: str) -> None:
        from optuna.samplers._tpe._erf import erf

        x = np.linspace(-4, 4, 50)
        result = erf(x)
        expected = np.array([math.erf(v) for v in x])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_large_array(self, numba_mode: str) -> None:
        from optuna.samplers._tpe._erf import erf

        x = np.linspace(-6, 6, 5000)
        result = erf(x)
        expected = np.array([math.erf(v) for v in x])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_edge_values(self, numba_mode: str) -> None:
        from optuna.samplers._tpe._erf import erf

        x = np.array([0.0, 1e-30, -1e-30, 6.0, -6.0, 100.0, -100.0])
        result = erf(x)
        expected = np.array([math.erf(v) for v in x])
        np.testing.assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests for _truncnorm functions
# ---------------------------------------------------------------------------


class TestTruncNorm:
    """Test _log_ndtr and _ndtri_exp in both numba and fallback modes."""

    def test_log_ndtr_matches_scalar(self, numba_mode: str) -> None:
        from optuna.samplers._tpe._truncnorm import _log_ndtr
        from optuna.samplers._tpe._truncnorm import _log_ndtr_single

        values = np.array([-30.0, -10.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        result = _log_ndtr(values)
        expected = np.array([_log_ndtr_single(float(v)) for v in values])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ndtri_exp_roundtrip(self, numba_mode: str) -> None:
        """ndtri_exp(log_ndtr(x)) should approximately recover x."""
        from optuna.samplers._tpe._truncnorm import _log_ndtr
        from optuna.samplers._tpe._truncnorm import _ndtri_exp

        x_orig = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        log_cdf = _log_ndtr(x_orig)
        x_recovered = _ndtri_exp(log_cdf)
        np.testing.assert_allclose(x_recovered, x_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for Pareto front
# ---------------------------------------------------------------------------


class TestParetoFront:
    """Test _is_pareto_front_nd in both numba and fallback modes."""

    def test_simple_2d(self, numba_mode: str) -> None:
        from optuna.study._multi_objective import _is_pareto_front_nd

        vals = np.array([[0.0, 3.0], [1.0, 1.0], [2.0, 0.0]])
        front = _is_pareto_front_nd(vals)
        assert front[-1]  # Point with lowest loss in col 1 must be on front.

    def test_all_dominated(self, numba_mode: str) -> None:
        from optuna.study._multi_objective import _is_pareto_front_nd

        vals = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        front = _is_pareto_front_nd(vals)
        assert front[0]
        assert not front[1]
        assert not front[2]

    def test_numba_matches_python(self, numba_mode: str) -> None:
        """Verify both paths agree on a non-trivial input."""
        if not HAS_NUMBA:
            pytest.skip("need numba to compare both paths")
        from optuna.study._multi_objective import _is_pareto_front_nd
        from optuna.study._multi_objective import _is_pareto_front_nd_numba

        rng = np.random.RandomState(99)
        vals = rng.rand(20, 4)
        vals = vals[np.lexsort(vals.T[::-1])]
        # Prepend a sort-key column (the function expects lexsorted with col 0 as key).
        keyed = np.column_stack([np.arange(len(vals), dtype=float), vals])

        front_numba = _is_pareto_front_nd_numba(keyed)
        # Force Python path
        with mock.patch("optuna.study._multi_objective.HAS_NUMBA", False):
            front_python = _is_pareto_front_nd(keyed)

        np.testing.assert_array_equal(front_numba, front_python)


# ---------------------------------------------------------------------------
# Tests for hypervolume
# ---------------------------------------------------------------------------


class TestHypervolume:
    """Test hypervolume computation in both numba and fallback modes."""

    def test_single_point_2d(self, numba_mode: str) -> None:
        from optuna._hypervolume.wfg import compute_hypervolume

        vals = np.array([[1.0, 1.0]])
        ref = np.array([3.0, 3.0])
        assert compute_hypervolume(vals, ref) == pytest.approx(4.0)

    def test_two_points_2d(self, numba_mode: str) -> None:
        from optuna._hypervolume.wfg import compute_hypervolume

        vals = np.array([[1.0, 3.0], [3.0, 1.0]])
        ref = np.array([5.0, 5.0])
        assert compute_hypervolume(vals, ref) == pytest.approx(12.0)

    def test_4d_numba_vs_python(self, numba_mode: str) -> None:
        """For 4D+ problems, compare numba path against the Python fallback."""
        if not HAS_NUMBA:
            pytest.skip("need numba to compare both paths")

        from optuna._hypervolume.wfg import _compute_hv
        from optuna._hypervolume.wfg import _compute_hv_numba
        from optuna.study._multi_objective import _is_pareto_front

        rng = np.random.RandomState(42)
        n_points, n_dim = 8, 4
        vals = rng.rand(n_points, n_dim)
        ref = np.ones(n_dim) * 2.0

        on_front = _is_pareto_front(vals, assume_unique_lexsorted=False)
        pareto_vals = vals[on_front]
        pareto_vals = pareto_vals[np.lexsort(pareto_vals.T[::-1])]

        hv_numba = _compute_hv_numba(pareto_vals, ref)
        hv_python = _compute_hv(pareto_vals, ref)
        assert hv_numba == pytest.approx(hv_python, rel=1e-10)
