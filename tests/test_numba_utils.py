"""Tests verifying that the optional numba acceleration paths produce correct results.

When numba is not installed, the decorated functions should behave identically to
the pure-Python/NumPy fallbacks. When numba *is* installed, we verify numerical
correctness against known reference values.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from optuna._numba_utils import HAS_NUMBA
from optuna._numba_utils import njit
from optuna._numba_utils import numba_vectorize


# ---------------------------------------------------------------------------
# Tests for the decorator utilities themselves
# ---------------------------------------------------------------------------


def test_njit_passthrough_when_no_numba() -> None:
    """When numba is absent the njit decorator must be a no-op."""
    if HAS_NUMBA:
        pytest.skip("numba is installed; passthrough path not exercised")

    @njit(cache=True)
    def _add(a: float, b: float) -> float:
        return a + b

    assert _add(1.0, 2.0) == 3.0


def test_numba_vectorize_passthrough_when_no_numba() -> None:
    if HAS_NUMBA:
        pytest.skip("numba is installed; passthrough path not exercised")

    @numba_vectorize()
    def _double(x: float) -> float:
        return x * 2.0

    assert _double(3.0) == 6.0


# ---------------------------------------------------------------------------
# Tests for numba-accelerated _erf
# ---------------------------------------------------------------------------


class TestErfNumba:
    """Test the numba-accelerated erf against math.erf."""

    def test_small_array(self) -> None:
        from optuna.samplers._tpe._erf import erf

        x = np.linspace(-4, 4, 50)
        result = erf(x)
        expected = np.array([math.erf(v) for v in x])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_large_array(self) -> None:
        from optuna.samplers._tpe._erf import erf

        x = np.linspace(-6, 6, 5000)
        result = erf(x)
        expected = np.array([math.erf(v) for v in x])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_edge_values(self) -> None:
        from optuna.samplers._tpe._erf import erf

        x = np.array([0.0, 1e-30, -1e-30, 6.0, -6.0, 100.0, -100.0])
        result = erf(x)
        expected = np.array([math.erf(v) for v in x])
        np.testing.assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Tests for numba-accelerated _truncnorm functions
# ---------------------------------------------------------------------------


class TestTruncNormNumba:
    """Test numba-accelerated _log_ndtr and _ndtri_exp against pure-Python paths."""

    def test_log_ndtr_matches_scalar(self) -> None:
        from optuna.samplers._tpe._truncnorm import _log_ndtr
        from optuna.samplers._tpe._truncnorm import _log_ndtr_single

        values = np.array([-30.0, -10.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        result = _log_ndtr(values)
        expected = np.array([_log_ndtr_single(float(v)) for v in values])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_ndtri_exp_roundtrip(self) -> None:
        """ndtri_exp(log_ndtr(x)) should approximately recover x."""
        from optuna.samplers._tpe._truncnorm import _log_ndtr
        from optuna.samplers._tpe._truncnorm import _ndtri_exp

        x_orig = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        log_cdf = _log_ndtr(x_orig)
        x_recovered = _ndtri_exp(log_cdf)
        np.testing.assert_allclose(x_recovered, x_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for numba-accelerated Pareto front
# ---------------------------------------------------------------------------


class TestParetoFrontNumba:
    """Test numba-accelerated _is_pareto_front_nd against known results."""

    def test_simple_2d(self) -> None:
        from optuna.study._multi_objective import _is_pareto_front_nd

        # Lexsorted loss values with first column as sort key.
        # Points: (0,3), (1,1), (2,0) — all are Pareto-optimal in the remaining cols.
        vals = np.array([[0.0, 3.0], [1.0, 1.0], [2.0, 0.0]])
        front = _is_pareto_front_nd(vals)
        # After removing first column for dominance check: [3], [1], [0]
        # [0] < [1] < [3] so only (2,0) is non-dominated — but the function
        # checks dominance on cols [1:], so point with val 0.0 dominates.
        assert front[-1]  # The point with lowest loss in col 1 must be on front.

    def test_all_dominated(self) -> None:
        from optuna.study._multi_objective import _is_pareto_front_nd

        # One point dominates all others in columns [1:].
        vals = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        front = _is_pareto_front_nd(vals)
        assert front[0]
        assert not front[1]
        assert not front[2]


# ---------------------------------------------------------------------------
# Tests for numba-accelerated hypervolume
# ---------------------------------------------------------------------------


class TestHypervolumeNumba:
    """Test numba-accelerated hypervolume computation."""

    def test_single_point_2d(self) -> None:
        from optuna._hypervolume.wfg import compute_hypervolume

        # Single point (1,1) with reference (3,3) → area = 2*2 = 4
        vals = np.array([[1.0, 1.0]])
        ref = np.array([3.0, 3.0])
        assert compute_hypervolume(vals, ref) == pytest.approx(4.0)

    def test_two_points_2d(self) -> None:
        from optuna._hypervolume.wfg import compute_hypervolume

        vals = np.array([[1.0, 3.0], [3.0, 1.0]])
        ref = np.array([5.0, 5.0])
        # Union of two rectangles: 4*2 + 2*4 - 2*2 = 8 + 8 - 4 = 12
        assert compute_hypervolume(vals, ref) == pytest.approx(12.0)

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_4d_numba_vs_python(self) -> None:
        """For 4D+ problems, compare numba path against the Python fallback."""
        from optuna._hypervolume.wfg import _compute_hv
        from optuna._hypervolume.wfg import _compute_hv_numba

        rng = np.random.RandomState(42)
        n_points, n_dim = 8, 4
        vals = rng.rand(n_points, n_dim)
        ref = np.ones(n_dim) * 2.0

        # Filter to Pareto front for valid input.
        from optuna.study._multi_objective import _is_pareto_front

        on_front = _is_pareto_front(vals, assume_unique_lexsorted=False)
        pareto_vals = vals[on_front]
        pareto_vals = pareto_vals[np.lexsort(pareto_vals.T[::-1])]

        hv_numba = _compute_hv_numba(pareto_vals, ref)
        hv_python = _compute_hv(pareto_vals, ref)
        assert hv_numba == pytest.approx(hv_python, rel=1e-10)
