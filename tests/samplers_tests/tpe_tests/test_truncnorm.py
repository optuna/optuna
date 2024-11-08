import sys

import numpy as np
import pytest
from scipy.stats import truncnorm as truncnorm_scipy

from optuna._imports import try_import
import optuna.samplers._tpe._truncnorm as truncnorm_ours


with try_import() as _imports:
    from scipy.stats._continuous_distns import _log_gauss_mass as _log_gauss_mass_scipy


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "a,b",
    [(-np.inf, np.inf), (-10, +10), (-1, +1), (-1e-3, +1e-3), (10, 100), (-100, -10), (0, 0)],
)
def test_ppf(a: float, b: float) -> None:
    for x in np.concatenate(
        [np.linspace(0, 1, num=100), np.array([sys.float_info.min, 1 - sys.float_info.epsilon])]
    ):
        assert truncnorm_ours.ppf(x, a, b) == pytest.approx(
            truncnorm_scipy.ppf(x, a, b), nan_ok=True
        ), f"ppf(x={x}, a={a}, b={b})"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    "a,b",
    [(-np.inf, np.inf), (-10, +10), (-1, +1), (-1e-3, +1e-3), (10, 100), (-100, -10), (0, 0)],
)
@pytest.mark.parametrize("loc", [-10, 0, 10])
@pytest.mark.parametrize("scale", [0.1, 1, 10])
def test_logpdf(a: float, b: float, loc: float, scale: float) -> None:
    for x in np.concatenate(
        [np.linspace(np.max([a, -100]), np.min([b, 100]), num=1000), np.array([-2000.0, +2000.0])]
    ):
        assert truncnorm_ours.logpdf(x, a, b, loc, scale) == pytest.approx(
            truncnorm_scipy.logpdf(x, a, b, loc, scale), nan_ok=True
        ), f"logpdf(x={x}, a={a}, b={b})"


@pytest.mark.skipif(
    not _imports.is_successful(), reason="Failed to import SciPy's internal function."
)
@pytest.mark.parametrize(
    "a,b",
    # we don't test (0, 0) as SciPy returns the incorrect value.
    [(-np.inf, np.inf), (-10, +10), (-1, +1), (-1e-3, +1e-3), (10, 100), (-100, -10)],
)
def test_log_gass_mass(a: float, b: float) -> None:
    for x in np.concatenate(
        [np.linspace(0, 1, num=100), np.array([sys.float_info.min, 1 - sys.float_info.epsilon])]
    ):
        assert truncnorm_ours._log_gauss_mass(np.array([a]), np.array([b])) == pytest.approx(
            np.atleast_1d(_log_gauss_mass_scipy(a, b)), nan_ok=True
        ), f"_log_gauss_mass(x={x}, a={a}, b={b})"
