import sys

import numpy as np
import pytest
from scipy.stats import truncnorm as truncnorm_scipy

import optuna.samplers._tpe._truncnorm as truncnorm_ours


@pytest.mark.skipif(
    sys.version_info < (3, 8, 0), reason="SciPy 1.9.2 is not supported in Python 3.7"
)
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


@pytest.mark.skipif(
    sys.version_info < (3, 8, 0), reason="SciPy 1.9.2 is not supported in Python 3.7"
)
@pytest.mark.parametrize(
    "a,b",
    [(-np.inf, np.inf), (-10, +10), (-1, +1), (-1e-3, +1e-3), (10, 100), (-100, -10), (0, 0)],
)
def test_logpdf(a: float, b: float) -> None:
    for x in np.concatenate(
        [np.linspace(np.max([a, -100]), np.min([b, 100]), num=1000), np.array([-2000.0, +2000.0])]
    ):
        assert truncnorm_ours.logpdf(x, a, b) == pytest.approx(
            truncnorm_scipy.logpdf(x, a, b), nan_ok=True
        ), f"logpdf(x={x}, a={a}, b={b})"


@pytest.mark.skipif(
    sys.version_info < (3, 8, 0), reason="SciPy 1.9.2 is not supported in Python 3.7"
)
@pytest.mark.parametrize(
    "a,b",
    [(-np.inf, np.inf), (-10, +10), (-1, +1), (-1e-3, +1e-3), (10, 100), (-100, -10), (0, 0)],
)
def test_cdf(a: float, b: float) -> None:
    for x in np.concatenate(
        [np.linspace(0, 1, num=100), np.array([sys.float_info.min, 1 - sys.float_info.epsilon])]
    ):
        assert truncnorm_ours.cdf(x, a, b) == pytest.approx(
            truncnorm_scipy.cdf(x, a, b), nan_ok=True
        ), f"cdf(x={x}, a={a}, b={b})"
