import numpy as np

from optuna.samplers.cmaes.cma import _compress_symmetric
from optuna.samplers.cmaes.cma import _decompress_symmetric


def test_compress_symmetric_odd() -> None:
    sym2d = np.array([[1, 2], [2, 3]])
    actual = _compress_symmetric(sym2d)
    expected = np.array([1, 2, 3])
    assert np.all(np.equal(actual, expected))


def test_compress_symmetric_even() -> None:
    sym2d = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    actual = _compress_symmetric(sym2d)
    expected = np.array([1, 2, 3, 4, 5, 6])
    assert np.all(np.equal(actual, expected))


def test_decompress_symmetric_odd() -> None:
    sym1d = np.array([1, 2, 3])
    actual = _decompress_symmetric(sym1d)
    expected = np.array([[1, 2], [2, 3]])
    assert np.all(np.equal(actual, expected))


def test_decompress_symmetric_even() -> None:
    sym1d = np.array([1, 2, 3, 4, 5, 6])
    actual = _decompress_symmetric(sym1d)
    expected = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    assert np.all(np.equal(actual, expected))
