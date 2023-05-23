from __future__ import annotations

import abc
from typing import Callable
from typing import Union

import numpy as np


class BaseIdenticalTransformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, y: float) -> float:
        raise NotImplementedError


class BaseBiasTransformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, y: float) -> float:
        raise NotImplementedError


class BaseShiftTransformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, y: float) -> float:
        raise NotImplementedError


class BaseReductionTransformation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, y: np.ndarray) -> float:
        raise NotImplementedError


BaseTransformations = Union[
    BaseIdenticalTransformation,
    BaseBiasTransformation,
    BaseShiftTransformation,
    BaseReductionTransformation,
]


class IdenticalTransformation(BaseIdenticalTransformation):
    def __init__(self) -> None:
        pass

    def __call__(self, y: float) -> float:
        return y


class PolynomialBiasTransformation(BaseBiasTransformation):
    def __init__(self, alpha: float) -> None:
        assert alpha > 0 and alpha != 1.0
        self._alpha = alpha

    def __call__(self, y: float) -> float:
        return np.power(y, self._alpha)


class FlatRegionBiasTransformation(BaseBiasTransformation):
    def __init__(self, a: float, b: float, c: float) -> None:
        assert 0 <= a <= 1
        assert 0 <= b <= 1
        assert 0 <= c <= 1
        assert b < c
        assert not (b == 0) or (a == 0 and c != 1)
        assert not (c == 1) or (a == 1 and b != 0)

        self._a = a
        self._b = b
        self._c = c

    def __call__(self, y: float) -> float:
        a = self._a
        b = self._b
        c = self._c
        return (
            a
            + min(0, np.floor(y - b)) * a * (b - y) / b
            - min(0, np.floor(c - y)) * (1.0 - a) * (y - c) / (1.0 - c)
        )


class ParameterDependentBiasTransformation(BaseReductionTransformation):
    def __init__(
        self,
        w: np.ndarray,
        input_converter: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        c: float,
        i: int,
    ) -> None:
        assert 0 < a < 1
        assert 0 < b < c

        self._w = w
        self._input_converter = input_converter
        self._a = a
        self._b = b
        self._c = c
        self._i = i

    def __call__(self, y: np.ndarray) -> float:
        w = self._w
        a = self._a
        b = self._b
        c = self._c
        i = self._i

        u = (self._input_converter(y) * w).sum() / w.sum()
        v = a - (1.0 - 2 * u) * np.fabs(np.floor(0.5 - u) + a)
        return np.power(y[i], b + (c - b) * v)


class LinearShiftTransformation(BaseShiftTransformation):
    def __init__(self, a: float) -> None:
        assert 0 < a < 1

        self._a = a

    def __call__(self, y: float) -> float:
        return np.fabs(y - self._a) / np.fabs(np.floor(self._a - y) + self._a)


class DeceptiveShiftTransformation(BaseShiftTransformation):
    def __init__(self, a: float, b: float, c: float) -> None:
        assert 0 < a < 1
        assert 0 < b < 1
        assert 0 < c < 1
        assert a - b > 0
        assert a + b < 1

        self._a = a
        self._b = b
        self._c = c

    def __call__(self, y: float) -> float:
        a = self._a
        b = self._b
        c = self._c

        q1 = np.floor(y - a + b) * (1.0 - c + (a - b) / b)
        q2 = np.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b)
        return 1.0 + (np.fabs(y - a) - b) * (q1 / (a - b) + q2 / (1.0 - a - b) + 1.0 / b)


class MultiModalShiftTransformation(BaseShiftTransformation):
    def __init__(self, a: int, b: float, c: float) -> None:
        assert a > 0
        assert b >= 0
        assert (4 * a + 2) * np.pi >= 4 * b
        assert 0 < c < 1

        self._a = a
        self._b = b
        self._c = c

    def __call__(self, y: float) -> float:
        a = self._a
        b = self._b
        c = self._c

        q1 = np.fabs(y - c) / (2 * (np.floor(c - y) + c))
        q2 = (4 * a + 2) * np.pi * (0.5 - q1)
        return (1.0 + np.cos(q2) + 4 * b * (q1**2)) / (b + 2)


class WeightedSumReductionTransformation(BaseReductionTransformation):
    def __init__(self, w: np.ndarray, input_converter: Callable[[np.ndarray], np.ndarray]) -> None:
        assert all(w > 0)

        self._w = w
        self._input_converter = input_converter

    def __call__(self, y: np.ndarray) -> float:
        y = self._input_converter(y)
        return (y * self._w).sum() / self._w.sum()


class NonSeparableReductionTransformation(BaseReductionTransformation):
    def __init__(self, a: int, input_converter: Callable[[np.ndarray], np.ndarray]) -> None:
        assert a > 0

        self._a = a
        self._input_converter = input_converter

    def __call__(self, y: np.ndarray) -> float:
        a = float(self._a)
        y = self._input_converter(y)
        n = y.shape[0]
        indices = [(j + k + 1) % n for k in np.arange(n) for j in np.arange(n)]

        q = y.sum() + np.fabs(y[indices].reshape((n, n)) - y)[:, 0 : int(a) - 1].sum()
        r = n * np.ceil(a / 2) * (1.0 + 2 * a - 2 * np.ceil(a / 2)) / a

        return q / r
