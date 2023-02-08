import abc

import numpy as np


class BaseShapeFunction(object, metaclass=abc.ABCMeta):
    def __init__(self, n_objectives: int) -> None:
        self._n_objectives = n_objectives

    def __call__(self, m: int, x: np.ndarray) -> float:
        assert 1 <= m <= self.n_objectives
        assert x.shape == (self.n_objectives - 1,)
        return self._call(m, x)

    @abc.abstractmethod
    def _call(self, m: int, x: np.ndarray) -> float:
        raise NotImplementedError

    @property
    def n_objectives(self) -> int:
        return self._n_objectives


class LinearShapeFunction(BaseShapeFunction):
    def _call(self, m: int, x: np.ndarray) -> float:
        if m == 1:
            return x[:-1].prod()

        if m == self.n_objectives:
            return 1 - x[0]

        return x[: self.n_objectives - m].prod() * (1.0 - x[self.n_objectives - m])


class ConvexShapeFunction(BaseShapeFunction):
    def _call(self, m: int, x: np.ndarray) -> float:
        if m == 1:
            return (
                1
                - np.cos(
                    x * np.pi / 2,
                )
            )[:-1].prod()

        if m == self.n_objectives:
            return 1 - np.sin(x[0] * np.pi / 2.0)

        return (1.0 - np.cos(x * np.pi / 2.0))[: self.n_objectives - m].prod() * (
            1.0 - np.sin(x[self.n_objectives - m] * np.pi / 2.0)
        )


class ConcaveShapeFunction(BaseShapeFunction):
    def _call(self, m: int, x: np.ndarray) -> float:
        if m == 1:
            return np.sin(x * np.pi / 2.0)[:-1].prod()

        if m == self.n_objectives:
            return np.cos(x[0] * np.pi / 2.0)

        return np.sin(x * np.pi / 2.0)[: self.n_objectives - m].prod() * np.cos(
            x[self.n_objectives - m] * np.pi / 2.0
        )


class MixedConvexOrConcaveShapeFunction(BaseShapeFunction):
    def __init__(self, n_objectives: int, alpha: float, n_segments: int) -> None:
        super().__init__(n_objectives)
        self._alpha = alpha
        self._n_segments = n_segments

    def _call(self, m: int, x: np.ndarray) -> float:
        if m == self.n_objectives:
            two_A_pi = 2 * self._n_segments * np.pi
            return np.power(
                1 - x[0] - np.cos(two_A_pi * x[0] + np.pi / 2.0) / two_A_pi, self._alpha
            )

        raise ValueError("m should be the number of objectives")


class DisconnectedShapeFunction(BaseShapeFunction):
    def __init__(
        self, n_objectives: int, alpha: float, beta: float, n_disconnected_regions: int
    ) -> None:
        super().__init__(n_objectives)
        self._alpha = alpha
        self._beta = beta
        self._n_disconnected_regions = n_disconnected_regions

    def _call(self, m: int, x: np.ndarray) -> float:
        if m == self.n_objectives:
            return (
                1
                - np.power(x[0], self._alpha)
                * np.cos(self._n_disconnected_regions * np.power(x[0], self._beta) * np.pi) ** 2
            )

        raise ValueError("m should be the number of objectives")
