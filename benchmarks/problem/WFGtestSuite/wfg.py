from typing import List

import numpy as np


from . import shape_functions
from . import transformation_functions


class BaseWFG(object):
    def __init__(
        self,
        S: np.ndarray,
        A: np.ndarray,
        upper_bounds: np.ndarray,
        shapes: List[shape_functions.BaseShapeFunction],
        transformations: List[List[transformation_functions.BaseTransformations]],
    ) -> None:

        assert all(S > 0)
        assert all((A == 0) + (A == 1))
        assert all(upper_bounds > 0)

        self._S = S
        self._A = A
        self._upper_bounds = upper_bounds
        self._shapes = shapes
        self._transformations = transformations

    def __call__(self, z: np.ndarray) -> np.ndarray:

        S = self._S
        A = self._A
        unit_z = z / self._upper_bounds
        shapes = self._shapes
        transformations = self._transformations

        y = unit_z
        for t_p in transformations:
            _y = np.empty((len(t_p),))
            for i in range(len(t_p)):
                if isinstance(t_p[i], transformation_functions.BaseReductionTransformation):
                    _y[i] = t_p[i](y)
                else:
                    _y[i] = t_p[i](y[i])
            y = _y

        x = np.empty(y.shape)
        x[:-1] = np.maximum(y[-1], A) * (y[:-1] - 0.5) + 0.5
        x[-1] = y[-1]

        f = x[-1] + S * np.asarray([h(m + 1, x[:-1]) for m, h in enumerate(shapes)])
        return f


class WFG1(object):
    """WFG1

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConvexShapeFunction(M) for _ in range(M - 1)]
        shapes.append(shape_functions.MixedConvexOrConcaveShapeFunction(M, 1, 5))

        t_1 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_1.append(transformation_functions.LinearShiftTransformation(0.35))

        t_2 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_2.append(transformation_functions.FlatRegionBiasTransformation(0.8, 0.75, 0.85))

        t_3 = [transformation_functions.PolynomialBiasTransformation(0.02) for _ in range(n)]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_4 = [
            transformation_functions.WeightedSumReductionTransformation(
                2 * np.arange(i * k // (M - 1) + 1, (i + 1) * k // (M - 1) + 1),
                lambda y: _input_converter(i, y),
            )
            for i in range(M - 1)
        ]
        t_4.append(
            transformation_functions.WeightedSumReductionTransformation(
                2 * np.arange(k, n) + 1,
                lambda y: y[k:n],
            )
        )
        transformations = [t_1, t_2, t_3, t_4]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG2(object):
    """WFG2

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments // 2
        assert (n_arguments - k) % 2 == 0

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConvexShapeFunction(M) for _ in range(M - 1)]
        shapes.append(shape_functions.DisconnectedShapeFunction(M, 1, 1, 5))

        t_1 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_1.append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            indices = [k + 2 * (i + 1 - k) - 2, k + 2 * (i - k + 1) - 1]
            return y[indices]

        t_2 = [lambda y: y for _ in range(k)]
        for i in range(k, n // 2):
            t_2.append(
                transformation_functions.NonSeparableReductionTransformation(
                    2, lambda y: _input_converter0(i, y)
                )
            )

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_3 = [
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(k // (M - 1)),
                lambda y: _input_converter1(i, y),
            )
            for i in range(M - 1)
        ]
        t_3.append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n // 2 - k),
                lambda y: y[k : n // 2],
            )
        )

        transformations = [t_1, t_2, t_3]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG3(object):
    """WFG3

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments // 2
        assert (n_arguments - k) % 2 == 0

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.zeros(M - 1)
        A[0] = 1
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.LinearShapeFunction(M) for _ in range(M)]

        t_1 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_1.append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            indices = [k + 2 * (i + 1 - k) - 2, k + 2 * (i - k + 1) - 1]
            return y[indices]

        t_2 = [lambda y: y for _ in range(k)]
        for i in range(k, n // 2):
            t_2.append(
                transformation_functions.NonSeparableReductionTransformation(
                    2, lambda y: _input_converter0(i, y)
                )
            )

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_3 = [
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(k // (M - 1)),
                lambda y: _input_converter1(i, y),
            )
            for i in range(M - 1)
        ]
        t_3.append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n // 2 - k),
                lambda y: y[k : n // 2],
            )
        )

        transformations = [t_1, t_2, t_3]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG4(object):
    """WFG4

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        t_1 = [
            transformation_functions.MultiModalShiftTransformation(30, 10, 0.35) for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_2 = []
        for i in range(M - 1):
            t_2.append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        t_2.append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        transformations = [t_1, t_2]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG5(object):
    """WFG5

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        t_1 = [
            transformation_functions.DeceptiveShiftTransformation(0.35, 0.001, 0.05)
            for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_2 = []
        for i in range(M - 1):
            t_2.append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        t_2.append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        transformations = [t_1, t_2]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG6(object):
    """WFG6

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        t_1 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_1.append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_2 = []
        for i in range(M - 1):
            t_2.append(
                transformation_functions.NonSeparableReductionTransformation(
                    k // (M - 1), lambda y: _input_converter(i, y)
                )
            )
        t_2.append(
            transformation_functions.NonSeparableReductionTransformation(
                n - k,
                lambda y: y[k:n],
            )
        )

        transformations = [t_1, t_2]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG7(object):
    """WFG7

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[i:n]

        t_1 = [
            transformation_functions.ParameterDependentBiasTransformation(
                np.ones(n - i),
                lambda y: _input_converter0(i, y),
                0.98 / 49.98,
                0.02,
                50,
                i,
            )
            for i in range(k)
        ]
        for _ in range(n - k):
            t_1.append(lambda y: y)

        t_2 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_2.append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_3 = []
        for i in range(M - 1):
            t_3.append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter1(i, y)
                )
            )
        t_3.append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        transformations = [t_1, t_2, t_3]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG8(object):
    """WFG8

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[: i - 1]

        t_1 = [lambda y: y for _ in range(k)]
        for i in range(k, n):
            t_1.append(
                transformation_functions.ParameterDependentBiasTransformation(
                    np.ones(i - 1),
                    lambda y: _input_converter0(i, y),
                    0.98 / 49.98,
                    0.02,
                    50,
                    i,
                )
            )

        t_2 = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            t_2.append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_3 = []
        for i in range(M - 1):
            t_3.append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        t_3.append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        transformations = [t_1, t_2, t_3]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG9(object):
    """WFG9

    Args:
        n_arguments:
            The number of arguments.
        n_objectives:
            The number of objectives.
        k:
            The degree of the Pareto front.
    """

    def __init__(self, n_arguments: int, n_objectives: int, k: int):

        assert k % (n_objectives - 1) == 0
        assert k + 1 <= n_arguments

        self._n_arguments = n_arguments
        self._n_objectives = n_objectives
        self._k = k

        n = self._n_arguments
        M = self._n_objectives

        S = 2 * (np.arange(M) + 1)
        A = np.ones(M - 1)
        upper_bounds = 2 * (np.arange(n) + 1)

        self.domain = np.zeros((n, 2))
        self.domain[:, 1] = upper_bounds

        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[i:n]

        t_1 = [
            transformation_functions.ParameterDependentBiasTransformation(
                np.ones(n - i),
                lambda y: _input_converter0(i, y),
                0.98 / 49.98,
                0.02,
                50,
                i,
            )
            for i in range(n - 1)
        ]
        t_1.append(lambda y: y)

        t_2 = [
            transformation_functions.DeceptiveShiftTransformation(0.35, 0.001, 0.05)
            for _ in range(k)
        ]
        for _ in range(n - k):
            t_2.append(transformation_functions.MultiModalShiftTransformation(30, 95, 0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        t_3 = []
        for i in range(M - 1):
            t_3.append(
                transformation_functions.NonSeparableReductionTransformation(
                    k // (M - 1), lambda y: _input_converter(i, y)
                )
            )
        t_3.append(
            transformation_functions.NonSeparableReductionTransformation(
                n - k,
                lambda y: y[k:n],
            )
        )

        transformations = [t_1, t_2, t_3]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)
