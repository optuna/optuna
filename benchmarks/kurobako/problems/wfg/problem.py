from __future__ import annotations

import math
import sys

import numpy as np
import shape_functions
import transformation_functions

from kurobako import problem


class BaseWFG:
    def __init__(
        self,
        S: np.ndarray,
        A: np.ndarray,
        upper_bounds: np.ndarray,
        shapes: list[shape_functions.BaseShapeFunction],
        transformations: list[list[transformation_functions.BaseTransformations]],
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


class WFG1:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConvexShapeFunction(M) for _ in range(M - 1)]
        shapes.append(shape_functions.MixedConvexOrConcaveShapeFunction(M, 1, 5))

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(4)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        # transformations[0] = [lambda y: y for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        # transformations[1] = [lambda y: y for _ in range(k)]

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[1].append(
                transformation_functions.FlatRegionBiasTransformation(0.8, 0.75, 0.85)
            )

        transformations[2] = [
            transformation_functions.PolynomialBiasTransformation(0.02) for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[3] = [
            transformation_functions.WeightedSumReductionTransformation(
                2 * np.arange(i * k // (M - 1) + 1, (i + 1) * k // (M - 1) + 1),
                lambda y: _input_converter(i, y),
            )
            for i in range(M - 1)
        ]
        transformations[3].append(
            transformation_functions.WeightedSumReductionTransformation(
                2 * np.arange(k, n) + 1,
                lambda y: y[k:n],
            )
        )

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG2:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConvexShapeFunction(M) for _ in range(M - 1)]
        shapes.append(shape_functions.DisconnectedShapeFunction(M, 1, 1, 5))

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            indices = [k + 2 * (i + 1 - k) - 2, k + 2 * (i - k + 1) - 1]
            return y[indices]

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for i in range(k, n // 2):
            transformations[1].append(
                transformation_functions.NonSeparableReductionTransformation(
                    2, lambda y: _input_converter0(i, y)
                )
            )

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = [
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(k // (M - 1)),
                lambda y: _input_converter1(i, y),
            )
            for i in range(M - 1)
        ]
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n // 2 - k),
                lambda y: y[k : n // 2],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG3:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.LinearShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            indices = [k + 2 * (i + 1 - k) - 2, k + 2 * (i - k + 1) - 1]
            return y[indices]

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for i in range(k, n // 2):
            transformations[1].append(
                transformation_functions.NonSeparableReductionTransformation(
                    2, lambda y: _input_converter0(i, y)
                )
            )

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = [
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(k // (M - 1)),
                lambda y: _input_converter1(i, y),
            )
            for i in range(M - 1)
        ]
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n // 2 - k),
                lambda y: y[k : n // 2],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG4:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(2)]

        transformations[0] = [
            transformation_functions.MultiModalShiftTransformation(30, 10, 0.35) for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        # transformations[1] = []
        for i in range(M - 1):
            transformations[1].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        transformations[1].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG5:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(2)]

        transformations[0] = [
            transformation_functions.DeceptiveShiftTransformation(0.35, 0.001, 0.05)
            for _ in range(n)
        ]

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[1] = []
        for i in range(M - 1):
            transformations[1].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        transformations[1].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG6:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(2)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[0].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        # transformations[1] = []
        for i in range(M - 1):
            transformations[1].append(
                transformation_functions.NonSeparableReductionTransformation(
                    k // (M - 1), lambda y: _input_converter(i, y)
                )
            )
        transformations[1].append(
            transformation_functions.NonSeparableReductionTransformation(
                n - k,
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG7:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[i:n]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [
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
            transformations[0].append(transformation_functions.IdenticalTransformation())

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[1].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter1(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = []
        for i in range(M - 1):
            transformations[2].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter1(i, y)
                )
            )
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG8:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[: i - 1]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for i in range(k, n):
            transformations[0].append(
                transformation_functions.ParameterDependentBiasTransformation(
                    np.ones(i - 1),
                    lambda y: _input_converter0(i, y),
                    0.98 / 49.98,
                    0.02,
                    50,
                    i,
                )
            )

        transformations[1] = [transformation_functions.IdenticalTransformation() for _ in range(k)]
        for _ in range(n - k):
            transformations[1].append(transformation_functions.LinearShiftTransformation(0.35))

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = []
        for i in range(M - 1):
            transformations[2].append(
                transformation_functions.WeightedSumReductionTransformation(
                    np.ones(k // (M - 1)), lambda y: _input_converter(i, y)
                )
            )
        transformations[2].append(
            transformation_functions.WeightedSumReductionTransformation(
                np.ones(n - k),
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFG9:
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

        shapes: list[shape_functions.BaseShapeFunction]
        shapes = [shape_functions.ConcaveShapeFunction(M) for _ in range(M)]

        def _input_converter0(i: int, y: np.ndarray) -> np.ndarray:
            return y[i:n]

        transformations: list[list[transformation_functions.BaseTransformations]]
        transformations = [[] for _ in range(3)]

        transformations[0] = [
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
        transformations[0].append(transformation_functions.IdenticalTransformation())

        transformations[1] = [
            transformation_functions.DeceptiveShiftTransformation(0.35, 0.001, 0.05)
            for _ in range(k)
        ]
        for _ in range(n - k):
            transformations[1].append(
                transformation_functions.MultiModalShiftTransformation(30, 95, 0.35)
            )

        def _input_converter(i: int, y: np.ndarray) -> np.ndarray:
            indices = np.arange(i * k // (M - 1), (i + 1) * k // (M - 1))
            return y[indices]

        transformations[2] = []
        for i in range(M - 1):
            transformations[2].append(
                transformation_functions.NonSeparableReductionTransformation(
                    k // (M - 1), lambda y: _input_converter(i, y)
                )
            )
        transformations[2].append(
            transformation_functions.NonSeparableReductionTransformation(
                n - k,
                lambda y: y[k:n],
            )
        )

        # transformations = [transformations[0], transformations[1], transformations[2]]

        self.wfg = BaseWFG(S, A, upper_bounds, shapes, transformations)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.wfg.__call__(z)


class WFGProblemFactory(problem.ProblemFactory):
    def specification(self) -> problem.ProblemSpec:
        self._n_wfg = int(sys.argv[1])
        self._n_dim = int(sys.argv[2])

        self._low = 0
        self._high = 2
        params = [
            problem.Var(f"x{i}", problem.ContinuousRange(0, self._high * i))
            for i in range(self._n_dim)
        ]
        return problem.ProblemSpec(
            name=f"WFG{self._n_wfg}",
            params=params,
            values=[problem.Var("f1"), problem.Var("f2")],
        )

    def create_problem(self, seed: int) -> problem.Problem:
        return WFGProblem()


class WFGProblem(problem.Problem):
    def __init__(self) -> None:
        super().__init__()

    def create_evaluator(self, params: list[problem.Var]) -> problem.Evaluator:
        return WFGEvaluator(params)


class WFGEvaluator(problem.Evaluator):
    def __init__(self, params: list[problem.Var]) -> None:
        self._n_wfg = int(sys.argv[1])
        self._n_dim = int(sys.argv[2])
        self._n_obj = int(sys.argv[3])
        self._k = int(sys.argv[4])

        self._x = np.array(params)
        self._current_step = 0

        self.wfg: WFG1 | WFG2 | WFG3 | WFG4 | WFG5 | WFG6 | WFG7 | WFG8 | WFG9
        if self._n_wfg == 1:
            self.wfg = WFG1(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 2:
            self.wfg = WFG2(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 3:
            self.wfg = WFG3(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 4:
            self.wfg = WFG4(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 5:
            self.wfg = WFG5(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 6:
            self.wfg = WFG6(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 7:
            self.wfg = WFG7(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 8:
            self.wfg = WFG8(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        elif self._n_wfg == 9:
            self.wfg = WFG9(n_arguments=self._n_dim, n_objectives=self._n_obj, k=self._k)
        else:
            raise AssertionError("Invalid specification for WFG number.")

    def current_step(self) -> int:
        return self._current_step

    def evaluate(self, next_step: int) -> list[float]:
        self._current_step = 1
        v = self.wfg(self._x)
        v = v.tolist()

        if math.isnan(v[0]) or math.isinf(v[0]):
            raise ValueError
        if math.isnan(v[1]) or math.isinf(v[1]):
            raise ValueError
        return [v[0], v[1]]


if __name__ == "__main__":
    runner = problem.ProblemRunner(WFGProblemFactory())
    runner.run()
