from io import BytesIO
import itertools
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma.testutils
import pytest

import optuna
from optuna.trial import FrozenTrial
from optuna.visualization.matplotlib import plot_pareto_front


def allclose_as_set(
    points1: Union[List[List[float]], np.ndarray], points2: Union[List[List[float]], np.ndarray]
) -> bool:
    p1 = points1 if isinstance(points1, list) else points1.tolist()
    p2 = points2 if isinstance(points2, list) else points2.tolist()
    return np.allclose(sorted(p1), sorted(p2))


def _check_data(figure: Axes, axis: str, expected: Sequence[int]) -> None:
    """Compare `figure` against `expected`.

    Concatenate `data` in `figure` in reverse order, pick the desired `axis`, and compare with
    the `expected` result.

    Args:
        figure: A figure.
        axis: The axis to be checked.
        expected: The expected result.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    n_data = len(figure.collections)
    actual = tuple(
        itertools.chain(
            *list(
                map(
                    lambda i: figure.collections[i].get_offsets()[:, axis_map[axis]],
                    reversed(range(n_data)),
                )
            )
        )
    )
    numpy.ma.testutils.assert_array_equal(actual, expected)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1], [1, 0]])
@pytest.mark.parametrize("use_constraints_func", [False, True])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1])])
def test_plot_pareto_front_2d(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
    use_constraints_func: bool,
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> None:
    if axis_order is not None and targets is not None:
        pytest.skip("skip using both axis_order and targets")
    # Test with no trial.
    study = optuna.create_study(directions=["minimize", "minimize"])
    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
    )

    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with four trials.
    study.enqueue_trial({"x": 1, "y": 2})
    study.enqueue_trial({"x": 1, "y": 1})
    study.enqueue_trial({"x": 0, "y": 2})
    study.enqueue_trial({"x": 1, "y": 0})
    study.optimize(lambda t: [t.suggest_int("x", 0, 2), t.suggest_int("y", 0, 2)], n_trials=4)

    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]]
    if use_constraints_func:
        # (x, y) = (1, 0) is infeasible; others are feasible.
        def constraints_func(t: FrozenTrial) -> Sequence[float]:
            return [1.0] if t.params["x"] == 1 and t.params["y"] == 0 else [-1.0]

    else:
        constraints_func = None

    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        constraints_func=constraints_func,
        targets=targets,
    )
    assert len(figure.get_lines()) == 0

    actual_axis_order = axis_order or [0, 1]
    if use_constraints_func:
        if include_dominated_trials:
            # The enqueue order of trial is: infeasible, feasible non-best, then feasible best.
            assert len(figure.collections) == 3
            data = [(1, 0, 1, 1), (1, 2, 2, 0)]  # type: ignore
        else:
            # The enqueue order of trial is: infeasible, feasible.
            assert len(figure.collections) == 2
            data = [(1, 0, 1), (1, 2, 0)]  # type: ignore
    else:
        if include_dominated_trials:
            # The last elements come from dominated trial that is enqueued firstly.
            assert len(figure.collections) == 2
            data = [(0, 1, 1, 1), (2, 0, 2, 1)]  # type: ignore
        else:
            assert len(figure.collections) == 1
            data = [(0, 1), (2, 0)]  # type: ignore

    _check_data(figure, "x", data[actual_axis_order[0]])
    _check_data(figure, "y", data[actual_axis_order[1]])
    plt.savefig(BytesIO())

    # Test with `target_names` argument.
    error_message = "The length of `target_names` is supposed to be 2."

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            constraints_func=constraints_func,
            targets=targets,
        )

    target_names = ["Foo", "Bar"]
    figure = plot_pareto_front(
        study=study,
        target_names=target_names,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        constraints_func=constraints_func,
        targets=targets,
    )
    assert len(figure.get_lines()) == 0

    if axis_order is None:
        assert figure.get_xlabel() == target_names[0]
        assert figure.get_ylabel() == target_names[1]
    else:
        assert figure.get_xlabel() == target_names[axis_order[0]]
        assert figure.get_ylabel() == target_names[axis_order[1]]


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize(
    "axis_order", [None] + list(itertools.permutations(range(3), 3))  # type: ignore
)
@pytest.mark.parametrize("use_constraints_func", [False, True])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1], t.values[2])])
def test_plot_pareto_front_3d(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
    use_constraints_func: bool,
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> None:
    if axis_order is not None and targets is not None:
        pytest.skip("skip using both axis_order and targets")
    # Test with no trial.
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
    )
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with three trials.
    study.enqueue_trial({"x": 1, "y": 1, "z": 2})
    study.enqueue_trial({"x": 1, "y": 1, "z": 1})
    study.enqueue_trial({"x": 1, "y": 0, "z": 2})
    study.enqueue_trial({"x": 1, "y": 1, "z": 0})
    study.optimize(
        lambda t: [t.suggest_int("x", 0, 1), t.suggest_int("y", 0, 2), t.suggest_int("z", 0, 2)],
        n_trials=4,
    )

    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]]
    if use_constraints_func:
        # (x, y, z) = (1, 1, 0) is infeasible; others are feasible.
        def constraints_func(t: FrozenTrial) -> Sequence[float]:
            return (
                [1.0]
                if t.params["x"] == 1 and t.params["y"] == 1 and t.params["z"] == 0
                else [-1.0]
            )

    else:
        constraints_func = None

    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        constraints_func=constraints_func,
        targets=targets,
    )
    actual_axis_order = axis_order or [0, 1, 2]
    assert len(figure.get_lines()) == 0

    if use_constraints_func:
        if include_dominated_trials:
            # The enqueue order of trial is: infeasible, feasible non-best, then feasible best.
            assert len(figure.collections) == 3
            data = [(1, 1, 1, 1), (1, 0, 1, 1), (1, 2, 2, 0)]  # type: ignore
        else:
            # The enqueue order of trial is: infeasible, feasible.
            assert len(figure.collections) == 2
            data = [(1, 1, 1), (1, 0, 1), (1, 2, 0)]  # type: ignore
    else:
        if include_dominated_trials:
            # The last elements come from dominated trial that is enqueued firstly.
            assert len(figure.collections) == 2
            data = [(1, 1, 1, 1), (0, 1, 1, 1), (2, 0, 2, 1)]  # type: ignore
        else:
            assert len(figure.collections) == 1
            data = [(1, 1), (0, 1), (2, 0)]  # type: ignore

    _check_data(figure, "x", data[actual_axis_order[0]])
    _check_data(figure, "y", data[actual_axis_order[1]])
    # TODO(fukatani): check z data.
    # _check_data(figure, "z", data[actual_axis_order[2]])
    plt.savefig(BytesIO())

    # Test with `target_names` argument.
    error_message = "The length of `target_names` is supposed to be 3."

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            constraints_func=constraints_func,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            constraints_func=constraints_func,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            constraints_func=constraints_func,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz", "Qux"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            constraints_func=constraints_func,
            targets=targets,
        )

    target_names = ["Foo", "Bar", "Baz"]
    figure = plot_pareto_front(
        study=study,
        target_names=target_names,
        axis_order=axis_order,
        targets=targets,
        constraints_func=constraints_func,
    )

    assert len(figure.get_lines()) == 0

    if axis_order is None:
        assert figure.get_xlabel() == target_names[0]
        assert figure.get_ylabel() == target_names[1]
        assert figure.get_zlabel() == target_names[2]
    else:
        assert figure.get_xlabel() == target_names[axis_order[0]]
        assert figure.get_ylabel() == target_names[axis_order[1]]
        assert figure.get_zlabel() == target_names[axis_order[2]]


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("use_constraints_func", [False, True])
def test_plot_pareto_front_unsupported_dimensions(
    include_dominated_trials: bool, use_constraints_func: bool
) -> None:
    constraints_func = (lambda _: [-1.0]) if use_constraints_func else None

    error_message = (
        "`plot_pareto_front` function only supports 2 or 3 objective"
        " studies when using `targets` is `None`. Please use `targets`"
        " if your objective studies have more than 3 objectives."
    )

    # Unsupported: n_objectives == 1.
    with pytest.raises(ValueError, match=error_message):
        study = optuna.create_study(directions=["minimize"])
        study.optimize(lambda _: [0], n_trials=1)
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
        )

    with pytest.raises(ValueError, match=error_message):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda _: [0], n_trials=1)
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
        )

    # Unsupported: n_objectives == 4.
    with pytest.raises(ValueError, match=error_message):
        study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
        study.optimize(lambda _: [0, 0, 0, 0], n_trials=1)
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
        )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("use_constraints_func", [False, True])
def test_plot_pareto_front_invalid_axis_order(
    dimension: int, include_dominated_trials: bool, use_constraints_func: bool
) -> None:
    study = optuna.create_study(directions=["minimize"] * dimension)
    study.optimize(lambda _: [0] * dimension, n_trials=1)
    constraints_func = (lambda _: [-1.0]) if use_constraints_func else None

    # Invalid: len(axis_order) != dimension
    with pytest.raises(ValueError):
        invalid_axis_order = list(range(dimension + 1))
        assert len(invalid_axis_order) != dimension
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
            constraints_func=constraints_func,
        )

    # Invalid: np.unique(axis_order).size != dimension
    with pytest.raises(ValueError):
        invalid_axis_order = list(range(dimension))
        invalid_axis_order[1] = invalid_axis_order[0]
        assert np.unique(invalid_axis_order).size != dimension
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
            constraints_func=constraints_func,
        )

    # Invalid: max(axis_order) > (dimension - 1)
    with pytest.raises(ValueError):
        invalid_axis_order = list(range(dimension))
        invalid_axis_order[-1] += 1
        assert max(invalid_axis_order) > (dimension - 1)
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
            constraints_func=constraints_func,
        )

    # Invalid: min(axis_order) < 0
    with pytest.raises(ValueError):
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(lambda _: [0] * 2, n_trials=1)
        invalid_axis_order = list(range(dimension))
        invalid_axis_order[0] -= 1
        assert min(invalid_axis_order) < 0
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
            constraints_func=constraints_func,
        )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_plot_pareto_front_targets_without_target_names() -> None:
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    with pytest.raises(
        ValueError,
        match="If `targets` is specified for empty studies, `target_names` must be specified.",
    ):
        plot_pareto_front(
            study=study,
            target_names=None,
            targets=lambda t: (t.values[0], t.values[1], t.values[2]),
        )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_plot_pareto_front_invalid_target_values() -> None:
    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
    study.optimize(lambda _: [0, 0, 0, 0], n_trials=3)
    with pytest.raises(
        ValueError,
        match="targets` should return a sequence of target values. your `targets`"
        " returns <class 'float'>",
    ):
        plot_pareto_front(
            study=study,
            targets=lambda t: t.values[0],
        )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize(
    "targets",
    [
        lambda t: (t.values[0],),
        lambda t: (t.values[0], t.values[1], t.values[2], t.values[3]),
    ],
)
def test_plot_pareto_front_n_targets_unsupported(
    targets: Callable[[FrozenTrial], Sequence[float]]
) -> None:
    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
    study.optimize(lambda _: [0, 0, 0, 0], n_trials=3)
    n_targets = len(targets(study.best_trials[0]))
    with pytest.raises(
        ValueError,
        match="`plot_pareto_front` function only supports 2 or 3 targets."
        " you used {} targets now.".format(n_targets),
    ):
        plot_pareto_front(
            study=study,
            targets=targets,
        )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_plot_pareto_front_using_axis_order_and_targets() -> None:
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    with pytest.raises(
        ValueError,
        match="Using both `targets` and `axis_order` is not supported."
        " Use either `targets` or `axis_order`.",
    ):
        plot_pareto_front(
            study=study,
            axis_order=[0, 1, 2],
            targets=lambda t: (t.values[0], t.values[1], t.values[2]),
        )
