from io import BytesIO
import itertools
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import numpy as np
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


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1], [1, 0]])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1])])
def test_plot_pareto_front_2d(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
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

    # Test with three trials.
    study.enqueue_trial({"x": 1, "y": 1})
    study.enqueue_trial({"x": 1, "y": 0})
    study.enqueue_trial({"x": 0, "y": 1})
    study.optimize(lambda t: [t.suggest_int("x", 0, 1), t.suggest_int("y", 0, 1)], n_trials=3)

    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        targets=targets,
    )
    assert len(figure.get_lines()) == 0

    if axis_order is not None:
        pareto_front_points = np.array([[1.0, 0.0], [0.0, 1.0]])[:, axis_order]
    else:
        pareto_front_points = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert pareto_front_points.shape == (2, 2)

    path_offsets = list(map(lambda pc: pc.get_offsets(), figure.findobj(PathCollection)))
    exists_pareto_front = any(
        map(lambda po: allclose_as_set(po, pareto_front_points), path_offsets)
    )
    exists_dominated_trials = any(
        map(lambda po: allclose_as_set(po, np.array([[1.0, 1.0]])), path_offsets)
    )
    assert exists_pareto_front
    if include_dominated_trials:
        assert exists_dominated_trials
    plt.savefig(BytesIO())

    # Test with `target_names` argument.
    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            targets=targets,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            targets=targets,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    target_names = ["Foo", "Bar"]
    figure = plot_pareto_front(
        study=study,
        target_names=target_names,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        targets=targets,
    )
    assert len(figure.get_lines()) == 0

    if axis_order is None:
        assert figure.get_xlabel() == target_names[0]
        assert figure.get_ylabel() == target_names[1]
    else:
        assert figure.get_xlabel() == target_names[axis_order[0]]
        assert figure.get_ylabel() == target_names[axis_order[1]]

    if axis_order is not None:
        pareto_front_points = np.array([[1.0, 0.0], [0.0, 1.0]])[:, axis_order]
    else:
        pareto_front_points = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert pareto_front_points.shape == (2, 2)

    path_offsets = list(map(lambda pc: pc.get_offsets(), figure.findobj(PathCollection)))
    exists_pareto_front = any(
        map(lambda po: allclose_as_set(po, pareto_front_points), path_offsets)
    )
    exists_dominated_trials = any(
        map(lambda po: allclose_as_set(po, np.array([[1.0, 1.0]])), path_offsets)
    )
    assert exists_pareto_front
    if include_dominated_trials:
        assert exists_dominated_trials
    plt.savefig(BytesIO())


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize(
    "axis_order", [None] + list(itertools.permutations(range(3), 3))  # type: ignore
)
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1], t.values[2])])
def test_plot_pareto_front_3d(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
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
    study.enqueue_trial({"x": 1, "y": 1, "z": 1})
    study.enqueue_trial({"x": 1, "y": 0, "z": 1})
    study.enqueue_trial({"x": 1, "y": 1, "z": 0})
    study.optimize(
        lambda t: [t.suggest_int("x", 0, 1), t.suggest_int("y", 0, 1), t.suggest_int("z", 0, 1)],
        n_trials=3,
    )

    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        targets=targets,
    )
    assert len(figure.get_lines()) == 0

    if axis_order is not None:
        pareto_front_points = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])[:, axis_order][:, 0:2]
    else:
        pareto_front_points = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])[:, 0:2]
    assert pareto_front_points.shape == (2, 2)

    path_offsets = list(map(lambda pc: pc.get_offsets(), figure.findobj(PathCollection)))
    exists_pareto_front = any(
        map(lambda po: allclose_as_set(po, pareto_front_points), path_offsets)
    )
    exists_dominated_trials = any(map(lambda po: allclose_as_set(po, [[1.0, 1.0]]), path_offsets))
    assert exists_pareto_front
    if include_dominated_trials:
        assert exists_dominated_trials
    plt.savefig(BytesIO())

    # Test with `target_names` argument.
    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz", "Qux"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    target_names = ["Foo", "Bar", "Baz"]
    figure = plot_pareto_front(
        study=study, target_names=target_names, axis_order=axis_order, targets=targets
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

    if axis_order is not None:
        pareto_front_points = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])[:, axis_order][:, 0:2]
    else:
        pareto_front_points = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])[:, 0:2]
    assert pareto_front_points.shape == (2, 2)

    path_offsets = list(map(lambda pc: pc.get_offsets(), figure.findobj(PathCollection)))
    exists_pareto_front = any(
        map(lambda po: allclose_as_set(po, pareto_front_points), path_offsets)
    )
    exists_dominated_trials = any(map(lambda po: allclose_as_set(po, [[1.0, 1.0]]), path_offsets))
    assert exists_pareto_front
    if include_dominated_trials:
        assert exists_dominated_trials
    plt.savefig(BytesIO())


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("include_dominated_trials", [False, True])
def test_plot_pareto_front_unsupported_dimensions(include_dominated_trials: bool) -> None:
    # Unsupported: n_objectives == 1.
    with pytest.raises(ValueError):
        study = optuna.create_study(directions=["minimize"])
        study.optimize(lambda t: [0], n_trials=1)
        plot_pareto_front(study=study, include_dominated_trials=include_dominated_trials)

    with pytest.raises(ValueError):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: [0], n_trials=1)
        plot_pareto_front(study=study, include_dominated_trials=include_dominated_trials)

    # Unsupported: n_objectives == 4.
    with pytest.raises(ValueError):
        study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
        study.optimize(lambda t: [0, 0, 0, 0], n_trials=1)
        plot_pareto_front(study=study, include_dominated_trials=include_dominated_trials)


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("include_dominated_trials", [False, True])
def test_plot_pareto_front_invalid_axis_order(
    dimension: int, include_dominated_trials: bool
) -> None:
    study = optuna.create_study(directions=["minimize"] * dimension)
    study.optimize(lambda t: [0] * dimension, n_trials=1)

    # Invalid: len(axis_order) != dimension
    with pytest.raises(ValueError):
        invalid_axis_order = list(range(dimension + 1))
        assert len(invalid_axis_order) != dimension
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
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
        )

    # Invalid: min(axis_order) < 0
    with pytest.raises(ValueError):
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(lambda t: [0] * 2, n_trials=1)
        invalid_axis_order = list(range(dimension))
        invalid_axis_order[0] -= 1
        assert min(invalid_axis_order) < 0
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
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
@pytest.mark.parametrize(
    "targets",
    [
        lambda t: (t.values[0]),
    ],
)
def test_plot_pareto_front_invalid_target_values(
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]]
) -> None:
    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
    study.optimize(lambda t: [0, 0, 0, 0], n_trials=3)
    with pytest.raises(
        ValueError,
        match="targets` should return a sequence of target values. your `targets`"
        " returns <class 'float'>",
    ):
        plot_pareto_front(
            study=study,
            targets=targets,
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
    study.optimize(lambda t: [0, 0, 0, 0], n_trials=3)
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
