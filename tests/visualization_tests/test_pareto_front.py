import itertools
from typing import List
from typing import Optional

import numpy as np
import pytest

import optuna
from optuna.visualization import plot_pareto_front


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1], [1, 0]])
def test_plot_pareto_front_2d(
    include_dominated_trials: bool, axis_order: Optional[List[int]]
) -> None:
    # Test with no trial.
    study = optuna.create_study(directions=["minimize", "minimize"])
    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
    )
    assert len(figure.data) == 2
    assert (figure.data[1]["x"] + figure.data[0]["x"]) == ()
    assert (figure.data[1]["y"] + figure.data[0]["y"]) == ()

    # Test with three trials.
    study.enqueue_trial({"x": 1, "y": 1})
    study.enqueue_trial({"x": 1, "y": 0})
    study.enqueue_trial({"x": 0, "y": 1})
    study.optimize(lambda t: [t.suggest_int("x", 0, 1), t.suggest_int("y", 0, 1)], n_trials=3)

    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
    )
    assert len(figure.data) == 2
    if include_dominated_trials:
        # The last elements come from dominated trial that is enqueued firstly.
        data = [(1, 0, 1), (0, 1, 1)]  # type: ignore
        if axis_order is None:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[0]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[1]
        else:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[axis_order[0]]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[axis_order[1]]
    else:
        data = [(1, 0), (0, 1)]  # type: ignore
        if axis_order is None:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[0]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[1]
        else:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[axis_order[0]]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[axis_order[1]]

    titles = ["Objective {}".format(i) for i in range(2)]
    if axis_order is None:
        assert figure.layout.xaxis.title.text == titles[0]
        assert figure.layout.yaxis.title.text == titles[1]
    else:
        assert figure.layout.xaxis.title.text == titles[axis_order[0]]
        assert figure.layout.yaxis.title.text == titles[axis_order[1]]

    # Test with `target_names` argument.
    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study, target_names=[], include_dominated_trials=include_dominated_trials
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study, target_names=["Foo"], include_dominated_trials=include_dominated_trials
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
        )

    target_names = ["Foo", "Bar"]
    figure = plot_pareto_front(
        study=study,
        target_names=target_names,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
    )
    if axis_order is None:
        assert figure.layout.xaxis.title.text == target_names[0]
        assert figure.layout.yaxis.title.text == target_names[1]
    else:
        assert figure.layout.xaxis.title.text == target_names[axis_order[0]]
        assert figure.layout.yaxis.title.text == target_names[axis_order[1]]


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize(
    "axis_order", [None] + list(itertools.permutations(range(3), 3))  # type: ignore
)
def test_plot_pareto_front_3d(
    include_dominated_trials: bool, axis_order: Optional[List[int]]
) -> None:
    # Test with no trial.
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    figure = plot_pareto_front(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
    )
    assert len(figure.data) == 2
    assert (figure.data[1]["x"] + figure.data[0]["x"]) == ()
    assert (figure.data[1]["y"] + figure.data[0]["y"]) == ()
    assert (figure.data[1]["z"] + figure.data[0]["z"]) == ()

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
    )
    assert len(figure.data) == 2
    if include_dominated_trials:
        # The last elements come from dominated trial that is enqueued firstly.
        data = [(1, 1, 1), (0, 1, 1), (1, 0, 1)]  # type: ignore
        if axis_order is None:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[0]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[1]
            assert (figure.data[1]["z"] + figure.data[0]["z"]) == data[2]
        else:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[axis_order[0]]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[axis_order[1]]
            assert (figure.data[1]["z"] + figure.data[0]["z"]) == data[axis_order[2]]
    else:
        data = [(1, 1), (0, 1), (1, 0)]  # type: ignore
        if axis_order is None:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[0]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[1]
            assert (figure.data[1]["z"] + figure.data[0]["z"]) == data[2]
        else:
            assert (figure.data[1]["x"] + figure.data[0]["x"]) == data[axis_order[0]]
            assert (figure.data[1]["y"] + figure.data[0]["y"]) == data[axis_order[1]]
            assert (figure.data[1]["z"] + figure.data[0]["z"]) == data[axis_order[2]]

    titles = ["Objective {}".format(i) for i in range(3)]
    if axis_order is None:
        assert figure.layout.scene.xaxis.title.text == titles[0]
        assert figure.layout.scene.yaxis.title.text == titles[1]
        assert figure.layout.scene.zaxis.title.text == titles[2]
    else:
        assert figure.layout.scene.xaxis.title.text == titles[axis_order[0]]
        assert figure.layout.scene.yaxis.title.text == titles[axis_order[1]]
        assert figure.layout.scene.zaxis.title.text == titles[axis_order[2]]

    # Test with `target_names` argument.
    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
        )

    with pytest.raises(ValueError):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz", "Qux"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
        )

    target_names = ["Foo", "Bar", "Baz"]
    figure = plot_pareto_front(study=study, target_names=target_names, axis_order=axis_order)
    if axis_order is None:
        assert figure.layout.scene.xaxis.title.text == target_names[0]
        assert figure.layout.scene.yaxis.title.text == target_names[1]
        assert figure.layout.scene.zaxis.title.text == target_names[2]
    else:
        assert figure.layout.scene.xaxis.title.text == target_names[axis_order[0]]
        assert figure.layout.scene.yaxis.title.text == target_names[axis_order[1]]
        assert figure.layout.scene.zaxis.title.text == target_names[axis_order[2]]


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


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("include_dominated_trials", [False, True])
def test_plot_pareto_front_invalid_axis_order(
    dimension: int, include_dominated_trials: bool
) -> None:
    study = optuna.create_study(directions=["minimize"] * dimension)

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
        invalid_axis_order = list(range(dimension))
        invalid_axis_order[0] -= 1
        assert min(invalid_axis_order) < 0
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
        )
