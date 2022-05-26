import datetime
import itertools
from textwrap import dedent
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pytest

import optuna
from optuna.distributions import FloatDistribution
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization import plot_pareto_front
from optuna.visualization._pareto_front import _make_hovertext
from optuna.visualization._plotly_imports import go
from optuna.visualization._utils import COLOR_SCALE


def _check_data(figure: "go.Figure", axis: str, expected: Sequence[int]) -> None:
    """Compare `figure` against `expected`.

    Concatenate `data` in `figure` in reverse order, pick the desired `axis`, and compare with
    the `expected` result.

    Args:
        figure: A figure.
        axis: The axis to be checked.
        expected: The expected result.
    """

    n_data = len(figure.data)
    actual = tuple(
        itertools.chain(*list(map(lambda i: figure.data[i][axis], reversed(range(n_data)))))
    )
    assert actual == expected


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("use_constraints_func", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1], [1, 0]])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1])])
def test_plot_pareto_front_2d(
    include_dominated_trials: bool,
    use_constraints_func: bool,
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
    assert len(figure.data) == 2
    assert (figure.data[1]["x"] + figure.data[0]["x"]) == ()
    assert (figure.data[1]["y"] + figure.data[0]["y"]) == ()

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
    actual_axis_order = axis_order or [0, 1]
    if use_constraints_func:
        assert len(figure.data) == 3
        if include_dominated_trials:
            # The enqueue order of trial is: infeasible, feasible non-best, then feasible best.
            data = [(1, 0, 1, 1), (1, 2, 2, 0)]  # type: ignore
        else:
            # The enqueue order of trial is: infeasible, feasible.
            data = [(1, 0, 1), (1, 2, 0)]  # type: ignore
    else:
        assert len(figure.data) == 2
        if include_dominated_trials:
            # The last elements come from dominated trial that is enqueued firstly.
            data = [(0, 1, 1, 1), (2, 0, 2, 1)]  # type: ignore
        else:
            data = [(0, 1), (2, 0)]  # type: ignore

    _check_data(figure, "x", data[actual_axis_order[0]])
    _check_data(figure, "y", data[actual_axis_order[1]])

    titles = ["Objective {}".format(i) for i in range(2)]
    assert figure.layout.xaxis.title.text == titles[actual_axis_order[0]]
    assert figure.layout.yaxis.title.text == titles[actual_axis_order[1]]

    # Test with `target_names` argument.
    error_message = "The length of `target_names` is supposed to be 2."

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
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
        constraints_func=constraints_func,
        targets=targets,
    )
    assert figure.layout.xaxis.title.text == target_names[actual_axis_order[0]]
    assert figure.layout.yaxis.title.text == target_names[actual_axis_order[1]]


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("use_constraints_func", [False, True])
@pytest.mark.parametrize(
    "axis_order", [None] + list(itertools.permutations(range(3), 3))  # type: ignore
)
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1], t.values[2])])
def test_plot_pareto_front_3d(
    include_dominated_trials: bool,
    use_constraints_func: bool,
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
    assert len(figure.data) == 2
    assert (figure.data[1]["x"] + figure.data[0]["x"]) == ()
    assert (figure.data[1]["y"] + figure.data[0]["y"]) == ()
    assert (figure.data[1]["z"] + figure.data[0]["z"]) == ()

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
    if use_constraints_func:
        assert len(figure.data) == 3
        if include_dominated_trials:
            # The enqueue order of trial is: infeasible, feasible non-best, then feasible best.
            data = [(1, 1, 1, 1), (1, 0, 1, 1), (1, 2, 2, 0)]  # type: ignore
        else:
            # The enqueue order of trial is: infeasible, feasible.
            data = [(1, 1, 1), (1, 0, 1), (1, 2, 0)]  # type: ignore
    else:
        assert len(figure.data) == 2
        if include_dominated_trials:
            # The last elements come from dominated trial that is enqueued firstly.
            data = [(1, 1, 1, 1), (0, 1, 1, 1), (2, 0, 2, 1)]  # type: ignore
        else:
            data = [(1, 1), (0, 1), (2, 0)]  # type: ignore

    _check_data(figure, "x", data[actual_axis_order[0]])
    _check_data(figure, "y", data[actual_axis_order[1]])
    _check_data(figure, "z", data[actual_axis_order[2]])

    titles = ["Objective {}".format(i) for i in range(3)]
    assert figure.layout.scene.xaxis.title.text == titles[actual_axis_order[0]]
    assert figure.layout.scene.yaxis.title.text == titles[actual_axis_order[1]]
    assert figure.layout.scene.zaxis.title.text == titles[actual_axis_order[2]]

    # Test with `target_names` argument.
    error_message = "The length of `target_names` is supposed to be 3."

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=[],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    with pytest.raises(ValueError, match=error_message):
        plot_pareto_front(
            study=study,
            target_names=["Foo", "Bar", "Baz", "Qux"],
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            targets=targets,
        )

    target_names = ["Foo", "Bar", "Baz"]
    figure = plot_pareto_front(study=study, target_names=target_names, axis_order=axis_order)
    assert figure.layout.scene.xaxis.title.text == target_names[actual_axis_order[0]]
    assert figure.layout.scene.yaxis.title.text == target_names[actual_axis_order[1]]
    assert figure.layout.scene.zaxis.title.text == target_names[actual_axis_order[2]]


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


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("use_constraints_func", [False, True])
def test_plot_pareto_front_invalid_axis_order(
    dimension: int, include_dominated_trials: bool, use_constraints_func: bool
) -> None:
    study = optuna.create_study(directions=["minimize"] * dimension)
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
        invalid_axis_order = list(range(dimension))
        invalid_axis_order[0] -= 1
        assert min(invalid_axis_order) < 0
        plot_pareto_front(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=invalid_axis_order,
            constraints_func=constraints_func,
        )


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


def test_make_hovertext() -> None:
    trial_no_user_attrs = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": FloatDistribution(5, 12)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    assert (
        _make_hovertext(trial_no_user_attrs)
        == dedent(
            """
        {
          "number": 0,
          "values": [
            0.2
          ],
          "params": {
            "x": 10
          }
        }
        """
        )
        .strip()
        .replace("\n", "<br>")
    )

    trial_user_attrs_valid_json = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": FloatDistribution(5, 12)},
        user_attrs={"a": 42, "b": 3.14},
        system_attrs={},
        intermediate_values={},
    )
    assert (
        _make_hovertext(trial_user_attrs_valid_json)
        == dedent(
            """
        {
          "number": 0,
          "values": [
            0.2
          ],
          "params": {
            "x": 10
          },
          "user_attrs": {
            "a": 42,
            "b": 3.14
          }
        }
        """
        )
        .strip()
        .replace("\n", "<br>")
    )

    trial_user_attrs_invalid_json = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": FloatDistribution(5, 12)},
        user_attrs={"a": 42, "b": 3.14, "c": np.zeros(1), "d": np.nan},
        system_attrs={},
        intermediate_values={},
    )
    assert (
        _make_hovertext(trial_user_attrs_invalid_json)
        == dedent(
            """
        {
          "number": 0,
          "values": [
            0.2
          ],
          "params": {
            "x": 10
          },
          "user_attrs": {
            "a": 42,
            "b": 3.14,
            "c": "[0.]",
            "d": NaN
          }
        }
        """
        )
        .strip()
        .replace("\n", "<br>")
    )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = prepare_study_with_trials(with_c_d=False, direction=direction, n_objectives=2)

    # Since `plot_pareto_front`'s colormap depends on only trial.number,
    # `reversecale` is not in the plot.
    marker = plot_pareto_front(study).data[0]["marker"]
    assert COLOR_SCALE == [v[1] for v in marker["colorscale"]]
    assert "reversecale" not in marker
