import datetime
from io import BytesIO
from textwrap import dedent
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pytest

import optuna
from optuna import create_study
from optuna import create_trial
from optuna.distributions import FloatDistribution
from optuna.study.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization import plot_pareto_front
import optuna.visualization._pareto_front
from optuna.visualization._pareto_front import _get_pareto_front_info
from optuna.visualization._pareto_front import _make_hovertext
from optuna.visualization._pareto_front import _ParetoFrontInfo
from optuna.visualization._plotly_imports import go
from optuna.visualization._utils import COLOR_SCALE
from optuna.visualization.matplotlib._matplotlib_imports import plt
import optuna.visualization.matplotlib._pareto_front


def test_get_pareto_front_info_infer_n_targets() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])
    assert _get_pareto_front_info(study).n_targets == 2

    study = optuna.create_study(directions=["minimize"] * 5)
    assert (
        _get_pareto_front_info(
            study, target_names=["target1", "target2"], targets=lambda _: [0.0, 1.0]
        ).n_targets
        == 2
    )

    study = optuna.create_study(directions=["minimize"] * 5)
    study.optimize(lambda _: [0] * 5, n_trials=1)
    assert _get_pareto_front_info(study, targets=lambda _: [0.0, 1.0]).n_targets == 2

    study = optuna.create_study(directions=["minimize"] * 2)
    with pytest.raises(ValueError):
        _get_pareto_front_info(study, targets=lambda _: [0.0, 1.0])


def create_study_2d() -> Study:

    study = optuna.create_study(directions=["minimize", "minimize"])

    study.enqueue_trial({"x": 1, "y": 2})
    study.enqueue_trial({"x": 1, "y": 1})
    study.enqueue_trial({"x": 0, "y": 2})
    study.enqueue_trial({"x": 1, "y": 0})
    study.optimize(lambda t: [t.suggest_int("x", 0, 2), t.suggest_int("y", 0, 2)], n_trials=4)
    return study


def create_study_3d() -> Study:

    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])

    study.enqueue_trial({"x": 1, "y": 2})
    study.enqueue_trial({"x": 1, "y": 1})
    study.enqueue_trial({"x": 0, "y": 2})
    study.enqueue_trial({"x": 1, "y": 0})
    study.optimize(
        lambda t: [t.suggest_int("x", 0, 2), t.suggest_int("y", 0, 2), 1.0],
        n_trials=4,
    )
    return study


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1], [1, 0]])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1])])
@pytest.mark.parametrize("target_names", [None, ["Foo", "Bar"]])
def test_get_pareto_front_info_unconstrained(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]],
    target_names: Optional[List[str]],
) -> None:
    if axis_order is not None and targets is not None:
        pytest.skip("skip using both axis_order and targets")

    study = create_study_2d()
    trials = study.get_trials(deepcopy=False)

    info = _get_pareto_front_info(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        targets=targets,
        target_names=target_names,
    )

    assert info == _ParetoFrontInfo(
        n_targets=2,
        target_names=target_names or ["Objective 0", "Objective 1"],
        best_trials_with_values=[(trials[2], [0, 2]), (trials[3], [1, 0])],
        non_best_trials_with_values=[(trials[0], [1, 2]), (trials[1], [1, 1])]
        if include_dominated_trials
        else [],
        infeasible_trials_with_values=[],
        axis_order=axis_order or [0, 1],
        include_dominated_trials=include_dominated_trials,
        has_constraints_func=False,
    )


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1], [1, 0]])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1])])
@pytest.mark.parametrize("target_names", [None, ["Foo", "Bar"]])
def test_get_pareto_front_info_constrained(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]],
    target_names: Optional[List[str]],
) -> None:
    if axis_order is not None and targets is not None:
        pytest.skip("skip using both axis_order and targets")

    study = create_study_2d()
    trials = study.get_trials(deepcopy=False)

    # (x, y) = (1, 0) is infeasible; others are feasible.
    def constraints_func(t: FrozenTrial) -> Sequence[float]:
        return [1.0] if t.params["x"] == 1 and t.params["y"] == 0 else [-1.0]

    info = _get_pareto_front_info(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        targets=targets,
        target_names=target_names,
        constraints_func=constraints_func,
    )

    assert info == _ParetoFrontInfo(
        n_targets=2,
        target_names=target_names or ["Objective 0", "Objective 1"],
        best_trials_with_values=[(trials[1], [1, 1]), (trials[2], [0, 2])],
        non_best_trials_with_values=[(trials[0], [1, 2])] if include_dominated_trials else [],
        infeasible_trials_with_values=[(trials[3], [1, 0])],
        axis_order=axis_order or [0, 1],
        include_dominated_trials=include_dominated_trials,
        has_constraints_func=True,
    )


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("axis_order", [None, [0, 1, 2], [2, 1, 0]])
@pytest.mark.parametrize("targets", [None, lambda t: (t.values[0], t.values[1], t.values[2])])
@pytest.mark.parametrize("target_names", [None, ["Foo", "Bar", "Baz"]])
def test_get_pareto_front_info_3d(
    include_dominated_trials: bool,
    axis_order: Optional[List[int]],
    targets: Optional[Callable[[FrozenTrial], Sequence[float]]],
    target_names: Optional[List[str]],
) -> None:
    if axis_order is not None and targets is not None:
        pytest.skip("skip using both axis_order and targets")

    study = create_study_3d()
    trials = study.get_trials(deepcopy=False)

    info = _get_pareto_front_info(
        study=study,
        include_dominated_trials=include_dominated_trials,
        axis_order=axis_order,
        targets=targets,
        target_names=target_names,
    )

    assert info == _ParetoFrontInfo(
        n_targets=3,
        target_names=target_names or ["Objective 0", "Objective 1", "Objective 2"],
        best_trials_with_values=[(trials[2], [0, 2, 1]), (trials[3], [1, 0, 1])],
        non_best_trials_with_values=[(trials[0], [1, 2, 1]), (trials[1], [1, 1, 1])]
        if include_dominated_trials
        else [],
        infeasible_trials_with_values=[],
        axis_order=axis_order or [0, 1, 2],
        include_dominated_trials=include_dominated_trials,
        has_constraints_func=False,
    )


def test_get_pareto_front_info_invalid_number_of_target_names() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_pareto_front_info(study=study, target_names=["Foo"])


@pytest.mark.parametrize("n_dims", [1, 4])
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("constraints_func", [None, lambda _: [-1.0]])
def test_get_pareto_front_info_unsupported_dimensions(
    n_dims: int,
    include_dominated_trials: bool,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> None:
    study = optuna.create_study(directions=["minimize"] * n_dims)
    with pytest.raises(ValueError):
        _get_pareto_front_info(
            study=study,
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
        )


@pytest.mark.parametrize("axis_order", [[0, 1, 1], [0, 0], [0, 2], [-1, 1]])
@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("constraints_func", [None, lambda _: [-1.0]])
def test_get_pareto_front_info_invalid_axis_order(
    axis_order: List[int],
    include_dominated_trials: bool,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_pareto_front_info(
            study=study,
            include_dominated_trials=include_dominated_trials,
            axis_order=axis_order,
            constraints_func=constraints_func,
        )


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("constraints_func", [None, lambda _: [-1.0]])
def test_get_pareto_front_info_invalid_target_values(
    include_dominated_trials: bool,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(lambda _: [0, 0], n_trials=3)
    with pytest.raises(ValueError):
        _get_pareto_front_info(
            study=study,
            targets=lambda t: t.values[0],
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
        )


@pytest.mark.parametrize("include_dominated_trials", [False, True])
@pytest.mark.parametrize("constraints_func", [None, lambda _: [-1.0]])
def test_get_pareto_front_info_using_axis_order_and_targets(
    include_dominated_trials: bool,
    constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]],
) -> None:
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_pareto_front_info(
            study=study,
            axis_order=[0, 1, 2],
            targets=lambda t: (t.values[0], t.values[1], t.values[2]),
            include_dominated_trials=include_dominated_trials,
            constraints_func=constraints_func,
        )


def test_constraints_func_experimental_warning() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"])

    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        _get_pareto_front_info(
            study=study,
            constraints_func=lambda _: [1.0],
        )


@pytest.mark.parametrize(
    "plotter",
    [
        optuna.visualization._pareto_front._get_pareto_front_plot,
        optuna.visualization.matplotlib._pareto_front._get_pareto_front_plot,
    ],
)
@pytest.mark.parametrize(
    "info_template",
    [
        _get_pareto_front_info(create_study_2d()),
        _get_pareto_front_info(create_study_3d()),
    ],
)
@pytest.mark.parametrize("include_dominated_trials", [True, False])
@pytest.mark.parametrize("has_constraints_func", [True, False])
def test_get_pareto_front_plot(
    plotter: Callable[[_ParetoFrontInfo], Any],
    info_template: _ParetoFrontInfo,
    include_dominated_trials: bool,
    has_constraints_func: bool,
) -> None:

    info = info_template
    if not include_dominated_trials:
        info = info._replace(include_dominated_trials=False, non_best_trials_with_values=[])
    if not has_constraints_func:
        info = info._replace(has_constraints_func=False, infeasible_trials_with_values=[])

    figure = plotter(info)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())


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
    study = create_study(directions=[direction, direction])
    for i in range(3):
        study.add_trial(
            create_trial(
                values=[float(i), float(i)],
                params={"param_a": 1.0, "param_b": 2.0},
                distributions={
                    "param_a": FloatDistribution(0.0, 3.0),
                    "param_b": FloatDistribution(0.0, 3.0),
                },
            )
        )

    # Since `plot_pareto_front`'s colormap depends on only trial.number,
    # `reversecale` is not in the plot.
    marker = plot_pareto_front(study).data[0]["marker"]
    assert COLOR_SCALE == [v[1] for v in marker["colorscale"]]
    assert "reversecale" not in marker
