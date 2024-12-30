from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
import math
from typing import Any

import numpy as np
import pytest

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.study import Study
from optuna.testing.objectives import fail_objective
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.visualization import matplotlib
from optuna.visualization import plot_parallel_coordinate as plotly_plot_parallel_coordinate
from optuna.visualization._parallel_coordinate import _DimensionInfo
from optuna.visualization._parallel_coordinate import _get_parallel_coordinate_info
from optuna.visualization._parallel_coordinate import _ParallelCoordinateInfo
from optuna.visualization._plotly_imports import go
from optuna.visualization._utils import COLOR_SCALE
from optuna.visualization.matplotlib._matplotlib_imports import plt


parametrize_plot_parallel_coordinate = pytest.mark.parametrize(
    "plot_parallel_coordinate",
    [plotly_plot_parallel_coordinate, matplotlib.plot_parallel_coordinate],
)


def _create_study_with_failed_trial() -> Study:
    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    return study


def _create_study_with_categorical_params() -> Study:
    study_categorical_params = create_study()
    distributions: dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution(("preferred", "opt")),
        "category_b": CategoricalDistribution(("net", "una")),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "category_b": "net"},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "opt", "category_b": "una"},
            distributions=distributions,
        )
    )
    return study_categorical_params


def _create_study_with_numeric_categorical_params() -> Study:
    study_categorical_params = create_study()
    distributions: dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution((1, 2)),
        "category_b": CategoricalDistribution((10, 20, 30)),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": 2, "category_b": 20},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=1.0,
            params={"category_a": 1, "category_b": 30},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": 2, "category_b": 10},
            distributions=distributions,
        )
    )
    return study_categorical_params


def _create_study_with_log_params() -> Study:
    study_log_params = create_study()
    distributions: dict[str, BaseDistribution] = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": FloatDistribution(1, 1000, log=True),
    }
    study_log_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": 10},
            distributions=distributions,
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2e-5, "param_b": 200},
            distributions=distributions,
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=0.1,
            params={"param_a": 1e-4, "param_b": 30},
            distributions=distributions,
        )
    )
    return study_log_params


def _create_study_with_log_scale_and_str_and_numeric_category() -> Study:
    study_multi_distro_params = create_study()
    distributions: dict[str, BaseDistribution] = {
        "param_a": CategoricalDistribution(("preferred", "opt")),
        "param_b": CategoricalDistribution((1, 2, 10)),
        "param_c": FloatDistribution(1, 1000, log=True),
        "param_d": CategoricalDistribution((1, -1, 2)),
    }
    study_multi_distro_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": "preferred", "param_b": 2, "param_c": 30, "param_d": 2},
            distributions=distributions,
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": "opt", "param_b": 1, "param_c": 200, "param_d": 2},
            distributions=distributions,
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=2.0,
            params={"param_a": "preferred", "param_b": 10, "param_c": 10, "param_d": 1},
            distributions=distributions,
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=3.0,
            params={"param_a": "opt", "param_b": 2, "param_c": 10, "param_d": -1},
            distributions=distributions,
        )
    )
    return study_multi_distro_params


def test_target_is_none_and_study_is_multi_obj() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _get_parallel_coordinate_info(study)


def test_plot_parallel_coordinate_customized_target_name() -> None:
    study = prepare_study_with_trials()
    figure = plotly_plot_parallel_coordinate(study, target_name="Target Name")
    assert figure.data[0]["dimensions"][0]["label"] == "Target Name"

    figure = matplotlib.plot_parallel_coordinate(study, target_name="Target Name")
    assert figure.get_figure().axes[1].get_ylabel() == "Target Name"


@parametrize_plot_parallel_coordinate
@pytest.mark.parametrize(
    "specific_create_study, params",
    [
        [create_study, None],
        [prepare_study_with_trials, ["param_a", "param_b"]],
        [prepare_study_with_trials, ["param_a", "param_b", "param_c"]],
        [prepare_study_with_trials, ["param_a", "param_b", "param_c", "param_d"]],
        [_create_study_with_failed_trial, None],
        [_create_study_with_categorical_params, None],
        [_create_study_with_numeric_categorical_params, None],
        [_create_study_with_log_params, None],
        [_create_study_with_log_scale_and_str_and_numeric_category, None],
    ],
)
def test_plot_parallel_coordinate(
    plot_parallel_coordinate: Callable[..., Any],
    specific_create_study: Callable[[], Study],
    params: list[str] | None,
) -> None:
    study = specific_create_study()
    figure = plot_parallel_coordinate(study, params=params)
    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()


def test_get_parallel_coordinate_info() -> None:
    # Test with no trial.
    study = create_study()
    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(),
            range=(0, 0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[],
        reverse_scale=True,
        target_name="Objective Value",
    )

    study = prepare_study_with_trials()

    # Test with no parameters.
    info = _get_parallel_coordinate_info(study, params=[])
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0, 2.0, 1.0),
            range=(0.0, 2.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[],
        reverse_scale=True,
        target_name="Objective Value",
    )

    # Test with a trial.
    info = _get_parallel_coordinate_info(study, params=["param_a", "param_b"])
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0, 1.0),
            range=(0.0, 1.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="param_a",
                values=(1.0, 2.5),
                range=(1.0, 2.5),
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            ),
            _DimensionInfo(
                label="param_b",
                values=(2.0, 1.0),
                range=(1.0, 2.0),
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )

    # Test with a trial to select parameter.
    info = _get_parallel_coordinate_info(study, params=["param_a"])
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0, 1.0),
            range=(0.0, 1.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="param_a",
                values=(1.0, 2.5),
                range=(1.0, 2.5),
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        info = _get_parallel_coordinate_info(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(2.0, 1.0),
            range=(1.0, 2.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="param_a",
                values=(1.0, 2.5),
                range=(1.0, 2.5),
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )

    # Test with a customized target name.
    info = _get_parallel_coordinate_info(
        study, params=["param_a", "param_b"], target_name="Target Name"
    )
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Target Name",
            values=(0.0, 1.0),
            range=(0.0, 1.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="param_a",
                values=(1.0, 2.5),
                range=(1.0, 2.5),
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            ),
            _DimensionInfo(
                label="param_b",
                values=(2.0, 1.0),
                range=(1.0, 2.0),
                is_log=False,
                is_cat=False,
                tickvals=[],
                ticktext=[],
            ),
        ],
        reverse_scale=True,
        target_name="Target Name",
    )

    # Test with wrong params that do not exist in trials.
    with pytest.raises(ValueError, match="Parameter optuna does not exist in your study."):
        _get_parallel_coordinate_info(study, params=["optuna", "optuna"])

    # Ignore failed trials.
    study = _create_study_with_failed_trial()
    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(),
            range=(0, 0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_parallel_coordinate_info_categorical_params() -> None:
    # Test with categorical params that cannot be converted to numeral.
    study = _create_study_with_categorical_params()
    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0, 2.0),
            range=(0.0, 2.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="category_a",
                values=(0, 1),
                range=(0, 1),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1],
                ticktext=["preferred", "opt"],
            ),
            _DimensionInfo(
                label="category_b",
                values=(0, 1),
                range=(0, 1),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1],
                ticktext=["net", "una"],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_parallel_coordinate_info_categorical_numeric_params() -> None:
    # Test with categorical params that can be interpreted as numeric params.
    study = _create_study_with_numeric_categorical_params()

    # Trials are sorted by using param_a and param_b, i.e., trial#1, trial#2, and trial#0.
    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(1.0, 2.0, 0.0),
            range=(0.0, 2.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="category_a",
                values=(0, 1, 1),
                range=(0, 1),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1],
                ticktext=["1", "2"],
            ),
            _DimensionInfo(
                label="category_b",
                values=(2, 0, 1),
                range=(0, 2),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1, 2],
                ticktext=["10", "20", "30"],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_parallel_coordinate_info_log_params() -> None:
    # Test with log params.
    study = _create_study_with_log_params()
    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0, 1.0, 0.1),
            range=(0.0, 1.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="param_a",
                values=(-6, math.log10(2e-5), -4),
                range=(-6.0, -4.0),
                is_log=True,
                is_cat=False,
                tickvals=[-6, -5, -4.0],
                ticktext=["1e-06", "1e-05", "0.0001"],
            ),
            _DimensionInfo(
                label="param_b",
                values=(1.0, math.log10(200), math.log10(30)),
                range=(1.0, math.log10(200)),
                is_log=True,
                is_cat=False,
                tickvals=[1.0, 2.0, math.log10(200)],
                ticktext=["10", "100", "200"],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_parallel_coordinate_info_unique_param() -> None:
    # Test case when one unique value is suggested during the optimization.

    study_categorical_params = create_study()
    distributions: dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution(("preferred", "opt")),
        "param_b": FloatDistribution(1, 1000, log=True),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "param_b": 30},
            distributions=distributions,
        )
    )

    # Both parameters contain unique values.
    info = _get_parallel_coordinate_info(study_categorical_params)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0,),
            range=(0.0, 0.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="category_a",
                values=(0.0,),
                range=(0, 0),
                is_log=False,
                is_cat=True,
                tickvals=[0],
                ticktext=["preferred"],
            ),
            _DimensionInfo(
                label="param_b",
                values=(math.log10(30),),
                range=(math.log10(30), math.log10(30)),
                is_log=True,
                is_cat=False,
                tickvals=[math.log10(30)],
                ticktext=["30"],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "preferred", "param_b": 20},
            distributions=distributions,
        )
    )

    # Still "category_a" contains unique suggested value during the optimization.
    info = _get_parallel_coordinate_info(study_categorical_params)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(0.0, 2.0),
            range=(0.0, 2.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="category_a",
                values=(0, 0),
                range=(0, 0),
                is_log=False,
                is_cat=True,
                tickvals=[0],
                ticktext=["preferred"],
            ),
            _DimensionInfo(
                label="param_b",
                values=(math.log10(30), math.log10(20)),
                range=(math.log10(20), math.log10(30)),
                is_log=True,
                is_cat=False,
                tickvals=[math.log10(20), math.log10(30)],
                ticktext=["20", "30"],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


def test_get_parallel_coordinate_info_with_log_scale_and_str_and_numeric_category() -> None:
    # Test with sample from multiple distributions including categorical params
    # that can be interpreted as numeric params.
    study = _create_study_with_log_scale_and_str_and_numeric_category()
    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(1.0, 3.0, 0.0, 2.0),
            range=(0.0, 3.0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[
            _DimensionInfo(
                label="param_a",
                values=(1, 1, 0, 0),
                range=(0, 1),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1],
                ticktext=["preferred", "opt"],
            ),
            _DimensionInfo(
                label="param_b",
                values=(0, 1, 1, 2),
                range=(0, 2),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1, 2],
                ticktext=["1", "2", "10"],
            ),
            _DimensionInfo(
                label="param_c",
                values=(math.log10(200), 1.0, math.log10(30), 1.0),
                range=(1.0, math.log10(200)),
                is_log=True,
                is_cat=False,
                tickvals=[1, 2, math.log10(200)],
                ticktext=["10", "100", "200"],
            ),
            _DimensionInfo(
                label="param_d",
                values=(2, 0, 2, 1),
                range=(0, 2),
                is_log=False,
                is_cat=True,
                tickvals=[0, 1, 2],
                ticktext=["-1", "1", "2"],
            ),
        ],
        reverse_scale=True,
        target_name="Objective Value",
    )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_color_map(direction: str) -> None:
    study = create_study(direction=direction)
    for i in range(3):
        study.add_trial(
            create_trial(
                value=float(i),
                params={"param_a": float(i), "param_b": float(i)},
                distributions={
                    "param_a": FloatDistribution(0.0, 3.0),
                    "param_b": FloatDistribution(0.0, 3.0),
                },
            )
        )

    # `target` is `None`.
    line = plotly_plot_parallel_coordinate(study).data[0]["line"]
    assert COLOR_SCALE == [v[1] for v in line["colorscale"]]
    if direction == "minimize":
        assert line["reversescale"]
    else:
        assert not line["reversescale"]

    # When `target` is not `None`, `reversescale` is always `True`.
    line = plotly_plot_parallel_coordinate(
        study, target=lambda t: t.number, target_name="Target Name"
    ).data[0]["line"]
    assert COLOR_SCALE == [v[1] for v in line["colorscale"]]
    assert line["reversescale"]

    # Multi-objective optimization.
    study = create_study(directions=[direction, direction])
    for i in range(3):
        study.add_trial(
            create_trial(
                values=[float(i), float(i)],
                params={"param_a": float(i), "param_b": float(i)},
                distributions={
                    "param_a": FloatDistribution(0.0, 3.0),
                    "param_b": FloatDistribution(0.0, 3.0),
                },
            )
        )
    line = plotly_plot_parallel_coordinate(
        study, target=lambda t: t.number, target_name="Target Name"
    ).data[0]["line"]
    assert COLOR_SCALE == [v[1] for v in line["colorscale"]]
    assert line["reversescale"]


def test_get_parallel_coordinate_info_only_missing_params() -> None:
    # When all trials contain only a part of parameters,
    # the plot returns an empty figure.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_b": 200},
            distributions={
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )

    info = _get_parallel_coordinate_info(study)
    assert info == _ParallelCoordinateInfo(
        dim_objective=_DimensionInfo(
            label="Objective Value",
            values=(),
            range=(0, 0),
            is_log=False,
            is_cat=False,
            tickvals=[],
            ticktext=[],
        ),
        dims_params=[],
        reverse_scale=True,
        target_name="Objective Value",
    )


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_nonfinite_removed(value: float) -> None:
    study = prepare_study_with_trials(value_for_first_trial=value)
    info = _get_parallel_coordinate_info(study)
    assert all(np.isfinite(info.dim_objective.values))


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_nonfinite_multiobjective(objective: int, value: float) -> None:
    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    info = _get_parallel_coordinate_info(
        study, target=lambda t: t.values[objective], target_name="Target Name"
    )
    assert all(np.isfinite(info.dim_objective.values))
