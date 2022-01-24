import itertools
from typing import List
from typing import Optional

import numpy as np
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib._contour import _create_zmap
from optuna.visualization.matplotlib._contour import _interpolate_zmap


def test_create_zmap() -> None:

    x_values = np.arange(10)
    y_values = np.arange(10)
    z_values = list(np.random.rand(10))

    # we are testing for exact placement of z_values
    # so also passing x_values and y_values as xi and yi
    zmap = _create_zmap(x_values, y_values, z_values, x_values, y_values)

    assert len(zmap) == len(z_values)
    for coord, value in zmap.items():
        # test if value under coordinate
        # still refers to original trial value
        xidx = coord[0]
        yidx = coord[1]
        assert xidx == yidx
        assert z_values[xidx] == value


def test_interpolate_zmap() -> None:

    contour_point_num = 2
    zmap = {(0, 0): 1.0, (1, 1): 4.0}
    expected = np.array([[1.0, 2.5], [2.5, 4.0]])

    actual = _interpolate_zmap(zmap, contour_point_num)

    assert np.allclose(expected, actual)


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_contour(study)


def test_target_is_not_none_and_study_is_multi_obj() -> None:

    # Multiple sub-figures.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=True)
    plot_contour(study, target=lambda t: t.values[0])

    # Single figure.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=False)
    plot_contour(study, target=lambda t: t.values[0])


@pytest.mark.parametrize(
    "params",
    [
        [],
        ["param_a"],
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_c", "param_d"],
        None,
    ],
)
def test_plot_contour(params: Optional[List[str]]) -> None:

    # Test with no trial.
    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = plot_contour(study_without_trials, params=params)
    assert len(figure.get_lines()) == 0

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    assert len(figure.get_lines()) == 0

    # Test with some trials.
    study = prepare_study_with_trials()

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        plot_contour(study, ["optuna", "Optuna"])

    figure = plot_contour(study, params=params)
    if params is not None and len(params) < 3:
        if len(params) <= 1:
            assert len(figure.get_lines()) == 0
        elif len(params) == 2:
            assert len(figure.get_lines()) == 0
    elif params is None:
        assert figure.shape == (len(study.best_params), len(study.best_params))
        for i in range(len(study.best_params)):
            assert figure[i][0].yaxis.label.get_text() == list(study.best_params)[i]
    else:
        assert figure.shape == (len(params), len(params))
        for i in range(len(params)):
            assert figure[i][0].yaxis.label.get_text() == list(params)[i]


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target(params: List[str]) -> None:

    study = prepare_study_with_trials()
    with pytest.warns(UserWarning):
        figure = plot_contour(study, params=params, target=lambda t: t.params["param_d"])
    if len(params) == 2:
        assert len(figure.get_lines()) == 0
    else:
        assert figure.shape == (len(params), len(params))
        for i in range(len(params)):
            assert figure[i][0].yaxis.label.get_text() == list(params)[i]


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target_name(params: List[str]) -> None:

    study = prepare_study_with_trials()
    figure = plot_contour(study, params=params, target_name="Target Name")
    if len(params) == 2:
        assert len(figure.get_lines()) == 0
    else:
        assert figure.shape == (len(params), len(params))
        for i in range(len(params)):
            assert figure[i][0].yaxis.label.get_text() == list(params)[i]


def test_plot_contour_log_scale_and_str_category() -> None:

    # If the search space has three parameters, plot_contour generates nine plots.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": "100", "param_c": "one"},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": CategoricalDistribution(["100", "101"]),
                "param_c": CategoricalDistribution(["one", "two"]),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 1e-5, "param_b": "101", "param_c": "two"},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": CategoricalDistribution(["100", "101"]),
                "param_c": CategoricalDistribution(["one", "two"]),
            },
        )
    )

    figure = plot_contour(study)
    subplots = [plot for plot in figure.flatten() if plot.has_data()]
    expected = {"param_a": [1e-6, 1e-5], "param_b": [0.0, 1.0], "param_c": [0.0, 1.0]}
    ranges = itertools.permutations(expected.keys(), 2)

    for plot, (yrange, xrange) in zip(subplots, ranges):
        # Take 5% axis padding into account.
        np.testing.assert_allclose(plot.get_xlim(), expected[xrange], atol=5e-2)
        np.testing.assert_allclose(plot.get_ylim(), expected[yrange], atol=5e-2)


def test_plot_contour_mixture_category_types() -> None:

    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": None, "param_b": 101},
            distributions={
                "param_a": CategoricalDistribution([None, "100"]),
                "param_b": CategoricalDistribution([101, 102.0]),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=0.5,
            params={"param_a": "100", "param_b": 102.0},
            distributions={
                "param_a": CategoricalDistribution([None, "100"]),
                "param_b": CategoricalDistribution([101, 102.0]),
            },
        )
    )

    figure = plot_contour(study)
    assert figure.get_xlim() == (-0.05, 1.05)
    assert figure.get_ylim() == (100.95, 102.05)


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_b", "param_a"],
    ],
)
def test_generate_contour_plot_for_few_observations(params: List[str]) -> None:

    study = prepare_study_with_trials(less_than_two=True)
    figure = plot_contour(study, params)
    assert not figure.has_data()
