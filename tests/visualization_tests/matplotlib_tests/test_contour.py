from io import BytesIO
import itertools
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pytest import WarningsRecorder

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib._contour import _create_zmap
from optuna.visualization.matplotlib._contour import _interpolate_zmap
from optuna.visualization.matplotlib._contour import AXES_PADDING_RATIO


def test_create_zmap() -> None:

    x_values = np.arange(10)
    y_values = np.arange(10)
    z_values = list(np.random.rand(10))

    # we are testing for exact placement of z_values
    # so also passing x_values and y_values as xi and yi
    zmap = _create_zmap(x_values.tolist(), y_values.tolist(), z_values, x_values, y_values)

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
    plt.savefig(BytesIO())

    # Single figure.
    study = prepare_study_with_trials(more_than_three=True, n_objectives=2, with_c_d=False)
    plot_contour(study, target=lambda t: t.values[0])
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

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
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())


def all_equal(iterable: Iterable) -> bool:
    """Returns True if all the elements are equal to each other"""
    return len(set(iterable)) == 1


def range_covers(range1: Tuple[float, float], range2: Tuple[float, float]) -> bool:
    """Returns True if `range1` covers `range2`"""
    min1, max1 = sorted(range1)
    min2, max2 = sorted(range2)
    return min1 <= min2 and max1 >= max2


def test_contour_subplots_have_correct_axis_labels_and_ranges() -> None:
    study = prepare_study_with_trials()
    params = ["param_a", "param_b", "param_c"]
    subplots = plot_contour(study, params=params)
    # `subplots` should look like this:
    # param_a [[subplot 1, subplot 2, subplot 3],
    # param_b  [subplot 4, subplot 4, subplot 6],
    # param_c  [subplot 7, subplot 8, subplot 9]]
    #           param_a    param_b    param_c
    #
    # The folowing block ensures:
    # - The y-axis label of subplot 1 is "param_a"
    # - The x-axis label of subplot 7 is "param_a"
    # - Subplot 1, 2, and 3 have the same y-axis range that covers the search space for `param_a`
    # - Subplot 1, 4, and 7 have the same x-axis range that covers the search space for `param_a`
    # - The y-axis label of subplot 4 is "param_b"
    # - ...
    # - The y-axis label of subplot 7 is "param_c"
    # - ...
    param_ranges = {
        "param_a": (0.0, 3.0),
        "param_b": (0.0, 3.0),
        "param_c": (2.0, 5.0),
    }
    for index, (param_name, param_range) in enumerate(param_ranges.items()):
        minimum, maximum = param_range
        padding = (maximum - minimum) * AXES_PADDING_RATIO
        param_range_with_padding = (minimum - padding, maximum + padding)
        assert subplots[index, 0].get_ylabel() == param_name
        assert subplots[-1, index].get_xlabel() == param_name
        ylims = [ax.get_ylim() for ax in subplots[index, :]]
        assert all_equal(ylims)
        assert all(range_covers(param_range_with_padding, ylim) for ylim in ylims)
        xlims = [ax.get_xlim() for ax in subplots[:, index]]
        assert all_equal(xlims)
        assert all(range_covers(param_range_with_padding, xlim) for xlim in xlims)
    plt.savefig(BytesIO())


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_nonfinite_removed(recwarn: WarningsRecorder, value: float) -> None:

    # To check if contour lines have been rendered (meaning +-inf trials were removed),
    # we should be looking if artists responsible for drawing them are preset in the final plot.
    # Turns out it's difficult to do reliably (No information which artists are responsible for
    # drawing contours) so instead we are checking for warning raised by matplotlib
    # when contour plot fails. TODO(xadrianzetx) Find a better way to test this.
    study = prepare_study_with_trials(with_c_d=True, value_for_first_trial=value)
    plot_contour(study)
    for record in recwarn.list:
        assert "No contour levels were found within the data range" not in str(record.message)
    plt.savefig(BytesIO())


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf")))
def test_nonfinite_multiobjective(recwarn: WarningsRecorder, objective: int, value: float) -> None:

    study = prepare_study_with_trials(with_c_d=True, n_objectives=2, value_for_first_trial=value)
    plot_contour(study, target=lambda t: t.values[objective])
    for record in recwarn.list:
        assert "No contour levels were found within the data range" not in str(record.message)
    plt.savefig(BytesIO())
