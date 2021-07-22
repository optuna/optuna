from typing import List
from typing import Optional

import numpy as np
import pytest

from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib._contour import _create_zmap
from optuna.visualization.matplotlib._contour import _create_zmatrix_from_zmap
from optuna.visualization.matplotlib._contour import _find_coordinates_where_empty
from optuna.visualization.matplotlib._contour import _interpolate_zmap
from optuna.visualization.matplotlib._contour import _run_iteration


def test_create_zmap() -> None:

    x_values = np.arange(10)
    y_values = np.arange(10)
    z_values = list(np.random.randn(10))

    # we are testing for exact placement of z_values
    # so also passing x_values and y_values as xi and yi
    zmap = _create_zmap(x_values, y_values, z_values, x_values, y_values)

    assert len(zmap) == len(z_values)
    for coord, value in zmap.items():
        # test if value under coordinate
        # still refers to original trial value
        xidx = int(coord.real)
        yidx = int(coord.imag)
        assert xidx == yidx
        assert z_values[xidx] == value


def test_create_zmatrix_from_zmap() -> None:

    contour_point_num = 2
    zmap = {0 + 0j: 1, 1 + 0j: 2, 0 + 1j: 3, 1 + 1j: 4}
    expected = np.array([[1, 2], [3, 4]])
    zmatrix = _create_zmatrix_from_zmap(zmap, contour_point_num)  # type: ignore

    assert zmatrix.shape == (contour_point_num, contour_point_num)
    assert np.array_equal(zmatrix, expected)


def test_find_coordinates_where_empty() -> None:

    contour_point_num = 2
    zmap = {0 + 0j: 1, 1 + 1j: 4}
    n_missing = (contour_point_num ** 2) - len(zmap)
    empties = _find_coordinates_where_empty(zmap, contour_point_num)  # type: ignore

    # test if all missing are found
    assert len(empties) == n_missing

    # test if iter queue follows C style iteration
    assert empties[0] == 1 + 0j
    assert empties[1] == 0 + 1j


def test_interpolation_first_iteration() -> None:

    zmap = {0 + 0j: 1, 1 + 1j: 4}
    empties = [1 + 0j, 0 + 1j]
    initial_zmap_len = len(zmap)

    zmap, max_fractional_change = _run_iteration(zmap, empties)  # type: ignore

    # test loss after first iter
    assert max_fractional_change == 1.0

    # test if initial pass filled all values
    assert len(zmap) == initial_zmap_len + len(empties)


def test_interpolate_zmap() -> None:

    contour_point_num = 2
    zmap = {0 + 0j: 1, 1 + 1j: 4}
    interpolated = {1 + 0j: 2.5, 1 + 1j: 4, 0 + 0j: 1, 0 + 1j: 2.5}

    zmap = _interpolate_zmap(zmap, contour_point_num)  # type: ignore

    for coord, value in zmap.items():
        expected_at_coord = interpolated.get(coord)
        assert value == expected_at_coord


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
        ["param_b", "param_d"],
        ["param_a", "param_b", "param_c"],
        ["param_a", "param_b", "param_d"],
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
    study = prepare_study_with_trials(more_than_three=True)

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
            assert figure[i][0].get_yaxis().label.get_text() == list(study.best_params)[i]
    else:
        assert figure.shape == (len(params), len(params))
        for i in range(len(params)):
            assert figure[i][0].get_yaxis().label.get_text() == list(params)[i]


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target(params: List[str]) -> None:

    study = prepare_study_with_trials(more_than_three=True)
    with pytest.warns(UserWarning):
        figure = plot_contour(study, params=params, target=lambda t: t.params["param_d"])
    if len(params) == 2:
        assert len(figure.get_lines()) == 0
    else:
        assert figure.shape == (len(params), len(params))
        for i in range(len(params)):
            assert figure[i][0].get_yaxis().label.get_text() == list(params)[i]


@pytest.mark.parametrize(
    "params",
    [
        ["param_a", "param_b"],
        ["param_a", "param_b", "param_c"],
    ],
)
def test_plot_contour_customized_target_name(params: List[str]) -> None:

    study = prepare_study_with_trials(more_than_three=True)
    figure = plot_contour(study, params=params, target_name="Target Name")
    if len(params) == 2:
        assert len(figure.get_lines()) == 0
    else:
        assert figure.shape == (len(params), len(params))
        for i in range(len(params)):
            assert figure[i][0].get_yaxis().label.get_text() == list(params)[i]
