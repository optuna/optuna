from typing import List
from typing import Optional

import pytest

from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_contour


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
    assert not figure.has_data()

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    assert not figure.has_data()

    # Test with some trials.
    study = prepare_study_with_trials(more_than_three=True)

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        plot_contour(study, ["optuna", "Optuna"])

    figure = plot_contour(study, params=params)
    if params is not None and len(params) < 3:
        if len(params) <= 1:
            assert not figure.has_data()
        elif len(params) == 2:
            # TODO(ytknzw): Add more specific assertion with the test case.
            assert figure.has_data()
    elif params is None:
        # TODO(ytknzw): Add more specific assertion with the test case.
        assert figure.shape == (len(study.best_params), len(study.best_params))
    else:
        # TODO(ytknzw): Add more specific assertion with the test case.
        assert figure.shape == (len(params), len(params))


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
        assert figure.has_data()
    else:
        assert figure.shape == (len(params), len(params))


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
        assert figure.has_data()
    else:
        assert figure.shape == (len(params), len(params))
