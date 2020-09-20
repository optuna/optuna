from typing import List
from typing import Optional

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
from optuna.study import create_study
from optuna.study import StudyDirection
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib._contour import _generate_contour_subplot
from optuna.visualization.matplotlib import plot_contour


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
    assert figure.has_data() is False

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_contour(study, params=params)
    assert figure.has_data() is False

    # Test with some trials.
    study = prepare_study_with_trials()

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        plot_contour(study, ["optuna", "Optuna"])

    figure = plot_contour(study, params=params)
    if params is not None and len(params) < 3:
        if len(params) <= 1:
            assert figure.has_data() is False
        elif len(params) == 2:
            # TODO(ytknzw): Add more specific assertion with the test case.
            assert figure.has_data() is True
    elif params is None:
        # TODO(ytknzw): Add more specific assertion with the test case.
        assert figure.shape == (len(study.best_params), len(study.best_params))
    else:
        # TODO(ytknzw): Add more specific assertion with the test case.
        assert figure.shape == (len(params), len(params))
