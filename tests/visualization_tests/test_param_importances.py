import math

import pytest

from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization import plot_param_importances


def test_plot_param_importances() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_param_importances(study)
    assert len(figure.data) == 0

    study = prepare_study_with_trials(with_c_d=True)

    # Test with a trial.
    figure = plot_param_importances(study)
    assert len(figure.data) == 1
    assert set(figure.data[0].y) == set(
        ("param_b", "param_d")
    )  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in figure.data[0].x), abs_tol=1e-5)

    # Test with an evaluator.
    plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    assert len(figure.data) == 1
    assert set(figure.data[0].y) == set(
        ("param_b", "param_d")
    )  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in figure.data[0].x), abs_tol=1e-5)

    # Test with a trial to select parameter.
    figure = plot_param_importances(study, params=["param_b"])
    assert len(figure.data) == 1
    assert figure.data[0].y == ("param_b",)
    assert math.isclose(1.0, sum(i for i in figure.data[0].x), abs_tol=1e-5)

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_param_importances(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_param_importances(study)
    assert len(figure.data) == 0
