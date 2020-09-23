import pytest

from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_param_importances


def test_plot_param_importances() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_param_importances(study)
    assert not figure.has_data()

    study = prepare_study_with_trials(with_c_d=True)

    # Test with a trial.
    # TODO(ytknzw): Add more specific assertion with the test case.
    figure = plot_param_importances(study)
    assert figure.has_data()

    # Test with an evaluator.
    # TODO(ytknzw): Add more specific assertion with the test case.
    plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    assert figure.has_data()

    # Test with a trial to select parameter.
    # TODO(ytknzw): Add more specific assertion with the test case.
    figure = plot_param_importances(study, params=["param_b"])
    assert figure.has_data()

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_param_importances(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_param_importances(study)
    assert not figure.has_data()
