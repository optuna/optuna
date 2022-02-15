import pytest
import math

from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_param_importances
from matplotlib.patches import Rectangle


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_param_importances(study)


def test_plot_param_importances() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_param_importances(study)
    assert len(figure.get_lines()) == 0

    study = prepare_study_with_trials(with_c_d=True)

    # Test with a trial.
    figure = plot_param_importances(study)

    bars = figure.findobj(Rectangle)[:-1]  # the last Rectangle is the plot itself
    plotted_data = [bar.get_width() for bar in bars]
    # get_yticklabels return a data structure of Text(0, 0, 'param_d')
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(figure.get_lines()) == 0
    assert len(bars) == 2
    assert set(labels) == set(
        ("param_b", "param_d")
    )  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert figure.xaxis.label.get_text() == "Importance for Objective Value"

    # Test with an evaluator.
    plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())

    bars = figure.findobj(Rectangle)[:-1]  # the last Rectangle is the plot itself
    plotted_data = [bar.get_width() for bar in bars]
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(figure.get_lines()) == 0
    assert len(bars) == 2
    assert set(labels) == set(
        ("param_b", "param_d")
    )  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert figure.xaxis.label.get_text() == "Importance for Objective Value"

    # Test with a trial to select parameter.
    figure = plot_param_importances(study, params=["param_b"])

    bars = figure.findobj(Rectangle)[:-1]  # the last Rectangle is the plot itself
    plotted_data = [bar.get_width() for bar in bars]
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(figure.get_lines()) == 0
    assert len(bars) == 1
    assert set(labels) == set(
        ("param_b",)
    )  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert figure.xaxis.label.get_text() == "Importance for Objective Value"

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_param_importances(
            study, target=lambda t: t.params["param_b"] + t.params["param_d"]
        )
    bars = figure.findobj(Rectangle)[:-1]  # the last Rectangle is the plot itself
    plotted_data = [bar.get_width() for bar in bars]
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(bars) == 2
    assert set(labels) == set(
        ("param_b", "param_d")
    )  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert len(figure.get_lines()) == 0

    # Test with a customized target name.
    figure = plot_param_importances(study, target_name="Target Name")
    assert len(figure.get_lines()) == 0
    assert figure.xaxis.label.get_text() == "Importance for Target Name"

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_param_importances(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_param_importances(study)
    assert len(figure.get_lines()) == 0


def test_importance_scores_rendering() -> None:

    study = prepare_study_with_trials()
    ax = plot_param_importances(study)

    # Test if importance scores are rendered.
    text_objects = ax.figure.findobj(lambda obj: "Text" in str(obj))
    importances = [patch.get_width() for patch in ax.patches]
    labels = [obj for obj in text_objects if obj.get_position()[0] in importances]
    assert len(labels) == 2


def test_switch_label_when_param_insignificant() -> None:
    def _objective(trial: Trial) -> int:
        x = trial.suggest_int("x", 0, 2)
        _ = trial.suggest_int("y", -1, 1)
        return x**2

    study = create_study()
    for x in range(1, 3):
        study.enqueue_trial({"x": x, "y": 0})

    study.optimize(_objective, n_trials=2)
    ax = plot_param_importances(study)

    # Test if label for `y` param has been switched to `<0.01`.
    labels = ax.figure.findobj(lambda obj: "<0.01" in str(obj))
    assert len(labels) == 1
