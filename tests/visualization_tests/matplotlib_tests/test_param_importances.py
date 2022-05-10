from io import BytesIO
import math
from typing import Tuple

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pytest

from optuna.distributions import FloatDistribution
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance._base import BaseImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_param_importances


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_param_importances(study)


def test_plot_param_importances() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_param_importances(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    study = prepare_study_with_trials(with_c_d=True)

    # Test with a trial.
    figure = plot_param_importances(study)

    bars = figure.findobj(Rectangle)[:-1]  # The last Rectangle is the plot itself.
    plotted_data = [bar.get_width() for bar in bars]
    # get_yticklabels returns a data structure of Text(0, 0, 'param_d').
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(figure.get_lines()) == 0
    assert len(bars) == 2
    assert set(labels) == set(("param_b", "param_d"))  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert figure.xaxis.label.get_text() == "Importance for Objective Value"
    plt.savefig(BytesIO())

    # Test with an evaluator.
    plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())

    bars = figure.findobj(Rectangle)[:-1]  # The last Rectangle is the plot itself.
    plotted_data = [bar.get_width() for bar in bars]
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(figure.get_lines()) == 0
    assert len(bars) == 2
    assert set(labels) == set(("param_b", "param_d"))  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert figure.xaxis.label.get_text() == "Importance for Objective Value"
    plt.savefig(BytesIO())

    # Test with a trial to select parameter.
    figure = plot_param_importances(study, params=["param_b"])

    bars = figure.findobj(Rectangle)[:-1]  # The last Rectangle is the plot itself.
    plotted_data = [bar.get_width() for bar in bars]
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(figure.get_lines()) == 0
    assert len(bars) == 1
    assert set(labels) == set(("param_b",))
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert figure.xaxis.label.get_text() == "Importance for Objective Value"
    plt.savefig(BytesIO())

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_param_importances(
            study, target=lambda t: t.params["param_b"] + t.params["param_d"]
        )
    bars = figure.findobj(Rectangle)[:-1]  # The last Rectangle is the plot itself.
    plotted_data = [bar.get_width() for bar in bars]
    labels = [label.get_text() for label in figure.get_yticklabels()]

    assert len(bars) == 2
    assert set(labels) == set(("param_b", "param_d"))  # "param_a", "param_c" are conditional.
    assert math.isclose(1.0, sum(i for i in plotted_data), abs_tol=1e-5)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with a customized target name.
    figure = plot_param_importances(study, target_name="Target Name")
    assert len(figure.get_lines()) == 0
    assert figure.xaxis.label.get_text() == "Importance for Target Name"
    plt.savefig(BytesIO())

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
    plt.savefig(BytesIO())


def test_importance_scores_rendering() -> None:

    study = prepare_study_with_trials()
    ax = plot_param_importances(study)

    # Test if importance scores are rendered.
    text_objects = ax.figure.findobj(lambda obj: "Text" in str(obj))
    importances = [patch.get_width() for patch in ax.patches]
    labels = [obj for obj in text_objects if obj.get_position()[0] in importances]
    assert len(labels) == 2
    plt.savefig(BytesIO())


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
    plt.savefig(BytesIO())


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "evaluator",
    [MeanDecreaseImpurityImportanceEvaluator(seed=10), FanovaImportanceEvaluator(seed=10)],
)
@pytest.mark.parametrize("n_trial", [0, 10])
def test_trial_with_infinite_value_ignored(
    inf_value: float, evaluator: BaseImportanceEvaluator, n_trial: int
) -> None:
    def _objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1 + x2 * x3

    seed = 13

    study = create_study(sampler=RandomSampler(seed=seed))
    study.optimize(_objective, n_trials=n_trial)

    # A figure is created without a trial with an inf value.
    plot_param_importances(study, evaluator=evaluator)
    with BytesIO() as byte_io:
        plt.savefig(byte_io)
        figure_with_inf = byte_io.getvalue()

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            value=inf_value,
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )

    # A figure is created with a trial with an inf value.
    plot_param_importances(study, evaluator=evaluator)
    with BytesIO() as byte_io:
        plt.savefig(byte_io)
        figure_without_inf = byte_io.getvalue()

    # Obtained figures should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    assert len(figure_without_inf) > 0
    assert figure_with_inf == figure_without_inf


@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "evaluator",
    [MeanDecreaseImpurityImportanceEvaluator(seed=10), FanovaImportanceEvaluator(seed=10)],
)
@pytest.mark.parametrize("n_trial", [0, 10])
def test_multi_objective_trial_with_infinite_value_ignored(
    target_idx: int, inf_value: float, evaluator: BaseImportanceEvaluator, n_trial: int
) -> None:
    def _multi_objective_function(trial: Trial) -> Tuple[float, float]:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1, x2 * x3

    seed = 13

    study = create_study(directions=["minimize", "minimize"], sampler=RandomSampler(seed=seed))
    study.optimize(_multi_objective_function, n_trials=n_trial)

    # A figure is created without a trial with an inf value.
    plot_param_importances(study, evaluator=evaluator, target=lambda t: t.values[target_idx])
    with BytesIO() as byte_io:
        plt.savefig(byte_io)
        figure_with_inf = byte_io.getvalue()

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            values=[inf_value, inf_value],
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )

    # A figure is created with a trial with an inf value.
    plot_param_importances(study, evaluator=evaluator, target=lambda t: t.values[target_idx])
    with BytesIO() as byte_io:
        plt.savefig(byte_io)
        figure_without_inf = byte_io.getvalue()

    # Obtained figures should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    assert len(figure_without_inf) > 0
    assert figure_without_inf == figure_with_inf
