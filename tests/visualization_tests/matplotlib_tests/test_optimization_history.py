import pytest

from optuna.study import create_study
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_optimization_history


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_optimization_history(study)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history(direction: str) -> None:
    # Test with no trial.
    study = create_study(direction=direction)
    figure = plot_optimization_history(study)
    assert len(figure.get_lines()) == 0

    def objective(trial: Trial) -> float:

        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with a trial.
    study = create_study(direction=direction)
    study.optimize(objective, n_trials=3)
    figure = plot_optimization_history(study)
    assert len(figure.get_lines()) == 1
    assert list(figure.get_lines()[0].get_xdata()) == [0, 1, 2]

    # Test customized target.
    with pytest.warns(UserWarning):
        figure = plot_optimization_history(study, target=lambda t: t.number)
    assert len(figure.get_lines()) == 0

    # Test customized target name.
    figure = plot_optimization_history(study, target_name="Target Name")
    assert len(figure.get_lines()) == 1
    assert figure.yaxis.label.get_text() == "Target Name"

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:
        raise ValueError

    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    figure = plot_optimization_history(study)
    assert len(figure.get_lines()) == 0


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_plot_optimization_history_with_multiple_studies(direction: str) -> None:
    n_studies = 10

    # Test with no trial.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    figure = plot_optimization_history(studies)
    assert len(figure.get_lines()) == 0

    def objective(trial: Trial) -> float:

        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with trials.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    for study in studies:
        study.optimize(objective, n_trials=3)
    figure = plot_optimization_history(studies)
    assert len(figure.get_lines()) == n_studies

    legend_texts = []
    expected_legend_texts = []
    for i, legend in enumerate(figure.legend().get_texts()):
        legend_texts.append(legend.get_text())

        if i < n_studies:
            expected_legend_texts.append(f"Best Values of {studies[i].study_name}")
        else:
            expected_legend_texts.append(f"Objective Value of {studies[i-n_studies].study_name}")

    assert sorted(legend_texts) == sorted(expected_legend_texts)

    # Test customized target.
    with pytest.warns(UserWarning):
        figure = plot_optimization_history(studies, target=lambda t: t.number)
    assert len(figure.get_lines()) == 0
    assert len(figure.get_legend().get_texts()) == n_studies

    # Test customized target name.
    custom_target_name = "Target Name"
    figure = plot_optimization_history(studies, target_name=custom_target_name)

    legend_texts = []
    expected_legend_texts = []
    for i, legend in enumerate(figure.legend().get_texts()):
        legend_texts.append(legend.get_text())

        if i < n_studies:
            expected_legend_texts.append(f"Best Values of {studies[i].study_name}")
        else:
            expected_legend_texts.append(
                f"{custom_target_name} of {studies[i-n_studies].study_name}"
            )

    assert sorted(legend_texts) == sorted(expected_legend_texts)
    assert figure.get_ylabel() == custom_target_name

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:
        raise ValueError

    studies = [create_study(direction=direction) for _ in range(n_studies)]
    for study in studies:
        study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    figure = plot_optimization_history(studies)
    assert len(figure.get_lines()) == 0
