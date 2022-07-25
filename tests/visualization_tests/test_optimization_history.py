import numpy as np
import pytest

from optuna.study import create_study
from optuna.testing.objectives import fail_objective
from optuna.trial import Trial
from optuna.visualization import plot_optimization_history
from optuna.visualization._optimization_history import _get_optimization_history_info_list


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_optimization_history(study)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("error_bar", [False, True])
def test_warn_default_target_name_with_customized_target(direction: str, error_bar: bool) -> None:
    # Single study.
    study = create_study(direction=direction)
    with pytest.warns(UserWarning):
        _get_optimization_history_info_list(
            study, target=lambda t: t.number, target_name="Objective Value", error_bar=error_bar
        )

    # Multiple studies.
    studies = [create_study(direction=direction) for _ in range(10)]
    with pytest.warns(UserWarning):
        _get_optimization_history_info_list(
            studies, target=lambda t: t.number, target_name="Objective Value", error_bar=error_bar
        )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
def test_plot_optimization_history(direction: str, target_name: str) -> None:
    # Test with no trial.
    study = create_study(direction=direction)
    figure = plot_optimization_history(study, target_name=target_name)
    assert len(figure.data) == 0

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
    figure = plot_optimization_history(study, target_name=target_name)
    assert len(figure.data) == 2
    assert np.array_equal(figure.data[0].x, [0, 1, 2])
    assert np.array_equal(figure.data[0].y, [1.0, 2.0, 0.0])
    assert np.array_equal(figure.data[1].x, [0, 1, 2])
    ydata = figure.data[1].y
    if direction == "minimize":
        assert np.array_equal(ydata, [1.0, 1.0, 0.0])
    else:
        assert np.array_equal(ydata, [1.0, 2.0, 2.0])
    legend_texts = [x.name for x in figure.data]
    assert legend_texts == [target_name, "Best Value"]
    assert figure.layout.yaxis.title.text == target_name

    # Test customized target.
    figure = plot_optimization_history(study, target=lambda t: t.number, target_name=target_name)
    assert len(figure.data) == 1
    assert np.array_equal(figure.data[0].x, [0, 1, 2])
    assert np.array_equal(figure.data[0].y, [0.0, 1.0, 2.0])

    # Ignore failed trials.
    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_optimization_history(study, target_name=target_name)
    assert len(figure.data) == 0


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
def test_plot_optimization_history_with_multiple_studies(direction: str, target_name: str) -> None:
    n_studies = 10

    # Test with no trial.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    figure = plot_optimization_history(studies, target_name=target_name)
    assert len(figure.data) == 0

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
    figure = plot_optimization_history(studies, target_name=target_name)
    assert len(figure.data) == 2 * n_studies
    assert np.array_equal(figure.data[0].x, [0, 1, 2])
    assert np.array_equal(figure.data[0].y, [1.0, 2.0, 0.0])
    assert np.array_equal(figure.data[1].x, [0, 1, 2])
    ydata = figure.data[1].y
    if direction == "minimize":
        assert np.array_equal(ydata, [1.0, 1.0, 0.0])
    else:
        assert np.array_equal(ydata, [1.0, 2.0, 2.0])
    expected_legend_texts = []
    for i in range(n_studies):
        expected_legend_texts.append(f"Best Value of {studies[i].study_name}")
        expected_legend_texts.append(f"{target_name} of {studies[i].study_name}")
    legend_texts = [scatter.name for scatter in figure.data]
    assert sorted(legend_texts) == sorted(expected_legend_texts)
    assert figure.layout.yaxis.title.text == target_name

    # Test customized target.
    figure = plot_optimization_history(studies, target=lambda t: t.number, target_name=target_name)
    assert len(figure.data) == 1 * n_studies
    assert np.array_equal(figure.data[0].x, [0, 1, 2])
    assert np.array_equal(figure.data[0].y, [0, 1, 2])

    # Ignore failed trials.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    for study in studies:
        study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_optimization_history(studies, target_name=target_name)
    assert len(figure.data) == 0


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize("target_name", ["Objective Value", "Target Name"])
def test_plot_optimization_history_with_error_bar(direction: str, target_name: str) -> None:
    n_studies = 10

    # Test with no trial.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    figure = plot_optimization_history(studies, error_bar=True, target_name=target_name)
    assert len(figure.data) == 0

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
    figure = plot_optimization_history(studies, error_bar=True, target_name=target_name)
    assert len(figure.data) == 4
    assert np.array_equal(figure.data[0].x, [0, 1, 2])
    assert np.array_equal(figure.data[0].y, [1.0, 2.0, 0.0])
    assert np.array_equal(figure.data[1].x, [0, 1, 2])
    ydata = figure.data[1].y
    if direction == "minimize":
        assert np.array_equal(ydata, [1.0, 1.0, 0.0])
    else:
        assert np.array_equal(ydata, [1.0, 2.0, 2.0])
    # Scatters for error bar don't have `name`.
    legend_texts = [scatter.name for scatter in figure.data if scatter.name is not None]
    assert sorted(legend_texts) == ["Best Value", target_name]
    assert figure.layout.yaxis.title.text == target_name

    # Test customized target.
    figure = plot_optimization_history(
        studies, target=lambda t: t.number, error_bar=True, target_name=target_name
    )
    assert len(figure.data) == 1
    assert np.array_equal(figure.data[0].x, [0, 1, 2])
    assert np.array_equal(figure.data[0].y, [0, 1, 2])

    # Ignore failed trials.
    studies = [create_study(direction=direction) for _ in range(n_studies)]
    for study in studies:
        study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_optimization_history(studies, error_bar=True, target_name=target_name)
    assert len(figure.data) == 0


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_error_bar_in_optimization_history(direction: str) -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("x", 0, 1)

    studies = [create_study(direction=direction) for _ in range(3)]
    suggested_params = [0.1, 0.3, 0.2]
    for x, study in zip(suggested_params, studies):
        study.enqueue_trial({"x": x})
        study.optimize(objective, n_trials=1)
    figure = plot_optimization_history(studies, error_bar=True)

    mean = np.mean(suggested_params)
    std = np.std(suggested_params)

    np.testing.assert_almost_equal(figure.data[0].y, mean)
    np.testing.assert_almost_equal(figure.data[2].y, mean + std)
    np.testing.assert_almost_equal(figure.data[3].y, mean - std)
