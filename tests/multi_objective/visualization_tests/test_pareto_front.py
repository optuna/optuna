import pytest

import optuna
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.visualization import plot_pareto_front


def test_plot_pareto_front_2d() -> None:
    # Test with no trial.
    study = optuna.multi_objective.create_study(["minimize", "minimize"])

    with pytest.raises(ValueError):
        plot_pareto_front(study)

    study.enqueue_trial({"x": 1, "y": 1})
    study.enqueue_trial({"x": 1, "y": 0})
    study.enqueue_trial({"x": 0, "y": 1})
    study.optimize(lambda t: [t.suggest_int("x", 0, 1), t.suggest_int("y", 0, 1)], n_trials=3)

    figure = plot_pareto_front(study)
    assert len(figure.data) == 1
    assert figure.data[0]["x"] == (1, 0)
    assert figure.data[0]["y"] == (0, 1)

    # TODO: name check


def test_plot_pareto_front_3d() -> None:
    # Test with no trial.
    study = optuna.multi_objective.create_study(["minimize", "maximize", "minimize"])

    with pytest.raises(ValueError):
        plot_pareto_front(study)

    # study.enqueue_trial({"x": 1, "y": 1})
    # study.enqueue_trial({"x": 1, "y": 0})
    # study.enqueue_trial({"x": 0, "y": 1})
    # study.optimize(lambda t: [t.suggest_int("x", 0, 1), t.suggest_int("y", 0, 1)], n_trials=3)

    # figure = plot_pareto_front(study)
    # assert len(figure.data) == 1
    # assert figure.data[0]["x"] == (1, 0)
    # assert figure.data[0]["y"] == (0, 1)

    # TODO: name check
