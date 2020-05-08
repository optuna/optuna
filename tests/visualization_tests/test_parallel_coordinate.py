import pytest

from optuna.distributions import CategoricalDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna import type_checking
from optuna.visualization.parallel_coordinate import plot_parallel_coordinate

if type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


def test_plot_parallel_coordinate():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 0

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 2.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][2]["values"] == (2.0, 0.0, 1.0)

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    assert len(figure.data[0]["dimensions"]) == 2
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 2.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)

    # Test with wrong params that do not exist in trials
    with pytest.raises(ValueError, match="Parameter optuna does not exist in your study."):
        plot_parallel_coordinate(study, params=["optuna", "optuna"])

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 0

    # Test with categorical params that cannot be converted to numeral.
    study_categorical_params = create_study()
    study_categorical_params._append_trial(
        value=0.0,
        params={"category_a": "preferred", "category_b": "net",},
        distributions={
            "category_a": CategoricalDistribution(("preferred", "opt")),
            "category_b": CategoricalDistribution(("net", "una")),
        },
    )
    study_categorical_params._append_trial(
        value=2.0,
        params={"category_a": "opt", "category_b": "una",},
        distributions={
            "category_a": CategoricalDistribution(("preferred", "opt")),
            "category_b": CategoricalDistribution(("net", "una")),
        },
    )
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][1]["label"] == "category_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["values"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["ticktext"] == (["preferred", 0], ["opt", 1])
    assert figure.data[0]["dimensions"][2]["label"] == "category_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["values"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["ticktext"] == (["net", 0], ["una", 1])
