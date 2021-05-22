import math

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization import plot_parallel_coordinate


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_parallel_coordinate(study)


def test_plot_parallel_coordinate() -> None:

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

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    assert len(figure.data[0]["dimensions"]) == 2
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (2.0, 0.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Target Name"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 2.0, 1.0)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][1]["values"] == (1.0, 2.5)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][2]["values"] == (2.0, 0.0, 1.0)

    # Test with wrong params that do not exist in trials
    with pytest.raises(ValueError, match="Parameter optuna does not exist in your study."):
        plot_parallel_coordinate(study, params=["optuna", "optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 0


def test_plot_parallel_coordinate_categorical_params() -> None:
    # Test with categorical params that cannot be converted to numeral.
    study_categorical_params = create_study()
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "category_b": "net"},
            distributions={
                "category_a": CategoricalDistribution(("preferred", "opt")),
                "category_b": CategoricalDistribution(("net", "una")),
            },
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "opt", "category_b": "una"},
            distributions={
                "category_a": CategoricalDistribution(("preferred", "opt")),
                "category_b": CategoricalDistribution(("net", "una")),
            },
        )
    )
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 2.0)
    assert figure.data[0]["dimensions"][1]["label"] == "category_a"
    assert figure.data[0]["dimensions"][1]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["values"] == (0, 1)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("preferred", "opt")
    assert figure.data[0]["dimensions"][2]["label"] == "category_b"
    assert figure.data[0]["dimensions"][2]["range"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["values"] == (0, 1)
    assert figure.data[0]["dimensions"][2]["ticktext"] == ("net", "una")


def test_plot_parallel_coordinate_log_params() -> None:
    # Test with log params.
    study_log_params = create_study()
    study_log_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": 10},
            distributions={
                "param_a": LogUniformDistribution(1e-7, 1e-2),
                "param_b": LogUniformDistribution(1, 1000),
            },
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2e-5, "param_b": 200},
            distributions={
                "param_a": LogUniformDistribution(1e-7, 1e-2),
                "param_b": LogUniformDistribution(1, 1000),
            },
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=0.1,
            params={"param_a": 1e-4, "param_b": 30},
            distributions={
                "param_a": LogUniformDistribution(1e-7, 1e-2),
                "param_b": LogUniformDistribution(1, 1000),
            },
        )
    )
    figure = plot_parallel_coordinate(study_log_params)
    assert len(figure.data[0]["dimensions"]) == 3
    assert figure.data[0]["dimensions"][0]["label"] == "Objective Value"
    assert figure.data[0]["dimensions"][0]["range"] == (0.0, 1.0)
    assert figure.data[0]["dimensions"][0]["values"] == (0.0, 1.0, 0.1)
    assert figure.data[0]["dimensions"][1]["label"] == "param_a"
    assert figure.data[0]["dimensions"][1]["range"] == (-6.0, -4.0)
    assert figure.data[0]["dimensions"][1]["values"] == (-6, math.log10(2e-5), -4)
    assert figure.data[0]["dimensions"][1]["ticktext"] == ("1e-06", "1e-05", "0.0001")
    assert figure.data[0]["dimensions"][1]["tickvals"] == (-6, -5, -4.0)
    assert figure.data[0]["dimensions"][2]["label"] == "param_b"
    assert figure.data[0]["dimensions"][2]["range"] == (1.0, math.log10(200))
    assert figure.data[0]["dimensions"][2]["values"] == (1.0, math.log10(200), math.log10(30))
    assert figure.data[0]["dimensions"][2]["ticktext"] == ("10", "100", "200")
    assert figure.data[0]["dimensions"][2]["tickvals"] == (1.0, 2.0, math.log10(200))
