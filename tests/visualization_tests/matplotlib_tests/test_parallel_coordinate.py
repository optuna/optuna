from io import BytesIO

import matplotlib.pyplot as plt
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_parallel_coordinate


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_parallel_coordinate(study)


def test_plot_parallel_coordinate() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_parallel_coordinate(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    # Test with wrong params that do not exist in trials
    with pytest.raises(ValueError, match="Parameter optuna does not exist in your study."):
        plot_parallel_coordinate(study, params=["optuna", "optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_parallel_coordinate(study)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())


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
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_categorical_numeric_params() -> None:
    # Test with categorical params that can be interpreted as numeric params.
    study_categorical_params = create_study()
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": 2, "category_b": 20},
            distributions={
                "category_a": CategoricalDistribution((1, 2)),
                "category_b": CategoricalDistribution((10, 20, 30)),
            },
        )
    )

    study_categorical_params.add_trial(
        create_trial(
            value=1.0,
            params={"category_a": 1, "category_b": 30},
            distributions={
                "category_a": CategoricalDistribution((1, 2)),
                "category_b": CategoricalDistribution((10, 20, 30)),
            },
        )
    )

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": 2, "category_b": 10},
            distributions={
                "category_a": CategoricalDistribution((1, 2)),
                "category_b": CategoricalDistribution((10, 20, 30)),
            },
        )
    )
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_log_params() -> None:
    # Test with log params
    study_log_params = create_study()
    study_log_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": 10},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2e-5, "param_b": 200},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=0.1,
            params={"param_a": 1e-4, "param_b": 30},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2, log=True),
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )
    figure = plot_parallel_coordinate(study_log_params)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_unique_hyper_param() -> None:
    # Test case when one unique value is suggested during the optimization.

    study_categorical_params = create_study()
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "param_b": 30},
            distributions={
                "category_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )

    # both hyperparameters contain unique values
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "preferred", "param_b": 20},
            distributions={
                "category_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": FloatDistribution(1, 1000, log=True),
            },
        )
    )

    # still "category_a" contains unique suggested value during the optimization.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.get_lines()) == 0
    plt.savefig(BytesIO())


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_removed(value: float) -> None:

    study = prepare_study_with_trials(value_for_first_trial=value)
    plot_parallel_coordinate(study)
    plt.savefig(BytesIO())


@pytest.mark.parametrize("objective", (0, 1))
@pytest.mark.parametrize("value", (float("inf"), -float("inf"), float("nan")))
def test_nonfinite_multiobjective(objective: int, value: float) -> None:

    study = prepare_study_with_trials(n_objectives=2, value_for_first_trial=value)
    plot_parallel_coordinate(study, target=lambda t: t.values[objective])
    plt.savefig(BytesIO())
