from matplotlib.collections import LineCollection
import math

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
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
    assert len(figure.get_figure().axes) == 0 + 1

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    assert [
        axes[2].get_lines()[0].get_ydata()[0],
        axes[2].get_lines()[0].get_ydata()[-1],
    ] == [1.0, 2.5]
    assert axes[3].get_ylim() == (0.0, 2.0)
    assert axes[3].get_lines()[0].get_ydata().tolist() == [2.0, 0.0, 1.0]
    line_collections = figure.findobj(LineCollection)
    assert len(line_collections) == 1
    assert line_collections[0].get_array().tolist()[:-1] == [0.0, 2.0, 1.0]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    assert xticklabels[1].get_text() == "param_a"
    assert xticklabels[2].get_text() == "param_b"

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    axes = figure.get_figure().axes
    assert len(axes) == 2 + 1
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0, 1.0]
    assert axes[2].get_ylim() == (1.0, 2.5)
    assert [
        axes[2].get_lines()[0].get_ydata()[0],
        axes[2].get_lines()[0].get_ydata()[-1],
    ] == [1.0, 2.5]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    assert xticklabels[1].get_text() == "param_a"

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    axes = figure.get_figure().axes
    assert len(axes) == 2 + 1
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [2.0, 0.0, 1.0]
    assert axes[2].get_ylim() == (1.0, 2.5)
    assert [
        axes[2].get_lines()[0].get_ydata()[0],
        axes[2].get_lines()[0].get_ydata()[-1],
    ] == [1.0, 2.5]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    assert xticklabels[1].get_text() == "param_a"

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[1].get_ylabel() == "Target Name"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0, 1.0]
    assert axes[2].get_ylim() == (1.0, 2.5)
    assert [
        axes[2].get_lines()[0].get_ydata()[0],
        axes[2].get_lines()[0].get_ydata()[-1],
    ] == [1.0, 2.5]
    assert axes[3].get_ylim() == (0.0, 2.0)
    assert axes[3].get_lines()[0].get_ydata().tolist() == [2.0, 0.0, 1.0]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Target Name"
    assert xticklabels[1].get_text() == "param_a"
    assert xticklabels[2].get_text() == "param_b"

    # Test with wrong params that do not exist in trials.
    with pytest.raises(ValueError, match="Parameter optuna does not exist in your study."):
        plot_parallel_coordinate(study, params=["optuna", "optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_parallel_coordinate(study)
    assert len(figure.get_lines()) == 0


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
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0]
    assert axes[2].get_ylim() == (0, 1)
    assert axes[2].get_lines()[0].get_ydata().tolist() == [0, 1]
    assert [l.get_text() for l in axes[2].get_yticklabels()] == ["preferred", "opt"]
    assert axes[3].get_ylim() == (0, 1)
    assert axes[3].get_lines()[0].get_ydata().tolist() == [0, 1]
    assert [l.get_text() for l in axes[3].get_yticklabels()] == ["net", "una"]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    assert xticklabels[1].get_text() == "category_a"
    assert xticklabels[2].get_text() == "category_b"


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

    # Trials are sorted by using param_a and param_b, i.e., trial#1, trial#2, and trial#0.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    # TODO(nzw0301): implement the validation of values of objectives
    # assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [1.0, 2.0,]
    assert axes[2].get_ylim() == (0, 1)
    assert axes[2].get_lines()[0].get_ydata().tolist() == [0, 1, 1]
    assert [int(l.get_text()) for l in axes[2].get_yticklabels()] == [1, 2]
    assert [l.get_position()[1] for l in axes[2].get_yticklabels()] == [0, 1]
    assert axes[3].get_ylim() == (0, 2)
    assert axes[3].get_lines()[0].get_ydata().tolist() == [2, 0, 1]
    assert [int(l.get_text()) for l in axes[3].get_yticklabels()] == [10, 20, 30]
    assert [l.get_position()[1] for l in axes[3].get_yticklabels()] == [0, 1, 2]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    assert xticklabels[1].get_text() == "category_a"
    assert xticklabels[2].get_text() == "category_b"


def test_plot_parallel_coordinate_log_params() -> None:
    # Test with log params
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

    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 1.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 1.0, 0.1]
    assert axes[2].get_ylim() == (-6.0, -4.0)
    assert [
        axes[2].get_lines()[0].get_ydata()[0],
        axes[2].get_lines()[0].get_ydata()[1],
        axes[2].get_lines()[0].get_ydata()[-1],
    ] == [-6, math.log10(2e-5), -4]
    assert axes[3].get_ylim() == (1.0, math.log10(200))
    assert axes[3].get_lines()[0].get_ydata().tolist() == [
        1.0,
        math.log10(200),
        math.log10(30),
    ]
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    assert xticklabels[1].get_text() == "param_a"
    assert xticklabels[2].get_text() == "param_b"


def test_plot_parallel_coordinate_unique_hyper_param() -> None:
    # Test case when one unique value is suggested during the optimization.
    study_categorical_params = create_study()
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "param_b": 30},
            distributions={
                "category_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": LogUniformDistribution(1, 1000),
            },
        )
    )

    # Both hyperparameters contain unique values.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.get_lines()) == 0

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "preferred", "param_b": 20},
            distributions={
                "category_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": LogUniformDistribution(1, 1000),
            },
        )
    )

    # Still "category_a" contains unique suggested value during the optimization.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.get_lines()) == 0


def test_plot_parallel_coordinate_with_categorical_numeric_params() -> None:
    # Test with sample from mulitiple distributions including categorical params
    # that can be interpreted as numeric params.
    study_multi_distro_params = create_study()
    study_multi_distro_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": "preferred", "param_b": 2, "param_c": 30, "param_d": 2},
            distributions={
                "param_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": CategoricalDistribution((1, 2, 10)),
                "param_c": LogUniformDistribution(1, 1000),
                "param_d": CategoricalDistribution((1, -1, 2)),
            },
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": "opt", "param_b": 1, "param_c": 200, "param_d": 2},
            distributions={
                "param_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": CategoricalDistribution((1, 2, 10)),
                "param_c": LogUniformDistribution(1, 1000),
                "param_d": CategoricalDistribution((1, -1, 2)),
            },
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=2.0,
            params={"param_a": "preferred", "param_b": 10, "param_c": 10, "param_d": 1},
            distributions={
                "param_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": CategoricalDistribution((1, 2, 10)),
                "param_c": LogUniformDistribution(1, 1000),
                "param_d": CategoricalDistribution((1, -1, 2)),
            },
        )
    )

    study_multi_distro_params.add_trial(
        create_trial(
            value=3.0,
            params={"param_a": "opt", "param_b": 2, "param_c": 10, "param_d": -1},
            distributions={
                "param_a": CategoricalDistribution(("preferred", "opt")),
                "param_b": CategoricalDistribution((1, 2, 10)),
                "param_c": LogUniformDistribution(1, 1000),
                "param_d": CategoricalDistribution((-1, 1, 2)),
            },
        )
    )
