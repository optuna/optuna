from matplotlib.collections import PathCollection
from matplotlib.collections import LineCollection
import math
import numpy as np

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
    assert len(list(figure.get_figure().axes)) == 0 + 1


    study = prepare_study_with_trials(with_c_d=False)
    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    assert len(list(figure.get_figure().axes)) == 3 + 1
    fig = figure.get_figure()
    assert fig.axes[1].get_ylabel() == "Objective Value"
    assert fig.axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0, 1.0]
    assert fig.axes[2].get_ylim() == (1.0, 2.5)
    assert [ fig.axes[2].get_lines()[0].get_ydata()[0],
             fig.axes[2].get_lines()[0].get_ydata()[-1] ] == [1.0, 2.5]
    assert fig.axes[3].get_ylim() == (0.0, 2.0)
    assert fig.axes[3].get_lines()[0].get_ydata().tolist() == [2.0, 0.0, 1.0]

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    assert len(list(figure.get_figure().axes)) == 2 + 1
    fig = figure.get_figure()
    assert fig.axes[1].get_ylabel() == "Objective Value"
    assert fig.axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0, 1.0]
    assert fig.axes[2].get_ylim() == (1.0, 2.5)
    assert [ fig.axes[2].get_lines()[0].get_ydata()[0],
            fig.axes[2].get_lines()[0].get_ydata()[-1] ] == [1.0, 2.5]

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )

    assert len(list(figure.get_figure().axes)) == 2 + 1
    fig = figure.get_figure()
    assert fig.axes[1].get_ylabel() == "Objective Value"
    assert fig.axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [2.0, 0.0, 1.0]
    assert fig.axes[2].get_ylim() == (1.0, 2.5)
    assert [ fig.axes[2].get_lines()[0].get_ydata()[0],
            fig.axes[2].get_lines()[0].get_ydata()[-1] ] == [1.0, 2.5]

    

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")

    assert len(list(figure.get_figure().axes)) == 3 + 1
    fig = figure.get_figure()
    assert fig.axes[1].get_ylabel() == "Target Name"
    assert fig.axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0, 1.0]
    assert fig.axes[2].get_ylim() == (1.0, 2.5)
    assert [ fig.axes[2].get_lines()[0].get_ydata()[0],
             fig.axes[2].get_lines()[0].get_ydata()[-1] ] == [1.0, 2.5]
    assert fig.axes[3].get_ylim() == (0.0, 2.0)
    assert fig.axes[3].get_lines()[0].get_ydata().tolist() == [2.0, 0.0, 1.0]

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

    assert len(list(figure.get_figure().axes)) == 3 + 1
    fig = figure.get_figure()
    assert fig.axes[1].get_ylabel() == "Objective Value"
    assert fig.axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 2.0]
    assert fig.axes[2].get_ylim() == (0, 1)
    assert [ fig.axes[2].get_lines()[0].get_ydata()[0],
             fig.axes[2].get_lines()[0].get_ydata()[-1] ] == [0, 1]
    assert fig.axes[3].get_ylim() == (0, 1)
    assert fig.axes[3].get_lines()[0].get_ydata().tolist() == [0, 1]


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

    assert len(list(figure.get_figure().axes)) == 3 + 1
    fig = figure.get_figure()
    assert fig.axes[1].get_ylabel() == "Objective Value"
    assert fig.axes[1].get_ylim() == (0.0, 1.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist()[:-1] == [0.0, 1.0, 0.1]
    assert fig.axes[2].get_ylim() == (-6.0, -4.0)
    assert [ fig.axes[2].get_lines()[0].get_ydata()[0],
             fig.axes[2].get_lines()[0].get_ydata()[1],
             fig.axes[2].get_lines()[0].get_ydata()[-1] ] == [-6, math.log10(2e-5), -4]
    assert fig.axes[3].get_ylim() == (1.0, math.log10(200))
    assert fig.axes[3].get_lines()[0].get_ydata().tolist() == [1.0, math.log10(200), math.log10(30)]


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

    # both hyperparameters contain unique values
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

    # still "category_a" contains unique suggested value during the optimization.
    figure = plot_parallel_coordinate(study_categorical_params)
    assert len(figure.get_lines()) == 0
