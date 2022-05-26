from io import BytesIO
import string
from typing import Dict
from typing import List

from matplotlib.axes._axes import Axes
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import pytest

from optuna.distributions import BaseDistribution
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


def _fetch_objectives_from_figure(figure: Axes) -> List[float]:
    # Fetch line plots in parallel coordinate.
    line_collections = figure.findobj(LineCollection)
    assert len(line_collections) == 1

    # Fetch objective values from line plots.
    objectives = [line[0, 1] for line in line_collections[0].get_segments()]
    return objectives


def _test_xtick_labels(axes: Axes, expected_labels: List[str]) -> None:
    xtick_labels = axes[0].get_xticklabels()

    assert len(expected_labels) == len(xtick_labels)
    for expected_label, xtick_label in zip(expected_labels, xtick_labels):
        assert expected_label == xtick_label.get_text()


def test_plot_parallel_coordinate() -> None:

    # Test with no trial.
    study = create_study()
    figure = plot_parallel_coordinate(study)
    assert len(figure.get_figure().axes) == 0 + 1
    plt.savefig(BytesIO())

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    # axes[0] is the objective vertical line.
    assert axes[0].get_ylim() == (0.0, 1.0)
    # axes[1] is colorbar.
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 1.0)
    # axes[2] is `param_a`'s vertical line.
    assert axes[2].get_ylim() == (1.0, 2.5)
    # axes[3] is `param_b`'s vertical line.
    assert axes[3].get_ylim() == (1.0, 2.0)
    objectives = _fetch_objectives_from_figure(figure)
    assert objectives == [0.0, 1.0]
    expected_labels = ["Objective Value", "param_a", "param_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    axes = figure.get_figure().axes
    assert len(axes) == 2 + 1
    assert axes[0].get_ylim() == (0.0, 1.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 1.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    objectives = _fetch_objectives_from_figure(figure)
    assert objectives == [0.0, 1.0]
    expected_labels = ["Objective Value", "param_a"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    axes = figure.get_figure().axes
    assert len(axes) == 2 + 1
    assert axes[0].get_ylim() == (1.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (1.0, 2.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    objectives = _fetch_objectives_from_figure(figure)
    assert objectives == [2.0, 1.0]
    expected_labels = ["Objective Value", "param_a"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 1.0)
    assert axes[1].get_ylabel() == "Target Name"
    assert axes[1].get_ylim() == (0.0, 1.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    assert axes[3].get_ylim() == (1.0, 2.0)
    objectives = _fetch_objectives_from_figure(figure)
    assert objectives == [0.0, 1.0]
    expected_labels = ["Target Name", "param_a", "param_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())

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
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_categorical_params() -> None:
    # Test with categorical params that cannot be converted to numeral.
    study_categorical_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution(("preferred", "opt")),
        "category_b": CategoricalDistribution(("net", "una")),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "category_b": "net"},
            distributions=distributions,
        )
    )
    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "opt", "category_b": "una"},
            distributions=distributions,
        )
    )
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert axes[2].get_ylim() == (0, 1)
    assert [line.get_text() for line in axes[2].get_yticklabels()] == ["preferred", "opt"]
    assert axes[3].get_ylim() == (0, 1)
    assert [line.get_text() for line in axes[3].get_yticklabels()] == ["net", "una"]
    objectives = _fetch_objectives_from_figure(figure)
    assert objectives == [0.0, 2.0]
    expected_labels = ["Objective Value", "category_a", "category_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_categorical_numeric_params() -> None:
    # Test with categorical params that can be interpreted as numeric params.
    study_categorical_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution((1, 2)),
        "category_b": CategoricalDistribution((10, 20, 30)),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": 2, "category_b": 20},
            distributions=distributions,
        )
    )

    study_categorical_params.add_trial(
        create_trial(
            value=1.0,
            params={"category_a": 1, "category_b": 30},
            distributions=distributions,
        )
    )

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": 2, "category_b": 10},
            distributions=distributions,
        )
    )

    # Trials are sorted by using param_a and param_b, i.e., trial#1, trial#2, and trial#0.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    objectives = _fetch_objectives_from_figure(figure)
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert objectives == [0.0, 1.0, 2.0]
    num_choices_category_a = 2
    assert axes[2].get_ylim() == (0, num_choices_category_a - 1)
    assert [int(label.get_text()) for label in axes[2].get_yticklabels()] == [1, 2]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == list(
        range(num_choices_category_a)
    )
    num_choices__category_b = 3
    assert axes[3].get_ylim() == (0, num_choices__category_b - 1)
    assert [int(label.get_text()) for label in axes[3].get_yticklabels()] == [10, 20, 30]
    assert [label.get_position()[1] for label in axes[3].get_yticklabels()] == list(
        range(num_choices__category_b)
    )
    expected_labels = ["Objective Value", "category_a", "category_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_log_params() -> None:
    # Test with log params.
    study_log_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "param_a": FloatDistribution(1e-7, 1e-2, log=True),
        "param_b": FloatDistribution(1, 1000, log=True),
    }
    study_log_params.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6, "param_b": 10},
            distributions=distributions,
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": 2e-5, "param_b": 200},
            distributions=distributions,
        )
    )
    study_log_params.add_trial(
        create_trial(
            value=0.1,
            params={"param_a": 1e-4, "param_b": 30},
            distributions=distributions,
        )
    )
    figure = plot_parallel_coordinate(study_log_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 1.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 1.0)
    objectives = _fetch_objectives_from_figure(figure)
    assert objectives == [0.0, 1.0, 0.1]
    assert axes[2].get_ylim() == (1e-6, 1e-4)
    assert axes[3].get_ylim() == (10.0, 200)
    expected_labels = ["Objective Value", "param_a", "param_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_unique_hyper_param() -> None:
    # Test case when one unique value is suggested during the optimization.

    study_categorical_params = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "category_a": CategoricalDistribution(("preferred", "opt")),
        "param_b": FloatDistribution(1, 1000, log=True),
    }
    study_categorical_params.add_trial(
        create_trial(
            value=0.0,
            params={"category_a": "preferred", "param_b": 30},
            distributions=distributions,
        )
    )

    # Both hyperparameters contain unique values.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    # Default padding is 5% in Matplotlib.
    default_padding_fraction = plt.margins()[0]
    assert axes[0].get_ylim() == (-default_padding_fraction, default_padding_fraction)
    assert axes[1].get_ylabel() == "Objective Value"
    # Optuna's parallel coordinate uses 10% padding for color map.
    assert axes[1].get_ylim() == (-0.1, 0.1)
    objectives = _fetch_objectives_from_figure(figure)
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert objectives == [0.0]
    assert axes[2].get_ylim() == (-default_padding_fraction, default_padding_fraction)
    assert [line.get_text() for line in axes[2].get_yticklabels()] == ["preferred"]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == [0]
    assert axes[3].get_ylim() == (
        30 * (1.0 - default_padding_fraction),
        30 * (1.0 + default_padding_fraction),
    )
    expected_labels = ["Objective Value", "category_a", "param_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())

    study_categorical_params.add_trial(
        create_trial(
            value=2.0,
            params={"category_a": "preferred", "param_b": 20},
            distributions=distributions,
        )
    )

    # Still "category_a" contains unique suggested value during the optimization.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    objectives = _fetch_objectives_from_figure(figure)
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert objectives == [0.0, 2.0]
    assert axes[2].get_ylim() == (-default_padding_fraction, default_padding_fraction)
    assert [line.get_text() for line in axes[2].get_yticklabels()] == ["preferred"]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == [0]
    assert axes[3].get_ylim() == (20, 30)
    expected_labels = ["Objective Value", "category_a", "param_b"]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_with_categorical_numeric_params() -> None:
    # Test with sample from multiple distributions including categorical params
    # that can be interpreted as numeric params.
    study = create_study()
    distributions: Dict[str, BaseDistribution] = {
        "param_a": CategoricalDistribution(("preferred", "opt")),
        "param_b": CategoricalDistribution((1, 2, 10)),
        "param_c": FloatDistribution(1, 1000, log=True),
        "param_d": CategoricalDistribution((1, -1, 2)),
    }
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": "preferred", "param_b": 2, "param_c": 30, "param_d": 2},
            distributions=distributions,
        )
    )

    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_a": "opt", "param_b": 1, "param_c": 200, "param_d": 2},
            distributions=distributions,
        )
    )

    study.add_trial(
        create_trial(
            value=2.0,
            params={"param_a": "preferred", "param_b": 10, "param_c": 10, "param_d": 1},
            distributions=distributions,
        )
    )

    study.add_trial(
        create_trial(
            value=3.0,
            params={"param_a": "opt", "param_b": 2, "param_c": 10, "param_d": -1},
            distributions=distributions,
        )
    )
    figure = plot_parallel_coordinate(study)
    axes = figure.get_figure().axes
    assert len(axes) == 5 + 1
    assert axes[0].get_ylim() == (0.0, 3.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 3.0)
    objectives = _fetch_objectives_from_figure(figure)
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert objectives == [0.0, 1.0, 2.0, 3.0]
    num_choices_param_a = 2
    assert axes[2].get_ylim() == (0, num_choices_param_a - 1)
    assert [line.get_text() for line in axes[2].get_yticklabels()] == ["preferred", "opt"]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == list(
        range(num_choices_param_a)
    )
    num_choices_param_b = 3
    assert axes[3].get_ylim() == (0.0, num_choices_param_b - 1)
    assert [int(label.get_text()) for label in axes[3].get_yticklabels()] == [1, 2, 10]
    assert [label.get_position()[1] for label in axes[3].get_yticklabels()] == list(
        range(num_choices_param_b)
    )
    assert axes[4].get_ylim() == (10, 200)
    num_choices_param_d = 3
    assert axes[5].get_ylim() == (0.0, num_choices_param_d - 1)
    assert [int(label.get_text()) for label in axes[5].get_yticklabels()] == [-1, 1, 2]
    assert [label.get_position()[1] for label in axes[5].get_yticklabels()] == list(
        range(num_choices_param_d)
    )
    expected_labels = ["Objective Value"] + [
        f"param_{postfix}" for postfix in string.ascii_lowercase[:4]
    ]
    _test_xtick_labels(axes, expected_labels)
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_only_missing_params() -> None:
    # When all trials contain only a part of parameters,
    # the plot returns an empty figure.
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_a": 1e-6},
            distributions={
                "param_a": FloatDistribution(1e-7, 1e-2),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_b": 200},
            distributions={
                "param_b": FloatDistribution(1, 1000),
            },
        )
    )

    figure = plot_parallel_coordinate(study)
    axes = figure.get_figure().axes
    assert len(axes) == 0 + 1
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
