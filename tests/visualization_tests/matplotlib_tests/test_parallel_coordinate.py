from io import BytesIO

from matplotlib.collections import LineCollection
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
    assert len(figure.get_figure().axes) == 0 + 1
    plt.savefig(BytesIO())

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_parallel_coordinate(study)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    # axes[0] contains line plots.
    assert axes[0].get_ylim() == (0.0, 2.0)
    # axes[1] is colorbar.
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    # axes[2] is `param_a`'s vertical line.
    assert axes[2].get_ylim() == (1.0, 2.5)
    # axes[3] is `param_b`'s vertical line.
    assert axes[3].get_ylim() == (0.0, 2.0)
    line_collections = figure.findobj(LineCollection)
    assert len(line_collections) == 1
    # `objective`'s vertical line.
    assert line_collections[0].get_array().tolist() == [0.0, 2.0, 1.0]
    expected_labels = ("Objective Value", "param_a", "param_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
    plt.savefig(BytesIO())

    # Test with a trial to select parameter.
    figure = plot_parallel_coordinate(study, params=["param_a"])
    axes = figure.get_figure().axes
    assert len(axes) == 2 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    line_collections = figure.findobj(LineCollection)
    assert len(line_collections) == 1
    # `objective`'s vertical line.
    assert line_collections[0].get_array().tolist() == [0.0, 2.0, 1.0]
    expected_labels = ("Objective Value", "param_a", "param_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
    plt.savefig(BytesIO())

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_parallel_coordinate(
            study, params=["param_a"], target=lambda t: t.params["param_b"]
        )
    axes = figure.get_figure().axes
    assert len(axes) == 2 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    line_collections = figure.findobj(LineCollection)
    assert len(line_collections) == 1
    assert line_collections[0].get_array().tolist() == [2.0, 0.0, 1.0]
    expected_labels = ("Objective Value", "param_a")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
    plt.savefig(BytesIO())

    # Test with a customized target name.
    figure = plot_parallel_coordinate(study, target_name="Target Name")
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Target Name"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert axes[2].get_ylim() == (1.0, 2.5)
    assert axes[3].get_ylim() == (0.0, 2.0)
    line_collections = figure.findobj(LineCollection)
    assert len(line_collections) == 1
    assert line_collections[0].get_array().tolist() == [0.0, 2.0, 1.0]
    expected_labels = ("Target Name", "param_a", "param_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
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
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert axes[2].get_ylim() == (0, 1)
    assert [lien.get_text() for lien in axes[2].get_yticklabels()] == ["preferred", "opt"]
    assert axes[3].get_ylim() == (0, 1)
    assert [lien.get_text() for lien in axes[3].get_yticklabels()] == ["net", "una"]
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist() == [0.0, 2.0]
    expected_labels = ("Objective Value", "category_a", "category_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
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

    # Trials are sorted by using param_a and param_b, i.e., trial#1, trial#2, and trial#0.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert figure.findobj(LineCollection)[0].get_array().tolist() == [0.0, 1.0, 2.0]
    num_choices = 2
    assert axes[2].get_ylim() == (0, num_choices - 1)
    assert [int(label.get_text()) for label in axes[2].get_yticklabels()] == [1, 2]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == list(
        range(num_choices)
    )
    num_choices = 3
    assert axes[3].get_ylim() == (0, num_choices - 1)
    assert [int(label.get_text()) for label in axes[3].get_yticklabels()] == [10, 20, 30]
    assert [label.get_position()[1] for label in axes[3].get_yticklabels()] == list(
        range(num_choices)
    )
    expected_labels = ("Objective Value", "category_a", "category_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_log_params() -> None:
    # Test with log params.
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
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 1.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 1.0)
    assert len(figure.findobj(LineCollection)) == 1
    assert figure.findobj(LineCollection)[0].get_array().tolist() == [0.0, 1.0, 0.1]
    assert axes[2].get_ylim() == (1e-6, 1e-4)
    assert axes[3].get_ylim() == (10.0, 200)
    expected_labels = ("Objective Value", "param_a", "param_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
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

    # Both hyperparameters contain unique values.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    # Default padding is 5% in Matplotlib.
    default_padding_fraction = 0.05
    assert axes[0].get_ylim() == (-default_padding_fraction, default_padding_fraction)
    assert axes[1].get_ylabel() == "Objective Value"
    # Optuna's parallel coordinate uses 10% padding for color map.
    assert axes[1].get_ylim() == (-0.1, 0.1)
    assert len(figure.findobj(LineCollection)) == 1
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert figure.findobj(LineCollection)[0].get_array().tolist() == [0.0]
    assert axes[2].get_ylim() == (-default_padding_fraction, default_padding_fraction)
    assert [lien.get_text() for lien in axes[2].get_yticklabels()] == ["preferred"]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == [0]
    assert axes[3].get_ylim() == (
        30 * (1.0 - default_padding_fraction),
        30 * (1.0 + default_padding_fraction),
    )
    expected_labels = ("Objective Value", "category_a", "param_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
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

    # Still "category_a" contains unique suggested value during the optimization.
    figure = plot_parallel_coordinate(study_categorical_params)
    axes = figure.get_figure().axes
    assert len(axes) == 3 + 1
    assert axes[0].get_ylim() == (0.0, 2.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 2.0)
    assert len(figure.findobj(LineCollection)) == 1
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert figure.findobj(LineCollection)[0].get_array().tolist() == [0.0, 2.0]
    assert axes[2].get_ylim() == (-default_padding_fraction, default_padding_fraction)
    assert [lien.get_text() for lien in axes[2].get_yticklabels()] == ["preferred"]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == [0]
    assert axes[3].get_ylim() == (20, 30)
    expected_labels = ("Objective Value", "category_a", "param_b")
    xticklabels = axes[0].get_xticklabels()
    for expected_label, xticklabel in zip(expected_labels, xticklabels):
        assert expected_label == xticklabel.get_text()
    plt.savefig(BytesIO())


def test_plot_parallel_coordinate_with_categorical_numeric_params() -> None:
    # Test with sample from multiple distributions including categorical params
    # that can be interpreted as numeric params.
    study = create_study()
    study.add_trial(
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

    study.add_trial(
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

    study.add_trial(
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

    study.add_trial(
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
    figure = plot_parallel_coordinate(study)
    axes = figure.get_figure().axes
    assert len(axes) == 5 + 1
    assert axes[0].get_ylim() == (0.0, 3.0)
    assert axes[1].get_ylabel() == "Objective Value"
    assert axes[1].get_ylim() == (0.0, 3.0)
    assert len(figure.findobj(LineCollection)) == 1
    # Objective values are not sorted by the other parameters,
    # unlike Plotly's parallel_coordinate.
    assert figure.findobj(LineCollection)[0].get_array().tolist() == [0.0, 1.0, 2.0, 3.0]
    num_choices = 2
    assert axes[2].get_ylim() == (0, num_choices - 1)
    assert [lien.get_text() for lien in axes[2].get_yticklabels()] == ["preferred", "opt"]
    assert [label.get_position()[1] for label in axes[2].get_yticklabels()] == list(
        range(num_choices)
    )
    num_choices = 3
    assert axes[3].get_ylim() == (0.0, num_choices - 1)
    assert [int(label.get_text()) for label in axes[3].get_yticklabels()] == [1, 2, 10]
    assert [label.get_position()[1] for label in axes[3].get_yticklabels()] == list(
        range(num_choices)
    )
    assert axes[4].get_ylim() == (10, 200)
    num_choices = 3
    assert axes[5].get_ylim() == (0.0, num_choices - 1)
    assert [int(label.get_text()) for label in axes[5].get_yticklabels()] == [-1, 1, 2]
    assert [label.get_position()[1] for label in axes[5].get_yticklabels()] == list(
        range(num_choices)
    )
    xticklabels = axes[0].get_xticklabels()
    assert xticklabels[0].get_text() == "Objective Value"
    for index, postfix in zip(range(1, 5), string.ascii_lowercase):
        assert xticklabels[index].get_text() == f"param_{postfix}"
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
                "param_a": LogUniformDistribution(1e-7, 1e-2),
            },
        )
    )
    study.add_trial(
        create_trial(
            value=1.0,
            params={"param_b": 200},
            distributions={
                "param_b": LogUniformDistribution(1, 1000),
            },
        )
    )

    figure = plot_parallel_coordinate(study)
    assert len(figure.data) == 1
    plt.savefig(BytesIO())
