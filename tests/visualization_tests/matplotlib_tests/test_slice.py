from io import BytesIO

from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
import numpy as np
import pytest

from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import Trial
from optuna.visualization.matplotlib import plot_slice


def test_target_is_none_and_study_is_multi_obj() -> None:

    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        plot_slice(study)


def test_plot_slice() -> None:

    # Test with no trial.
    study = prepare_study_with_trials(no_trials=True)
    figure = plot_slice(study)
    assert len(figure.findobj(PathCollection)) == 0
    plt.savefig(BytesIO())

    study = prepare_study_with_trials(with_c_d=False)

    # Test with a trial.
    figure = plot_slice(study)
    assert len(figure) == 2
    assert len(figure[0].findobj(PathCollection)) == 1
    assert len(figure[1].findobj(PathCollection)) == 1
    assert figure[0].yaxis.label.get_text() == "Objective Value"
    plt.savefig(BytesIO())

    # Scatter plot data is available as PathCollection.
    data0 = figure[0].findobj(PathCollection)[0].get_offsets().data
    data1 = figure[1].findobj(PathCollection)[0].get_offsets().data
    assert np.allclose(data0, [[1.0, 0.0], [2.5, 1.0]])
    assert np.allclose(data1, [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    # Test with a trial to select parameter.
    figure = plot_slice(study, params=["param_a"])
    assert len(figure.findobj(PathCollection)) == 1
    assert figure.yaxis.label.get_text() == "Objective Value"

    data0 = figure.findobj(PathCollection)[0].get_offsets().data
    assert np.allclose(data0, [[1.0, 0.0], [2.5, 1.0]])
    plt.savefig(BytesIO())

    # Test with a customized target value.
    with pytest.warns(UserWarning):
        figure = plot_slice(study, params=["param_a"], target=lambda t: t.params["param_b"])
    assert len(figure.findobj(PathCollection)) == 1
    assert figure.yaxis.label.get_text() == "Objective Value"

    data0 = figure.findobj(PathCollection)[0].get_offsets().data
    assert np.allclose(data0, [[1.0, 2.0], [2.5, 1.0]])
    plt.savefig(BytesIO())

    # Test with a customized target name.
    figure = plot_slice(study, target_name="Target Name")
    assert len(figure) == 2
    assert len(figure[0].findobj(PathCollection)) == 1
    assert len(figure[1].findobj(PathCollection)) == 1
    assert figure[0].yaxis.label.get_text() == "Target Name"
    plt.savefig(BytesIO())

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        plot_slice(study, params=["optuna"])

    # Ignore failed trials.
    def fail_objective(_: Trial) -> float:

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = plot_slice(study)
    assert len(figure.get_lines()) == 0
    assert len(figure.findobj(PathCollection)) == 0
    plt.savefig(BytesIO())


def test_plot_slice_log_scale() -> None:

    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"x_linear": 1.0, "y_log": 1e-3},
            distributions={
                "x_linear": FloatDistribution(0.0, 3.0),
                "y_log": FloatDistribution(1e-5, 1.0, log=True),
            },
        )
    )

    # Plot a parameter.
    figure = plot_slice(study, params=["y_log"])
    assert len(figure.findobj(PathCollection)) == 1
    assert figure.xaxis.label.get_text() == "y_log"
    assert figure.xaxis.get_scale() == "log"
    plt.savefig(BytesIO())
    figure = plot_slice(study, params=["x_linear"])
    assert len(figure.findobj(PathCollection)) == 1
    assert figure.xaxis.label.get_text() == "x_linear"
    assert figure.xaxis.get_scale() == "linear"
    plt.savefig(BytesIO())

    # Plot multiple parameters.
    figure = plot_slice(study)
    assert len(figure) == 2
    assert len(figure[0].findobj(PathCollection)) == 1
    assert len(figure[1].findobj(PathCollection)) == 1
    assert figure[0].xaxis.label.get_text() == "x_linear"
    assert figure[1].xaxis.label.get_text() == "y_log"
    assert figure[0].xaxis.get_scale() == "linear"
    assert figure[1].xaxis.get_scale() == "log"
    plt.savefig(BytesIO())
