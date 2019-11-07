import pytest

from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.study import create_study
from optuna.trial import Trial  # NOQA
from optuna import visualization
from optuna.visualization import _get_contour_plot
from optuna.visualization import _get_intermediate_plot
from optuna.visualization import _get_optimization_history_plot
from optuna.visualization import _get_parallel_coordinate_plot
from optuna.visualization import _get_slice_plot


def test_get_intermediate_plot():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = _get_intermediate_plot(study)
    assert len(figure.data) == 0

    def objective(trial, report_intermediate_values):
        # type: (Trial, bool) -> float

        if report_intermediate_values:
            trial.report(1.0, step=0)
            trial.report(2.0, step=1)
        return 0.0

    # Test with a trial with intermediate values.
    study = create_study()
    study.optimize(lambda t: objective(t, True), n_trials=1)
    figure = _get_intermediate_plot(study)
    assert len(figure.data) == 1
    assert figure.data[0].x == (0, 1)
    assert figure.data[0].y == (1.0, 2.0)

    # Test with trials, one of which contains no intermediate value.
    study = create_study()
    study.optimize(lambda t: objective(t, False), n_trials=1)
    figure = _get_intermediate_plot(study)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 0
    assert len(figure.data[0].y) == 0

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_intermediate_plot(study)
    assert len(figure.data) == 0


def test_get_optimization_history_plot():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = _get_optimization_history_plot(study)
    assert len(figure.data) == 0

    def objective(trial):
        # type: (Trial) -> float

        if trial.number == 0:
            return 1.0
        elif trial.number == 1:
            return 2.0
        elif trial.number == 2:
            return 0.0
        return 0.0

    # Test with a trial.
    study = create_study()
    study.optimize(objective, n_trials=3)
    figure = _get_optimization_history_plot(study)
    assert len(figure.data) == 2
    assert figure.data[0].x == (0, 1, 2)
    assert figure.data[0].y == (1.0, 2.0, 0.0)
    assert figure.data[1].x == (0, 1, 2)
    assert figure.data[1].y == (1.0, 1.0, 0.0)

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_optimization_history_plot(study)
    assert len(figure.data) == 0


def test_get_contour_plot():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = _get_contour_plot(study)
    assert len(figure.data) == 0

    study._append_trial(
        value=0.0,
        params={
            'param_a': 1.0,
            'param_b': 2.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=2.0,
        params={
            'param_b': 0.0,
        },
        distributions={
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=1.0,
        params={
            'param_a': 2.5,
            'param_b': 1.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )

    # Test with a trial.
    figure = _get_contour_plot(study)
    assert figure.data[0]['x'] == (1.0, 2.5)
    assert figure.data[0]['y'] == (0.0, 1.0, 2.0)
    assert figure.data[1]['x'] == (1.0, 2.5)
    assert figure.data[1]['y'] == (2.0, 1.0)
    assert figure.layout['xaxis']['range'] == (1.0, 2.5)
    assert figure.layout['yaxis']['range'] == (0.0, 2.0)

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        _get_contour_plot(study, ['optuna', 'Optuna'])

    # Test with a trial to select parameter.
    figure = _get_contour_plot(study, params=['param_a', 'param_b'])
    assert figure.data[0]['x'] == (1.0, 2.5)
    assert figure.data[0]['y'] == (0.0, 1.0, 2.0)
    assert figure.data[1]['x'] == (1.0, 2.5)
    assert figure.data[1]['y'] == (2.0, 1.0)
    assert figure.layout['xaxis']['range'] == (1.0, 2.5)
    assert figure.layout['yaxis']['range'] == (0.0, 2.0)

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_contour_plot(study)
    assert len(figure.data) == 0


def test_get_contour_plot_log_scale():
    # type: () -> None

    # If the search space has two parameters, _get_contour_plot generates a single plot.
    study = create_study()
    study._append_trial(
        value=0.0,
        params={
            'param_a': 1e-6,
            'param_b': 1e-4,
        },
        distributions={
            'param_a': LogUniformDistribution(1e-7, 1e-2),
            'param_b': LogUniformDistribution(1e-5, 1e-1),
        }
    )
    study._append_trial(
        value=1.0,
        params={
            'param_a': 1e-5,
            'param_b': 1e-3,
        },
        distributions={
            'param_a': LogUniformDistribution(1e-7, 1e-2),
            'param_b': LogUniformDistribution(1e-5, 1e-1),
        }
    )

    figure = _get_contour_plot(study)
    assert figure.layout['xaxis']['range'] == (-6, -5)
    assert figure.layout['yaxis']['range'] == (-4, -3)
    assert figure.layout['xaxis_type'] == 'log'
    assert figure.layout['yaxis_type'] == 'log'

    # If the search space has three parameters, _get_contour_plot generates nine plots.
    study = create_study()
    study._append_trial(
        value=0.0,
        params={
            'param_a': 1e-6,
            'param_b': 1e-4,
            'param_c': 1e-2,
        },
        distributions={
            'param_a': LogUniformDistribution(1e-7, 1e-2),
            'param_b': LogUniformDistribution(1e-5, 1e-1),
            'param_c': LogUniformDistribution(1e-3, 10),
        }
    )
    study._append_trial(
        value=1.0,
        params={
            'param_a': 1e-5,
            'param_b': 1e-3,
            'param_c': 1e-1,
        },
        distributions={
            'param_a': LogUniformDistribution(1e-7, 1e-2),
            'param_b': LogUniformDistribution(1e-5, 1e-1),
            'param_c': LogUniformDistribution(1e-3, 10),
        }
    )

    figure = _get_contour_plot(study)
    param_a_range = (-6, -5)
    param_b_range = (-4, -3)
    param_c_range = (-2, -1)
    axis_to_range = {
        'xaxis': param_a_range,
        'xaxis2': param_b_range,
        'xaxis3': param_c_range,
        'xaxis4': param_a_range,
        'xaxis5': param_b_range,
        'xaxis6': param_c_range,
        'xaxis7': param_a_range,
        'xaxis8': param_b_range,
        'xaxis9': param_c_range,
        'yaxis': param_a_range,
        'yaxis2': param_a_range,
        'yaxis3': param_a_range,
        'yaxis4': param_b_range,
        'yaxis5': param_b_range,
        'yaxis6': param_b_range,
        'yaxis7': param_c_range,
        'yaxis8': param_c_range,
        'yaxis9': param_c_range,
    }

    for axis, param_range in axis_to_range.items():
        assert figure.layout[axis]['range'] == param_range
        assert figure.layout[axis]['type'] == 'log'


def test_get_parallel_coordinate_plot():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = _get_parallel_coordinate_plot(study)
    assert len(figure.data) == 0

    study._append_trial(
        value=0.0,
        params={
            'param_a': 1.0,
            'param_b': 2.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=2.0,
        params={
            'param_b': 0.0,
        },
        distributions={
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=1.0,
        params={
            'param_a': 2.5,
            'param_b': 1.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )

    # Test with a trial.
    figure = _get_parallel_coordinate_plot(study)
    assert len(figure.data[0]['dimensions']) == 3
    assert figure.data[0]['dimensions'][0]['label'] == 'Objective Value'
    assert figure.data[0]['dimensions'][0]['range'] == (0.0, 2.0)
    assert figure.data[0]['dimensions'][0]['values'] == (0.0, 2.0, 1.0)
    assert figure.data[0]['dimensions'][1]['label'] == 'param_a'
    assert figure.data[0]['dimensions'][1]['range'] == (1.0, 2.5)
    assert figure.data[0]['dimensions'][1]['values'] == (1.0, 2.5)
    assert figure.data[0]['dimensions'][2]['label'] == 'param_b'
    assert figure.data[0]['dimensions'][2]['range'] == (0.0, 2.0)
    assert figure.data[0]['dimensions'][2]['values'] == (2.0, 0.0, 1.0)

    # Test with a trial to select parameter.
    figure = _get_parallel_coordinate_plot(study, params=['param_a'])
    assert len(figure.data[0]['dimensions']) == 2
    assert figure.data[0]['dimensions'][0]['label'] == 'Objective Value'
    assert figure.data[0]['dimensions'][0]['range'] == (0.0, 2.0)
    assert figure.data[0]['dimensions'][0]['values'] == (0.0, 2.0, 1.0)
    assert figure.data[0]['dimensions'][1]['label'] == 'param_a'
    assert figure.data[0]['dimensions'][1]['range'] == (1.0, 2.5)
    assert figure.data[0]['dimensions'][1]['values'] == (1.0, 2.5)

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_parallel_coordinate_plot(study)
    assert len(figure.data) == 0


def test_get_slice_plot():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = _get_slice_plot(study)
    assert len(figure.data) == 0

    study._append_trial(
        value=0.0,
        params={
            'param_a': 1.0,
            'param_b': 2.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=2.0,
        params={
            'param_b': 0.0,
        },
        distributions={
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=1.0,
        params={
            'param_a': 2.5,
            'param_b': 1.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )

    # Test with a trial.
    figure = _get_slice_plot(study)
    assert len(figure.data) == 2
    assert figure.data[0]['x'] == (1.0, 2.5)
    assert figure.data[0]['y'] == (0.0, 1.0)
    assert figure.data[1]['x'] == (2.0, 0.0, 1.0)
    assert figure.data[1]['y'] == (0.0, 2.0, 1.0)

    # Test with a trial to select parameter.
    figure = _get_slice_plot(study, params=['param_a'])
    assert len(figure.data) == 1
    assert figure.data[0]['x'] == (1.0, 2.5)
    assert figure.data[0]['y'] == (0.0, 1.0)

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_slice_plot(study)
    assert len(figure.data) == 0


def test_get_slice_plot_log_scale():
    # type: () -> None

    study = create_study()
    study._append_trial(
        value=0.0,
        params={
            'x_linear': 1.0,
            'y_log': 1e-3,
        },
        distributions={
            'x_linear': UniformDistribution(0.0, 3.0),
            'y_log': LogUniformDistribution(1e-5, 1.),
        }
    )

    # Plot a parameter.
    figure = _get_slice_plot(study, params=['y_log'])
    assert figure.layout['xaxis_type'] == 'log'
    figure = _get_slice_plot(study, params=['x_linear'])
    assert figure.layout['xaxis_type'] is None

    # Plot multiple parameters.
    figure = _get_slice_plot(study)
    assert figure.layout['xaxis_type'] is None
    assert figure.layout['xaxis2_type'] == 'log'


def test_is_log_scale():
    # type: () -> None

    study = create_study()
    study._append_trial(
        value=0.0,
        params={
            'param_linear': 1.0,
        },
        distributions={
            'param_linear': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=2.0,
        params={
            'param_linear': 2.0,
            'param_log': 1e-3,
        },
        distributions={
            'param_linear': UniformDistribution(0.0, 3.0),
            'param_log': LogUniformDistribution(1e-5, 1.),
        }
    )
    assert visualization._is_log_scale(study.trials, 'param_log')
    assert not visualization._is_log_scale(study.trials, 'param_linear')


def _is_plotly_available():
    # type: () -> bool

    try:
        import plotly  # NOQA
        available = True
    except Exception:
        available = False
    return available


def test_visualization_is_available():
    # type: () -> None

    assert visualization.is_available() == _is_plotly_available()
