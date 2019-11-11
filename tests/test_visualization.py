from typing import List  # NOQA
from typing import Optional  # NOQA

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.structs import StudyDirection
from optuna.study import create_study
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA
from optuna import visualization
from optuna.visualization import _generate_contour_subplot
from optuna.visualization import _get_contour_plot
from optuna.visualization import _get_intermediate_plot
from optuna.visualization import _get_optimization_history_plot
from optuna.visualization import _get_parallel_coordinate_plot
from optuna.visualization import _get_slice_plot


def _prepare_study_with_trials(no_trials=False, less_than_two=False, with_c_d=True):
    # type: (bool, bool, bool) -> Study
    """Prepare a study for tests.

    Args:
        no_trials: If ``False``, create a study with no trials.
        less_than_two: If ``True``, create a study with two/four hyperparameters where
            'param_a' (and 'param_c') appear(s) only once while 'param_b' (and 'param_b')
            appear(s) twice in `study.trials`.
        with_c_d: If ``True``, the study has four hyperparameters named 'param_a',
            'param_b', 'param_c', and 'param_d'. Otherwise, there are only two
            hyperparameters ('param_a' and 'param_b').

    Returns:
        :class:`~optuna.study.Study`

    """

    study = create_study()
    if no_trials:
        return study
    study._append_trial(
        value=0.0,
        params={
            'param_a': 1.0,
            'param_b': 2.0,
            'param_c': 3.0,
            'param_d': 4.0,
        } if with_c_d else {
            'param_a': 1.0,
            'param_b': 2.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
            'param_c': UniformDistribution(2.0, 5.0),
            'param_d': UniformDistribution(2.0, 5.0),
        } if with_c_d else {
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value=2.0,
        params={
            'param_b': 0.0,
            'param_d': 4.0,
        } if with_c_d else {
            'param_b': 0.0,
        },
        distributions={
            'param_b': UniformDistribution(0.0, 3.0),
            'param_d': UniformDistribution(2.0, 5.0),
        } if with_c_d else {
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    if less_than_two:
        return study

    study._append_trial(
        value=1.0,
        params={
            'param_a': 2.5,
            'param_b': 1.0,
            'param_c': 4.5,
            'param_d': 2.0,
        } if with_c_d else {
            'param_a': 2.5,
            'param_b': 1.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
            'param_c': UniformDistribution(2.0, 5.0),
            'param_d': UniformDistribution(2.0, 5.0),
        } if with_c_d else {
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    return study


def test_get_intermediate_plot():
    # type: () -> None

    # Test with no trials.
    study = _prepare_study_with_trials(no_trials=True)
    figure = _get_intermediate_plot(study)
    assert not figure.data

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

    # Test a study with one trial with intermediate values and
    # one trial without intermediate values.
    # Expect the trial with no intermediate values to be ignored.
    study.optimize(lambda t: objective(t, False), n_trials=1)
    assert len(study.trials) == 2
    figure = _get_intermediate_plot(study)
    assert len(figure.data) == 1
    assert figure.data[0].x == (0, 1)
    assert figure.data[0].y == (1.0, 2.0)

    # Test a study of only one trial that has no intermediate values.
    study = create_study()
    study.optimize(lambda t: objective(t, False), n_trials=1)
    figure = _get_intermediate_plot(study)
    assert not figure.data

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_intermediate_plot(study)
    assert not figure.data


@pytest.mark.parametrize('direction', ['minimize', 'maximize'])
def test_get_optimization_history_plot(direction):
    # type: (str) -> None

    # Test with no trial.
    study = create_study(direction=direction)
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
    study = create_study(direction=direction)
    study.optimize(objective, n_trials=3)
    figure = _get_optimization_history_plot(study)
    assert len(figure.data) == 2
    assert figure.data[0].x == (0, 1, 2)
    assert figure.data[0].y == (1.0, 2.0, 0.0)
    assert figure.data[1].x == (0, 1, 2)
    if direction == 'minimize':
        assert figure.data[1].y == (1.0, 1.0, 0.0)
    else:
        assert figure.data[1].y == (1.0, 2.0, 2.0)

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study(direction=direction)
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))

    figure = _get_optimization_history_plot(study)
    assert len(figure.data) == 0


@pytest.mark.parametrize('params',
                         [
                             [],
                             ['param_a'],
                             ['param_a', 'param_b'],
                             ['param_a', 'param_b', 'param_c'],
                             ['param_a', 'param_b', 'param_c', 'param_d'],
                             None,
                         ])
def test_get_contour_plot(params):
    # type: (Optional[List[str]]) -> None

    # Test with no trial.
    study_without_trials = _prepare_study_with_trials(no_trials=True)
    figure = _get_contour_plot(study_without_trials, params=params)
    assert len(figure.data) == 0

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_contour_plot(study, params=params)
    assert len(figure.data) == 0

    # Test with some trials.
    study = _prepare_study_with_trials()

    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        _get_contour_plot(study, ['optuna', 'Optuna'])

    figure = _get_contour_plot(study, params=params)
    if params is not None and len(params) < 3:
        if len(params) <= 1:
            assert not figure.data
        elif len(params) == 2:
            assert figure.data[0]['x'] == (1.0, 2.5)
            assert figure.data[0]['y'] == (0.0, 1.0, 2.0)
            assert figure.data[1]['x'] == (1.0, 2.5)
            assert figure.data[1]['y'] == (2.0, 1.0)
            assert figure.layout['xaxis']['range'] == (1.0, 2.5)
            assert figure.layout['yaxis']['range'] == (0.0, 2.0)
    else:
        # TODO(crcrpar): Add more checks. Currently this only checks the number of data.
        n_params = len(params) if params is not None else 4
        assert len(figure.data) == n_params ** 2 + n_params * (n_params - 1)


def test_generate_contour_plot_for_few_observations():
    # type: () -> None

    study = _prepare_study_with_trials(less_than_two=True)
    trials = study.trials

    # `x_axis` has one observation.
    params = ['param_a', 'param_b']
    contour, scatter = _generate_contour_subplot(
        trials, params[0], params[1], StudyDirection.MINIMIZE)
    assert contour.x is None and contour.y is None and scatter.x is None and scatter.y is None

    # `y_axis` has one observation.
    params = ['param_b', 'param_a']
    contour, scatter = _generate_contour_subplot(
        trials, params[0], params[1], StudyDirection.MINIMIZE)
    assert contour.x is None and contour.y is None and scatter.x is None and scatter.y is None


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

    study = _prepare_study_with_trials(with_c_d=False)

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

    # Test with wrong params that do not exist in trials
    with pytest.raises(ValueError):
        _get_parallel_coordinate_plot(study, params=['optuna', 'optuna'])

    # Ignore failed trials.
    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1, catch=(ValueError,))
    figure = _get_parallel_coordinate_plot(study)
    assert len(figure.data) == 0

    # Test with categorical params that cannot be converted to numeral.
    study_categorical_params = create_study()
    study_categorical_params._append_trial(
        value=0.0,
        params={
            'category_a': 'preferred',
            'category_b': 'net',
        },
        distributions={
            'category_a': CategoricalDistribution(('preferred', 'opt')),
            'category_b': CategoricalDistribution(('net', 'una')),
        }
    )
    study_categorical_params._append_trial(
        value=2.0,
        params={
            'category_a': 'opt',
            'category_b': 'una',
        },
        distributions={
            'category_a': CategoricalDistribution(('preferred', 'opt')),
            'category_b': CategoricalDistribution(('net', 'una')),
        }
    )
    figure = _get_parallel_coordinate_plot(study_categorical_params)
    assert len(figure.data[0]['dimensions']) == 3
    assert figure.data[0]['dimensions'][0]['label'] == 'Objective Value'
    assert figure.data[0]['dimensions'][0]['range'] == (0.0, 2.0)
    assert figure.data[0]['dimensions'][0]['values'] == (0.0, 2.0)
    assert figure.data[0]['dimensions'][1]['label'] == 'category_a'
    assert figure.data[0]['dimensions'][1]['range'] == (0, 1)
    assert figure.data[0]['dimensions'][1]['values'] == (0, 1)
    assert figure.data[0]['dimensions'][1]['ticktext'] == (['preferred', 0], ['opt', 1])
    assert figure.data[0]['dimensions'][2]['label'] == 'category_b'
    assert figure.data[0]['dimensions'][2]['range'] == (0, 1)
    assert figure.data[0]['dimensions'][2]['values'] == (0, 1)
    assert figure.data[0]['dimensions'][2]['ticktext'] == (['net', 0], ['una', 1])


def test_get_slice_plot():
    # type: () -> None

    # Test with no trial.
    study = _prepare_study_with_trials(no_trials=True)
    figure = _get_slice_plot(study)
    assert len(figure.data) == 0

    study = _prepare_study_with_trials(with_c_d=False)

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

    # Test with wrong parameters.
    with pytest.raises(ValueError):
        _get_slice_plot(study, params=['optuna'])

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
