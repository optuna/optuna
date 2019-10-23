from typing import List  # NOQA
from typing import Optional  # NOQA

import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import UniformDistribution
from optuna.structs import StudyDirection
from optuna.study import create_study
from optuna.study import Study  # NOQA
from optuna.trial import Trial  # NOQA
from optuna.visualization import _generate_contour_subplot
from optuna.visualization import _get_contour_plot
from optuna.visualization import _get_intermediate_plot
from optuna.visualization import _get_optimization_history_plot
from optuna.visualization import _get_parallel_coordinate_plot
from optuna.visualization import _get_slice_plot


def prepare_study_with_trials(no_trials=False, less_than_two=False, with_c_d=True):
    # type: (bool, bool, bool) -> Study

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

    # Test with no trial.
    study = prepare_study_with_trials(no_trials=True)
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
    study.optimize(fail_objective, n_trials=1)
    figure = _get_intermediate_plot(study)
    assert len(figure.data) == 0


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
    study.optimize(fail_objective, n_trials=1)
    figure = _get_optimization_history_plot(study)
    assert len(figure.data) == 0

    study = create_study()
    study._append_trial(
        value='abc',  # type: ignore
        params={
            'param_a': 1.0,
            'param_b': 1.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study._append_trial(
        value='def',  # type: ignore
        params={
            'param_a': 2.0,
            'param_b': 2.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    with pytest.raises(ValueError):
        _get_optimization_history_plot(study)


@pytest.mark.parametrize('params',
                         [
                             ([],),
                             (['param_a'],),
                             (['param_a', 'param_b'],),
                             (['param_a', 'param_b', 'param_c'],),
                             (['param_a', 'param_b', 'param_c', 'param_d'],),
                             (None,),
                         ])
def test_get_contour_plot(params):
    # type: (Optional[List[str]]) -> None

    # Test with no trial.
    study_without_trials = prepare_study_with_trials(no_trials=True)
    figure = _get_contour_plot(study_without_trials, params=params)
    assert len(figure.data) == 0

    # Test whether trials with `ValueError`s are ignored.

    def fail_objective(_):
        # type: (Trial) -> float

        raise ValueError

    study = create_study()
    study.optimize(fail_objective, n_trials=1)
    figure = _get_contour_plot(study, params=params)
    assert not figure.data
    
    # Test ValueError due to wrong params.
    with pytest.raises(ValueError):
        _get_contour_plot(study, ['optuna', 'Optuna'])

    # Test with some trials.
    study = prepare_study_with_trials(no_trials=False)
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
        # TODO(crcrpar): Add more checks.
        n_params = len(params) if params is not None else 4
        assert len(figure.data) == n_params ** 2 + n_params * (n_params - 1)


def test_generate_contour_plot_for_few_observations():
    # type: () -> None

    direction = 'minimize'

    study = prepare_study_with_trials(less_than_two=True)
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

    # Not numeral trial's value.
    study_str_trial_value = create_study(direction=direction)
    study_str_trial_value._append_trial(
        value='opt',  # type: ignore
        params={
            'param_a': 1.0,
            'param_b': 1.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    study_str_trial_value._append_trial(
        value='una',  # type: ignore
        params={
            'param_a': 2.0,
            'param_b': 2.0,
        },
        distributions={
            'param_a': UniformDistribution(0.0, 3.0),
            'param_b': UniformDistribution(0.0, 3.0),
        }
    )
    with pytest.raises(ValueError):
        _generate_contour_subplot(
            study_str_trial_value.trials, params[0], params[1], StudyDirection.MINIMIZE)


def test_get_parallel_coordinate_plot():
    # type: () -> None

    # Test with no trial.
    study = create_study()
    figure = _get_parallel_coordinate_plot(study)
    assert len(figure.data) == 0

    study = prepare_study_with_trials(with_c_d=False)

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
    study.optimize(fail_objective, n_trials=1)
    figure = _get_parallel_coordinate_plot(study)
    assert len(figure.data) == 0

    figure = _get_parallel_coordinate_plot(study, params=['optuna'])
    assert not figure.data

    # Test with categorical params that cannot be converted to numeral.
    study_str_trial_value = create_study()
    study_str_trial_value._append_trial(
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
    study_str_trial_value._append_trial(
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
    figure = _get_parallel_coordinate_plot(study_str_trial_value)
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

    # Test that the API raises when trials have str values.
    study_str_trial_value = create_study()
    study_str_trial_value._append_trial(
        value='opt',  # type: ignore
        params={
            'category_a': 'preferred',
            'category_b': 'net',
        },
        distributions={
            'category_a': CategoricalDistribution(('preferred', 'opt')),
            'category_b': CategoricalDistribution(('net', 'una')),
        }
    )
    study_str_trial_value._append_trial(
        value='una',  # type: ignore
        params={
            'category_a': 'opt',
            'category_b': 'una',
        },
        distributions={
            'category_a': CategoricalDistribution(('preferred', 'opt')),
            'category_b': CategoricalDistribution(('net', 'una')),
        }
    )
    with pytest.raises(ValueError):
        _get_parallel_coordinate_plot(study_str_trial_value)


def test_get_slice_plot():
    # type: () -> None

    # Test with no trial.
    study = prepare_study_with_trials(no_trials=True)
    figure = _get_slice_plot(study)
    assert len(figure.data) == 0

    study = prepare_study_with_trials(with_c_d=False)

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
    study.optimize(fail_objective, n_trials=1)
    figure = _get_slice_plot(study)
    assert len(figure.data) == 0
