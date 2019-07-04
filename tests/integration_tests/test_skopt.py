from mock import call
from mock import patch
import pytest
from skopt.space import space

import optuna
from optuna import distributions
from optuna.structs import FrozenTrial
from optuna.testing.sampler import FirstTrialOnlyRandomSampler

if optuna.types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA


def test_conversion_from_distribution_to_dimenstion():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)
    with patch('skopt.Optimizer') as mock_object:
        study.optimize(_objective, n_trials=2, catch=())

        dimensions = [
            # Original: trial.suggest_uniform('p0', -3.3, 5.2)
            space.Real(-3.3, 5.2),

            # Original: trial.suggest_uniform('p1', 2.0, 2.0)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_loguniform('p2', 0.0001, 0.3)
            space.Real(0.0001, 0.3, prior='log-uniform'),

            # Original: trial.suggest_loguniform('p3', 1.1, 1.1)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_int('p4', -100, 8)
            space.Integer(-100, 8),

            # Original: trial.suggest_int('p5', -20, -20)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_discrete_uniform('p6', 10, 20, 2)
            space.Integer(0, 5),

            # Original: trial.suggest_discrete_uniform('p7', 0.1, 1.0, 0.1)
            space.Integer(0, 8),

            # Original: trial.suggest_discrete_uniform('p8', 2.2, 2.2, 0.5)
            # => Skipped because `skopt.Optimizer` cannot handle an empty `Real` dimension.

            # Original: trial.suggest_categorical('p9', ['9', '3', '0', '8'])
            space.Categorical(('9', '3', '0', '8'))
        ]
        assert mock_object.mock_calls[0] == call(dimensions)


def test_suggested_value():
    # type: () -> None

    independent_sampler = FirstTrialOnlyRandomSampler()
    sampler = optuna.integration.SkoptSampler(independent_sampler=independent_sampler,
                                              skopt_kwargs={'n_initial_points': 5})

    # direction='minimize'
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(_objective, n_trials=10, catch=())
    for trial in study.trials:
        for param_name, param_value in trial.params_in_internal_repr.items():
            assert trial.distributions[param_name]._contains(param_value)

    # direction='maximize'
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(_objective, n_trials=10, catch=())
    for trial in study.trials:
        for param_name, param_value in trial.params_in_internal_repr.items():
            assert trial.distributions[param_name]._contains(param_value)


def test_sample_independent():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)

    # First trial.
    def objective0(trial):
        # type: (optuna.trial.Trial) -> float

        p0 = trial.suggest_uniform('p0', 0, 10)
        p1 = trial.suggest_loguniform('p1', 1, 10)
        p2 = trial.suggest_int('p2', 0, 10)
        p3 = trial.suggest_discrete_uniform('p3', 0, 9, 3)
        p4 = trial.suggest_categorical('p4', ['10', '20', '30'])
        return p0 + p1 + p2 + p3 + int(p4)

    with patch.object(sampler, 'sample_independent') as mock_object:
        mock_object.side_effect = [1, 2, 3, 3, '10']

        study.optimize(objective0, n_trials=1)

        # In first trial, all parameters were suggested via `sample_independent`.
        assert mock_object.call_count == 5

    # Second trial.
    def objective1(trial):
        # type: (optuna.trial.Trial) -> float

        # p0, p2 and p4 are deleted.
        p1 = trial.suggest_loguniform('p1', 1, 10)
        p3 = trial.suggest_discrete_uniform('p3', 0, 9, 3)

        # p5 is added.
        p5 = trial.suggest_uniform('p5', 0, 1)

        return p1 + p3 + p5

    with patch.object(sampler, 'sample_independent') as mock_object:
        mock_object.side_effect = [0]

        study.optimize(objective1, n_trials=1)

        assert [call[1][2] for call in mock_object.mock_calls] == ['p5']

    # Third trial.
    def objective2(trial):
        # type: (optuna.trial.Trial) -> float

        p1 = trial.suggest_loguniform('p1', 50, 100)  # The range has been changed
        p3 = trial.suggest_discrete_uniform('p3', 0, 9, 3)
        p5 = trial.suggest_uniform('p5', 0, 1)

        return p1 + p3 + p5

    with patch.object(sampler, 'sample_independent') as mock_object:
        mock_object.side_effect = [90, 0.2]

        study.optimize(objective2, n_trials=1)

        assert [call[1][2] for call in mock_object.mock_calls] == ['p1', 'p5']


def test_skopt_kwargs():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler(skopt_kwargs={'base_estimator': "GBRT"})
    study = optuna.create_study(sampler=sampler)

    with patch('skopt.Optimizer') as mock_object:
        study.optimize(lambda t: t.suggest_int('x', -10, 10), n_trials=2)

        dimensions = [space.Integer(-10, 10)]
        assert mock_object.mock_calls[0] == call(dimensions, base_estimator="GBRT")


def test_skopt_kwargs_dimenstions():
    # type: () -> None

    # User specified `dimensions` argument will be ignored in `SkoptSampler`.
    sampler = optuna.integration.SkoptSampler(skopt_kwargs={'dimensions': []})
    study = optuna.create_study(sampler=sampler)

    with patch('skopt.Optimizer') as mock_object:
        study.optimize(lambda t: t.suggest_int('x', -10, 10), n_trials=2)

        expected_dimensions = [space.Integer(-10, 10)]
        assert mock_object.mock_calls[0] == call(expected_dimensions)


def test_warn_independent_sampling():
    # type: () -> None

    # warn_independent_sampling=True
    sampler = optuna.integration.SkoptSampler(warn_independent_sampling=True)
    study = optuna.create_study(sampler=sampler)

    with patch('optuna.integration.skopt.SkoptSampler._log_independent_sampling') as mock_object:
        study.optimize(lambda t: t.suggest_uniform('p0', 0, 10), n_trials=1)
        assert mock_object.call_count == 0

    with patch('optuna.integration.skopt.SkoptSampler._log_independent_sampling') as mock_object:
        study.optimize(lambda t: t.suggest_uniform('p1', 0, 10), n_trials=1)
        assert mock_object.call_count == 1

    # warn_independent_sampling=False
    sampler = optuna.integration.SkoptSampler(warn_independent_sampling=False)
    study = optuna.create_study(sampler=sampler)

    with patch('optuna.integration.skopt.SkoptSampler._log_independent_sampling') as mock_object:
        study.optimize(lambda t: t.suggest_uniform('p0', 0, 10), n_trials=1)
        assert mock_object.call_count == 0

    with patch('optuna.integration.skopt.SkoptSampler._log_independent_sampling') as mock_object:
        study.optimize(lambda t: t.suggest_uniform('p1', 0, 10), n_trials=1)
        assert mock_object.call_count == 0


def test_is_compatible():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler()
    study = optuna.create_study(sampler=sampler)

    study.optimize(lambda t: t.suggest_uniform('p0', 0, 10), n_trials=1)
    search_space = optuna.samplers.product_search_space(study)
    assert search_space == {'p0': distributions.UniformDistribution(low=0, high=10)}

    optimizer = optuna.integration.skopt._Optimizer(search_space, {})

    # Compatible.
    trial = _create_frozen_trial({'p0': 5},
                                 {'p0': distributions.UniformDistribution(low=0, high=10)})
    assert optimizer._is_compatible(trial)

    # Compatible.
    trial = _create_frozen_trial({'p0': 5},
                                 {'p0': distributions.UniformDistribution(low=0, high=100)})
    assert optimizer._is_compatible(trial)

    # Compatible.
    trial = _create_frozen_trial({
        'p0': 5,
        'p1': 7
    }, {
        'p0': distributions.UniformDistribution(low=0, high=10),
        'p1': distributions.UniformDistribution(low=0, high=10)
    })
    assert optimizer._is_compatible(trial)

    # Incompatible ('p0' doesn't exist).
    trial = _create_frozen_trial({'p1': 5},
                                 {'p1': distributions.UniformDistribution(low=0, high=10)})
    assert not optimizer._is_compatible(trial)

    # Incompatible (the value of 'p0' is out of range).
    trial = _create_frozen_trial({'p0': 20},
                                 {'p0': distributions.UniformDistribution(low=0, high=100)})
    assert not optimizer._is_compatible(trial)

    # Error (different distribution class).
    trial = _create_frozen_trial({'p0': 5},
                                 {'p0': distributions.IntUniformDistribution(low=0, high=10)})
    with pytest.raises(ValueError):
        optimizer._is_compatible(trial)


def _objective(trial):
    # type: (optuna.trial.Trial) -> float

    p0 = trial.suggest_uniform('p0', -3.3, 5.2)
    p1 = trial.suggest_uniform('p1', 2.0, 2.0)
    p2 = trial.suggest_loguniform('p2', 0.0001, 0.3)
    p3 = trial.suggest_loguniform('p3', 1.1, 1.1)
    p4 = trial.suggest_int('p4', -100, 8)
    p5 = trial.suggest_int('p5', -20, -20)
    p6 = trial.suggest_discrete_uniform('p6', 10, 20, 2)
    p7 = trial.suggest_discrete_uniform('p7', 0.1, 1.0, 0.1)
    p8 = trial.suggest_discrete_uniform('p8', 2.2, 2.2, 0.5)
    p9 = trial.suggest_categorical('p9', ['9', '3', '0', '8'])

    return p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + int(p9)


def _create_frozen_trial(params, param_distributions):
    # type: (Dict[str, Any], Dict[str, distributions.BaseDistribution]) -> FrozenTrial

    params_in_internal_repr = {}
    for param_name, param_value in params.items():
        params_in_internal_repr[param_name] = param_distributions[param_name].to_internal_repr(
            param_value)

    return FrozenTrial(
        number=0,
        value=1.,
        state=optuna.structs.TrialState.COMPLETE,
        user_attrs={},
        system_attrs={},
        params=params,
        params_in_internal_repr=params_in_internal_repr,
        distributions=param_distributions,
        intermediate_values={},
        datetime_start=None,
        datetime_complete=None,
        trial_id=0,
    )
