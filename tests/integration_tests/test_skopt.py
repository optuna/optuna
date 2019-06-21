from mock import call
from mock import patch
from skopt.space import space

import optuna
from optuna.testing.sampler import FirstTrialOnlyRandomSampler


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


def test_nan_objective_value():
    # type: () -> None

    independent_sampler = FirstTrialOnlyRandomSampler()
    sampler = optuna.integration.SkoptSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)

    # Non NaN objective values.
    for i in range(10, 1, -1):
        objective = lambda trial: trial.suggest_uniform('x', 0.1, 0.2) + i
        study.optimize(objective, n_trials=1, catch=())
    assert int(study.best_value) == 2

    # NaN objective values.
    objective = lambda trial: trial.suggest_uniform('x', 0.1, 0.2) + float('nan')
    study.optimize(objective, n_trials=1, catch=())
    assert int(study.best_value) == 2

    # Non NaN objective value.
    objective = lambda trial: trial.suggest_uniform('x', 0.1, 0.2) + 1
    study.optimize(objective, n_trials=1, catch=())
    assert int(study.best_value) == 1


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
        mock_object.side_effect = [1, 2, 3, 3, 0]

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
