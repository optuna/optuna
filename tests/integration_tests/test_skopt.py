from mock import call
from mock import Mock
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
            space.Integer(-100, 9),

            # Original: trial.suggest_int('p5', -20, -20)
            space.Integer(-20, -19),

            # Original: trial.suggest_discrete_uniform('p6', 10, 20, 2)
            space.Integer(0, 6),

            # Original: trial.suggest_discrete_uniform('p7', 0.1, 1.0, 0.1)
            space.Integer(0, 9),

            # Original: trial.suggest_discrete_uniform('p8', 2.2, 2.2, 0.5)
            space.Integer(0, 1),

            # Original: trial.suggest_categorical('p9', ['9', '3', '0', '8'])
            space.Categorical(('9', '3', '0', '8'))
        ]
        assert mock_object.mock_calls[0] == call(dimensions)


def test_suggested_value():
    # type: () -> None

    independent_sampler = FirstTrialOnlyRandomSampler()
    sampler = optuna.integration.SkoptSampler(independent_sampler=independent_sampler,
                                              skopt_kwargs={'n_initial_points': 5})
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=10, catch=())

    for trial in study.trials:
        for param_name, param_value in trial.params_in_internal_repr:
            assert trial.distributions[param_name].contains(param_value)


def test_independent_sampler():
    # type: () -> None

    # first trial

    # second or later trials
    pass


def test_skopt_kwargs():
    # type: () -> None

    sampler = optuna.integration.SkoptSampler(skopt_kwargs={'base_estimator': "GBRT"})
    study = optuna.create_study(sampler=sampler)

    with patch('skopt.Optimizer') as mock_object:
        study.optimize(lambda t: t.suggest_int('x', -10, 10), n_trials=2)

        dimensions = [space.Integer(-10, 11)]
        assert mock_object.mock_calls[0] == call(dimensions, base_estimator="GBRT")


def test_skopt_kwargs_dimenstions():
    # type: () -> None

    # User specified `dimensions` argument will be ignored in `SkoptSampler`.
    sampler = optuna.integration.SkoptSampler(skopt_kwargs={'dimensions': []})
    study = optuna.create_study(sampler=sampler)

    with patch('skopt.Optimizer') as mock_object:
        study.optimize(lambda t: t.suggest_int('x', -10, 10), n_trials=2)

        expected_dimensions = [space.Integer(-10, 11)]
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
