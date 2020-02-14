import optuna
from optuna.distributions import IntUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers.search_space import intersection_search_space


def test_intersection_search_space():
    # type: () -> None

    study = optuna.create_study()

    # No trial.
    assert intersection_search_space(study) == {}

    # First trial.
    study.optimize(lambda t: t.suggest_int('x', 0, 10) + t.suggest_uniform('y', -3, 3), n_trials=1)
    assert intersection_search_space(study) == {
        'x': IntUniformDistribution(low=0, high=10),
        'y': UniformDistribution(low=-3, high=3)
    }

    # Second trial (only 'y' parameter is suggested in this trial).
    study.optimize(lambda t: t.suggest_uniform('y', -3, 3), n_trials=1)
    assert intersection_search_space(study) == {
        'y': UniformDistribution(low=-3, high=3)
    }

    # Failed or pruned trials are not considered in the calculation of
    # an intersection search space.
    def objective(trial, exception):
        # type: (optuna.trial.Trial, Exception) -> float

        trial.suggest_uniform('z', 0, 1)
        raise exception

    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, optuna.exceptions.TrialPruned()), n_trials=1)
    assert intersection_search_space(study) == {
        'y': UniformDistribution(low=-3, high=3)
    }

    # If two parameters have the same name but different distributions,
    # those are regarded as different trials.
    study.optimize(lambda t: t.suggest_uniform('y', -1, 1), n_trials=1)
    assert intersection_search_space(study) == {}
