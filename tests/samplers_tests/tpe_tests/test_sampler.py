import pytest

import optuna
from optuna.exceptions import TrialPruned
from optuna.samplers import tpe
from optuna.samplers import TPESampler

if optuna.type_checking.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


@pytest.mark.parametrize('use_hyperband', [False, True])
def test_hyperopt_parameters(use_hyperband):
    # type: (bool) -> None

    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    study = optuna.create_study(
        sampler=sampler, pruner=optuna.pruners.HyperbandPruner() if use_hyperband else None)
    study.optimize(lambda t: t.suggest_uniform('x', 10, 20), n_trials=50)


def test_get_observation_pairs():
    # type: () -> None

    def objective(trial):
        # type: (Trial) -> float

        x = trial.suggest_int('x', 5, 5)
        if trial.number == 0:
            return x
        elif trial.number == 1:
            trial.report(1, 4)
            trial.report(2, 7)
            raise TrialPruned()
        elif trial.number == 2:
            trial.report(float('nan'), 3)
            raise TrialPruned()
        elif trial.number == 3:
            raise TrialPruned()
        else:
            raise RuntimeError()

    # direction=minimize.
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5, catch=(RuntimeError,))
    trial_number = study._storage.create_new_trial(study._study_id)  # Create a running trial.
    trial = study._storage.get_trial(trial_number)

    assert tpe.sampler._get_observation_pairs(study, 'x', trial) == (
        [5.0, 5.0, 5.0, 5.0],
        [
            (-float('inf'), 5.0),   # COMPLETE
            (-7, 2),  # PRUNED (with intermediate values)
            (-3, float('inf')),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float('inf'), 0.0)  # PRUNED (without intermediate values)
        ])
    assert tpe.sampler._get_observation_pairs(study, 'y', trial) == ([], [])

    # direction=maximize.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=4)
    study._storage.create_new_trial(study._study_id)  # Create a running trial.

    assert tpe.sampler._get_observation_pairs(study, 'x', trial) == (
        [5.0, 5.0, 5.0, 5.0],
        [
            (-float('inf'), -5.0),   # COMPLETE
            (-7, -2),  # PRUNED (with intermediate values)
            (-3, float('inf')),  # PRUNED (with a NaN intermediate value; it's treated as infinity)
            (float('inf'), 0.0)  # PRUNED (without intermediate values)
        ])
    assert tpe.sampler._get_observation_pairs(study, 'y', trial) == ([], [])
