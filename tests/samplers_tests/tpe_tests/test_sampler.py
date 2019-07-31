import optuna

from optuna.samplers import tpe
from optuna.samplers import TPESampler
from optuna.structs import TrialPruned
from optuna.study import InTrialStudy

if optuna.types.TYPE_CHECKING:
    from optuna.trial import Trial  # NOQA


def test_hyperopt_parameters():
    # type: () -> None

    sampler = TPESampler(**TPESampler.hyperopt_parameters())
    study = optuna.create_study(sampler=sampler)
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
            raise TrialPruned()
        else:
            raise RuntimeError()

    # direction=minimize.
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=4)
    study.storage.create_new_trial_id(study.study_id)  # Create a running trial.

    in_trial_study = InTrialStudy(study)

    assert tpe.sampler._get_observation_pairs(in_trial_study, 'x') == (
        [5.0, 5.0, 5.0],
        [
            (-float('inf'), 5.0),   # COMPLETE
            (-7, 2),  # PRUNED (with intermediate values)
            (float('inf'), 0.0)  # PRUNED (without intermediate values)
        ])
    assert tpe.sampler._get_observation_pairs(in_trial_study, 'y') == ([], [])

    # direction=maximize.
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=4)
    study.storage.create_new_trial_id(study.study_id)  # Create a running trial.

    in_trial_study = InTrialStudy(study)

    assert tpe.sampler._get_observation_pairs(in_trial_study, 'x') == (
        [5.0, 5.0, 5.0],
        [
            (-float('inf'), -5.0),   # COMPLETE
            (-7, -2),  # PRUNED (with intermediate values)
            (float('inf'), 0.0)  # PRUNED (without intermediate values)
        ])
    assert tpe.sampler._get_observation_pairs(in_trial_study, 'y') == ([], [])
