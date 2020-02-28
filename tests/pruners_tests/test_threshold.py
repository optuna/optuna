import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA
    from typing import Tuple  # NOQA


def test_threshold_pruner_with_ub():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(upper_bound=2)

    trial.report(1, 1)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(3, 2)
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_lt():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(lower_bound=2)

    trial.report(3, 1)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1, 2)
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))
