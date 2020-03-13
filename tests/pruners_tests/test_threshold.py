import pytest

import optuna


def test_threshold_pruner_with_ub() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(upper=2, n_warmup_steps=0, interval_steps=1)

    trial.report(1, 1)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(3, 2)
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_lt() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(lower=2, n_warmup_steps=0, interval_steps=1)

    trial.report(3, 1)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1, 2)
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_two_side() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(lower=0, upper=1, n_warmup_steps=0, interval_steps=1)

    trial.report(-0.1, 1)
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(0, 2)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(0.4, 3)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1, 4)
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1.1, 5)
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_invalid_inputs() -> None:

    with pytest.raises(Exception):
        pruner = optuna.pruners.ThresholdPruner(lower='0', upper=1)  # NOQA

    with pytest.raises(Exception):
        pruner = optuna.pruners.ThresholdPruner(lower=0, upper=False)  # NOQA
