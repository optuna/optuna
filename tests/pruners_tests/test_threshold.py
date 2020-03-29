import pytest

import optuna


def test_threshold_pruner_with_ub() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(upper=2.0, n_warmup_steps=0, interval_steps=1)

    trial.report(1.0, 1)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(3.0, 2)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_lt() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(lower=2.0, n_warmup_steps=0, interval_steps=1)

    trial.report(3.0, 1)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1.0, 2)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_two_side() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(
        lower=0.0, upper=1.0, n_warmup_steps=0, interval_steps=1
    )

    trial.report(-0.1, 1)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(0.0, 2)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(0.4, 3)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1.0, 4)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1.1, 5)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_with_invalid_inputs() -> None:

    with pytest.raises(TypeError):
        optuna.pruners.ThresholdPruner(lower="val", upper=1.0)  # type: ignore

    with pytest.raises(TypeError):
        optuna.pruners.ThresholdPruner(lower=0.0, upper="val")  # type: ignore

    with pytest.raises(TypeError):
        optuna.pruners.ThresholdPruner(lower=None, upper=None)


def test_threshold_pruner_with_nan() -> None:

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(
        lower=0.0, upper=1.0, n_warmup_steps=0, interval_steps=1
    )

    trial.report(float("nan"), 1)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_n_warmup_steps() -> None:
    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(lower=0.0, upper=1.0, n_warmup_steps=2)

    trial.report(-10.0, 1)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(100.0, 2)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(-100.0, 3)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1.0, 4)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1000.0, 5)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_threshold_pruner_interval_steps() -> None:
    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    pruner = optuna.pruners.ThresholdPruner(lower=0.0, upper=1.0, interval_steps=2)

    trial.report(-10.0, 1)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(100.0, 2)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(-100.0, 3)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(10.0, 4)
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(1000.0, 5)
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
