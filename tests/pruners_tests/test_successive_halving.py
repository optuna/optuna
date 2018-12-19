import optuna


def test_successive_halving_pruner_with_one_trial():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)

    # A pruner is not activated at a first trial.
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)


def test_successive_halving_pruner_intermediate_values():
    # type: () -> None

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)

    # A pruner is not activated at a first trial.
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    # A pruner is not activated if a trial has no intermediate values.
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)

    trial.report(2, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)


def test_successive_halving_pruner_up_to_third_rung():
    # type: () -> None

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study()

    # Report 7 trials in advance.
    for i in range(7):
        trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
        trial.report(0.1 * (i + 1), step=7)
        pruner.prune(study.storage, study.study_id, trial.trial_id, step=7)

    # Report a trial that has the 7-th value from bottom.
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(0.75, step=7)
    pruner.prune(study.storage, study.study_id, trial.trial_id, step=7)
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs

    # Report a trial that has the third value from bottom.
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(0.25, step=7)
    pruner.prune(study.storage, study.study_id, trial.trial_id, step=7)
    assert 'completed_rung_1' in trial.system_attrs
    assert 'completed_rung_2' not in trial.system_attrs

    # Report a trial that has the lowest value.
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(0.05, step=7)
    pruner.prune(study.storage, study.study_id, trial.trial_id, step=7)
    assert 'completed_rung_2' in trial.system_attrs
    assert 'completed_rung_3' not in trial.system_attrs


def test_successive_halving_pruner_first_trial_always_wins():
    # type: () -> None

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    for i in range(10):
        trial.report(1, step=i)

        # The first trial always wins.
        assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=i)

    # The trial completed until rung 3.
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' in trial.system_attrs
    assert 'completed_rung_2' in trial.system_attrs
    assert 'completed_rung_3' in trial.system_attrs
    assert 'completed_rung_4' not in trial.system_attrs


def test_successive_halving_pruner_min_resource_parameter():
    # type: () -> None

    study = optuna.study.create_study()

    # min_resource=1: The rung 0 ends at step 1.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))

    trial.report(1, step=1)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs

    # min_resource=2: The rung 0 ends at step 2.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=2, reduction_factor=2, min_early_stopping_rate=0)
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))

    trial.report(1, step=1)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)
    assert 'completed_rung_0' not in trial.system_attrs

    trial.report(1, step=2)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=2)
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs


def test_successive_halving_pruner_reduction_factor_parameter():
    study = optuna.study.create_study()

    # reduction_factor=2: The rung 0 ends at step 1.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))

    trial.report(1, step=1)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs

    # reduction_factor=3: The rung 1 ends at step 3.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))

    trial.report(1, step=1)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs

    trial.report(1, step=2)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=2)
    assert 'completed_rung_1' not in trial.system_attrs

    trial.report(1, step=3)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=3)
    assert 'completed_rung_1' in trial.system_attrs
    assert 'completed_rung_2' not in trial.system_attrs


def test_successive_halving_pruner_min_early_stopping_rate_parameter():
    study = optuna.study.create_study()

    # min_early_stopping_rate=0: The rung 0 ends at step 1.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))

    trial.report(1, step=1)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)
    assert 'completed_rung_0' in trial.system_attrs

    # min_early_stopping_rate=1: The rung 0 ends at step 2.
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=1)
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))

    trial.report(1, step=1)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=1)
    assert 'completed_rung_0' not in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs

    trial.report(1, step=2)
    assert not pruner.prune(study.storage, study.study_id, trial.trial_id, step=2)
    assert 'completed_rung_0' in trial.system_attrs
    assert 'completed_rung_1' not in trial.system_attrs
