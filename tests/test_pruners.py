import pfnopt


def test_median_pruner_without_reports():
    # type: () -> None

    study = pfnopt.study.create_study()
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    pruner = pfnopt.pruners.MedianPruner(0, 0)

    # A trial is not pruned if it has no intermediate results.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)


def test_median_pruner_with_only_one_trial():
    # type: () -> None

    study = pfnopt.study.create_study()
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(0, 1)
    pruner = pfnopt.pruners.MedianPruner(0, 0)

    # A first trial is not pruned.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)


def test_median_pruner_n_startup_trials():
    # type: () -> None

    pruner = pfnopt.pruners.MedianPruner(2, 0)
    study = pfnopt.study.create_study()

    # A first trial is not pruned.
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(0, 1)

    # A trial is not pruned during the startup trials.
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)

    # A trial is pruned after the startup trials.
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(2, 1)
    assert pruner.prune(storage=study.storage, study_id=study.study_id,
                        trial_id=trial.trial_id, step=1)


def test_median_pruner_n_warmup_steps():
    # type: () -> None

    pruner = pfnopt.pruners.MedianPruner(0, 1)
    study = pfnopt.study.create_study()

    # A first trial is not pruned.
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(0, 1)
    trial.report(0, 2)

    # A trial is not pruned during the warm-up steps.
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)
    # A trial is pruned after the warm-up steps.
    trial.report(1, 2)
    assert pruner.prune(storage=study.storage, study_id=study.study_id,
                        trial_id=trial.trial_id, step=2)
