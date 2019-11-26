import optuna


def test_nop_pruner():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 1)
    pruner = optuna.pruners.NopPruner()

    # A NopPruner instance is always deactivated.
    assert not pruner.prune(study=study, trial=study.trials[0])
