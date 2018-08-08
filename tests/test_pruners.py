import pfnopt
from pfnopt.structs import TrialState


def test_median_pruner_without_reports():
    # type: () -> None

    study = pfnopt.study.create_study()
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    pruner = pfnopt.pruners.MedianPruner(0, 0)

    # A pruner is not activated if no trials are reported.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)


def test_median_pruner_with_one_trial():
    # type: () -> None

    study = pfnopt.study.create_study()
    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    pruner = pfnopt.pruners.MedianPruner(0, 0)

    # A pruner is not activated at a first trial.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)


def test_median_pruner_without_intermediate_values():
    # type: () -> None

    pruner = pfnopt.pruners.MedianPruner(0, 0)
    study = pfnopt.study.create_study()

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    study.storage.set_trial_state(trial.trial_id, TrialState.COMPLETE)

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    # A pruner is not activated if a trial has no intermediate values.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)

    trial.report(2, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert pruner.prune(storage=study.storage, study_id=study.study_id,
                        trial_id=trial.trial_id, step=1)


def test_median_pruner_n_startup_trials():
    # type: () -> None

    pruner = pfnopt.pruners.MedianPruner(2, 0)
    study = pfnopt.study.create_study()

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    study.storage.set_trial_state(trial.trial_id, TrialState.COMPLETE)

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(2, 1)
    # A pruner is not activated during startup trials.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)
    study.storage.set_trial_state(trial.trial_id, TrialState.COMPLETE)

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(3, 1)
    # A pruner is activated after startup trials.
    assert pruner.prune(storage=study.storage, study_id=study.study_id,
                        trial_id=trial.trial_id, step=1)


def test_median_pruner_n_warmup_steps():
    # type: () -> None

    pruner = pfnopt.pruners.MedianPruner(0, 1)
    study = pfnopt.study.create_study()

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    trial.report(1, 2)
    study.storage.set_trial_state(trial.trial_id, TrialState.COMPLETE)

    trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(2, 1)
    # A pruner is not activated during warm-up steps.
    assert not pruner.prune(storage=study.storage, study_id=study.study_id,
                            trial_id=trial.trial_id, step=1)

    trial.report(2, 2)
    # A pruner is activated after warm-up steps.
    assert pruner.prune(storage=study.storage, study_id=study.study_id,
                        trial_id=trial.trial_id, step=2)
