import pytest

import pfnopt


@pytest.mark.parametrize('n_startup_trials', [0, 5])
def test_median_pruner_n_startup_trials(n_startup_trials):
    # type: (int) -> None

    pruner = pfnopt.pruners.MedianPruner(n_startup_trials, 0)
    study = pfnopt.study.create_study()
    for count in range(n_startup_trials + 5):
        trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
        for step in range(1, 6):
            # Report a trial count as an objective value that always exceeds a pruner threshold.
            trial.report(count, step)
            result = pruner.prune(storage=study.storage, study_id=study.study_id,
                                  trial_id=trial.trial_id, step=step)
            if trial.trial_id == 0:
                # Pruning does not happen at the first trial.
                assert result is False
            elif trial.trial_id < n_startup_trials:
                # Pruning does not happen if the number of trials is less than n_startup_trials.
                assert result is False
            else:
                assert result is True


@pytest.mark.parametrize('n_warmup_steps', [0, 5])
def test_median_pruner_n_warmup_steps(n_warmup_steps):
    # type: (int) -> None

    pruner = pfnopt.pruners.MedianPruner(0, n_warmup_steps)
    study = pfnopt.study.create_study()
    for count in range(5):
        trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
        for step in range(1, n_warmup_steps + 6):
            # Report a trial count as an objective value that always exceeds a pruner threshold.
            trial.report(count, step)
            result = pruner.prune(storage=study.storage, study_id=study.study_id,
                                  trial_id=trial.trial_id, step=step)
            if trial.trial_id == 0:
                # Pruning does not happen at the first trial.
                assert result is False
            if step <= n_warmup_steps or trial.trial_id == 0:
                # Pruning does not happen if the number of steps is less than n_warmup_steps.
                assert result is False
            else:
                assert result is True
