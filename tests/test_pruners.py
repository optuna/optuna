import pytest

import pfnopt


parametrize_pruner_args = pytest.mark.parametrize(
    'n_startup_trials,n_warmup_steps',
    [(5, 0), (0, 5)]
)


@parametrize_pruner_args
def test_median_pruner(n_startup_trials, n_warmup_steps):
    # type: (int, int) -> None

    margin = 5
    pruner = pfnopt.pruners.MedianPruner(n_startup_trials, n_warmup_steps)
    study = pfnopt.study.create_study()
    for _ in range(n_startup_trials + margin):
        trial = pfnopt.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
        for step in range(1, n_warmup_steps + margin + 1):
            # Always greater than past trials.
            trial.report(trial.trial_id, step)
            result = pruner.prune(storage=study.storage, study_id=study.study_id,
                                  trial_id=trial.trial_id, step=step)
            if trial.trial_id < n_startup_trials:
                # Start up
                assert result is False
            elif step <= n_warmup_steps or trial.trial_id == 0:
                # Warm up
                assert result is False
            else:
                assert result is True
