from optuna.study import create_study
from optuna.study import Study
from optuna.terminator import BaseTerminator
from optuna.terminator import TerminatorCallback
from optuna.trial import TrialState


class _DeterministicTerminator(BaseTerminator):
    def __init__(self, termination_trial_number: int) -> None:
        self._termination_trial_number = termination_trial_number

    def should_terminate(self, study: Study) -> bool:
        trials = study.get_trials(states=[TrialState.COMPLETE])
        latest_number = max([t.number for t in trials])

        if latest_number >= self._termination_trial_number:
            return True
        else:
            return False


def test_terminator_callback_terminator() -> None:
    # This test case validates that the study is stopped when the `should_terminate` method of the
    # terminator returns `True` for the first time.
    termination_trial_number = 10

    callback = TerminatorCallback(
        terminator=_DeterministicTerminator(termination_trial_number),
    )

    study = create_study()
    study.optimize(lambda _: 0.0, callbacks=[callback], n_trials=100)

    assert len(study.trials) == termination_trial_number + 1
