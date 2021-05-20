from optuna import create_study
from optuna import Trial
from optuna import TrialPruned
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


def test_stop_with_MaxTrialsCallback() -> None:
    # Test stopping the optimization with MaxTrialsCallback.
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, callbacks=[MaxTrialsCallback(5)])
    assert len(study.trials) == 5

    # Test stopping the optimization with MaxTrialsCallback with pruned trials

    def pruned_objective(trial: Trial) -> float:
        raise TrialPruned()

    study = create_study()
    study.optimize(
        pruned_objective,
        n_trials=10,
        callbacks=[MaxTrialsCallback(5, states=(TrialState.PRUNED,))],
    )
    assert len(study.trials) == 5
