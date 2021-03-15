"""
Optuna example to demonstrate setting the maximum number of trials in
a shared database when multiple workers are used.

In this example, we optimize a simple quadratic function. We use multiple
script runs (workers) to demonstrate the use of max_trial_callbacks,
which allows the user to set a maximum number of trials
regardless of the number of workers/scripts running the Trials.

"""

from time import sleep

import optuna
from optuna.trial import TrialState


num_completed_trials = 10


def max_trial_callback(study, trial):
    # we consider all the running states and already completed states.
    n_complete = len(
        study.get_trials(deepcopy=False, states=[TrialState.COMPLETE, TrialState.RUNNING])
    )
    if n_complete >= num_completed_trials:
        study.stop()


def objective(trial):
    sleep(1)
    x = trial.suggest_uniform("x", 0, 10)
    return x ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="test",
        storage="sqlite:///database.sqlite",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50, callbacks=[max_trial_callback])
    trials = study.trials_dataframe()
    print("Number of completed trials: {}".format(len(trials[trials.state == "COMPLETE"])))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
