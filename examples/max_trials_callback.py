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
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


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

    study.optimize(
        objective, n_trials=50, callbacks=[MaxTrialsCallback(10, states=(TrialState.COMPLETE,))]
    )
    trials = study.trials_dataframe()
    print("Number of completed trials: {}".format(len(trials[trials.state == "COMPLETE"])))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
