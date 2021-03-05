"""
Optuna example that demonstrates setting maximum number of trials in shared database when multiple workers are used.

In this example, we optimize a simple quadratic function. We use multiple script runs(workers)
to demonstrate the use of max_trial_callbacks, which essentially allows user to set maximum number of trials
regardless of the number of workers/scripts running the Trials. This example also demonstrates the flexibility of 
using max_trial_callbacks with shared database for parallel runs.

"""

import optuna
from optuna.trial import TrialState

max_trials = 10

def max_trial_callback(study, trial):
    #we consider all the running states and already completed states in account for estimating number of n_complete.
    n_complete = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE, TrialState.RUNNING]))
    if n_complete >= max_trials:
        study.stop()
        

def objective(trial):
    x = trial.suggest_uniform("x", 0, 10)
    return x ** 2


if __name__ == '__main__':
    _ = optuna.create_study(
        study_name="test",
        storage="sqlite:///database.sqlite",
    )
    
    study = optuna.load_study(study_name="test", storage="sqlite:///database.sqlite")
    study.optimize(objective, n_trials=50,callbacks=[max_trial_callback])
    trials = study.trials_dataframe()
    assert len(trials[trials.state == "COMPLETE"]) == max_trials