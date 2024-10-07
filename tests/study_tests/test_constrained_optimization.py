from optuna.study._constrained_optimization import _CONSTRAINTS_KEY
from optuna.study._constrained_optimization import _get_feasible_trials
from optuna.trial import create_trial


def test_get_feasible_trials() -> None:
    trials = []
    trials.append(create_trial(value=0.0, system_attrs={_CONSTRAINTS_KEY: [0.0]}))
    trials.append(create_trial(value=0.0, system_attrs={_CONSTRAINTS_KEY: [1.0]}))
    trials.append(create_trial(value=0.0))
    feasible_trials = _get_feasible_trials(trials)
    assert len(feasible_trials) == 1
    assert feasible_trials[0] == trials[0]
