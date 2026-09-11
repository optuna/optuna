from optuna.study._constrained_optimization import _CONSTRAINTS_KEY
from optuna.study._constrained_optimization import _get_best_feasible_trial
from optuna.study._constrained_optimization import _get_feasible_trials
from optuna.study._study_direction import StudyDirection
from optuna.trial import create_trial


def test_get_feasible_trials() -> None:
    trials = []
    trials.append(create_trial(value=0.0, system_attrs={_CONSTRAINTS_KEY: [0.0]}))
    trials.append(create_trial(value=0.0, system_attrs={_CONSTRAINTS_KEY: [1.0]}))
    # A trial without constraint values is considered feasible.
    trials.append(create_trial(value=0.0))
    feasible_trials = _get_feasible_trials(trials)
    assert len(feasible_trials) == 2
    assert feasible_trials[0] == trials[0]
    assert feasible_trials[1] == trials[2]


def test_get_best_feasible_trial() -> None:
    infeasible = create_trial(value=10.0, system_attrs={_CONSTRAINTS_KEY: [1.0]})
    feasible_low = create_trial(value=0.0, system_attrs={_CONSTRAINTS_KEY: [0.0]})
    feasible_high = create_trial(value=1.0, system_attrs={_CONSTRAINTS_KEY: [-1.0]})
    unconstrained = create_trial(value=-2.0)

    assert _get_best_feasible_trial([], StudyDirection.MINIMIZE) is None
    assert _get_best_feasible_trial([infeasible], StudyDirection.MINIMIZE) is None
    assert (
        _get_best_feasible_trial(
            [infeasible, feasible_low, feasible_high], StudyDirection.MINIMIZE
        )
        == feasible_low
    )
    assert (
        _get_best_feasible_trial(
            [infeasible, feasible_low, feasible_high], StudyDirection.MAXIMIZE
        )
        == feasible_high
    )
    # A trial without constraint values is considered feasible.
    assert (
        _get_best_feasible_trial([infeasible, unconstrained], StudyDirection.MINIMIZE)
        == unconstrained
    )
