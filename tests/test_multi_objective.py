from __future__ import annotations

from optuna import create_study
from optuna import trial
from optuna.study._multi_objective import _get_pareto_front_trials_2d
from optuna.study._multi_objective import _get_pareto_front_trials_nd
from optuna.trial import FrozenTrial


def _trial_to_values(t: FrozenTrial) -> tuple[float, ...]:
    assert t.values is not None
    return tuple(t.values)


def test_get_pareto_front_trials_2d() -> None:
    study = create_study(directions=["minimize", "maximize"])
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == set()

    study.optimize(lambda t: [2, 2], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == {(2, 2)}

    study.optimize(lambda t: [1, 1], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == {(1, 1), (2, 2)}

    study.optimize(lambda t: [3, 1], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == {(1, 1), (2, 2)}

    study.optimize(lambda t: [3, 2], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == {(1, 1), (2, 2)}

    study.optimize(lambda t: [1, 3], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == {(1, 3)}
    assert len(_get_pareto_front_trials_2d(study.trials, study.directions)) == 1

    study.optimize(lambda t: [1, 3], n_trials=1)  # The trial result is the same as the above one.
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_2d(study.trials, study.directions)
    } == {(1, 3)}
    assert len(_get_pareto_front_trials_2d(study.trials, study.directions)) == 2


def test_get_pareto_front_trials_2d_with_constraint() -> None:
    study = create_study(directions=["minimize", "maximize"])
    trials = [
        trial.create_trial(values=[1, 1], system_attrs={"constraints": [0]}),
        trial.create_trial(values=[2, 2], system_attrs={"constraints": [1]}),
        trial.create_trial(values=[3, 2], system_attrs={"constraints": [0]}),
    ]
    study.add_trials(trials)
    assert {
        _trial_to_values(t)
        for t in _get_pareto_front_trials_2d(study.trials, study.directions, False)
    } == {(1, 1), (2, 2)}
    assert {
        _trial_to_values(t)
        for t in _get_pareto_front_trials_2d(study.trials, study.directions, True)
    } == {(1, 1), (3, 2)}


def test_get_pareto_front_trials_nd() -> None:
    study = create_study(directions=["minimize", "maximize", "minimize"])
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == set()

    study.optimize(lambda t: [2, 2, 2], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == {(2, 2, 2)}

    study.optimize(lambda t: [1, 1, 1], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == {
        (1, 1, 1),
        (2, 2, 2),
    }

    study.optimize(lambda t: [3, 1, 3], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == {
        (1, 1, 1),
        (2, 2, 2),
    }

    study.optimize(lambda t: [3, 2, 3], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == {
        (1, 1, 1),
        (2, 2, 2),
    }

    study.optimize(lambda t: [1, 3, 1], n_trials=1)
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == {(1, 3, 1)}
    assert len(_get_pareto_front_trials_nd(study.trials, study.directions)) == 1

    study.optimize(
        lambda t: [1, 3, 1], n_trials=1
    )  # The trial result is the same as the above one.
    assert {
        _trial_to_values(t) for t in _get_pareto_front_trials_nd(study.trials, study.directions)
    } == {(1, 3, 1)}
    assert len(_get_pareto_front_trials_nd(study.trials, study.directions)) == 2


def test_get_pareto_front_trials_nd_with_constraint() -> None:
    study = create_study(directions=["minimize", "maximize", "minimize"])
    trials = [
        trial.create_trial(values=[1, 1, 1], system_attrs={"constraints": [0]}),
        trial.create_trial(values=[2, 2, 2], system_attrs={"constraints": [1]}),
        trial.create_trial(values=[3, 2, 3], system_attrs={"constraints": [0]}),
    ]
    study.add_trials(trials)
    assert {
        _trial_to_values(t)
        for t in _get_pareto_front_trials_nd(study.trials, study.directions, False)
    } == {(1, 1, 1), (2, 2, 2)}
    assert {
        _trial_to_values(t)
        for t in _get_pareto_front_trials_nd(study.trials, study.directions, True)
    } == {(1, 1, 1), (3, 2, 3)}
