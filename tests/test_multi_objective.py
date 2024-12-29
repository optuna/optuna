from __future__ import annotations

from collections.abc import Sequence

import pytest

from optuna import create_study
from optuna import trial
from optuna.study import StudyDirection
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.trial import FrozenTrial


def _trial_to_values(t: FrozenTrial) -> tuple[float, ...]:
    assert t.values is not None
    return tuple(t.values)


def assert_is_output_equal_to_ans(
    trials: list[FrozenTrial], directions: Sequence[StudyDirection], ans: set[tuple[int]]
) -> None:
    res = {_trial_to_values(t) for t in _get_pareto_front_trials_by_trials(trials, directions)}
    assert res == ans


@pytest.mark.parametrize(
    "directions,values_set,ans_set",
    [
        (
            ["minimize", "maximize"],
            [[2, 2], [1, 1], [3, 1], [3, 2], [1, 3]],
            [{(2, 2)}] + [{(1, 1), (2, 2)}] * 3 + [{(1, 3)}],
        ),
        (
            ["minimize", "maximize", "minimize"],
            [[2, 2, 2], [1, 1, 1], [3, 1, 3], [3, 2, 3], [1, 3, 1]],
            [{(2, 2, 2)}] + [{(1, 1, 1), (2, 2, 2)}] * 3 + [{(1, 3, 1)}],
        ),
    ],
)
def test_get_pareto_front_trials(
    directions: list[str], values_set: list[list[int]], ans_set: list[set[tuple[int]]]
) -> None:
    study = create_study(directions=directions)
    assert_is_output_equal_to_ans(study.trials, study.directions, set())
    for values, ans in zip(values_set, ans_set):
        study.optimize(lambda t: values, n_trials=1)
        assert_is_output_equal_to_ans(study.trials, study.directions, ans)

    assert len(_get_pareto_front_trials_by_trials(study.trials, study.directions)) == 1
    # The trial result is the same as the last one.
    study.optimize(lambda t: values_set[-1], n_trials=1)
    assert_is_output_equal_to_ans(study.trials, study.directions, ans_set[-1])
    assert len(_get_pareto_front_trials_by_trials(study.trials, study.directions)) == 2


@pytest.mark.parametrize(
    "directions,values_set,ans_set",
    [
        (["minimize", "maximize"], [[1, 1], [2, 2], [3, 2]], [{(1, 1), (2, 2)}, {(1, 1), (3, 2)}]),
        (
            ["minimize", "maximize", "minimize"],
            [[1, 1, 1], [2, 2, 2], [3, 2, 3]],
            [{(1, 1, 1), (2, 2, 2)}, {(1, 1, 1), (3, 2, 3)}],
        ),
    ],
)
def test_get_pareto_front_trials_with_constraint(
    directions: list[str], values_set: list[list[int]], ans_set: list[set[tuple[int]]]
) -> None:
    study = create_study(directions=directions)
    trials = [
        trial.create_trial(values=values, system_attrs={"constraints": [i % 2]})
        for i, values in enumerate(values_set)
    ]
    study.add_trials(trials)
    for consider_constraint, ans in zip([False, True], ans_set):
        trials = study.trials
        study_dirs = study.directions
        assert ans == {
            _trial_to_values(t)
            for t in _get_pareto_front_trials_by_trials(trials, study_dirs, consider_constraint)
        }
