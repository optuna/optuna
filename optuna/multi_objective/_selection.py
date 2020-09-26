from typing import List
from typing import Optional

import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState


def _get_pareto_front_trials(study) -> List["optuna.Trial"]:
    pareto_front = []
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    direction = study.direction

    # TODO(vincent): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
    for trial in trials:
        dominated = False
        for other in trials:
            if _dominates(other, trial, direction):
                dominated = True
                break

        if not dominated:
            pareto_front.append(trial)

    return pareto_front


def _dominates(
    trial0,
    trial1,
    directions: List[StudyDirection],
) -> bool:
    assert not isinstance(trial0.value, float), "Trial should have multiple values."
    assert not isinstance(trial1.value, float), "Trial should have multiple values."

    if len(trial0.value) != len(trial1.value):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(trial0.value) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    value0 = [_normalize_value(v, d) for v, d in zip(trial0.value, directions)]
    value1 = [_normalize_value(v, d) for v, d in zip(trial1.value, directions)]

    if value0 == value1:
        return False

    return all([v0 <= v1 for v0, v1 in zip(value0, value1)])


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
