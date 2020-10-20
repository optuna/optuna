from typing import List
from typing import Optional

import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState


def _get_pareto_front_trials(study: "optuna.Study") -> List["optuna.trial.FrozenTrial"]:
    pareto_front = []
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if isinstance(study.direction, str):
        direction = [study.direction]
    else:
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
    trial0: "optuna.trial.FrozenTrial",
    trial1: "optuna.trial.FrozenTrial",
    directions: List[StudyDirection],
) -> bool:
    value0 = trial0.value
    value1 = trial1.value

    assert not isinstance(value0, float), "Trial should have multiple values."
    assert not isinstance(value1, float), "Trial should have multiple values."

    if len(value0) != len(value1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(value0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    value0 = [_normalize_value(v, d) for v, d in zip(value0, directions)]
    value1 = [_normalize_value(v, d) for v, d in zip(value1, directions)]

    if value0 == value1:
        return False

    return all([v0 <= v1 for v0, v1 in zip(value0, value1)])


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
