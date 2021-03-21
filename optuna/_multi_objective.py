from typing import List
from typing import Optional
from typing import Sequence

import optuna
from optuna._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _get_pareto_front_trials_2d(study: "optuna.study.BaseStudy") -> List[FrozenTrial]:
    trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    points = sorted(
        (
            +_normalize_value(trial.values[0], study.directions[0]),
            -_normalize_value(trial.values[1], study.directions[1]),
            index,
        )
        for index, trial in enumerate(trials)
    )

    mask = [False] * len(trials)

    def set_mask(width: int, hi: int) -> None:
        for k in range(hi - width, hi):
            _, _, index = points[k]
            mask[index] = True

    width = 0
    best_y = float("inf")
    curr_x = float("nan")
    for i, (x, y, _) in enumerate(points):
        y = -y
        if curr_x != x:
            set_mask(width, hi=i)
            width = 0
        if y > best_y or (y == best_y and width == 0):
            continue
        if y < best_y:
            width = 0
        width += 1
        best_y = y
        curr_x = x
    set_mask(width, hi=len(points))

    pareto_front = [trial for trial, keep in zip(trials, mask) if keep]
    return pareto_front


def _get_pareto_front_trials_nd(study: "optuna.study.BaseStudy") -> List[FrozenTrial]:
    pareto_front = []
    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    # TODO(vincent): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
    for trial in trials:
        dominated = False
        for other in trials:
            if _dominates(other, trial, study.directions):
                dominated = True
                break

        if not dominated:
            pareto_front.append(trial)

    return pareto_front


def _get_pareto_front_trials(study: "optuna.study.BaseStudy") -> List[FrozenTrial]:
    if len(study.directions) == 2:
        return _get_pareto_front_trials_2d(study)  # Log-linear in number of trials.
    return _get_pareto_front_trials_nd(study)  # Quadratic in number of trials.


def _dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    values0 = trial0.values
    values1 = trial1.values

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
