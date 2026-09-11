from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from optuna._warnings import optuna_warn
from optuna.study._study_direction import StudyDirection


if TYPE_CHECKING:
    from optuna.trial import FrozenTrial


_CONSTRAINTS_KEY = "constraints"


def _is_constrained_optimization(trials: Sequence[FrozenTrial]) -> bool:
    """Return whether the given trials are created in constrained optimization."""

    return any(len(trial.constraints) > 0 for trial in trials)


def _get_feasible_trials(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    """Return feasible trials from given trials.

    Returns:
        A list of feasible trials.
    """

    feasible_trials = []
    for trial in trials:
        constraints = trial.constraints.values()
        if all(x <= 0.0 for x in constraints):
            feasible_trials.append(trial)
    return feasible_trials


def _get_best_feasible_trial(
    trials: Sequence[FrozenTrial], direction: StudyDirection
) -> FrozenTrial | None:
    """Return the best feasible trial from given trials.

    Returns:
        The best feasible trial, or ``None`` if no feasible trial exists.
    """

    feasible_trials = _get_feasible_trials(trials)
    if len(feasible_trials) == 0:
        return None
    if direction == StudyDirection.MAXIMIZE:
        return max(feasible_trials, key=lambda t: cast("float", t.value))
    return min(feasible_trials, key=lambda t: cast("float", t.value))


def _get_constraints_from_system_attrs(system_attrs: dict[str, Any]) -> dict[str, float]:
    constraints_dict: dict[str, float] = {}

    # Load constraints from old format (list)
    con = system_attrs.get(_CONSTRAINTS_KEY)
    if con is not None:
        for i, c in enumerate(con):
            constraints_dict[str(i)] = c

    # Load constraints from new format (individual keys)
    prefix = f"{_CONSTRAINTS_KEY}:"
    for key, value in system_attrs.items():
        if key.startswith(prefix):
            constraint_key = key[len(prefix) :]
            if constraint_key in constraints_dict:
                optuna_warn("Overwrite an old format constraint.")
            constraints_dict[constraint_key] = value
    return constraints_dict
