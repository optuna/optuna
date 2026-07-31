from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna.study._multi_objective import _dominates
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.study import StudyDirection
    from optuna.trial import FrozenTrial


def _constrained_dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    """Checks constrained-domination.

    A trial x is said to constrained-dominate a trial y, if any of the following conditions is
    true:
    1) Trial x is feasible and trial y is not.
    2) Trial x and y are both infeasible, but solution x has a smaller overall constraint
    violation.
    3) Trial x and y are feasible and trial x dominates trial y.
    """

    constraints0 = trial0.constraints
    constraints1 = trial1.constraints

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    satisfy_constraints0 = all(v <= 0 for v in constraints0.values())
    satisfy_constraints1 = all(v <= 0 for v in constraints1.values())

    if satisfy_constraints0 and satisfy_constraints1:
        # Both trials satisfy the constraints.
        return _dominates(trial0, trial1, directions)

    if satisfy_constraints0:
        # trial0 satisfies the constraints, but trial1 violates them.
        return True

    if satisfy_constraints1:
        # trial1 satisfies the constraints, but trial0 violates them.
        return False

    # Both trials violate the constraints.
    violation0 = sum(v for v in constraints0.values() if v > 0)
    violation1 = sum(v for v in constraints1.values() if v > 0)
    return violation0 < violation1


def _evaluate_penalty(population: Sequence[FrozenTrial]) -> np.ndarray:
    """Evaluate penalty values of trials in population.
    Returns:
        A list of penalty values of trials in population, where a trial with constraint values of
        zero or less is feasible.
    """

    penalty: list[float] = []
    for trial in population:
        penalty.append(sum(v for v in trial.constraints.values() if v > 0))
    return np.array(penalty)


def _validate_constraints(
    population: list[FrozenTrial],
    *,
    is_constrained: bool = False,
) -> None:
    if not is_constrained:
        return

    for _trial in population:
        _constraints = _trial.constraints
        if np.any(np.isnan(list(_constraints.values()))):
            raise ValueError("NaN is not acceptable as constraint value.")
