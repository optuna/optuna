import itertools
from typing import List
from typing import Sequence

import pytest

import optuna
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


MINIMIZE = StudyDirection.MINIMIZE
MAXIMIZE = StudyDirection.MAXIMIZE


def _create_trial(values: Sequence[float], state: TrialState = TrialState.COMPLETE) -> FrozenTrial:
    return optuna.trial.create_trial(values=list(values), state=state)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_dominates_grid(n_dims: int) -> None:
    # Check all pairs of trials consisting of these values, i.e.,
    # [-inf, -inf], [-inf, -1], [-inf, 1], [-inf, inf], [-1, -inf], ...
    # These values should be specified in ascending order.
    vals_1d = [-float("inf"), -1, 1, float("inf")]

    # Check all possible directions.
    for directions in itertools.product([MINIMIZE, MAXIMIZE], repeat=n_dims):

        # Create list of values of each dimension in good-to-bad order.
        dim_vals = [vals_1d if directions[i] == MINIMIZE else vals_1d[::-1] for i in range(n_dims)]

        # This function converts integer indices to values.
        def get_values(index: Sequence[int]) -> List[float]:
            return [dim_vals[d][index[d]] for d in range(n_dims)]

        # Generate the set of all possible indices.
        all_indices = set(itertools.product(*[range(len(vals)) for vals in dim_vals]))
        for index in all_indices:

            # Generate the set of all indices that dominates the current index.
            dominating_indices = set(itertools.product(*[range(i + 1) for i in index]))
            dominating_indices -= {index}

            for dominating_index in dominating_indices:
                t1 = _create_trial(get_values(dominating_index))
                t2 = _create_trial(get_values(index))
                assert _dominates(t1, t2, directions)

            for other_index in all_indices - dominating_indices:
                t1 = _create_trial(get_values(other_index))
                t2 = _create_trial(get_values(index))
                assert not _dominates(t1, t2, directions)


def test_dominates_invalid() -> None:

    directions = [MINIMIZE, MAXIMIZE]

    # The numbers of objectives for `t0` and `t1` don't match.
    with pytest.raises(ValueError):
        t0 = _create_trial([1])  # One objective.
        t1 = _create_trial([1, 2])  # Two objectives.
        _dominates(t0, t1, directions)

    # The numbers of objectives and directions don't match.
    with pytest.raises(ValueError):
        t0 = _create_trial([1])  # One objective.
        t1 = _create_trial([1])  # One objective.
        _dominates(t0, t1, directions)


def test_dominates_various_states() -> None:

    directions = [MINIMIZE, MAXIMIZE]

    for t0_state in [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED]:
        t0 = _create_trial([1, 1], t0_state)

        for t1_state in [
            TrialState.COMPLETE,
            TrialState.FAIL,
            TrialState.WAITING,
            TrialState.PRUNED,
        ]:
            # If `t0` has not the COMPLETE state, it never dominates other trials.
            t1 = _create_trial([0, 2], t1_state)
            if t1_state == TrialState.COMPLETE:
                # If `t0` isn't COMPLETE and `t1` is COMPLETE, `t1` dominates `t0`.
                assert _dominates(t1, t0, directions)
                assert not _dominates(t0, t1, directions)
            else:
                # If `t1` isn't COMPLETE, it doesn't dominate others.
                assert not _dominates(t0, t1, directions)
                assert not _dominates(t1, t0, directions)
