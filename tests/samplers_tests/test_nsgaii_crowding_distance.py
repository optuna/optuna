from __future__ import annotations

import datetime
import math

import numpy as np

from optuna.samplers.nsgaii._elite_population_selection_strategy import _calc_crowding_distance
from optuna.samplers.nsgaii._elite_population_selection_strategy import _crowding_distance_sort
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _make_trial(number: int, values: list[float]) -> FrozenTrial:
    return FrozenTrial(
        number=number,
        state=TrialState.COMPLETE,
        value=None,
        values=values,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


def test_calc_crowding_distance_multi_objective_ties_and_infinities() -> None:
    values = [
        [-float("inf"), 1.0, 5.0],
        [-float("inf"), 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, float("inf")],
        [float("inf"), 1.0, float("inf")],
    ]
    trials = [_make_trial(i, v) for i, v in enumerate(values)]
    dist = _calc_crowding_distance(trials)

    for i in range(len(values)):
        assert math.isinf(dist[i]) and dist[i] > 0


def test_crowding_distance_sort_matches_reference_order() -> None:
    rng = np.random.default_rng(0)
    raw_values = rng.choice([-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf], size=(40, 4)).tolist()
    trials = [_make_trial(i, v) for i, v in enumerate(raw_values)]

    ref_trials = list(trials)
    ref_dist: dict[int, float] = {t.number: 0.0 for t in ref_trials}
    for dim in range(4):
        ref_trials.sort(key=lambda x: x.values[dim])
        if ref_trials[0].values[dim] == ref_trials[-1].values[dim]:
            continue
        vs = [-float("inf")] + [t.values[dim] for t in ref_trials] + [float("inf")]
        v_min = next(x for x in vs if x != -float("inf"))
        v_max = next(x for x in reversed(vs) if x != float("inf"))
        width = v_max - v_min
        if width <= 0:
            width = 1.0
        for j, t in enumerate(ref_trials):
            gap = 0.0 if vs[j] == vs[j + 2] else vs[j + 2] - vs[j]
            ref_dist[t.number] += gap / width
    ref_trials.sort(key=lambda x: ref_dist[x.number])
    ref_trials.reverse()

    dist = _calc_crowding_distance(list(trials))
    for k, v in ref_dist.items():
        assert (dist[k] == v) or math.isclose(dist[k], v, rel_tol=1e-14, abs_tol=1e-14)

    _crowding_distance_sort(trials)
    assert [t.number for t in trials] == [t.number for t in ref_trials]
