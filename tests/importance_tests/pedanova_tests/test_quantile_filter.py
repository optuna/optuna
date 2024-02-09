from __future__ import annotations

from collections.abc import Callable

import pytest

import optuna
from optuna.importance._ped_anova.evaluator import _QuantileFilter
from optuna.trial import FrozenTrial


_VALUES = list([float(i) for i in range(10)])[::-1]
_MULTI_VALUES = [[float(i), float(j)] for i, j in zip(range(10), reversed(range(10)))]


@pytest.mark.parametrize(
    "quantile,is_lower_better,values,target,filtered_indices",
    [
        (0.1, True, [1.0, 2.0], None, [0, 1]),  # Check min_n_trials = 2
        (0.49, True, _VALUES[:], None, list(range(10))[-5:]),
        (0.5, True, _VALUES[:], None, list(range(10))[-5:]),
        (0.51, True, _VALUES[:], None, list(range(10))[-6:]),
        (1.0, True, [1.0, 2.0], None, [0, 1]),
        (0.49, False, _VALUES[:], None, list(range(10))[:5]),
        (0.5, False, _VALUES[:], None, list(range(10))[:5]),
        (0.51, False, _VALUES[:], None, list(range(10))[:6]),
        # No tests for target!=None and is_lower_better=False because it is not used.
        (0.49, True, _MULTI_VALUES[:], lambda t: t.values[0], list(range(10))[:5]),
        (0.5, True, _MULTI_VALUES[:], lambda t: t.values[0], list(range(10))[:5]),
        (0.51, True, _MULTI_VALUES[:], lambda t: t.values[0], list(range(10))[:6]),
        (0.49, True, _MULTI_VALUES[:], lambda t: t.values[1], list(range(10))[-5:]),
        (0.5, True, _MULTI_VALUES[:], lambda t: t.values[1], list(range(10))[-5:]),
        (0.51, True, _MULTI_VALUES[:], lambda t: t.values[1], list(range(10))[-6:]),
    ],
)
def test_filter(
    quantile: float,
    is_lower_better: bool,
    values: list[float] | list[list[float]],
    target: Callable[[FrozenTrial], float] | None,
    filtered_indices: list[int],
) -> None:
    _filter = _QuantileFilter(quantile, is_lower_better, min_n_top_trials=2, target=target)

    def _create_trial(v: float | list[float]) -> FrozenTrial:
        if isinstance(v, float):
            return optuna.create_trial(value=v)
        elif isinstance(v, list):
            return optuna.create_trial(values=v)
        else:
            assert False, f"Unexpected Type for {v}"

    trials = [_create_trial(v) for v in values]
    for i, t in enumerate(trials):
        t.set_user_attr("index", i)

    indices = [t.user_attrs["index"] for t in _filter.filter(trials)]
    assert len(indices) == len(filtered_indices)
    assert all(i == j for i, j in zip(indices, filtered_indices))