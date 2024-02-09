from __future__ import annotations

from collections.abc import Callable

import pytest

import optuna
from optuna.importance._ped_anova._evaluator import QuantileFilter
from optuna.trial import FrozenTrial


def test_init() -> None:
    with pytest.raises(ValueError, match=r"quantile must be in *"):
        QuantileFilter(quantile=10.0, is_lower_better=True, min_n_top_trials=2, target=None)
    with pytest.raises(ValueError, match=r"quantile must be in *"):
        QuantileFilter(quantile=-1, is_lower_better=True, min_n_top_trials=2, target=None)
    with pytest.raises(ValueError, match=r"min_n_top_trials must be positive*"):
        QuantileFilter(quantile=0.1, is_lower_better=True, min_n_top_trials=-1, target=None)


def test_filter_must_have_target_for_multi_objective() -> None:
    trials = [optuna.create_trial(values=[1.0, 1.0])]
    _filter = QuantileFilter(quantile=0.1, is_lower_better=True, min_n_top_trials=1, target=None)
    with pytest.raises(ValueError, match=".*used for multi-objective.*"):
        _filter.filter(trials)


def test_len_trials_must_be_larger_than_or_equal_to_min_n_top_trials() -> None:
    trials = [optuna.create_trial(value=1.0) for _ in range(2)]
    _filter = QuantileFilter(quantile=0.1, is_lower_better=True, min_n_top_trials=2, target=None)
    _filter.filter(trials)  # This is OK.

    trials.pop(0)
    assert len(trials) == 1
    with pytest.raises(ValueError, match=r"len\(trials\) must be larger than or equal to.*"):
        _filter.filter(trials)


@pytest.mark.parametrize(
    "quantile,is_lower_better,values,target,filtered_indices",
    [
        (0.1, True, [1.0, 2.0], None, [0, 1]),  # Check min_n_trials = 2
        (0.5, True, list([float(i) for i in range(10)])[::-1], None, list(range(10))[5:]),
        (1.0, True, [1.0, 2.0], None, [0, 1]),
    ],
)
def test_filter(
    quantile: float,
    is_lower_better: bool,
    values: list[float] | list[list[float]],
    target: Callable[[FrozenTrial], float] | None,
    filtered_indices: list[int],
) -> None:
    _filter = QuantileFilter(quantile, is_lower_better, min_n_top_trials=2, target=target)

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
