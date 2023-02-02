import copy

import pytest

from optuna import create_study
from optuna import get_all_study_summaries
from optuna.storages import RDBStorage


def test_study_summary_eq_ne() -> None:
    storage = RDBStorage("sqlite:///:memory:")

    create_study(storage=storage)
    study = create_study(storage=storage)

    summaries = get_all_study_summaries(study._storage, include_best_trial=True)
    assert len(summaries) == 2

    assert summaries[0] == copy.deepcopy(summaries[0])
    assert not summaries[0] != copy.deepcopy(summaries[0])

    assert not summaries[0] == summaries[1]
    assert summaries[0] != summaries[1]

    assert not summaries[0] == 1
    assert summaries[0] != 1


def test_study_summary_lt_le() -> None:
    storage = RDBStorage("sqlite:///:memory:")

    create_study(storage=storage)
    study = create_study(storage=storage)

    summaries = get_all_study_summaries(study._storage, include_best_trial=True)
    assert len(summaries) == 2

    summary_0 = summaries[0]
    summary_1 = summaries[1]

    assert summary_0 < summary_1
    assert not summary_1 < summary_0

    with pytest.raises(TypeError):
        summary_0 < 1

    assert summary_0 <= summary_0
    assert not summary_1 <= summary_0

    with pytest.raises(TypeError):
        summary_0 <= 1

    # A list of StudySummaries is sortable.
    summaries.reverse()
    summaries.sort()
    assert summaries[0] == summary_0
    assert summaries[1] == summary_1
