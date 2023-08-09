from __future__ import annotations

import pytest

import optuna
from optuna.preferential._system_attrs import get_preferences
from optuna.preferential._system_attrs import report_preferences
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_report_and_get_preferences(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.ask()
        study.ask()

        assert len(get_preferences(study)) == 0

        better, worse = study.trials[0], study.trials[1]
        report_preferences(study, [(better, worse)])
        assert len(get_preferences(study)) == 1

        actual_better, actual_worse = get_preferences(study)[0]
        assert actual_better.number == better.number
        assert actual_worse.number == worse.number
