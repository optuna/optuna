from __future__ import annotations

import pytest

from optuna.storages import BaseStorage
from optuna.study._study_direction import StudyDirection
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


def _test_set_and_get_compatibility(
    storage_set: BaseStorage, storage_get: BaseStorage, values: list[float] | None
) -> None:
    study_id = storage_set.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage_set.create_new_trial(study_id)
    assert storage_get.get_trial(trial_id).state == TrialState.RUNNING
    storage_set.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=values)
    assert storage_get.get_trial(trial_id).state == TrialState.COMPLETE
    assert storage_get.get_trial(trial_id).values == values


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("values", [None, [0.0]])
def test_set_and_get_trial_state_values(storage_mode: str, values: list[float] | None) -> None:
    with StorageSupplier(storage_mode) as storage_direct:
        with StorageSupplier("grpc_proxy", base_storage=storage_direct) as storage_grpc_proxy:
            _test_set_and_get_compatibility(storage_grpc_proxy, storage_direct, values)
            _test_set_and_get_compatibility(storage_direct, storage_grpc_proxy, values)
