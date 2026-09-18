from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

from optuna.storages import BaseStorage
from optuna.study._study_direction import StudyDirection
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _test_set_and_get_compatibility(
    storage_set: BaseStorage,
    storage_get: BaseStorage,
    values: list[float] | None,
    template_trial: FrozenTrial | None = None,
) -> None:
    study_id = storage_set.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage_set.create_new_trial(study_id, template_trial=template_trial)
    trial = storage_get.get_trial(trial_id)
    assert trial.state == (
        template_trial.state if template_trial is not None else TrialState.RUNNING
    )
    assert trial.datetime_start == storage_set.get_trial(trial_id).datetime_start
    if template_trial is not None:
        assert template_trial.datetime_start is not None
        assert trial.datetime_start == template_trial.datetime_start.astimezone().replace(
            tzinfo=None
        )
    storage_set.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=values)
    trial = storage_get.get_trial(trial_id)
    assert trial.state == TrialState.COMPLETE
    assert trial.values == values
    assert trial.datetime_start == storage_set.get_trial(trial_id).datetime_start
    assert trial.datetime_complete == storage_set.get_trial(trial_id).datetime_complete


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    ("values", "template_timezone"),
    [
        pytest.param(None, None, id="without_template_without_values"),
        pytest.param([0.0], None, id="without_template_with_values"),
        pytest.param([0.0], timezone.utc, id="template_trial_utc"),
        pytest.param([0.0], timezone(timedelta(hours=9)), id="template_trial_utc_plus_9"),
        pytest.param([0.0], timezone(timedelta(hours=-8)), id="template_trial_utc_minus_8"),
    ],
)
def test_set_and_get_trial_state_values(
    storage_mode: str, values: list[float] | None, template_timezone: timezone | None
) -> None:
    template_trial: FrozenTrial | None = None
    if template_timezone is not None:
        datetime_start = datetime.now(template_timezone).replace(microsecond=123456)
        template_trial = FrozenTrial(
            number=0,
            state=TrialState.RUNNING,
            value=None,
            datetime_start=datetime_start,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            intermediate_values={},
            trial_id=0,
        )

    with StorageSupplier(storage_mode) as storage_direct:
        with StorageSupplier("grpc_proxy", base_storage=storage_direct) as storage_grpc_proxy:
            _test_set_and_get_compatibility(
                storage_grpc_proxy, storage_direct, values, template_trial
            )
            _test_set_and_get_compatibility(storage_direct, storage_grpc_proxy, values)
