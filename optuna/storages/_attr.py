from typing import Any
from typing import Dict

import optuna


class SystemAttributeStorage:
    def __init__(
        self, storage: "optuna.storages.BaseStorage", study_id: int, trial_id: int
    ) -> None:
        self._storage = storage
        self._study_id = study_id
        self._trial_id = trial_id

    def get_study_attrs(self) -> Dict[str, Any]:
        return self._storage.get_study_system_attrs(self._study_id)

    def set_study_attr(self, key: str, value: Any) -> None:
        self._storage.set_study_system_attr(
            self._study_id,
            key,
            value,
        )

    def get_trial_attrs(self) -> Dict[str, Any]:
        return self._storage.get_trial_system_attrs(self._trial_id)

    def set_trial_attr(self, key: str, value: Any) -> None:
        self._storage.set_trial_system_attr(
            self._trial_id,
            key,
            value,
        )


def _create_system_attr_storage(
    study: "optuna.study.Study", trial: "optuna.trial.Trial"
) -> SystemAttributeStorage:
    return SystemAttributeStorage(
        storage=study._storage,
        study_id=study._study_id,
        trial_id=None if trial is None else trial._trial_id,
    )
