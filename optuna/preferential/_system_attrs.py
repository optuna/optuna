from __future__ import annotations

import uuid

import optuna
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_SYSTEM_ATTR_PREFIX_PREFERENCE = "preference:values"


def report_preferences(
    study: optuna.Study,
    preferences: list[tuple[FrozenTrial, FrozenTrial]],
) -> None:
    key = _SYSTEM_ATTR_PREFIX_PREFERENCE + str(uuid.uuid4())
    study._storage.set_study_system_attr(
        study_id=study._study_id,
        key=key,
        value=[(better.number, worse.number) for better, worse in preferences],
    )

    values = [0 for _ in study.directions]
    for better, worse in preferences:
        for t in (better, worse):
            study.tell(
                t.number,
                values=values,
                state=TrialState.COMPLETE,
                skip_if_finished=True,
            )


def get_preferences(
    study: optuna.Study,
    *,
    deepcopy: bool = True,
) -> list[tuple[FrozenTrial, FrozenTrial]]:
    preferences: list[tuple[int, int]] = []
    for k, v in study.system_attrs.items():
        if not k.startswith(_SYSTEM_ATTR_PREFIX_PREFERENCE):
            continue
        preferences.extend(v)  # type: ignore
    trials = study.get_trials(deepcopy=deepcopy)
    return [(trials[better], trials[worse]) for (better, worse) in preferences]
