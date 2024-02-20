from __future__ import annotations

from typing import Any

import optuna
from optuna import Study
from optuna.trial import TrialState


def _create_study(
    trial_states_list: list[TrialState],
    trial_sys_attrs: dict[str, Any] | None = None,
) -> Study:
    study = optuna.create_study()
    fmax = float(len(trial_states_list))
    for i, s in enumerate(trial_states_list):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(i)},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, fmax)},
                value=0.0 if s == TrialState.COMPLETE else None,
                state=s,
                system_attrs=trial_sys_attrs,
            )
        )
    return study
