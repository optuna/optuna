from __future__ import annotations

from unittest.mock import patch

import pytest

import optuna
from optuna.trial import TrialState


@pytest.mark.parametrize("storage", [None, "sqlite://"])
@pytest.mark.parametrize("state", [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL])
def test_finished_generation_lookup_does_not_write(storage: str | None, state: TrialState) -> None:
    sampler = optuna.samplers.NSGAIISampler(population_size=2, seed=123)
    study = optuna.create_study(sampler=sampler, storage=storage)
    key = sampler._get_generation_key()
    for value in (1.0, 2.0):
        study.add_trial(optuna.trial.create_trial(value=value, system_attrs={key: 0}))
    study.add_trial(
        optuna.trial.create_trial(state=state, value=3.0 if state == TrialState.COMPLETE else None)
    )
    frozen = study.trials[-1]
    before = dict(frozen.system_attrs)
    with patch.object(
        study._storage, "set_trial_system_attr", wraps=study._storage.set_trial_system_attr
    ) as write:
        assert sampler.get_trial_generation(study, frozen) == 1
        assert sampler.get_trial_generation(study, frozen) == 1
        write.assert_not_called()
    assert frozen.system_attrs == before
    assert study.trials[-1].system_attrs == before
    running = study.ask()
    assert sampler.get_trial_generation(study, study.trials[-1]) == 1
    assert study.trials[-1].system_attrs[key] == 1
    study.tell(running, 4.0)
