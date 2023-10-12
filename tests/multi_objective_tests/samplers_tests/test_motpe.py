from typing import Dict
from typing import Tuple
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import optuna
from optuna import multi_objective
from optuna.multi_objective.samplers import MOTPEMultiObjectiveSampler
from optuna.samplers import MOTPESampler


class MockSystemAttr:
    def __init__(self) -> None:
        self.value: Dict[str, dict] = {}

    def set_trial_system_attr(self, _: int, key: str, value: dict) -> None:
        self.value[key] = value


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_reseed_rng() -> None:
    sampler = MOTPEMultiObjectiveSampler()
    original_random_state = sampler._motpe_sampler._rng.rng.get_state()

    with patch.object(
        sampler._motpe_sampler, "reseed_rng", wraps=sampler._motpe_sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
    assert str(original_random_state) != str(sampler._motpe_sampler._rng.rng.get_state())


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sample_relative() -> None:
    sampler = MOTPEMultiObjectiveSampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.sample_relative(study, frozen_trial, {}) == {}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_infer_relative_search_space() -> None:
    sampler = MOTPEMultiObjectiveSampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.infer_relative_search_space(study, frozen_trial) == {}


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sample_independent() -> None:
    sampler = MOTPEMultiObjectiveSampler()
    study = optuna.multi_objective.create_study(
        directions=["minimize", "maximize"], sampler=sampler
    )

    def _objective(trial: multi_objective.trial.MultiObjectiveTrial) -> Tuple[float, float]:
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x, y

    with patch.object(
        MOTPESampler,
        "sample_independent",
        wraps=sampler._motpe_sampler.sample_independent,
    ) as mock:
        study.optimize(_objective, n_trials=10)
        assert mock.call_count == 20
