import random
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from unittest.mock import Mock
from unittest.mock import patch
from unittest.mock import PropertyMock

import pytest

import optuna
from optuna import multi_objective
from optuna.multi_objective.samplers import MOTPEMultiObjectiveSampler


class MockSystemAttr:
    def __init__(self) -> None:
        self.value = {}  # type: Dict[str, dict]

    def set_trial_system_attr(self, _: int, key: str, value: dict) -> None:
        self.value[key] = value


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_reseed_rng() -> None:
    sampler = MOTPEMultiObjectiveSampler()
    original_seed = sampler._motpe_sampler._rng.seed

    with patch.object(
        sampler._motpe_sampler, "reseed_rng", wraps=sampler._motpe_sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
        assert original_seed != sampler._motpe_sampler._rng.seed


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
    study = optuna.multi_objective.create_study(directions=["minimize", "maximize"])
    dist = optuna.distributions.UniformDistribution(1.0, 100.0)

    random.seed(128)
    past_trials = [frozen_trial_factory(i, [random.random(), random.random()]) for i in range(16)]
    # Prepare a trial and a sample for later checks.
    trial = multi_objective.trial.FrozenMultiObjectiveTrial(2, frozen_trial_factory(16, [0, 0]))
    sampler = MOTPEMultiObjectiveSampler(seed=0)
    attrs = MockSystemAttr()
    with patch.object(study._storage, "get_all_trials", return_value=past_trials), patch.object(
        study._storage, "set_trial_system_attr", side_effect=attrs.set_trial_system_attr
    ), patch.object(study._storage, "get_trial", return_value=trial), patch(
        "optuna.multi_objective.trial.MultiObjectiveTrial.system_attrs", new_callable=PropertyMock
    ) as mock1, patch(
        "optuna.multi_objective.trial.FrozenMultiObjectiveTrial.system_attrs",
        new_callable=PropertyMock,
    ) as mock2:
        mock1.return_value = attrs.value
        mock2.return_value = attrs.value
        _ = sampler.sample_independent(study, trial, "param-a", dist)


def frozen_trial_factory(
    number: int,
    values: List[float],
    dist: optuna.distributions.BaseDistribution = optuna.distributions.UniformDistribution(
        1.0, 100.0
    ),
    value_fn: Optional[Callable[[int], Union[int, float]]] = None,
) -> multi_objective.trial.FrozenTrial:
    if value_fn is None:
        value = random.random() * 99.0 + 1.0
    else:
        value = value_fn(number)

    trial = optuna.trial.FrozenTrial(
        number=number,
        trial_id=number,
        state=optuna.trial.TrialState.COMPLETE,
        value=None,
        datetime_start=None,
        datetime_complete=None,
        params={"param-a": value},
        distributions={"param-a": dist},
        user_attrs={},
        system_attrs={},
        intermediate_values=dict(enumerate(values)),
    )
    return trial
