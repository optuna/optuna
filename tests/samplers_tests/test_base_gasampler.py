from __future__ import annotations

from typing import Any
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers._ga._base import BaseGASampler
from optuna.samplers._random import RandomSampler
from optuna.study.study import Study
from optuna.trial import FrozenTrial
from optuna.trial._state import TrialState


class BaseGASamplerTestSampler(BaseGASampler):
    def __init__(self, population_size: int):
        super().__init__(population_size=population_size)
        self._random_sampler = RandomSampler()

    def select_parent(self, study: optuna.Study, generation: int) -> list[FrozenTrial]:
        raise NotImplementedError

    def sample_relative(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        raise NotImplementedError

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        raise NotImplementedError

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        raise NotImplementedError


def test_systemattr_keys() -> None:
    assert BaseGASamplerTestSampler._get_generation_key() == "BaseGASamplerTestSampler:generation"
    assert (
        BaseGASamplerTestSampler._get_parent_cache_key_prefix()
        == "BaseGASamplerTestSampler:parent:"
    )

    test_sampler = BaseGASamplerTestSampler(population_size=42)

    assert test_sampler.population_size == 42


@pytest.mark.parametrize(
    "args",
    [
        {
            "population_size": 3,
            "trials": [],
            "generation": 0,
        },
        {
            "population_size": 4,
            "trials": [0, 0, 0, 0],
            "generation": 1,
        },
        {
            "population_size": 3,
            "trials": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
            "generation": 3,
        },
        {
            "population_size": 3,
            "trials": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2],
            "generation": 3,
        },
    ],
)
def test_get_generation(args: dict[str, Any]) -> None:
    test_sampler = BaseGASamplerTestSampler(population_size=args["population_size"])
    mock_study = Mock(
        _get_trials=Mock(
            return_value=[
                Mock(system_attrs={"BaseGASamplerTestSampler:generation": i})
                for i in args["trials"]
            ]
        )
    )
    mock_trial = Mock(system_attrs={})

    assert test_sampler.get_trial_generation(mock_study, mock_trial) == args["generation"]

    mock_study._get_trials.assert_called_once_with(
        deepcopy=False, states=[TrialState.COMPLETE], use_cache=True
    )
    mock_study._storage.set_trial_system_attr.assert_called_once_with(
        mock_trial._trial_id,
        "BaseGASamplerTestSampler:generation",
        args["generation"],
    )
    assert len(mock_study.mock_calls) == 2  # Check if only the two calls above were made
    assert len(mock_trial.mock_calls) == 0


def test_get_generation_already_set() -> None:
    test_sampler = BaseGASamplerTestSampler(population_size=42)

    mock_study = MagicMock()
    mock_trial = MagicMock()
    generation_key = MagicMock()

    mock_trial.system_attrs = {test_sampler._get_generation_key(): generation_key}

    assert test_sampler.get_trial_generation(mock_study, mock_trial) == generation_key


@pytest.mark.parametrize(
    "args",
    [
        {
            "population_size": 3,
            "trials": [],
            "generation": 0,
            "length": 0,
        },
        {
            "population_size": 4,
            "trials": [0, 0, 0, 0],
            "generation": 0,
            "length": 4,
        },
        {
            "population_size": 3,
            "trials": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
            "generation": 0,
            "length": 4,
        },
        {
            "population_size": 3,
            "trials": [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
            "generation": 1,
            "length": 5,
        },
    ],
)
def test_get_population(args: dict[str, Any]) -> None:
    test_sampler = BaseGASamplerTestSampler(population_size=args["population_size"])
    mock_study = Mock(
        _get_trials=Mock(
            return_value=[
                Mock(system_attrs={"BaseGASamplerTestSampler:generation": i})
                for i in args["trials"]
            ]
        )
    )
    mock_trial = Mock(system_attrs={})

    population = test_sampler.get_population(mock_study, args["generation"])

    assert all(
        [
            trial.system_attrs["BaseGASamplerTestSampler:generation"] == args["generation"]
            for trial in population
        ]
    )
    assert len(population) == args["length"]
    assert mock_study._get_trials.call_count == 1
    assert mock_trial.mock_calls == []


@pytest.mark.parametrize(
    "args",
    [
        {
            "study_system_attrs": {
                BaseGASamplerTestSampler._get_parent_cache_key_prefix() + "1": [0, 1, 2]
            },
            "parent_population": [0, 1, 2],
            "generation": 1,
            "cache": True,
        },
        {
            "study_system_attrs": {
                BaseGASamplerTestSampler._get_parent_cache_key_prefix() + "1": [0, 1, 2]
            },
            "parent_population": [3, 4, 6],
            "generation": 2,
            "cache": False,
        },
        {
            "study_system_attrs": {},
            "parent_population": [0, 1, 2],
            "generation": 1,
            "cache": False,
        },
        {
            "study_system_attrs": {},
            "parent_population": [0, 1, 2],
            "generation": 0,
            "cache": None,
        },
    ],
)
def test_get_parent_population(args: dict[str, Any]) -> None:
    test_sampler = BaseGASamplerTestSampler(population_size=3)

    mock_study = MagicMock()
    mock_study._storage.get_study_system_attrs.return_value = args["study_system_attrs"]

    with patch.object(
        BaseGASamplerTestSampler,
        "select_parent",
        return_value=[Mock(_trial_id=i) for i in args["parent_population"]],
    ) as mock_select_parent:
        return_value = test_sampler.get_parent_population(mock_study, args["generation"])

    if args["generation"] == 0:
        assert mock_select_parent.call_count == 0
        assert mock_study._storage.get_study_system_attrs.call_count == 0
        assert mock_study._get_trials.call_count == 0
        assert return_value == []
        return

    mock_study._storage.get_study_system_attrs.assert_called_once_with(mock_study._study_id)

    if args["cache"]:
        mock_study._get_trials.assert_has_calls(
            [call(deepcopy=False)] + [call().__getitem__(i) for i in args["parent_population"]]
        )
    else:
        mock_select_parent.assert_called_once_with(mock_study, args["generation"])
        mock_study._storage.set_study_system_attr.assert_called_once_with(
            mock_study._study_id,
            BaseGASamplerTestSampler._get_parent_cache_key_prefix() + str(args["generation"]),
            [i._trial_id for i in mock_select_parent.return_value],
        )
