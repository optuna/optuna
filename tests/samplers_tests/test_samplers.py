from __future__ import annotations

import multiprocessing
from multiprocessing.managers import DictProxy
import os
import pickle
from typing import Any
from typing import Callable
from unittest.mock import patch

from _pytest.fixtures import SubRequest
import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.testing.pytest_samplers import BasicSamplerTestCase
from optuna.testing.pytest_samplers import FixedSampler
from optuna.testing.pytest_samplers import MultiObjectiveSamplerTestCase
from optuna.testing.pytest_samplers import RelativeSamplerTestCase
from optuna.testing.pytest_samplers import SingleOnlySamplerTestCase
from optuna.trial import Trial


def get_gp_sampler(
    *, n_startup_trials: int = 0, deterministic_objective: bool = False, seed: int | None = None
) -> optuna.samplers.GPSampler:
    return optuna.samplers.GPSampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
        deterministic_objective=deterministic_objective,
    )


sampler_class_with_seed: dict[str, Callable[[int], BaseSampler]] = {
    "RandomSampler": lambda seed: optuna.samplers.RandomSampler(seed=seed),
    "TPESampler": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "multivariate TPESampler": lambda seed: optuna.samplers.TPESampler(
        multivariate=True, seed=seed
    ),
    "CmaEsSampler": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
    "separable CmaEsSampler": lambda seed: optuna.samplers.CmaEsSampler(
        seed=seed, use_separable_cma=True
    ),
    "NSGAIISampler": lambda seed: optuna.samplers.NSGAIISampler(seed=seed),
    "NSGAIIISampler": lambda seed: optuna.samplers.NSGAIIISampler(seed=seed),
    "QMCSampler": lambda seed: optuna.samplers.QMCSampler(seed=seed),
    "GPSampler": lambda seed: get_gp_sampler(seed=seed, n_startup_trials=0),
}
param_sampler_with_seed = []
param_sampler_name_with_seed = []
for sampler_name, sampler_class in sampler_class_with_seed.items():
    param_sampler_with_seed.append(pytest.param(sampler_class, id=sampler_name))
    param_sampler_name_with_seed.append(pytest.param(sampler_name))
parametrize_sampler_with_seed = pytest.mark.parametrize(
    "sampler_class_with_seed", param_sampler_with_seed
)
parametrize_sampler_name_with_seed = pytest.mark.parametrize(
    "sampler_name", param_sampler_name_with_seed
)


# This function is used only in test_reproducible_in_other_process, but declared at top-level
# because local function cannot be pickled, which occurs within multiprocessing.
def run_optimize(
    k: int,
    sampler_name: str,
    sequence_dict: DictProxy,
    hash_dict: DictProxy,
) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return a + b + c + d + e + f + g

    hash_dict[k] = hash("nondeterministic hash")
    sampler = sampler_class_with_seed[sampler_name](1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=15)
    sequence_dict[k] = list(study.trials[-1].params.values())


class TestBasicSampler(BasicSamplerTestCase):
    @pytest.fixture(
        params=[
            optuna.samplers.RandomSampler,
            lambda: optuna.samplers.TPESampler(n_startup_trials=0),
            lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True),
            lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
            lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0, use_separable_cma=True),
            optuna.samplers.NSGAIISampler,
            optuna.samplers.NSGAIIISampler,
            optuna.samplers.QMCSampler,
            lambda: get_gp_sampler(n_startup_trials=0),
            lambda: get_gp_sampler(n_startup_trials=0, deterministic_objective=True),
        ]
    )
    def sampler(self, request: SubRequest) -> Callable[[], BaseSampler]:
        return request.param


class TestRelativeSampler(RelativeSamplerTestCase):
    @pytest.fixture(
        params=[
            lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True),
            lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
            lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0, use_separable_cma=True),
            lambda: get_gp_sampler(n_startup_trials=0),
            lambda: get_gp_sampler(n_startup_trials=0, deterministic_objective=True),
        ]
    )
    def sampler(self, request: SubRequest) -> Callable[[], BaseSampler]:
        return request.param


class TestMultiObjectiveSampler(MultiObjectiveSamplerTestCase):
    @pytest.fixture(
        params=[
            optuna.samplers.NSGAIISampler,
            optuna.samplers.NSGAIIISampler,
            lambda: optuna.samplers.TPESampler(n_startup_trials=0),
            lambda: get_gp_sampler(deterministic_objective=True),
            lambda: get_gp_sampler(deterministic_objective=False),
        ]
    )
    def sampler(
        self,
        request: SubRequest,
    ) -> Callable[[], BaseSampler]:
        return request.param


class TestSingleOnlySampler(SingleOnlySamplerTestCase):
    @pytest.fixture(
        params=[
            lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
        ]
    )
    def sampler(
        self,
        request: SubRequest,
    ) -> Callable[[], BaseSampler]:
        return request.param


@pytest.fixture
def unset_seed_in_test(request: SubRequest) -> None:
    # Unset the hashseed at beginning and restore it at end regardless of an exception
    # in the test.
    # See https://docs.pytest.org/en/stable/how-to/fixtures.html#adding-finalizers-directly
    # for details.

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is not None:
        del os.environ["PYTHONHASHSEED"]

    def restore_seed() -> None:
        if hash_seed is not None:
            os.environ["PYTHONHASHSEED"] = hash_seed

    request.addfinalizer(restore_seed)


@pytest.mark.parametrize(
    "sampler_class_reseed_rng,expected_has_rng,expected_has_another_sampler",
    [
        (optuna.samplers.RandomSampler, True, False),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=0), True, True),
        (
            lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True),
            True,
            True,
        ),
        (lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0), True, True),
        (optuna.samplers.NSGAIISampler, True, True),
        (optuna.samplers.NSGAIIISampler, True, True),
        (
            lambda: optuna.samplers.PartialFixedSampler(
                fixed_params={"x": 0}, base_sampler=optuna.samplers.RandomSampler()
            ),
            False,
            True,
        ),
        (lambda: optuna.samplers.GridSampler(search_space={"x": [0]}), True, False),
        (lambda: optuna.samplers.QMCSampler(), False, True),
        (lambda: get_gp_sampler(n_startup_trials=0), True, True),
    ],
)
def test_sampler_reseed_rng(
    sampler_class_reseed_rng: Callable[[], BaseSampler],
    expected_has_rng: bool,
    expected_has_another_sampler: bool,
) -> None:
    def _extract_attr_name_from_sampler_by_cls(sampler: BaseSampler, cls: Any) -> str | None:
        for name, attr in sampler.__dict__.items():
            if isinstance(attr, cls):
                return name
        return None

    sampler = sampler_class_reseed_rng()

    rng_name = _extract_attr_name_from_sampler_by_cls(sampler, LazyRandomState)
    has_rng = rng_name is not None
    assert expected_has_rng == has_rng
    if has_rng:
        rng_name = str(rng_name)
        original_random_state = sampler.__dict__[rng_name].rng.get_state()
        sampler.reseed_rng()
        random_state = sampler.__dict__[rng_name].rng.get_state()
        if not isinstance(sampler, optuna.samplers.CmaEsSampler):
            assert str(original_random_state) != str(random_state)
        else:
            # CmaEsSampler has a RandomState that is not reseed by its reseed_rng method.
            assert str(original_random_state) == str(random_state)

    had_sampler_name = _extract_attr_name_from_sampler_by_cls(sampler, BaseSampler)
    has_another_sampler = had_sampler_name is not None
    assert expected_has_another_sampler == has_another_sampler

    if has_another_sampler:
        had_sampler_name = str(had_sampler_name)
        had_sampler = sampler.__dict__[had_sampler_name]
        had_sampler_rng_name = _extract_attr_name_from_sampler_by_cls(had_sampler, LazyRandomState)
        original_had_sampler_random_state = had_sampler.__dict__[
            had_sampler_rng_name
        ].rng.get_state()
        with patch.object(
            had_sampler,
            "reseed_rng",
            wraps=had_sampler.reseed_rng,
        ) as mock_object:
            sampler.reseed_rng()
            assert mock_object.call_count == 1

        had_sampler = sampler.__dict__[had_sampler_name]
        had_sampler_random_state = had_sampler.__dict__[had_sampler_rng_name].rng.get_state()
        assert str(original_had_sampler_random_state) != str(had_sampler_random_state)


@pytest.mark.slow
@parametrize_sampler_with_seed
def test_reseed_rng_change_sampling(sampler_class_with_seed: Callable[[int], BaseSampler]) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return a + b + c + d + e + f + g

    sampler = sampler_class_with_seed(1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=15)

    sampler_different_seed = sampler_class_with_seed(1)
    sampler_different_seed.reseed_rng()
    study_different_seed = optuna.create_study(sampler=sampler_different_seed)
    study_different_seed.optimize(objective, n_trials=15)
    assert any(
        [study.trials[i].params != study_different_seed.trials[i].params for i in range(15)]
    )


# We add tests for constant objective functions to ensure the reproducibility of sorting.
@parametrize_sampler_with_seed
@pytest.mark.slow
@pytest.mark.parametrize("objective_func", [lambda *args: sum(args), lambda *args: 0.0])
def test_reproducible(
    sampler_class_with_seed: Callable[[int], BaseSampler], objective_func: Any
) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return objective_func(a, b, c, d, e, f, g)

    study = optuna.create_study(sampler=sampler_class_with_seed(1))
    study.optimize(objective, n_trials=15)

    study_same_seed = optuna.create_study(sampler=sampler_class_with_seed(1))
    study_same_seed.optimize(objective, n_trials=15)
    for i in range(15):
        assert study.trials[i].params == study_same_seed.trials[i].params

    study_different_seed = optuna.create_study(sampler=sampler_class_with_seed(2))
    study_different_seed.optimize(objective, n_trials=15)
    assert any(
        [study.trials[i].params != study_different_seed.trials[i].params for i in range(15)]
    )


@pytest.mark.slow
@parametrize_sampler_name_with_seed
def test_reproducible_in_other_process(sampler_name: str, unset_seed_in_test: None) -> None:
    # This test should be tested without `PYTHONHASHSEED`. However, some tool such as tox
    # set the environmental variable "PYTHONHASHSEED" by default.
    # To do so, this test calls a finalizer: `unset_seed_in_test`.

    # Multiprocessing supports three way to start a process.
    # We use `spawn` option to create a child process as a fresh python process.
    # For more detail, see https://github.com/optuna/optuna/pull/3187#issuecomment-997673037.
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    sequence_dict: DictProxy = manager.dict()
    hash_dict: DictProxy = manager.dict()
    for i in range(3):
        p = multiprocessing.Process(
            target=run_optimize, args=(i, sampler_name, sequence_dict, hash_dict)
        )
        p.start()
        p.join()

    # Hashes are expected to be different because string hashing is nondeterministic
    # per process.
    assert not (hash_dict[0] == hash_dict[1] == hash_dict[2])
    # But the sequences are expected to be the same.
    assert sequence_dict[0] == sequence_dict[1] == sequence_dict[2]


@pytest.mark.parametrize("seed", [0, 169208])
def test_pickle_random_sampler(seed: int) -> None:
    sampler = optuna.samplers.RandomSampler(seed)
    restored_sampler = pickle.loads(pickle.dumps(sampler))
    assert sampler._rng.rng.bytes(10) == restored_sampler._rng.rng.bytes(10)


def test_sample_relative() -> None:
    relative_search_space: dict[str, BaseDistribution] = {
        "a": FloatDistribution(low=0, high=5),
        "b": CategoricalDistribution(choices=("foo", "bar", "baz")),
        "c": IntDistribution(low=20, high=50),  # Not exist in `relative_params`.
    }
    relative_params = {
        "a": 3.2,
        "b": "baz",
    }
    unknown_param_value = 30

    sampler = FixedSampler(relative_search_space, relative_params, unknown_param_value)
    study = optuna.study.create_study(sampler=sampler)

    def objective(trial: Trial) -> float:
        # Predefined parameters are sampled by `sample_relative()` method.
        assert trial.suggest_float("a", 0, 5) == 3.2
        assert trial.suggest_categorical("b", ["foo", "bar", "baz"]) == "baz"

        # Other parameters are sampled by `sample_independent()` method.
        assert trial.suggest_int("c", 20, 50) == unknown_param_value
        assert trial.suggest_float("d", 1, 100, log=True) == unknown_param_value
        assert trial.suggest_float("e", 20, 40) == unknown_param_value

        return 0.0

    study.optimize(objective, n_trials=10, catch=())
    for trial in study.trials:
        assert trial.params == {"a": 3.2, "b": "baz", "c": 30, "d": 30, "e": 30}
