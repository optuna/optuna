from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.trial import Trial
from optuna.trial import TrialState


_SEARCH_SPACE = {
    "x1": optuna.distributions.IntDistribution(0, 10),
    "x2": optuna.distributions.IntDistribution(1, 10, log=True),
    "x3": optuna.distributions.FloatDistribution(0, 10),
    "x4": optuna.distributions.FloatDistribution(1, 10, log=True),
    "x5": optuna.distributions.FloatDistribution(1, 10, step=3),
    "x6": optuna.distributions.CategoricalDistribution([1, 4, 7, 10]),
}


# TODO(kstoneriv3): `QMCSampler` can be initialized without this wrapper
# Remove this after the experimental warning is removed.
def _init_QMCSampler_without_exp_warning(**kwargs: Any) -> optuna.samplers.QMCSampler:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.QMCSampler(**kwargs)
    return sampler


def test_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.QMCSampler()


@pytest.mark.parametrize("qmc_type", ["sobol", "halton", "non-qmc"])
def test_invalid_qmc_type(qmc_type: str) -> None:
    if qmc_type == "non-qmc":
        with pytest.raises(ValueError):
            _init_QMCSampler_without_exp_warning(qmc_type=qmc_type)
    else:
        _init_QMCSampler_without_exp_warning(qmc_type=qmc_type)


def test_initial_seeding() -> None:
    with patch.object(optuna.samplers.QMCSampler, "_log_asynchronous_seeding") as mock_log_async:
        sampler = _init_QMCSampler_without_exp_warning(scramble=True)
    mock_log_async.assert_called_once()
    assert isinstance(sampler._seed, int)


def test_infer_relative_search_space() -> None:
    def objective(trial: Trial) -> float:
        ret: float = trial.suggest_int("x1", 0, 10)
        ret += trial.suggest_int("x2", 1, 10, log=True)
        ret += trial.suggest_float("x3", 0, 10)
        ret += trial.suggest_float("x4", 1, 10, log=True)
        ret += trial.suggest_float("x5", 1, 10, step=3)
        _ = trial.suggest_categorical("x6", [1, 4, 7, 10])
        return ret

    sampler = _init_QMCSampler_without_exp_warning()
    study = optuna.create_study(sampler=sampler)
    trial = Mock()
    # In case no past trials.
    assert sampler.infer_relative_search_space(study, trial) == {}
    # In case there is a past trial.
    study.optimize(objective, n_trials=1)
    relative_search_space = sampler.infer_relative_search_space(study, trial)
    assert len(relative_search_space.keys()) == 5
    assert set(relative_search_space.keys()) == {"x1", "x2", "x3", "x4", "x5"}
    # In case self._initial_trial already exists.
    new_search_space: dict[str, BaseDistribution] = {"x": Mock()}
    sampler._initial_search_space = new_search_space
    assert sampler.infer_relative_search_space(study, trial) == new_search_space


def test_infer_initial_search_space() -> None:
    trial = Mock()
    sampler = _init_QMCSampler_without_exp_warning()
    # Can it handle empty search space?
    trial.distributions = {}
    initial_search_space = sampler._infer_initial_search_space(trial)
    assert initial_search_space == {}
    # Does it exclude only categorical distribution?
    search_space = _SEARCH_SPACE.copy()
    trial.distributions = search_space
    initial_search_space = sampler._infer_initial_search_space(trial)
    search_space.pop("x6")
    assert initial_search_space == search_space


def test_sample_independent() -> None:
    objective: Callable[[Trial], float] = lambda t: t.suggest_categorical("x", [1.0, 2.0])
    independent_sampler = optuna.samplers.RandomSampler()

    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_sample_indep:
        sampler = _init_QMCSampler_without_exp_warning(independent_sampler=independent_sampler)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=1)
        assert mock_sample_indep.call_count == 1

        # Relative sampling of `QMCSampler` does not support categorical distribution.
        # Thus, `independent_sampler.sample_independent` is called twice.
        study.optimize(objective, n_trials=1)
        assert mock_sample_indep.call_count == 2

        # Unseen parameter is sampled by independent sampler.
        new_objective: Callable[[Trial], int] = lambda t: t.suggest_int("y", 0, 10)
        study.optimize(new_objective, n_trials=1)
        assert mock_sample_indep.call_count == 3


def test_warn_asynchronous_seeding() -> None:
    # Relative sampling of `QMCSampler` does not support categorical distribution.
    # Thus, `independent_sampler.sample_independent` is called twice.
    # '_log_independent_sampling is not called in the first trial so called once in total.
    objective: Callable[[Trial], float] = lambda t: t.suggest_categorical("x", [1.0, 2.0])

    with patch.object(optuna.samplers.QMCSampler, "_log_asynchronous_seeding") as mock_log_async:
        sampler = _init_QMCSampler_without_exp_warning(
            scramble=True, warn_asynchronous_seeding=False
        )
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=2)

        assert mock_log_async.call_count == 0

        sampler = _init_QMCSampler_without_exp_warning(scramble=True)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=2)

        assert mock_log_async.call_count == 1


def test_warn_independent_sampling() -> None:
    # Relative sampling of `QMCSampler` does not support categorical distribution.
    # Thus, `independent_sampler.sample_independent` is called twice.
    # '_log_independent_sampling is not called in the first trial so called once in total.
    objective: Callable[[Trial], float] = lambda t: t.suggest_categorical("x", [1.0, 2.0])

    with patch.object(optuna.samplers.QMCSampler, "_log_independent_sampling") as mock_log_indep:
        sampler = _init_QMCSampler_without_exp_warning(warn_independent_sampling=False)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=2)

        assert mock_log_indep.call_count == 0

        sampler = _init_QMCSampler_without_exp_warning()
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=2)

        assert mock_log_indep.call_count == 1


def test_sample_relative() -> None:
    search_space = _SEARCH_SPACE.copy()
    search_space.pop("x6")
    sampler = _init_QMCSampler_without_exp_warning()
    study = optuna.create_study(sampler=sampler)
    trial = Mock()
    # Make sure that sample type, shape is OK.
    for _ in range(3):
        sample = sampler.sample_relative(study, trial, search_space)
        assert 0 <= sample["x1"] <= 10
        assert 1 <= sample["x2"] <= 10
        assert 0 <= sample["x3"] <= 10
        assert 1 <= sample["x4"] <= 10
        assert 1 <= sample["x5"] <= 10

        assert isinstance(sample["x1"], int)
        assert isinstance(sample["x2"], int)
        assert sample["x5"] in (1, 4, 7, 10)

    # If empty search_space, return {}.
    assert sampler.sample_relative(study, trial, {}) == {}


def test_sample_relative_halton() -> None:
    n, d = 8, 5
    search_space: dict[str, BaseDistribution] = {
        f"x{i}": optuna.distributions.FloatDistribution(0, 1) for i in range(d)
    }
    sampler = _init_QMCSampler_without_exp_warning(scramble=False, qmc_type="halton")
    study = optuna.create_study(sampler=sampler)
    trial = Mock()
    # Make sure that sample type, shape is OK.
    samples = np.zeros((n, d))
    for i in range(n):
        sample = sampler.sample_relative(study, trial, search_space)
        for j in range(d):
            samples[i, j] = sample[f"x{j}"]
    ref_samples = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.33333333, 0.2, 0.14285714, 0.09090909],
            [0.25, 0.66666667, 0.4, 0.28571429, 0.18181818],
            [0.75, 0.11111111, 0.6, 0.42857143, 0.27272727],
            [0.125, 0.44444444, 0.8, 0.57142857, 0.36363636],
            [0.625, 0.77777778, 0.04, 0.71428571, 0.45454545],
            [0.375, 0.22222222, 0.24, 0.85714286, 0.54545455],
            [0.875, 0.55555556, 0.44, 0.02040816, 0.63636364],
        ]
    )
    # If empty search_space, return {}.
    np.testing.assert_allclose(samples, ref_samples, rtol=1e-6)


def test_sample_relative_sobol() -> None:
    n, d = 8, 5
    search_space: dict[str, BaseDistribution] = {
        f"x{i}": optuna.distributions.FloatDistribution(0, 1) for i in range(d)
    }
    sampler = _init_QMCSampler_without_exp_warning(scramble=False, qmc_type="sobol")
    study = optuna.create_study(sampler=sampler)
    trial = Mock()
    # Make sure that sample type, shape is OK.
    samples = np.zeros((n, d))
    for i in range(n):
        sample = sampler.sample_relative(study, trial, search_space)
        for j in range(d):
            samples[i, j] = sample[f"x{j}"]
    ref_samples = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.75, 0.25],
            [0.375, 0.375, 0.625, 0.875, 0.375],
            [0.875, 0.875, 0.125, 0.375, 0.875],
            [0.625, 0.125, 0.875, 0.625, 0.625],
            [0.125, 0.625, 0.375, 0.125, 0.125],
        ]
    )

    # If empty search_space, return {}.
    np.testing.assert_allclose(samples, ref_samples, rtol=1e-6)


@pytest.mark.parametrize("scramble", [True, False])
@pytest.mark.parametrize("qmc_type", ["sobol", "halton"])
@pytest.mark.parametrize("seed", [0, 12345])
def test_sample_relative_seeding(scramble: bool, qmc_type: str, seed: int) -> None:
    objective: Callable[[Trial], float] = lambda t: t.suggest_float("x", 0, 1)

    # Base case.
    sampler = _init_QMCSampler_without_exp_warning(scramble=scramble, qmc_type=qmc_type, seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10, n_jobs=1)
    past_trials = study._storage.get_all_trials(study._study_id, states=(TrialState.COMPLETE,))
    past_trials = [t for t in past_trials if t.number > 0]
    values = [t.params["x"] for t in past_trials]

    # Sequential case.
    sampler = _init_QMCSampler_without_exp_warning(scramble=scramble, qmc_type=qmc_type, seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10, n_jobs=1)
    past_trials_sequential = study._storage.get_all_trials(
        study._study_id, states=(TrialState.COMPLETE,)
    )
    past_trials_sequential = [t for t in past_trials_sequential if t.number > 0]
    values_sequential = [t.params["x"] for t in past_trials_sequential]
    np.testing.assert_allclose(values, values_sequential, rtol=1e-6)

    # Parallel case (n_jobs=3):
    # Same parameters might be evaluated multiple times.
    sampler = _init_QMCSampler_without_exp_warning(scramble=scramble, qmc_type=qmc_type, seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=30, n_jobs=3)
    past_trials_parallel = study._storage.get_all_trials(
        study._study_id, states=(TrialState.COMPLETE,)
    )
    past_trials_parallel = [t for t in past_trials_parallel if t.number > 0]
    values_parallel = [t.params["x"] for t in past_trials_parallel]
    for v in values:
        assert np.any(
            np.isclose(v, values_parallel, rtol=1e-6)
        ), f"v: {v} of values: {values} is not included in values_parallel: {values_parallel}."


def test_call_after_trial() -> None:
    sampler = _init_QMCSampler_without_exp_warning()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._independent_sampler, "after_trial", wraps=sampler._independent_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


@pytest.mark.parametrize("qmc_type", ["sobol", "halton"])
def test_sample_qmc(qmc_type: str) -> None:
    sampler = _init_QMCSampler_without_exp_warning(qmc_type=qmc_type)
    study = Mock()
    search_space = _SEARCH_SPACE.copy()
    search_space.pop("x6")

    with patch.object(sampler, "_find_sample_id", side_effect=[0, 1, 2, 4, 9]) as _:
        # Make sure that the shape of sample is correct.
        sample = sampler._sample_qmc(study, search_space)
        assert sample.shape == (1, 5)


def test_find_sample_id() -> None:
    sampler = _init_QMCSampler_without_exp_warning(qmc_type="halton", seed=0)
    study = optuna.create_study()
    for i in range(5):
        assert sampler._find_sample_id(study) == i

    # Change seed but without scramble. The hash should remain the same.
    with patch.object(sampler, "_seed", 1) as _:
        assert sampler._find_sample_id(study) == 5

        # Seed is considered only when scrambling is enabled.
        with patch.object(sampler, "_scramble", True) as _:
            assert sampler._find_sample_id(study) == 0

    # Change qmc_type.
    with patch.object(sampler, "_qmc_type", "sobol") as _:
        assert sampler._find_sample_id(study) == 0
