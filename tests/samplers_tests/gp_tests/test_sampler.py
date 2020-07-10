from unittest.mock import Mock
from unittest.mock import patch

import optuna
from optuna.samplers import GPSampler


def test_optimize():
    # type: (bool) -> None

    sampler = GPSampler(
        model_kwargs={
            "max_optimize_iters": 10,
            "hmc_burnin": 1,
            "hmc_n_samples": 1,
            "hmc_subsample_interval": 1,
            "hmc_iters": 1,
        },
        optimizer_kwargs={"maxiter": 10, "n_samples_for_anchor": 1, "n_anchor": 1},
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: t.suggest_uniform("x", 10, 20), n_trials=20)


def test_sample_relative() -> None:
    sampler = GPSampler()
    # Study and frozen-trial are not supposed to be accessed.
    study = Mock(spec=[])
    frozen_trial = Mock(spec=[])
    assert sampler.sample_relative(study, frozen_trial, {}) == {}


def test_reseed_rng() -> None:
    sampler = GPSampler()
    original_seed = sampler._rng.seed

    with patch.object(
        sampler._independent_sampler, "reseed_rng", wraps=sampler._independent_sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
        assert original_seed != sampler._rng.seed
