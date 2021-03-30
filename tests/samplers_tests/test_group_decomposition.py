from typing import Callable
from unittest.mock import patch

import pytest

from optuna import create_study
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import GroupDecompositionSampler
from optuna.samplers import TPESampler


@pytest.mark.parametrize(
    "base_sampler_class",
    [
        lambda: TPESampler(n_startup_trials=0),
        lambda: TPESampler(n_startup_trials=0, multivariate=True),
        lambda: CmaEsSampler(n_startup_trials=0),
    ],
)
def test_group_decomposition_sampler(base_sampler_class: Callable[[], BaseSampler]) -> None:
    base_sampler = base_sampler_class()
    sampler = GroupDecompositionSampler(base_sampler=base_sampler)
    study = create_study(sampler=sampler)

    with patch.object(base_sampler, "sample_relative", wraps=base_sampler.sample_relative) as mock:
        study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=2)
        assert mock.call_count == 1
    assert study.trials[-1].distributions == {"x": IntUniformDistribution(low=0, high=10)}

    with patch.object(base_sampler, "sample_relative", wraps=base_sampler.sample_relative) as mock:
        study.optimize(
            lambda t: t.suggest_int("y", 0, 10) + t.suggest_float("z", -3, 3), n_trials=1
        )
        assert mock.call_count == 1
    assert study.trials[-1].distributions == {
        "y": IntUniformDistribution(low=0, high=10),
        "z": UniformDistribution(low=-3, high=3),
    }

    with patch.object(base_sampler, "sample_relative", wraps=base_sampler.sample_relative) as mock:
        study.optimize(
            lambda t: t.suggest_int("y", 0, 10)
            + t.suggest_float("z", -3, 3)
            + t.suggest_float("u", 1e-2, 1e2, log=True)
            + bool(t.suggest_categorical("v", ["A", "B", "C"])),
            n_trials=1,
        )
        assert mock.call_count == 2
    assert study.trials[-1].distributions == {
        "u": LogUniformDistribution(low=1e-2, high=1e2),
        "v": CategoricalDistribution(choices=["A", "B", "C"]),
        "y": IntUniformDistribution(low=0, high=10),
        "z": UniformDistribution(low=-3, high=3),
    }

    with patch.object(base_sampler, "sample_relative", wraps=base_sampler.sample_relative) as mock:
        study.optimize(lambda t: t.suggest_float("u", 1e-2, 1e2, log=True), n_trials=1)
        assert mock.call_count == 3
    assert study.trials[-1].distributions == {"u": LogUniformDistribution(low=1e-2, high=1e2)}

    with patch.object(base_sampler, "sample_relative", wraps=base_sampler.sample_relative) as mock:
        study.optimize(
            lambda t: t.suggest_int("y", 0, 10) + t.suggest_int("w", 2, 8, log=True), n_trials=1
        )
        assert mock.call_count == 4
    assert study.trials[-1].distributions == {
        "y": IntUniformDistribution(low=0, high=10),
        "w": IntLogUniformDistribution(low=2, high=8),
    }

    with patch.object(base_sampler, "sample_relative", wraps=base_sampler.sample_relative) as mock:
        study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=1)
        assert mock.call_count == 6
    assert study.trials[-1].distributions == {"x": IntUniformDistribution(low=0, high=10)}
