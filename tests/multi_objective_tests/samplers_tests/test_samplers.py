from typing import Callable

import numpy as np
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna import multi_objective
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler

parametrize_sampler = pytest.mark.parametrize(
    "sampler_class",
    [
        optuna.multi_objective.samplers.RandomMultiObjectiveSampler,
        optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler,
    ],
)


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        UniformDistribution(-1.0, 1.0),
        UniformDistribution(0.0, 1.0),
        UniformDistribution(-1.0, 0.0),
        LogUniformDistribution(1e-7, 1.0),
        DiscreteUniformDistribution(-10, 10, 0.1),
        DiscreteUniformDistribution(-10.2, 10.2, 0.1),
        IntUniformDistribution(-10, 10),
        IntUniformDistribution(0, 10),
        IntUniformDistribution(-10, 0),
        IntUniformDistribution(-10, 10, 2),
        IntUniformDistribution(0, 10, 2),
        IntUniformDistribution(-10, 0, 2),
        CategoricalDistribution((1, 2, 3)),
        CategoricalDistribution(("a", "b", "c")),
        CategoricalDistribution((1, "a")),
    ],
)
def test_sample_independent(
    sampler_class: Callable[[], BaseMultiObjectiveSampler], distribution: UniformDistribution
) -> None:
    study = optuna.multi_objective.study.create_study(
        ["minimize", "maximize"], sampler=sampler_class()
    )
    for i in range(100):
        value = study.sampler.sample_independent(
            study, _create_new_trial(study), "x", distribution
        )
        assert distribution._contains(distribution.to_internal_repr(value))

        if not isinstance(distribution, CategoricalDistribution):
            # Please see https://github.com/optuna/optuna/pull/393 why this assertion is needed.
            assert not isinstance(value, np.floating)

        if isinstance(distribution, DiscreteUniformDistribution):
            # Check the value is a multiple of `distribution.q` which is
            # the quantization interval of the distribution.
            value -= distribution.low
            value /= distribution.q
            round_value = np.round(value)
            np.testing.assert_almost_equal(round_value, value)


def _create_new_trial(
    study: multi_objective.study.MultiObjectiveStudy,
) -> multi_objective.trial.FrozenMultiObjectiveTrial:
    trial_id = study._study._storage.create_new_trial(study._study._study_id)
    trial = study._study._storage.get_trial(trial_id)
    return multi_objective.trial.FrozenMultiObjectiveTrial(study.n_objectives, trial)
