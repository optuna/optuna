from typing import Callable
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna import multi_objective
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

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
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(0.0, 1.0),
        FloatDistribution(-1.0, 0.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.1),
        FloatDistribution(-10.2, 10.2, step=0.1),
        IntDistribution(-10, 10),
        IntDistribution(0, 10),
        IntDistribution(-10, 0),
        IntDistribution(-10, 10, step=2),
        IntDistribution(0, 10, step=2),
        IntDistribution(-10, 0, step=2),
        CategoricalDistribution((1, 2, 3)),
        CategoricalDistribution(("a", "b", "c")),
        CategoricalDistribution((1, "a")),
    ],
)
def test_sample_independent(
    sampler_class: Callable[[], BaseMultiObjectiveSampler], distribution: BaseDistribution
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

        if isinstance(distribution, FloatDistribution):
            if distribution.step is not None:
                # Check the value is a multiple of `distribution.q` which is
                # the quantization interval of the distribution.
                value -= distribution.low
                value /= distribution.step
                round_value = np.round(value)
                np.testing.assert_almost_equal(round_value, value)


def test_random_mo_sampler_reseed_rng() -> None:
    sampler = optuna.multi_objective.samplers.RandomMultiObjectiveSampler()
    original_random_state = sampler._sampler._rng.rng.get_state()

    with patch.object(
        sampler._sampler, "reseed_rng", wraps=sampler._sampler.reseed_rng
    ) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
    assert str(original_random_state) != str(sampler._sampler._rng.rng.get_state())


@pytest.mark.parametrize(
    "sampler_class",
    [
        optuna.multi_objective.samplers.RandomMultiObjectiveSampler,
        optuna.multi_objective.samplers.NSGAIIMultiObjectiveSampler,
        optuna.multi_objective.samplers.MOTPEMultiObjectiveSampler,
    ],
)
def test_deprecated_warning(sampler_class: Callable[[], BaseMultiObjectiveSampler]) -> None:
    with pytest.warns(FutureWarning):
        sampler_class()


def _create_new_trial(
    study: multi_objective.study.MultiObjectiveStudy,
) -> multi_objective.trial.FrozenMultiObjectiveTrial:
    trial_id = study._study._storage.create_new_trial(study._study._study_id)
    trial = study._study._storage.get_trial(trial_id)
    return multi_objective.trial.FrozenMultiObjectiveTrial(study.n_objectives, trial)
