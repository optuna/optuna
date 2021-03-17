from typing import Any
from typing import Dict
from typing import List
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
from optuna import create_trial
from optuna._transform import _SearchSpaceTransform
from optuna.samplers._cmaes import _concat_optimizer_attrs
from optuna.samplers._cmaes import _split_optimizer_str
from optuna.testing.sampler import DeterministicRelativeSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def test_qmc_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.QMCSampler()


def test_initial_seeding() -> None:
    with patch.object(optuna.samplers.QMCSampler, "_log_asyncronous_seeding") as mock_log_async:
        sampler = _init_QMCSampler_without_warnings(scramble=True)
    mock_log_async.assert_called_once()
    assert isinstance(sampler._seed, int)


def test_reseed_rng() -> None:
    sampler = _init_QMCSampler_without_warnings()
    with patch.object(sampler._independent_sampler, "reseed_rng") as mock_reseed_rng, patch.object(
        sampler, "_log_incomplete_reseeding"
    ) as mock_log_reseed:
        sampler.reseed_rng()
    mock_reseed_rng.assert_called_once()
    mock_log_reseed.assert_called_once()


def test_infer_relative_search_space() -> None:
    # in case no past trials
    # in case there is several trials
    # in case self._initial_trial exists.
    pass


def test_infer_initial_search_space() -> None:
    # without categorical dist?
    # can handle empty search space?
    pass


def test_sample_independent() -> None:
    # Relative sampling of `QMCSampler` does not support categorical distribution.
    # Thus, `independent_sampler.sample_independent` is called twice.
    independent_sampler = optuna.samplers.RandomSampler()

    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_sample_indep:

        sampler = _init_QMCSampler_without_warnings(independent_sampler=independent_sampler)
        study = optuna.create_study(sampler=sampler)
        study.optimize(lambda t: t.suggest_categorical("x", [1, 2]), n_trials=2)

    assert mock_sample_indep.call_count == 2


def test_log_independent_sampling() -> None:
    # Relative sampling of `QMCSampler` does not support categorical distribution.
    # Thus, `independent_sampler.sample_independent` is called twice.
    # '_log_independent_sampling is not called in the first trial so called once in total.
    with patch.object(optuna.samplers.QMCSampler, "_log_independent_sampling") as mock_log_indep:

        sampler = _init_QMCSampler_without_warnings()
        study = optuna.create_study(sampler=sampler)
        study.optimize(lambda t: t.suggest_categorical("x", [1, 2]), n_trials=2)

    mock_log_indep.called_once()


def test_sample_relative():
    # if empty search_space, return {}
    # else make sure that sample type, shape is OK.
    pass


@pytest.mark.parametrize("scramble", [True, False])
@pytest.mark.parametrize("qmc_type", ["sobol", "halton"])
def test_sample_relative_seeding(scramble, qmc_type):
    # seed = 12345
    # test if the samples taken by parallel workers are the same
    # as the ones taken by sequencial workers
    pass


@pytest.mark.parametrize("qmc_type", ["sobol", "halton", "non-qmc"])
def test_sample_qmc(qmc_type):
    # make sure that the qmc_engine._num_generated is consistent
    # make sure that behavior of cached engine is OK
    pass


def test_find_sample_id():
    # different hash for different config?
    # sequential execution and correct functions called?
    pass


def test_is_engine_cached():
    # change one of the condition of engine and see what happens.
    pass


def test_call_after_trial_of_base_sampler() -> None:
    sampler = _init_QMCSampler_without_warnings()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._independent_sampler, "after_trial", wraps=sampler._independent_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


# TODO(kstoneriv3): `QMCSampler` can be initialized without this wrapper
# after the experimental warning is removed.
def _init_QMCSampler_without_warnings(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.QMCSampler(**kwargs)
    return sampler
