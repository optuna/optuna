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


def test_reseed_rng():
    pass


def test_infer_relative_search_space():
    pass


def test_infer_initial_search_space():
    pass


def test_log_independent_sampling():
    # also test sample_independent
    pass


def test_log_asyncronous_seeding():
    # also test __init__
    pass


def test_sample_relative():
    pass


def test_sample_qmc():
    pass


def test_find_sample_id():
    pass


def test_is_engine_cached():
    pass


def test_parallel_workers():
    # test if the samples taken by parallel workers are the same
    # as the ones taken by sequencial workers
    pass
