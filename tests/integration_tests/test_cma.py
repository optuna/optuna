import math
from unittest.mock import call
from unittest.mock import patch

import cma
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.integration.cma import _Optimizer
from optuna.study import StudyDirection
from optuna.testing.distribution import UnsupportedDistribution
from optuna.testing.sampler import DeterministicRelativeSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

if optuna.type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA


class TestCmaEsSampler(object):
    @staticmethod
    def test_init_cma_opts():
        # type: () -> None

        sampler = optuna.integration.CmaEsSampler(
            x0={"x": 0, "y": 0},
            sigma0=0.1,
            cma_stds={"x": 1, "y": 1},
            seed=1,
            cma_opts={"popsize": 5},
            independent_sampler=DeterministicRelativeSampler({}, {}),
        )
        study = optuna.create_study(sampler=sampler)

        with patch("optuna.integration.cma._Optimizer") as mock_obj:
            mock_obj.ask.return_value = {"x": -1, "y": -1}
            study.optimize(
                lambda t: t.suggest_int("x", -1, 1) + t.suggest_int("y", -1, 1), n_trials=2
            )
            assert mock_obj.mock_calls[0] == call(
                {
                    "x": IntUniformDistribution(low=-1, high=1),
                    "y": IntUniformDistribution(low=-1, high=1),
                },
                {"x": 0, "y": 0},
                0.1,
                {"x": 1, "y": 1},
                {"popsize": 5, "seed": 1, "verbose": -2},
            )

    @staticmethod
    def test_init_default_values():
        # type: () -> None

        sampler = optuna.integration.CmaEsSampler()
        seed = sampler._cma_opts.get("seed")
        assert isinstance(seed, int)
        assert 0 < seed

        assert isinstance(sampler._independent_sampler, optuna.samplers.RandomSampler)

    @staticmethod
    def test_reseed_rng() -> None:
        sampler = optuna.integration.CmaEsSampler()
        original_seed = sampler._cma_opts["seed"]
        sampler._independent_sampler.reseed_rng()

        with patch.object(
            sampler._independent_sampler,
            "reseed_rng",
            wraps=sampler._independent_sampler.reseed_rng,
        ) as mock_object:
            sampler.reseed_rng()
            assert mock_object.call_count == 1
            assert original_seed != sampler._cma_opts["seed"]

    @staticmethod
    def test_infer_relative_search_space_1d():
        # type: () -> None

        sampler = optuna.integration.CmaEsSampler()
        study = optuna.create_study(sampler=sampler)

        # The distribution has only one candidate.
        study.optimize(lambda t: t.suggest_int("x", 1, 1), n_trials=1)
        assert sampler.infer_relative_search_space(study, study.best_trial) == {}

    @staticmethod
    def test_sample_relative_1d():
        # type: () -> None

        independent_sampler = DeterministicRelativeSampler({}, {})
        sampler = optuna.integration.CmaEsSampler(independent_sampler=independent_sampler)
        study = optuna.create_study(sampler=sampler)

        # If search space is one dimensional, the independent sampler is always used.
        with patch.object(
            independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
        ) as mock_object:
            study.optimize(lambda t: t.suggest_int("x", -1, 1), n_trials=2)
            assert mock_object.call_count == 2

    @staticmethod
    def test_sample_relative_n_startup_trials():
        # type: () -> None

        independent_sampler = DeterministicRelativeSampler({}, {})
        sampler = optuna.integration.CmaEsSampler(
            n_startup_trials=2, independent_sampler=independent_sampler
        )
        study = optuna.create_study(sampler=sampler)

        # The independent sampler is used for Trial#0 and Trial#1.
        # The CMA-ES is used for Trial#2.
        with patch.object(
            independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
        ) as mock_independent, patch.object(
            sampler, "sample_relative", wraps=sampler.sample_relative
        ) as mock_relative:
            study.optimize(
                lambda t: t.suggest_int("x", -1, 1) + t.suggest_int("y", -1, 1), n_trials=3
            )
            assert mock_independent.call_count == 4  # The objective function has two parameters.
            assert mock_relative.call_count == 3

    @staticmethod
    def test_initialize_x0_with_unsupported_distribution():
        # type: () -> None

        with pytest.raises(NotImplementedError):
            optuna.integration.CmaEsSampler._initialize_x0({"x": UnsupportedDistribution()})

    @staticmethod
    def test_initialize_sigma0_with_unsupported_distribution():
        # type: () -> None

        with pytest.raises(NotImplementedError):
            optuna.integration.CmaEsSampler._initialize_sigma0({"x": UnsupportedDistribution()})


class TestOptimizer(object):
    @staticmethod
    @pytest.fixture
    def search_space():
        # type: () -> Dict[str, BaseDistribution]

        return {
            "c": CategoricalDistribution(("a", "b")),
            "d": DiscreteUniformDistribution(-1, 9, 2),
            "i": IntUniformDistribution(-1, 1),
            "ii": IntUniformDistribution(-1, 3, 2),
            "l": LogUniformDistribution(0.001, 0.1),
            "u": UniformDistribution(-2, 2),
        }

    @staticmethod
    @pytest.fixture
    def x0():
        # type: () -> Dict[str, Any]

        return {
            "c": "a",
            "d": -1,
            "i": -1,
            "ii": -1,
            "l": 0.001,
            "u": -2,
        }

    @staticmethod
    def test_init(search_space, x0):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        with patch("cma.CMAEvolutionStrategy") as mock_obj:
            optuna.integration.cma._Optimizer(
                search_space, x0, 0.2, None, {"popsize": 5, "seed": 1}
            )
            assert mock_obj.mock_calls[0] == call(
                [0, 0, -1, -1, math.log(0.001), -2],
                0.2,
                {
                    "BoundaryHandler": cma.BoundTransform,
                    "bounds": [
                        [-0.5, -1.0, -1.5, -1.5, math.log(0.001), -2],
                        [1.5, 11.0, 1.5, 3.5, math.log(0.1), 2],
                    ],
                    "popsize": 5,
                    "seed": 1,
                },
            )

    @staticmethod
    def test_init_with_unsupported_distribution():
        # type: () -> None

        with pytest.raises(NotImplementedError):
            optuna.integration.cma._Optimizer(
                {"x": UnsupportedDistribution()}, {"x": 0}, 0.2, None, {}
            )

    @staticmethod
    @pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
    def test_tell(search_space, x0, direction):
        # type: (Dict[str, BaseDistribution], Dict[str, Any], StudyDirection) -> None

        optimizer = optuna.integration.cma._Optimizer(
            search_space, x0, 0.2, None, {"popsize": 3, "seed": 1}
        )

        trials = [_create_frozen_trial(x0, search_space)]
        assert -1 == optimizer.tell(trials, direction)

        trials = [_create_frozen_trial(x0, search_space, number=i) for i in range(3)]
        assert 2 == optimizer.tell(trials, direction)

    @staticmethod
    @pytest.mark.parametrize("state", [TrialState.FAIL, TrialState.RUNNING, TrialState.PRUNED])
    def test_tell_filter_by_state(search_space, x0, state):
        # type: (Dict[str, BaseDistribution], Dict[str, Any], TrialState) -> None

        optimizer = optuna.integration.cma._Optimizer(
            search_space, x0, 0.2, None, {"popsize": 2, "seed": 1}
        )

        trials = [_create_frozen_trial(x0, search_space)]
        trials.append(_create_frozen_trial(x0, search_space, state, len(trials)))
        assert -1 == optimizer.tell(trials, StudyDirection.MINIMIZE)

    @staticmethod
    def test_tell_filter_by_distribution(search_space, x0):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        optimizer = optuna.integration.cma._Optimizer(
            search_space, x0, 0.2, None, {"popsize": 2, "seed": 1}
        )

        trials = [_create_frozen_trial(x0, search_space)]
        distributions = trials[0].distributions.copy()
        distributions["additional"] = UniformDistribution(0, 100)
        trials.append(_create_frozen_trial(x0, distributions, number=1))
        assert 1 == optimizer.tell(trials, StudyDirection.MINIMIZE)

    @staticmethod
    def test_ask(search_space, x0):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        trials = [_create_frozen_trial(x0, search_space, number=i) for i in range(3)]

        # Create 0-th individual.
        optimizer = _Optimizer(search_space, x0, 0.2, None, {"popsize": 3, "seed": 1})
        last_told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        params0 = optimizer.ask(trials, last_told)

        # Ignore parameters with incompatible distributions and create new individual.
        optimizer = _Optimizer(search_space, x0, 0.2, None, {"popsize": 3, "seed": 1})
        last_told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        distributions = trials[0].distributions.copy()
        distributions["additional"] = UniformDistribution(0, 100)
        trials.append(_create_frozen_trial(x0, distributions, number=len(trials)))
        params1 = optimizer.ask(trials, last_told)

        assert params0 != params1
        assert "additional" not in params1

        # Create first individual.
        optimizer = _Optimizer(search_space, x0, 0.2, None, {"popsize": 3, "seed": 1})
        trials.append(_create_frozen_trial(x0, search_space, number=len(trials)))
        last_told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        params2 = optimizer.ask(trials, last_told)

        assert params0 != params2

        optimizer = _Optimizer(search_space, x0, 0.2, None, {"popsize": 3, "seed": 1})
        last_told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        # Other worker adds three trials.
        for _ in range(3):
            trials.append(_create_frozen_trial(x0, search_space, number=len(trials)))
        params3 = optimizer.ask(trials, last_told)

        assert params0 != params3
        assert params2 != params3

    @staticmethod
    def test_is_compatible(search_space, x0):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        optimizer = optuna.integration.cma._Optimizer(search_space, x0, 0.1, None, {})

        # Compatible.
        trial = _create_frozen_trial(x0, search_space)
        assert optimizer._is_compatible(trial)

        # Compatible.
        trial = _create_frozen_trial(x0, dict(search_space, u=UniformDistribution(-10, 10)))
        assert optimizer._is_compatible(trial)

        # Compatible.
        trial = _create_frozen_trial(
            dict(x0, unknown=7), dict(search_space, unknown=UniformDistribution(0, 10))
        )
        assert optimizer._is_compatible(trial)

        # Incompatible ('u' doesn't exist).
        param = dict(x0)
        del param["u"]
        dist = dict(search_space)
        del dist["u"]
        trial = _create_frozen_trial(param, dist)
        assert not optimizer._is_compatible(trial)

        # Incompatible (the value of 'u' is out of range).
        trial = _create_frozen_trial(
            dict(x0, u=20), dict(search_space, u=UniformDistribution(-100, 100))
        )
        assert not optimizer._is_compatible(trial)

        # Error (different distribution class).
        trial = _create_frozen_trial(x0, dict(search_space, u=IntUniformDistribution(-2, 2)))
        with pytest.raises(ValueError):
            optimizer._is_compatible(trial)


def _create_frozen_trial(params, param_distributions, state=TrialState.COMPLETE, number=0):
    # type: (Dict[str, Any], Dict[str, BaseDistribution], TrialState, int) -> FrozenTrial

    return FrozenTrial(
        number=number,
        value=1.0,
        state=state,
        user_attrs={},
        system_attrs={},
        params=params,
        distributions=param_distributions,
        intermediate_values={},
        datetime_start=None,
        datetime_complete=None,
        trial_id=number,
    )
