import cma
import math
from mock import call
from mock import patch
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.integration.cma import _Optimizer
from optuna.structs import FrozenTrial
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.testing.sampler import DeterministicRelativeSampler

if optuna.types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA


class TestCmaEsSampler(object):
    @staticmethod
    def test_init_cma_opts():
        # type: () -> None

        sampler = optuna.integration.CmaEsSampler(
            0.1,
            seed=1,
            cma_opts={'popsize': 5},
            independent_sampler=DeterministicRelativeSampler({}, {}))
        study = optuna.create_study(sampler=sampler)

        with patch('optuna.integration.cma._Optimizer') as mock_obj:
            mock_obj.ask.return_value = {'x': -1, 'y': -1}
            study.optimize(
                lambda t: t.suggest_int('x', -1, 1) + t.suggest_int('y', -1, 1), n_trials=2)
            assert mock_obj.mock_calls[0] == call({
                'x': IntUniformDistribution(low=-1, high=1),
                'y': IntUniformDistribution(low=-1, high=1)
            }, {
                'x': -1,
                'y': -1
            }, 0.1, {
                'popsize': 5,
                'seed': 1
            })

    @staticmethod
    def test_init_default_values():
        # type: () -> None

        sampler = optuna.integration.CmaEsSampler(0.1)
        seed = sampler._cma_opts.get('seed')
        assert isinstance(seed, int)
        assert 0 < seed

        assert isinstance(sampler._independent_sampler, optuna.samplers.RandomSampler)

    @staticmethod
    def test_infer_relative_search_space_single():
        # type: () -> None

        sampler = optuna.integration.CmaEsSampler(0.1)
        study = optuna.create_study(sampler=sampler)

        # The distribution has only one candidate.
        study.optimize(lambda t: t.suggest_int('x', 1, 1), n_trials=1)
        in_trial_study = optuna.study.InTrialStudy(study)
        assert sampler.infer_relative_search_space(in_trial_study, study.best_trial) == {}

    @staticmethod
    def test_sample_relative_1d():
        # type: () -> None

        independent_sampler = DeterministicRelativeSampler({}, {})
        sampler = optuna.integration.CmaEsSampler(0.1, independent_sampler=independent_sampler)
        study = optuna.create_study(sampler=sampler)

        # If search space is one dimensional, the independent sampler is always used.
        with patch.object(
                independent_sampler,
                'sample_independent',
                wraps=independent_sampler.sample_independent) as mock_object:
            study.optimize(lambda t: t.suggest_int('x', -1, 1), n_trials=2)
            assert mock_object.call_count == 2


class TestOptimizer(object):
    @staticmethod
    @pytest.fixture
    def search_space():
        # type: () -> Dict[str, BaseDistribution]

        return {
            'c': CategoricalDistribution(('a', 'b')),
            'd': DiscreteUniformDistribution(-1, 9, 2),
            'i': IntUniformDistribution(-1, 1),
            'l': LogUniformDistribution(0.001, 0.1),
            'u': UniformDistribution(-2, 2),
        }

    @staticmethod
    @pytest.fixture
    def initial_params():
        # type: () -> Dict[str, Any]

        return {
            'c': 'a',
            'd': -1,
            'i': -1,
            'l': 0.001,
            'u': -2,
        }

    @staticmethod
    def test_init(search_space, initial_params):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        with patch('cma.CMAEvolutionStrategy') as mock_obj:
            optuna.integration.cma._Optimizer(search_space, initial_params, 0.2, {
                'popsize': 4,
                'seed': 1
            })
            assert mock_obj.mock_calls[0] == call(
                [0, 0, -1, math.log(0.001), -2], 0.2, {
                    'BoundaryHandler':
                    cma.BoundTransform,
                    'bounds': [[-0.5, -1.0, -1.5, math.log(0.001), -2],
                               [1.5, 11.0, 1.5, math.log(0.1), 2]],
                    'popsize':
                    4,
                    'seed':
                    1
                })

    @staticmethod
    @pytest.mark.parametrize('direction', [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
    def test_tell(search_space, initial_params, direction):
        # type: (Dict[str, BaseDistribution], Dict[str, Any], StudyDirection) -> None

        optimizer = optuna.integration.cma._Optimizer(search_space, initial_params, 0.2, {
            'popsize': 3,
            'seed': 1
        })

        trials = [_create_frozen_trial(initial_params, search_space)]
        assert 0 == optimizer.tell(trials, direction)

        trials = [_create_frozen_trial(initial_params, search_space) for _ in range(3)]
        assert 3 == optimizer.tell(trials, direction)

    @staticmethod
    @pytest.mark.parametrize('state', [TrialState.FAIL, TrialState.RUNNING, TrialState.PRUNED])
    def test_tell_filter_by_state(search_space, initial_params, state):
        # type: (Dict[str, BaseDistribution], Dict[str, Any], TrialState) -> None

        optimizer = optuna.integration.cma._Optimizer(search_space, initial_params, 0.2, {
            'popsize': 2,
            'seed': 1
        })

        trials = [_create_frozen_trial(initial_params, search_space)]
        trials.append(trials[0]._replace(state=state))
        assert 0 == optimizer.tell(trials, StudyDirection.MINIMIZE)

    @staticmethod
    def test_tell_filter_by_distribution(search_space, initial_params):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        optimizer = optuna.integration.cma._Optimizer(search_space, initial_params, 0.2, {
            'popsize': 2,
            'seed': 1
        })

        trials = [_create_frozen_trial(initial_params, search_space)]
        distributions = trials[0].distributions.copy()
        distributions['additional'] = UniformDistribution(0, 100)
        trials.append(trials[0]._replace(distributions=distributions))
        assert 0 == optimizer.tell(trials, StudyDirection.MINIMIZE)

    @staticmethod
    def test_ask(search_space, initial_params):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        trials = [_create_frozen_trial(initial_params, search_space) for _ in range(3)]

        # Create 0-th individual.
        optimizer = _Optimizer(search_space, initial_params, 0.2, {'popsize': 3, 'seed': 1})
        told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        params0 = optimizer.ask(trials, told)

        # Ignore incompatible trial and create 0-th individual again.
        optimizer = _Optimizer(search_space, initial_params, 0.2, {'popsize': 3, 'seed': 1})
        told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        distributions = trials[0].distributions.copy()
        distributions['additional'] = UniformDistribution(0, 100)
        trials.append(trials[0]._replace(distributions=distributions))
        params1 = optimizer.ask(trials, told)

        assert params0 == params1

        # Create first individual.
        optimizer = _Optimizer(search_space, initial_params, 0.2, {'popsize': 3, 'seed': 1})
        trials.append(trials[0])
        told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        params2 = optimizer.ask(trials, told)

        assert params0 != params2

        optimizer = _Optimizer(search_space, initial_params, 0.2, {'popsize': 3, 'seed': 1})
        told = optimizer.tell(trials, StudyDirection.MINIMIZE)
        # Other worker adds three trials.
        for _ in range(3):
            trials.append(trials[0])
        params3 = optimizer.ask(trials, told)

        assert params0 != params3
        assert params2 != params3

    @staticmethod
    def test_n_target_trials():
        # type: () -> None

        pass

    @staticmethod
    def test_to_cma_params():
        # type: () -> None

        pass

    @staticmethod
    def test_to_optuna_params():
        # type: () -> None

        pass


def _create_frozen_trial(params, param_distributions):
    # type: (Dict[str, Any], Dict[str, BaseDistribution]) -> FrozenTrial

    params_in_internal_repr = {}
    for param_name, param_value in params.items():
        params_in_internal_repr[param_name] = param_distributions[param_name].to_internal_repr(
            param_value)

    return FrozenTrial(
        number=0,
        value=1.,
        state=optuna.structs.TrialState.COMPLETE,
        user_attrs={},
        system_attrs={},
        params=params,
        params_in_internal_repr=params_in_internal_repr,
        distributions=param_distributions,
        intermediate_values={},
        datetime_start=None,
        datetime_complete=None,
        trial_id=0,
    )
