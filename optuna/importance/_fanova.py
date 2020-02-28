from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna import logging
from optuna.samplers import intersection_search_space
from optuna.structs import TrialState
from optuna.study import Study

try:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import CategoricalHyperparameter
    from ConfigSpace.hyperparameters import Hyperparameter
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter
    from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
    from fanova import fANOVA

    _available = True
except ImportError as e:
    ConfigurationSpace = None
    Hyperparameter = None
    fANOVA = None

    _import_error = e
    _available = False

_logger = logging.get_logger(__name__)


def _get_distributions(study: Study) -> Dict[str, BaseDistribution]:
    # Return an ordered dict, ordered by parameter names lexicographically.
    # It must be sorted because the corresponding `ConfigurationSpace` container depends on the
    # order and also sorts.

    # Condtional hyperparameters are excluded here since we are taking the intersection.
    distributions = intersection_search_space(study)
    distributions = OrderedDict(sorted(distributions.items(), key=lambda x: x[0]))

    return distributions


def _get_configuration_space(distributions: Dict[str, BaseDistribution]) -> ConfigurationSpace:
    config_space = ConfigurationSpace()

    for name, distribution in distributions.items():
        config_space.add_hyperparameter(_distribution_to_hyperparameter(name, distribution))

    return config_space


def _distribution_to_hyperparameter(name: str, distribution: BaseDistribution) -> Hyperparameter:
    d = distribution

    if isinstance(d, UniformDistribution):
        hp = UniformFloatHyperparameter(name, lower=d.low, upper=d.high)
    elif isinstance(d, LogUniformDistribution):
        hp = UniformFloatHyperparameter(name, lower=d.low, upper=d.high, log=True)
    elif isinstance(d, DiscreteUniformDistribution):
        hp = UniformFloatHyperparameter(name, lower=d.low, upper=d.high, q=d.q)
    elif isinstance(d, IntUniformDistribution):
        hp = UniformIntegerHyperparameter(name, lower=d.low, upper=d.high)
    elif isinstance(d, CategoricalDistribution):
        hp = CategoricalHyperparameter(
            name, choices=[d.to_internal_repr(c) for c in d.choices])
    else:
        distribution_list = [
            UniformDistribution.__name__,
            LogUniformDistribution.__name__,
            DiscreteUniformDistribution.__name__,
            IntUniformDistribution.__name__,
            CategoricalDistribution.__name__
        ]
        raise NotImplementedError('The distribution {} is not implemented. '
                                  'The parameter distribution should be one of the {}'.format(
                                      d, distribution_list))
    return hp


def _get_evaluator(study: Study) -> fANOVA:
    # TODO(hvy): Support conditional hyperparameter.
    # TODO(hvy): Set cutoff based on minimization/maximization of study.
    # https://github.com/automl/ParameterImportance/blob/master/pimp/evaluator/fanova.py#L44

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    if len(trials) == 0:
        raise ValueError('Study must contain completed trials.')

    distributions = _get_distributions(study)
    config_space = _get_configuration_space(distributions)

    assert len(distributions) > 0
    assert len(distributions) == len(config_space.get_hyperparameters())
    for distribution_name, hyperparameter_name in zip(
            distributions.keys(), config_space.get_hyperparameter_names()):
        assert distribution_name == hyperparameter_name

    _logger.debug('Sorted parameter names {}.'.format(config_space.get_hyperparameter_names()))

    n_trials = len(trials)
    n_params = len(distributions)

    params = np.empty((n_trials, n_params), dtype=np.float64)
    values = np.empty((n_trials,), dtype=np.float64)

    for i, trial in enumerate(trials):
        trial_params = trial.params

        for j, (name, distribution) in enumerate(distributions.items()):
            # TODO(hvy): Impute missing values.
            if name not in trial_params:
                raise RuntimeError(
                    'Parameter \'{}\' was never suggested in trial number {}.'.format(
                        name, trial.number))

            param = trial_params[name]
            if isinstance(distribution, CategoricalDistribution):
                param = distribution.to_internal_repr(param)
            else:
                param = config_space.get_hyperparameter(name)._transform(param)
            params[i, j] = param

        values[i] = trial.value

    evaluator = fANOVA(
        X=params, Y=values, config_space=config_space, seed=0)

    return evaluator


class _Fanova(object):

    def __init__(self, study: Study, params: Optional[List[str]] = None) -> None:
        _check_fanova_availability()

        if params is not None:
            raise NotImplementedError

        self._evaluator = _get_evaluator(study)

    def get_param_importance(self) -> Dict[str, float]:
        importances = OrderedDict()
        evaluator = self._evaluator

        for i, name in enumerate(evaluator.cs.get_hyperparameter_names()):
            imp = evaluator.quantify_importance((i,))
            imp = imp[(i,)]['individual importance']

            _logger.debug('Individual importance for parameter {} is {}.'.format(name, imp))

            importances[name] = imp

        return importances


def _check_fanova_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'fanova is not available. Please install automl/fanova to use this feature. '
            'For further information, please refer to the installation guide of automl/fanova. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
