from collections import OrderedDict
from typing import Dict

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._base import _get_search_space
from optuna.importance._base import _get_trial_data
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


def _get_configuration_space(search_space: Dict[str, BaseDistribution]) -> ConfigurationSpace:
    config_space = ConfigurationSpace()

    for name, distribution in search_space.items():
        config_space.add_hyperparameter(_distribution_to_hyperparameter(name, distribution))

    return config_space


def _get_evaluator(study: Study) -> fANOVA:
    # TODO(hvy): Set cutoff based on minimization/maximization of study is needed.
    # https://github.com/automl/ParameterImportance/blob/master/pimp/evaluator/fanova.py#L44

    search_space = _get_search_space(study)
    config_space = _get_configuration_space(search_space)
    params, values = _get_trial_data(study, search_space)

    assert len(search_space) == len(config_space.get_hyperparameters())
    assert all(name == hyperparameter_name for name, hyperparameter_name in zip(
        search_space.keys(), config_space.get_hyperparameter_names()))

    evaluator = fANOVA(X=params, Y=values, config_space=config_space, seed=0)

    return evaluator


class FanovaImportanceEvaluator(BaseImportanceEvaluator):

    def get_param_importance(self, study: Study) -> Dict[str, float]:
        _check_fanova_availability()

        evaluator = _get_evaluator(study)

        individual_importances = {}
        for i, name in enumerate(evaluator.cs.get_hyperparameter_names()):
            imp = evaluator.quantify_importance((i,))
            imp = imp[(i,)]['individual importance']
            individual_importances[name] = imp

        param_importances = OrderedDict(sorted(individual_importances.items(), key=lambda x: x[1]))

        return param_importances


def _check_fanova_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'fanova is not available. Please install automl/fanova to use this feature. '
            'For further information, please refer to the installation guide of automl/fanova. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
