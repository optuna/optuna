from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_study_data
from optuna.importance._base import BaseImportanceEvaluator
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


@experimental("1.3.0")
class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    """fANOVA parameter importance evaluator.

    .. note::

        Requires the `fanova <https://github.com/automl/fanova>`_ Python package.

    .. seealso::

        `An Efficient Approach for Assessing Hyperparameter Importance
        <http://proceedings.mlr.press/v32/hutter14.html>`_.
    """

    def __init__(self) -> None:
        _check_fanova_availability()

    def evaluate(self, study: Study, params: Optional[List[str]]) -> Dict[str, float]:
        distributions = _get_distributions(study, params)
        params_data, values_data = _get_study_data(study, distributions)

        evaluator = fANOVA(
            X=params_data,
            Y=values_data,
            config_space=_get_configuration_space(distributions),
            max_features=max(1, int(params_data.shape[1] * 0.7)),
        )

        individual_importances = {}
        for i, name in enumerate(evaluator.cs.get_hyperparameter_names()):
            imp = evaluator.quantify_importance((i,))
            imp = imp[(i,)]["individual importance"]
            individual_importances[name] = imp

        tot_importance = sum(v for v in individual_importances.values())
        for name in individual_importances.keys():
            individual_importances[name] /= tot_importance

        param_importances = OrderedDict(
            reversed(
                sorted(
                    individual_importances.items(),
                    key=lambda name_and_importance: name_and_importance[1],
                )
            )
        )
        return param_importances


def _get_configuration_space(search_space: Dict[str, BaseDistribution]) -> ConfigurationSpace:
    config_space = ConfigurationSpace()

    for name, distribution in search_space.items():
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
        hp = UniformIntegerHyperparameter(name, lower=d.low, upper=d.high, q=d.step)
    elif isinstance(d, IntLogUniformDistribution):
        hp = UniformIntegerHyperparameter(name, lower=d.low, upper=d.high, q=d.step, log=True)
    elif isinstance(d, CategoricalDistribution):
        hp = CategoricalHyperparameter(name, choices=[d.to_internal_repr(c) for c in d.choices])
    else:
        distribution_list = [
            UniformDistribution.__name__,
            LogUniformDistribution.__name__,
            DiscreteUniformDistribution.__name__,
            IntUniformDistribution.__name__,
            IntLogUniformDistribution.__name__,
            CategoricalDistribution.__name__,
        ]
        raise NotImplementedError(
            "The distribution {} is not implemented. "
            "The parameter distribution should be one of the {}".format(d, distribution_list)
        )
    return hp


def _check_fanova_availability() -> None:
    if not _available:
        raise ImportError(
            "fanova is not available. Please install automl/fanova to use this feature. "
            "For further information, please refer to the installation guide of automl/fanova. "
            "(The actual import error is as follows: " + str(_import_error) + ")."
        )
