import abc
from typing import Any
from typing import Dict

from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import multi_objective


@experimental("1.4.0")
class BaseMultiObjectiveSampler(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        raise NotImplementedError
