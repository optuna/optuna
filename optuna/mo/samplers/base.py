import abc
from typing import Any
from typing import Dict

from optuna import mo
from optuna.distributions import BaseDistribution


class BaseMoSampler(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def infer_relative_search_space(
        self, study: "mo.study.MoStudy", trial: "mo.trial.FrozenMoTrial"
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(
        self,
        study: "mo.study.MoStudy",
        trial: "mo.trial.FrozenMoTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(
        self,
        study: "mo.study.MoStudy",
        trial: "mo.trial.FrozenMoTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        raise NotImplementedError
