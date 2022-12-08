import abc
from typing import Any
from typing import Dict

from optuna import multi_objective
from optuna._deprecated import deprecated_class
from optuna.distributions import BaseDistribution


@deprecated_class("2.4.0", "4.0.0")
class BaseMultiObjectiveSampler(abc.ABC):
    """Base class for multi-objective samplers.

    The abstract methods of this class are the same as ones defined by
    :class:`~optuna.samplers.BaseSampler` except for taking
    multi-objective versions of study and trial instances as the arguments.

    """

    @abc.abstractmethod
    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        """Infer the search space that will be used by relative sampling in the target trial.

        This method is called right before
        :func:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler.sample_relative`
        method, and the search space returned by this method is passed to it. The parameters not
        contained in the search space will be sampled by using
        :func:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler.sample_independent`
        method.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        .. seealso::
            Please refer to :func:`~optuna.samplers.intersection_search_space` as an
            implementation of
            :func:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler.infer_relative_search_space`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        """Sample parameters in a given search space.

        This method is called once at the beginning of each trial, i.e., right before the
        evaluation of the objective function. This method is suitable for sampling algorithms
        that use the relationship between parameters.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
            search_space:
                The search space returned by
                :func:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler.infer_relative_search_space`.

        Returns:
            A dictionary containing the parameter names and the values.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a parameter for a given distribution.

        This method is called only for the parameters not contained in the search space returned
        by :func:`~optuna.multi_objective.samplers.MultiObjectiveBaseSampler.sample_relative`
        method. This method is suitable for sampling algorithms that do not use the relationship
        between parameters such as random sampling.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A parameter value.

        """

        raise NotImplementedError

    def reseed_rng(self) -> None:
        """Reseed sampler's random number generator.

        This method is called by the :class:`~optuna.multi_objective.study.MultiObjectiveStudy`
        instance if trials are executed in parallel with the option ``n_jobs>1``. In that case, the
        sampler instance will be replicated including the state of the random number generator, and
        they may suggest the same values. To prevent this issue, this method assigns a different
        seed to each random number generator.
        """

        pass
