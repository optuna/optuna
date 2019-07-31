import abc
import six

from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers.

    Optuna combines two types of sampling strategies, which are called *relative sampling* and
    *independent sampling*.

    *The relative sampling* determines values of multiple parameters simultaneously so that
    sampling algorithms can use relationship between parameters (e.g., correlation).
    Target parameters of the relative sampling are described in a relative search space, which
    is determined by :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

    *The independent sampling* determines a value of a single parameter without considering any
    relationship between parameters. Target parameters of the independent sampling are the
    parameters not described in the relative search space.
    """

    @abc.abstractmethod
    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]
        """Infer the search space that will be used by relative sampling in the target trial.

        This method is called right before :func:`~optuna.samplers.BaseSampler.sample_relative`
        method, and the search space returned by this method is pass to it. The parameters not
        contained in the search space will be sampled by using
        :func:`~optuna.samplers.BaseSampler.sample_independent` method.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        .. seealso::
            Please refer to :func:`~optuna.samplers.intersection_search_space` as an
            implementation of :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]
        """Sample parameters in a given search space.

        This method is called once at the beginning of each trial, i.e., right before the
        evaluation of the objective function. This method is suitable for sampling algorithms
        that use relationship between parameters such as Gaussian Process and CMA-ES.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
            search_space:
                The search space returned by
                :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

        Returns:
            A dictionary containing the parameter names and the values.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> Any
        """Sample a parameter for a given distribution.

        This method is called only for the parameters not contained in the search space returned
        by :func:`~optuna.samplers.BaseSampler.sample_relative` method. This method is suitable
        for sampling algorithms that do not use relationship between parameters such as random
        sampling and TPE.

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
