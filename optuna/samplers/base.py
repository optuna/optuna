import abc
import six

from optuna import types

if types.TYPE_CHECKING:
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers."""

    @abc.abstractmethod
    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]
        """Infer the search space that will be used by the target trial.

        The search space returned by this method will be used as an argument of
        :func:`optuna.samplers.BaseSampler.sample_relative` method.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]
        """Sample parameters based on the previous trials and the given search space.

        This method is called once just after each trial has started.

        If a parameter that is not contained in the returned dictionary is
        requested in an objective function, the value will be sampled by using
        :func:`optuna.samplers.BaseSampler.sample_independent` method.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
            search_space:
                The search space returned by
                :func:`optuna.samplers.BaseSampler.infer_relative_search_space`.

        Returns:
            A dictionary containing the parameter names and the values that are the
            internal representations of Optuna.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> float
        """Sample a parameter based on the previous trials and the given distribution.

        The method is only called for the parameters that have not been contained in the dictionary
        returned by :func:`optuna.samplers.BaseSampler.sample_relative` method.

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
            A float value in the internal representation of Optuna.

        """

        raise NotImplementedError
