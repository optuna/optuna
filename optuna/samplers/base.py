import abc
import six

from optuna import types

if types.TYPE_CHECKING:
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import RunningStudy  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers."""

    @abc.abstractmethod
    def define_relative_search_space(self, study, trial):
        # type: (RunningStudy, FrozenTrial) -> Dict[str, BaseDistribution]
        """Define the search space used in the target trial.

        The search space defined by this method will be used as an argument of
        :func:`optuna.samplers.BaseSampler.sample_relative` method.

        If a parameter that is not contained in the space is requested in an object function,
        the value will be sampled by using :func:`optuna.samplers.BaseSampler.sample_independent`
        method.

        The default implementation returns the result of
        :func:`optuna.study.RunningStudy.full_search_space` call.

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
        # type: (RunningStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]
        """Sample parameters based on the previous trials and the given search space.

        This method is called once just after each trial has started.

        The default implementation of this method always returns :obj:`{}`.

        Args:
            study:
                Target study object.
            trial:
                Target trial object.
            search_space:
                The search space returned by
                :func:`optuna.samplers.BaseSampler.define_relative_search_space`.

        Returns:
            A dictionary containing the parameter names and the values that are the
            internal representations of Optuna.

        """

        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (RunningStudy, FrozenTrial, str, BaseDistribution) -> float
        """Sample a parameter based on the previous trials and the given distribution.

        The method is only called for the parameters that have not been contained in the dictionary
        returned by :func:`optuna.samplers.BaseSampler.sample_relative` method.

        The default implementation of this method uses
        :class:`optuna.samplers.TPESampler` for the sampling.

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
