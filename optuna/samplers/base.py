import abc
import six

import optuna
from optuna.distributions import BaseDistribution  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from optuna.study import RunningStudy  # NOQA
    from typing import Dict  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers."""

    def define_relative_search_space(self, trial):
        # type: (FrozenTrial) -> Dict[str, BaseDistribution]
        """Define the search space used in the target trial.

        The search space defined by this method will be used as an argument of
        :func:`optuna.samplers.BaseSampler.sample_relative` method.

        If a parameter that is not contained in the space is requested in an object function,
        the value will be sampled by using :func:`optuna.samplers.BaseSampler.sample_independent`
        method.

        The default implementation returns the result of
        :func:`optuna.study.RunningStudy.full_search_space` call.

        Args:
            trial:
                Target trial object.

        Returns:
            A dictionary containing the parameter names and parameter's distributions.

        """

        return self.study.full_search_space

    def sample_relative(self, trial, search_space):
        # type: (FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]
        """Sample parameters based on the previous trials and the given search space.

        This method is called once just after each trial has started.

        Note that this method is not supposed to be called by library users. Instead,
        :class:`optuna.trial.Trial` provides user interfaces to sample parameters in an objective
        function.

        The default implementation of this method always returns :obj:`{}`.

        Args:
            trial:
                Target trial object.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A float value in the internal representation of Optuna.

        """

        return {}

    def sample_independent(self, trial, param_name, param_distribution):
        # type: (FrozenTrial, str, BaseDistribution) -> float
        """Sample a parameter based on the previous trials and the given distribution.

        The method is only called for the parameters that have not been contained in the dictionary
        returned by :func:`optuna.samplers.BaseSampler.sample_relative` method.

        Note that this method is not supposed to be called by library users. Instead,
        :class:`optuna.trial.Trial` provides user interfaces to sample parameters in an objective
        function.

        The default implementation of this method uses
        :class:`optuna.samplers.TPESampler` for the sampling.

        Args:
            trial:
                Target trial object.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A float value in the internal representation of Optuna.

        """

        tpe = optuna.samplers.TPESampler()
        tpe._set_study(self.study)
        return tpe.sample_independent(trial, param_name, param_distribution)

    @property
    def study(self):
        # type: () -> RunningStudy
        """Return the target study."""

        if not hasattr(self, '_study'):
            raise RuntimeError('`_study` field has not yet been set.')

        return self._study

    def _set_study(self, study):
        # type: (RunningStudy) -> None

        self._study = study
