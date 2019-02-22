import abc
import six

from optuna.distributions import BaseDistribution  # NOQA
from optuna.storages.base import BaseStorage  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers."""

    @abc.abstractmethod
    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> float
        """Sample a parameter based on the previous trials and the given distribution.

        Note that this method is not supposed to be called by library users. Instead,
        :class:`optuna.trial.Trial` provides user interfaces to sample parameters in an objective
        function.

        Args:
            storage:
                Storage object.
            study_id:
                Identifier of the target study.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A float value in the internal representation of Optuna.

        """

        raise NotImplementedError
