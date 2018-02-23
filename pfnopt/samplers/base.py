import abc
import six

from pfnopt.distributions import BaseDistribution  # NOQA
from pfnopt.storage.base import BaseStorage  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):

    @abc.abstractmethod
    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> float
        raise NotImplementedError
